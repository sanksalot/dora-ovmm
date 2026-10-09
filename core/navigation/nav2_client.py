"""Map-frame Nav2 goals with path checks and cancellation before retrying."""

import math
import time
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from core.utils import events
from core.utils.transforms import matrix_from_transform

FRAME = "map"
BASE = "base_footprint"
# Use the single configured goal checker; the stock tree sends no checker ID.
# Its tolerances can be tightened temporarily for precise parking.
CONTROLLER = "/controller_server"
GOAL_CHECKER = "general_goal_checker"
# HSRC head joint limits. The pan range is asymmetric: about 100 deg right,
# 220 deg left. _move_head clamps into these rather than refusing to aim, so a
# target just outside still gets the closest look the joints allow.
PAN_LIMITS, TILT_LIMITS = (-3.84, 1.75), (-1.57, 0.52)


def _pose(x, y, yaw):
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError("Navigation goals must be finite")
    pose = PoseStamped()
    pose.header.frame_id = FRAME
    pose.pose.position.x, pose.pose.position.y = (float(x), float(y))
    # Encode the planar heading as a quaternion around the vertical axis.
    pose.pose.orientation.z = math.sin(yaw / 2)
    pose.pose.orientation.w = math.cos(yaw / 2)
    return pose


class Navigator(Node):

    def __init__(self, use_sim_time=True):
        super().__init__(
            "scene_graph_navigator",
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        # Create separate clients for checking paths and executing navigation goals.
        self.planner = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.driver = ActionClient(self, NavigateToPose, "navigate_to_pose")
        # Kept for the life of the node, as pick.py keeps its clients: a fresh client
        # per aim lost 3 of 8 head goals on the robot, one kept client 0 of 8.
        self.head = ActionClient(self, FollowJointTrajectory,
                                 "/head_trajectory_controller/follow_joint_trajectory")
        # Same fault, same cure: explore's torso lift opened a fresh client per view
        # and lost its goal on the robot (2026-09-24 03:24, "goal acceptance timed out").
        self.arm = ActionClient(self, FollowJointTrajectory,
                                "/arm_trajectory_controller/follow_joint_trajectory")
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)

    def _lifecycle_call(self, service_type, service, request, timeout=15):
        # Bound both service discovery and response time during a navigation handoff.
        client = self.create_client(service_type, service)
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"{service} unavailable; motion handoff stopped")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done():
                client.remove_pending_request(future)
                raise RuntimeError(f"{service} timed out; motion handoff stopped")
            return future.result()
        finally:
            self.destroy_client(client)

    def navigation_state(self, name):
        from lifecycle_msgs.srv import GetState
        return self._lifecycle_call(
            GetState, f"/{name}/get_state", GetState.Request()).current_state.id

    def _is_running(self, name, timeout=2.0):
        """Whether the lifecycle node exists at all.

        The launch decides which nodes there are: nav2_real.launch.py starts no
        waypoint_follower, and a node that is not running cannot be driving the base."""
        from lifecycle_msgs.srv import GetState
        client = self.create_client(GetState, f"/{name}/get_state")
        try:
            return client.wait_for_service(timeout_sec=timeout)
        finally:
            self.destroy_client(client)

    def pause_navigation(self):
        """Deactivate Nav2 motion nodes before MoveIt takes control of the base."""
        from lifecycle_msgs.msg import State
        from nav2_msgs.srv import ManageLifecycleNodes
        response = self._lifecycle_call(
            ManageLifecycleNodes, "/lifecycle_manager_navigation/manage_nodes",
            ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.PAUSE), timeout=30)
        if not response.success:
            raise RuntimeError("Nav2 pause rejected; refusing MoveIt handoff")
        # Confirm every motion node that exists is inactive before handing over.
        for name in ("controller_server", "behavior_server", "bt_navigator", "waypoint_follower"):
            if not self._is_running(name):
                print(f"[HANDOFF] {name} is not running; nothing to hand over", flush=True)
                continue
            if self.navigation_state(name) != State.PRIMARY_STATE_INACTIVE:
                raise RuntimeError(f"{name} is not inactive; refusing MoveIt handoff")
        print("[HANDOFF] Nav2 motion nodes inactive; MoveIt owns base motion", flush=True)

    def resume_navigation_if_paused(self):
        """Resume a previously paused navigation stack for a new search."""
        from lifecycle_msgs.msg import State
        from nav2_msgs.srv import ManageLifecycleNodes
        # Resume only when the controller is paused; leave an already-running stack alone.
        if self.navigation_state("controller_server") != State.PRIMARY_STATE_INACTIVE:
            return
        response = self._lifecycle_call(
            ManageLifecycleNodes, "/lifecycle_manager_navigation/manage_nodes",
            ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.RESUME), timeout=30)
        if not response.success:
            raise RuntimeError("Nav2 resume rejected")
        print("[READY] Nav2 resumed for search", flush=True)

    def _run(self, client, goal, timeout, what="Nav2"):
        """Send one action goal and wait for its result.

        `what` names the action in the errors: this is shared by the Nav2 goals and
        the head, and a head aim that failed read as a Nav2 fault twice."""
        if not client.wait_for_server(timeout_sec=10):
            raise RuntimeError(f"{what} action server is unavailable")
        # Send the goal and wait for the server to accept or reject it.
        sent = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=10)
        if not sent.done():

            def cancel_late(future):
                handle = future.result()
                if handle is not None and handle.accepted:
                    handle.cancel_goal_async()

            # A delayed acceptance must not leave an untracked goal running.
            sent.add_done_callback(cancel_late)
            rclpy.spin_until_future_complete(self, sent, timeout_sec=10)
            raise RuntimeError(f"{what} goal acceptance timed out")
        handle = sent.result()
        if handle is None:
            raise RuntimeError(f"{what} returned no goal acknowledgement")
        if not handle.accepted:
            return None
        # Wait for completion; cancel the motion on timeout or interruption.
        finished = handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(self, finished, timeout_sec=timeout)
        except BaseException:
            self._cancel(handle, finished)
            raise
        if not finished.done():
            self._cancel(handle, finished)
            return None
        return finished.result()

    def _cancel(self, handle, finished):
        cancelled = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancelled, timeout_sec=5)
        rclpy.spin_until_future_complete(self, finished, timeout_sec=5)
        # Confirm the previous motion ended before allowing another goal.
        if not finished.done() or finished.exception() is not None:
            raise RuntimeError(
                "Navigation cancellation unconfirmed; refusing another goal"
            )

    def robot_pose(self, timeout=10, max_age=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.tf_buffer.can_transform(FRAME, BASE, Time()):
                continue
            try:
                stamped = self.tf_buffer.lookup_transform(FRAME, BASE, Time())
            except TransformException:
                continue
            age = (self.get_clock().now() - Time.from_msg(stamped.header.stamp)).nanoseconds / 1e9
            # Reject stale poses; allow small clock differences in either direction.
            if abs(age) <= max_age:
                transform = stamped.transform
                break
        else:
            raise RuntimeError("No fresh map-to-base TF; check localization and clocks")
        translation, quaternion = (transform.translation, transform.rotation)
        yaw = math.atan2(
            2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        return (translation.x, translation.y, yaw)

    def robot_xy(self):
        return self.robot_pose()[:2]

    def frame_transform(self, target_frame, source_frame, timeout=10):
        """(4, 4) target_from_source, waited for rather than assumed published."""
        # Wait for the requested frame relationship instead of assuming it is already published.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.tf_buffer.can_transform(target_frame, source_frame, Time()):
                continue
            try:
                stamped = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, Time()
                )
            except TransformException:
                continue
            return matrix_from_transform(stamped.transform)
        raise RuntimeError(
            f"No {target_frame} <- {source_frame} TF; check localization and clocks"
        )

    def reachable(self, x, y, yaw):
        if self._has_path(x, y, yaw):
            return True
        # As Nav2's own tree does when planning fails: marks the laser left behind
        # outlive what made them, and enough of them close a doorway for good.
        print("[NAV] no path; clearing the global costmap and planning once more", flush=True)
        self._lifecycle_call(ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap",
                             ClearEntireCostmap.Request())
        return self._has_path(x, y, yaw)

    def _has_path(self, x, y, yaw):
        goal = ComputePathToPose.Goal()
        goal.goal = _pose(x, y, yaw)
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.use_start = False
        outcome = self._run(self.planner, goal, 15)
        if outcome is None or outcome.status != GoalStatus.STATUS_SUCCEEDED:
            return False
        path = outcome.result.path
        if not path.poses or path.header.frame_id != FRAME:
            return False
        # Check that the returned path actually ends near the requested position.
        end = path.poses[-1].pose.position
        return math.hypot(end.x - x, end.y - y) <= 0.1

    def drive_to(self, x, y, yaw, timeout=300, tolerance=0.3, yaw_tolerance=0.3):
        """Drive and verify. Tolerances must stay at or above the goal checker's."""
        goal = NavigateToPose.Goal()
        goal.pose = _pose(x, y, yaw)
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        events.emit("drive", pose=[x, y, yaw], reached=None)
        outcome = self._run(self.driver, goal, timeout)
        reached = False
        if outcome is not None and outcome.status == GoalStatus.STATUS_SUCCEEDED:
            # Verify the measured arrival pose after Nav2 reports success.
            offset, error = self.residual(x, y, yaw)
            reached = offset <= tolerance and error <= yaw_tolerance
        events.emit("drive", pose=[x, y, yaw], reached=reached)
        return reached

    def residual(self, x, y, yaw):
        """How far the base actually is from a goal, as (metres, radians)."""
        current_x, current_y, heading = self.robot_pose()
        # Wrap heading error so crossing -pi/pi does not look like a full turn.
        error = abs(math.atan2(math.sin(heading - yaw), math.cos(heading - yaw)))
        return (math.hypot(current_x - x, current_y - y), error)

    def set_goal_tolerance(self, xy, yaw, timeout=10):
        """Retune the running goal checker; both tolerances are dynamic parameters."""
        client = self.create_client(SetParameters, f"{CONTROLLER}/set_parameters")
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"{CONTROLLER} parameter service is unavailable")
            request = SetParameters.Request()
            for name, value in (("xy_goal_tolerance", xy), ("yaw_goal_tolerance", yaw)):
                parameter = ParameterMsg(name=f"{GOAL_CHECKER}.{name}")
                parameter.value.type = ParameterType.PARAMETER_DOUBLE
                parameter.value.double_value = float(value)
                request.parameters.append(parameter)
            # Allow a small reverse correction during precise parking.
            reverse = ParameterMsg(name='FollowPath.vx_min')
            reverse.value.type = ParameterType.PARAMETER_DOUBLE
            reverse.value.double_value = -.08 if xy < .1 else 0.
            request.parameters.append(reverse)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done():
                raise RuntimeError("Setting the goal tolerance timed out")
            for result in future.result().results:
                # Driving on with an unknown tolerance would make arrival meaningless.
                if not result.successful:
                    raise RuntimeError(f"Goal tolerance rejected: {result.reason}")
        finally:
            self.destroy_client(client)

    def look_at(self, point):
        """Aim the HSRC head approximately at a map-frame point."""
        x, y, yaw = self.robot_pose()
        if not self.tf_buffer.can_transform(FRAME, "head_pan_link", Time()):
            raise RuntimeError("No head TF available")
        pivot = self.tf_buffer.lookup_transform(FRAME, "head_pan_link", Time()).transform.translation
        bearing = math.atan2(point[1] - pivot.y, point[0] - pivot.x) - yaw
        pan = math.atan2(math.sin(bearing), math.cos(bearing))
        # Try the equivalent negative angle when it fits the head's wider left range.
        if pan > 1.75 and pan-2*math.pi >= -3.84:
            pan -= 2*math.pi
        # HSRC RGB-D camera sits roughly 0.25 m above the head pivot.
        distance = math.hypot(point[0] - pivot.x, point[1] - pivot.y)
        # Aim vertically from the camera height toward the target point.
        tilt = math.atan2(point[2] - pivot.z - 0.248, distance)
        return self._move_head(pan, tilt)

    def look_home(self):
        """Centre the head. Nothing else resets it, so a mission that aimed at one
        place leaves the camera there for every later step and for the next run."""
        return self._move_head(0.0, 0.0)

    def _move_head(self, pan, tilt):
        pan = min(max(pan, PAN_LIMITS[0]), PAN_LIMITS[1])
        tilt = min(max(tilt, TILT_LIMITS[0]), TILT_LIMITS[1])
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["head_pan_joint", "head_tilt_joint"]
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[pan, tilt], time_from_start=Duration(sec=5))]
        try:
            result = self._run(self.head, goal, 45, "head trajectory")
            return (result is not None and result.status == GoalStatus.STATUS_SUCCEEDED
                    and result.result.error_code == 0)
        except RuntimeError as error:
            # actions.search already handles "could not aim from this pose" and moves
            # to the next one. The head only looks; nothing it does is unsafe to
            # abandon, so a failed aim must not end the mission the way a base goal
            # left running would.
            print(f"[HEAD] {error}; treating as could not aim", flush=True)
            return False
