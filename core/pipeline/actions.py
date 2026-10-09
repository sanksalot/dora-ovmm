"""The actions core.pipeline.mission_tree runs, built from core's tools: wait for the
stack, home the arm, search remembered or LLM-ranked locations with Nav2 and SAM3,
park where the arm can reach what was found, and pick it up."""

import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from lifecycle_msgs.srv import GetState
from moveit_msgs.srv import GetPlanningScene
from moveit_task_constructor_msgs.action import ExecuteTaskSolution
from rclpy.action import ActionClient
from tmc_manipulation_msgs.srv import SolveIkWithCollision
from trajectory_msgs.msg import JointTrajectoryPoint
from core.navigation import base_placement, graspable, standoff
from core.perception import pointcloud, sam3_client
from core.perception.camera_ros2 import BASE_FRAME, grab_rgbd
from core.scene_graph import graph as sg
from core.scene_graph.instance import Instance, from_box
from core.scene_graph.relations import classify
from core.reasoner.query import Location, described, search_order
from core.utils import events

ROOT = Path(__file__).resolve().parents[2]
# Mission outcomes, from the object never being seen to standing ready to grasp.
NOT_FOUND = "not_found"
ARRIVED = "arrived"
FOUND = "found"
NOT_GRASPABLE = "not_graspable"
READY = "ready"
PICKED = "picked"
GRASPED = "grasped"
NOT_PICKED = "not_picked"
EXIT_CODES = {NOT_FOUND: 1, ARRIVED: 0, FOUND: 0, NOT_GRASPABLE: 2, READY: 0,
              PICKED: 0, GRASPED: 0, NOT_PICKED: 4}
# A same-label detection further than this from a remembered one is another instance.
SAME_INSTANCE = 1.0
# Nothing the HSR gripper can pick up measures more than this across.
MAX_SPAN = 0.6
# Taller furniture holds things on shelves the head cannot see from above.
TALL = 1.2
STORAGE_WORDS = ("shelf", "shelves", "bookcase", "rack")
# Cache viewing poses and outcomes for reuse during this run only.
VIEWS = {}
# After this many failed drives at one place, move on instead of trying every pose.
FAILED_DRIVES = 2


def home_goal():
    """The user's navigation home pose, as an arm trajectory goal."""
    goal = FollowJointTrajectory.Goal()
    # Name each joint in the same order as the home-pose positions below.
    goal.trajectory.joint_names = [
        'arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
        'wrist_flex_joint', 'wrist_roll_joint',
    ]
    goal.trajectory.points = [JointTrajectoryPoint(
        positions=[0.0, 0.0, -1.57, -1.57, 0.0],
        time_from_start=Duration(sec=3),
    )]
    return goal


def home_arm(navigator, wait, remaining):
    """Complete the user's navigation home pose before starting the search."""
    client = ActionClient(navigator, FollowJointTrajectory,
                         '/arm_trajectory_controller/follow_joint_trajectory')
    controllers = navigator.create_client(ListControllers,
                                          '/controller_manager/list_controllers')

    def arm_controller_running():
        """Check that the arm controller is active; a loaded action can still reject goals."""
        if not controllers.service_is_ready():
            return False
        future = controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(navigator, future,
                                         timeout_sec=min(1., remaining()))
        if not future.done():
            controllers.remove_pending_request(future)
            return False
        # A controller must be active, not just listed, before it can execute the goal.
        return any(state.name == 'arm_trajectory_controller' and state.state == 'active'
                   for state in future.result().controller)

    try:
        wait('arm trajectory action', client.server_is_ready)
        wait('arm_trajectory_controller active', arm_controller_running)
        goal = home_goal()
        print('[HOME] moving arm to home pose', flush=True)
        # Reuse the navigator's bounded action wait and cancellation handling.
        outcome = navigator._run(client, goal, timeout=30)
        if outcome is None or outcome.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError('Arm homing did not succeed; search will not start')
        if outcome.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f'Arm homing failed: {outcome.result.error_string}')
        print('[READY] arm home pose reached', flush=True)
    finally:
        client.destroy()
        navigator.destroy_client(controllers)


def waiter(navigator, deadline):
    """(remaining, wait) for readiness checks that share one deadline."""
    # Make every startup check use the same wall-clock deadline.
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Startup deadline exceeded; see the last WAIT message')
        return seconds

    def wait(label, check):
        print(f'[WAIT] {label}', flush=True)
        while True:
            remaining()
            if check():
                print(f'[READY] {label}', flush=True)
                return
            # Keep processing ROS messages while waiting for the next readiness check.
            rclpy.spin_once(navigator, timeout_sec=min(0.5, remaining()))

    return remaining, wait


def _nav_ready(navigator, remaining, wait):
    """What only the driving steps use: Nav2, the head action and the IK service."""
    def state(client):
        """The node's lifecycle state id, or None while it does not answer."""
        if not client.service_is_ready():
            return None
        future = client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(navigator, future, timeout_sec=min(1., remaining()))
        if not future.done():
            client.remove_pending_request(future)
            return None
        return future.result().current_state.id

    # Started by the same launch, controller_server answers nothing while its
    # activation waits for the robot's odom TF (35 s on 2026-09-21), and the
    # resume check below gives up on it after 15 s.
    client = navigator.create_client(GetState, '/controller_server/get_state')
    try:
        wait('controller_server answering', lambda: state(client) is not None)
    finally:
        navigator.destroy_client(client)
    navigator.resume_navigation_if_paused()
    # Wait for active navigation nodes before checking their actions and sensors.
    for name in ('amcl', 'planner_server', 'controller_server', 'behavior_server', 'bt_navigator'):
        client = navigator.create_client(GetState, f'/{name}/get_state')
        try:
            # Accept only the active lifecycle state.
            wait(name + ' active', lambda: state(client) == 3)  # lifecycle ACTIVE
        finally:
            navigator.destroy_client(client)
    wait('Nav2 actions', lambda: navigator.planner.server_is_ready() and navigator.driver.server_is_ready())
    head = ActionClient(navigator, FollowJointTrajectory,
                        '/head_trajectory_controller/follow_joint_trajectory')
    try:
        wait('head trajectory action', head.server_is_ready)
    finally:
        head.destroy()
    ik = navigator.create_client(SolveIkWithCollision, '/ik_solver_node/solve_ik_with_collision')
    try:
        wait('IK service', ik.service_is_ready)
    finally:
        navigator.destroy_client(ik)


def get_ready(navigator, target, timeout, perception=True, nav=True):
    """Wait for every service the mission uses; skip SAM3 without perception.

    nav=False is the pick alone: no Nav2 and no map frame, since
    core.grasping.pick itself works in odom."""
    remaining, wait = waiter(navigator, time.monotonic() + timeout)
    if nav:
        _nav_ready(navigator, remaining, wait)

    if perception:
        scene = navigator.create_client(GetPlanningScene, '/get_planning_scene')
        try:
            def answers():
                if not scene.service_is_ready():
                    return False
                future = scene.call_async(GetPlanningScene.Request())
                rclpy.spin_until_future_complete(navigator, future, timeout_sec=min(1., remaining()))
                if not future.done():
                    scene.remove_pending_request(future)
                    return False
                # A completed planning-scene response proves the service is responding.
                return True
            # Require a real reply; an advertised service may be unresponsive.
            wait('move_group planning scene', answers)
        finally:
            navigator.destroy_client(scene)
        # Check the MTC execution action before travelling to the object.
        solution = ActionClient(navigator, ExecuteTaskSolution, '/execute_task_solution')
        try:
            wait('MoveIt task solution action', solution.server_is_ready)
        finally:
            solution.destroy()

    def localized():
        try:
            navigator.robot_pose(timeout=min(1., remaining()))
            return True
        except RuntimeError:
            return False
    if nav:
        wait('fresh map-to-base localization', localized)

    print('[WAIT] synchronized RGB-D and camera TF', flush=True)
    while True:
        try:
            rgb, _, _, _ = grab_rgbd(target_frame='map' if nav else BASE_FRAME,
                                     timeout=min(5., remaining()))
            break
        except RuntimeError as error:
            print(f'[WAIT] {error}', flush=True)
    print('[READY] RGB-D and camera TF', flush=True)
    if perception:
        print('[WAIT] SAM3 response (start docker/sam3/run_sam3.sh on host)', flush=True)
        while True:
            # Take the budget outside the try: the deadline's TimeoutError must end
            # the wait, the RPC's ("no reply within 30s") must retry it.
            budget = min(30., remaining())
            try:
                sam3_client.detect(rgb, target, timeout=budget)
                break
            except sam3_client.ObjectNotFound:
                break  # A valid negative detection still proves the server is ready.
            except (RuntimeError, TimeoutError) as error:
                print(f'[WAIT] SAM3: {error}', flush=True)
                rclpy.spin_once(navigator, timeout_sec=min(1., remaining()))
        print('[READY] SAM3; startup detection discarded', flush=True)


def box(data):
    """The box centre differs from a segmented cloud's mean; prefer saved bounds."""
    lower, upper = np.asarray(data["bounds"])
    return ((lower + upper) / 2, upper - lower)


def is_storage(data):
    """Shelved furniture is searched level by level instead of from above."""
    height = data["bounds"][1][2] - data["bounds"][0][2]
    return height > TALL or any(word in data["label"].lower() for word in STORAGE_WORDS)


def targets(scene, obj, furniture=None, robot_xy=None, top_k=3):
    # Use an explicitly selected furniture item, or fall back to the object search order.
    if furniture:
        matches = []
        for node_id, data in sg.furniture(scene).items():
            if furniture in (data["name"], data["label"], str(node_id)):
                matches.append((node_id, data))
        if len(matches) != 1:
            raise ValueError("Use a unique furniture instance name or node ID")
        node, data = matches[0]
        yield Location(
            furniture_id=node,
            label=data["label"],
            room=data["room"],
            relation="near",
            source="furniture",
            centroid=data["centroid"],
        )
    else:
        yield from search_order(scene, obj, near=robot_xy, top_k=top_k)


def view_key(location):
    """One place to look: furniture, or a remembered object where it was last seen."""
    return (location.furniture_id, location.object_id,
            tuple(np.round(location.centroid, 2).tolist()))


def _views(scene, location):
    """Every clear pose around a place, facing it, and the surface it has to see."""
    pieces = sg.furniture(scene)
    # Keep the place being observed out of its own keep-out set.
    blockers = list(sg.blockers(scene, exclude=(location.object_id,)).values())
    if location.furniture_id is None:
        centre, dimensions, yaw = (location.centroid, [0, 0, 0], 0.0)
        storage = False
    else:
        centre, dimensions, yaw = sg.footprint(pieces[location.furniture_id])
        storage = is_storage(pieces[location.furniture_id])
    # Generate furniture-clear base poses and surface samples for view coverage.
    poses = standoff.candidates(
        centre, dimensions, yaw, blockers=blockers, storage=storage
    )
    points = standoff.surface_points(centre, dimensions, yaw)
    if location.object_id is not None:
        # Stand outside the furniture, but face the remembered object itself.
        aimed = []
        for x, y, yaw in poses:
            yaw = float(np.arctan2(location.centroid[1] - y, location.centroid[0] - x))
            aimed.append((x, y, yaw))
        poses = aimed
        points = np.array([location.centroid[:2]])
    return (poses, points)


def _shape(scene, location):
    """The footprint the observation poses were generated around."""
    if location.furniture_id is None:
        return (location.centroid, [0.0, 0.0, 0.0], 0.0)
    return sg.footprint(sg.furniture(scene)[location.furniture_id])


def _aim(scene, location, pose):
    """What that pose is meant to be looking at; _views re-aims at the object."""
    if location.object_id is not None:
        return location.centroid[:2]
    return standoff.aim_point(pose[:2], *_shape(scene, location))


def plan(scene, location, robot_xy=None, grid=None, sight_grid=None):
    """Order clear observation poses and report the view count; optionally filter by costmap.

    `grid` answers whether the base can stand somewhere, so its obstacles are
    grown by the robot radius. `sight_grid` answers whether the piece can be seen
    from there, which the collar has no say in: a pose can look straight across
    floor it could not park on. Pass the same costmap blocked at LETHAL instead of
    INSCRIBED (base_placement.sight_grid). It defaults to `grid`, which is how
    this behaved before and what callers with only one costmap still get."""
    sg.require_map(scene)
    if location.frame_id != "map":
        raise ValueError("Search locations must be in map")
    key = view_key(location)
    if key not in VIEWS:
        poses, points = _views(scene, location)
        # Remove candidate bases that land on blocked costmap cells, and those
        # that would look at the place through a wall: candidates() rings the
        # piece geometrically and cannot see walls at all.
        if grid is not None:
            seen_through = grid if sight_grid is None else sight_grid
            poses = [pose for pose in poses if standoff.free(grid, pose)]
            poses = [pose for pose in poses
                     if standoff.unobstructed(seen_through, pose,
                                              _aim(scene, location, pose),
                                              _shape(scene, location))]
        VIEWS[key] = {"poses": poses, "points": points, "outcomes": {}}
    # Ordered afresh each time: the nearest pose depends on where the robot is now.
    poses, views = standoff.order(VIEWS[key]["poses"], VIEWS[key]["points"], robot_xy)
    print(
        f"{location.source}: {location.label}, room={location.room},"
        f" {len(poses)} observation poses, {views} views needed"
    )
    if location.reason:
        print(f"  {location.reason}")
    # The dashboard draws the place, its neighbours and every pose it may park at.
    furniture = []
    for node, data in sg.furniture(scene).items():
        centre, dimensions, yaw = sg.footprint(data)
        furniture.append({"id": node, "name": data["name"] or data["label"],
                          "centre": centre, "dimensions": dimensions, "yaw": yaw})
    events.emit("plan", location=described(scene, [location])[0], poses=poses,
                views=views, furniture=furniture, robot_radius=standoff.ROBOT_RADIUS)
    return (poses, views)


def look_points(scene, location, pose):
    """Choose head targets for a remembered object, a tabletop, or successive shelf heights."""
    if location.object_id is not None or location.furniture_id is None:
        aims = [location.centroid]
        if location.object_id is not None and location.furniture_id is not None:
            data = scene.nodes[location.furniture_id]
            if not is_storage(data):
                # Also sweep the supporting surface in case the object moved.
                for x, y in standoff.sweep_points(pose[:2], *sg.footprint(data)):
                    aims.append([x, y, data['bounds'][1][2]+.06])
        return aims
    data = scene.nodes[location.furniture_id]
    lower, upper = (data["bounds"][0][2], data["bounds"][1][2])
    points = []
    # One aim sees only part of a long piece; pan across what is within range.
    for x, y in standoff.sweep_points(pose[:2], *sg.footprint(data)):
        if not is_storage(data):
            # Objects lie on top; aiming just above it keeps the whole top in view.
            points.append([x, y, upper + 0.1])
            continue
        distance = np.hypot(x - pose[0], y - pose[1])
        for height in standoff.shelf_heights(lower, upper, distance):
            points.append([x, y, height])
    return points


def observed_furniture(scene, points):
    ids = []
    pieces = []
    for node, data in sg.furniture(scene).items():
        centre, dimensions = box(data)
        ids.append(node)
        pieces.append(from_box(data["label"], centre, dimensions))
    # Associate the observed object with nearby supporting or containing furniture.
    index, relation = classify(Instance("target", points), pieces, near_limit=0.5)
    if index is None:
        return None, "near"
    return ids[index], relation


def locate(obj, with_camera=False):
    rgb, depth, intrinsics, transform = grab_rgbd(target_frame="map")
    try:
        mask, score = sam3_client.detect(rgb, obj)
    except sam3_client.ObjectNotFound:
        return None
    # Remove mask-edge pixels that may use background depth and stretch the box.
    core = pointcloud.shrink(mask)
    points = pointcloud.deproject(depth, intrinsics, core)
    if len(points) < 100 or len(points) < 0.25 * np.count_nonzero(core):
        print("Object detected, but depth is insufficient; trying another view.")
        return None
    # Keep one connected object body to remove depth points on the background.
    points = pointcloud.largest_cluster(points)
    if points is None or len(points) < 100:
        print("Detection did not form one solid body; trying another view.")
        return None
    print(f"Detected {obj}, score={score:.2f}, {len(points)} points")
    points = pointcloud.transform_points(transform, points)
    return (points, transform[:3, 3]) if with_camera else points


def make_graspable(scene, node_id, navigator, obstacles="--costmap", bearings=12, target=None,
                   hand_poses=None):
    """Park at a reachable base pose, using measured pregrasp geometry when available.

    `hand_poses` are map-frame pregrasp palm poses the caller already has, such
    as the grasp active perception found; the base is then solved for those."""
    centre, dimensions = box(scene.nodes[node_id])
    holder, relation = sg.location_of(scene, node_id)
    # The target is excluded: its own box is wider than graspable.PARK_DISTANCE
    # leaves, so as a blocker it would reject every base pose that can reach it.
    blockers = list(sg.blockers(scene, exclude=(node_id,)).values())
    solid = []
    for furniture_id, data in sg.furniture(scene).items():
        # Graph boxes are solid: the piece the object is on or in would swallow the hand.
        if furniture_id != holder or relation == "near":
            solid.append(sg.footprint(data))
    alternate_hand_poses = None
    if hand_poses is None and target:
        from core.grasping.pick import top_rectangle, cylinder, sphere, contact_candidates
        measured = locate(target, with_camera=True)
        if measured is not None:
            points, camera = measured
            if cylinder(points) is None and (top_rectangle(points) is not None or sphere(points) is not None):
                try:
                    _, hand_poses, _, kind = contact_candidates(points, np.empty((0,4,4)), camera)
                    # Back the palm away from contact to test reachable pregrasp poses.
                    back = np.eye(4); back[2,3] = -.08
                    hand_poses = hand_poses @ back
                    # Try overhead sphere grasps first, keeping side grasps as a fallback.
                    if kind == 'sphere' and np.any(hand_poses[:,2,2] < -.9):
                        overhead = hand_poses[:,2,2] < -.9
                        alternate_hand_poses = hand_poses[~overhead]
                        hand_poses = hand_poses[overhead]
                    print(f'[BASE] checking actual {kind} pregrasp orientations', flush=True)
                except RuntimeError as error:
                    print(f'[BASE] box geometry unresolved: {error}', flush=True)
                    hand_poses = None
    outcome = graspable.reposition(
        navigator, centre, dimensions, obstacles=obstacles, bearings=bearings,
        blockers=blockers, solid=solid, hand_poses=hand_poses,
    )
    # Retry with side grasps when the preferred overhead parking poses fail.
    if outcome is None and alternate_hand_poses is not None:
        print('[BASE] overhead parking unavailable; checking side grasps', flush=True)
        outcome = graspable.reposition(
            navigator, centre, dimensions, obstacles=obstacles, bearings=bearings,
            blockers=blockers, solid=solid, hand_poses=alternate_hand_poses)
    if outcome is None:
        print(
            "No clear, navigable IK base pose was reached within position and yaw "
            "tolerances. The detected object remains saved in the scene graph."
        )
        return NOT_GRASPABLE
    pose, offset = outcome
    if offset > graspable.PRECISE_TOLERANCE:
        print(
            f"Parked {offset:.3f} m from the certified pose, outside the"
            f" {graspable.PRECISE_TOLERANCE:.2f} m the solver certifies. The object is"
            " recorded, but the arm is not proven to reach it from here."
        )
        return NOT_GRASPABLE
    # Keep the object in view so grasp generation can capture it from here.
    if not navigator.look_at(centre):
        print("The head could not aim at the object from the parked pose, so the"
              " camera is not on it and a pick from here cannot capture it.")
        return NOT_GRASPABLE
    print(
        f"Ready to grasp from ({pose[0]:.2f}, {pose[1]:.2f}, yaw {pose[2]:.2f}),"
        f" parked {offset:.3f} m from it."
    )
    return READY


def pick_up(obj, mode='auto', cloud=None):
    """Run the pick in a separate process so MTC's C++ node has its own ROS context."""
    pick = subprocess.run([sys.executable, "-m", "core.grasping.pick", obj,
                           '--mode', mode] + (['--cloud', str(cloud)] if cloud else []), cwd=ROOT)
    return {0: PICKED, 3: GRASPED}.get(pick.returncode, NOT_PICKED)


def search(scene, obj, navigator, furniture=None, top_k=3, observe=locate, views=2):
    """Search viewpoints until an object is found; return (status, object node or None).

    Try at least views viewpoints per location, increasing this for coverage.
    Perception or server errors stop the search."""
    if views < 1:
        raise ValueError("views must be positive")
    # Advancing this iterator requests LLM fallback only when needed.
    for location in targets(scene, obj, furniture, navigator.robot_xy(), top_k):
        poses, needed = plan(scene, location, navigator.robot_xy(),
                             base_placement.costmap_grid(navigator),
                             base_placement.sight_grid(navigator))
        limit = max(views, needed)
        outcomes = VIEWS[view_key(location)]["outcomes"]
        observations = 0
        failed_drives = 0
        # Surface samples some view so far had within range. The view count assumes
        # every drive succeeds; this is what the views reached really took in.
        surface = np.asarray(VIEWS[view_key(location)]["points"], dtype=float)
        seen = np.zeros(len(surface), dtype=bool)
        current = navigator.robot_pose() if obj else None
        # Check the current view before driving to a new observation pose.
        for pose in ([current] if current is not None else []) + list(poses):
            at_current = current is not None and pose is current
            within = np.hypot(*(surface - pose[:2]).T) <= standoff.MAX_RANGE
            # Looking from where the robot already stands is only worth the head moves
            # when the place is within range: not the kitchen desk from the office.
            if at_current and not within.any():
                continue
            # Past the view budget, drive only where unseen surface comes within range.
            if not at_current and observations >= limit and not np.any(within & ~seen):
                continue
            if not at_current and not navigator.reachable(*pose):
                outcomes[pose] = "no path"
                continue
            if not at_current:
                # Drive facing forward; the head is still aimed at the last view.
                navigator.look_home()
            if not at_current and not navigator.drive_to(*pose):
                outcomes[pose] = "drive failed"
                failed_drives += 1
                if failed_drives >= FAILED_DRIVES:
                    break
                continue
            if not obj:
                return (ARRIVED, None)
            # Count each visited pose even if the head cannot aim at the target.
            observations += int(not at_current)
            points = None
            aimed = False
            aims = look_points(scene, location, pose)
            for index, aim in enumerate(aims, 1):
                print(f"[LOOK] aim {index}/{len(aims)} at {np.round(aim, 2).tolist()}", flush=True)
                # A shelf height beyond the head's tilt is skipped; the others still count.
                if not navigator.look_at(aim):
                    continue
                aimed = True
                # A missed detection allows retry; communication errors still stop us.
                points = observe(obj)
                if points is not None:
                    break
            if aimed:
                # look_points sweeps everything within range, so all of it was in a view.
                seen |= within
            if not aimed:
                print("The head cannot aim at the target from this pose.")
                outcomes[pose] = "cannot aim"
            elif points is None:
                outcomes[pose] = "not detected"
            else:
                outcomes[pose] = "detected"
            if points is not None:
                points = np.asarray(points, dtype=float)
                if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                    raise ValueError("Detection must contain finite map-frame 3D points")
                lower, upper = (points.min(axis=0), points.max(axis=0))
                span = float((upper - lower).max())
                if span > MAX_SPAN:
                    # Reject oversized detections before they change the saved object box.
                    print(
                        f"Detection spans {span:.2f} m, beyond the {MAX_SPAN:.2f} m"
                        " a graspable object can measure; discarding this view."
                    )
                    if observations >= limit:
                        break
                    continue
                # Reuse a nearby matching object; give distant detections a new node.
                object_id = location.object_id
                if object_id is not None and np.linalg.norm(
                    points.mean(axis=0)[:2]
                    - np.asarray(scene.nodes[object_id]["centroid"])[:2]
                ) > SAME_INSTANCE:
                    object_id = None
                if object_id is None:
                    object_id = sg.find_object(
                        scene, obj, near=points.mean(axis=0)[:2], limit=SAME_INSTANCE
                    )
                label, name = (obj, "")
                if object_id is not None:
                    # Preserve the saved label and name so future queries still match.
                    label = scene.nodes[object_id]["label"]
                    name = scene.nodes[object_id]["name"]
                furniture_id, relation = observed_furniture(scene, points)
                # Save the measured box and its furniture relation before returning success.
                object_id = sg.record_object(
                    scene,
                    label,
                    (lower + upper) / 2,
                    upper - lower,
                    furniture_id,
                    frame_id="map",
                    node_id=object_id,
                    relation=relation,
                    name=name,
                )
                print(f"Found {obj} in map at {points.mean(axis=0).round(3).tolist()}")
                return (FOUND, object_id)
            # Move to the next search location once its view budget is used and no
            # surface is left unseen.
            if observations >= limit and seen.all():
                break
        if observations == 0:
            print("Location could not be observed: navigation failed; trying the next location.")
        else:
            if not seen.all():
                print(f"[LOOK] {np.count_nonzero(~seen)}/{len(seen)} surface samples were never within"
                      f" {standoff.MAX_RANGE} m of a reached view.", flush=True)
            print("Object not detected in completed views; trying the next location.")
    return (NOT_FOUND, None)
