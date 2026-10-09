"""Drive the HSR to next best views around a queried object, then report the
best reachable grasp: active_grasp's GraspController.search_grasp for our stack.

    python3 -m core.active_perception.explore "pringles can"

Needs Nav2 localised in map, SAM3 and GraspGenX, and the IK solver
(launch/ik_solver.launch.py) unless --no-ik. In simulation:

    ros2 launch /home/ws/launch/search.launch.py web:=true      # sim, Nav2, IK, MoveIt, no mission
    python3 -m core.active_perception.explore "pringles can"    # in a second terminal

Each step fuses the current depth image, scores the views the head can take,
drives the base under the chosen one, lifts the torso to its height, aims the
head at the target and captures again, until the gain or grasp criterion stops it.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from core.active_perception.bbox import AABBox
from core.active_perception.grasps import graspgenx_grasps, ik_reachable
from core.active_perception.markers import Markers, Tee
from core.active_perception.nbv import NextBestView
from core.active_perception.record import Recording
from core.active_perception.views import CAMERA_HEIGHT, ViewHalfSphere
from core.navigation import base_placement
from core.navigation.nav2_client import Navigator
from core.scene_graph import graph as sg
from core.perception import sam3_client
from core.perception.camera_ros2 import grab_rgbd
from core.perception.pointcloud import deproject, largest_cluster, shrink, transform_points

# arm_lift_joint raises the torso, and with it the head, by half its travel.
ARM_LIFT_MAX = 0.69
HOME = [0.0, -1.57, -1.57, 0.0]  # arm_flex, arm_roll, wrist_flex, wrist_roll: actions.home_goal
OUTPUTS = Path("outputs/active_perception")
GRAPH = Path(__file__).resolve().parents[2] / "config/scene_graph/kitchen_objects.json"


def furniture_footprints(scene, exclude=()):
    """Every shape as (centre, dimensions, yaw): furniture and floor-standing
    objects both, since the base keeps out of them alike."""
    # Collect the outlines used to keep the moving base clear.
    return list(sg.blockers(scene, exclude=exclude).values())


def capture(anchor, sim):
    """RGB-D with the camera pose taken through odom, returned in the map frame.

    AMCL's map->odom correction jumps a few centimetres whenever it resamples.
    Fusion writes the space in front of every view as free, so views that
    disagree about where the robot stood carve the target away: a 2 cm
    disagreement cost us a third of a Pringles can's radius. odom does not jump,
    so the run is anchored to the map as it stood at the first capture and the
    camera is tracked through odom from there.
    """
    rgb, depth, k, odom_from_camera = grab_rgbd(target_frame="odom", use_sim_time=sim)
    return rgb, depth, k, anchor @ odom_from_camera


def detect_points(prompt, rgb, depth, k, map_from_camera):
    """The target's points in the map frame, from SAM3's mask."""
    # Detect the object, then keep its main cluster of valid depth points.
    mask, score = sam3_client.detect(rgb, prompt)
    points = largest_cluster(deproject(depth, k, shrink(mask)))
    if points is None:
        raise RuntimeError(f"'{prompt}' has too little depth to box")
    # Transform the object points into the map before boxing them.
    return transform_points(map_from_camera, points), score


class TargetBox:
    """The target's box, rebuilt from every view that has seen it.

    active_grasp is handed one box by its detector and keeps it. Ours comes from
    a SAM3 mask per view, so it has to survive a bad one. Taking the union of
    per-view boxes only ever grows, so a single bad mask or pose inflates the box
    for the rest of the run. Keeping the points instead lets a percentile drop
    those outliers, and lets the box shrink again when later views disagree.
    """

    def __init__(self, points, cube_min, cube_max, percentile=1.0, per_view=2000):
        self.cube_min, self.cube_max = (cube_min, cube_max)
        self.percentile, self.per_view = (percentile, per_view)
        self.clamped = False
        self.views = []
        self.add(points)

    def add(self, points):
        # Cap each view so one close-up cannot outvote every other view.
        if len(points) > self.per_view:
            chosen = np.random.default_rng(0).choice(len(points), self.per_view, replace=False)
            points = points[chosen]
        self.views.append(points)

    def box(self):
        points = np.concatenate(self.views)
        # A view whose mask landed on a neighbour contributes a whole blob, far
        # too large a share for a 1% percentile to trim: one bad view in eight
        # is 12% of the points, and run 20260924-013126 grew an 18.7 cm box
        # around a 7.7 cm can that way, then found no graspable geometry in it.
        # That blob is physically separate, so the cluster test that already
        # cleans up each view's mask cleans up their union too.
        clustered = largest_cluster(points)
        if clustered is not None:
            points = clustered
        low = np.percentile(points, self.percentile, axis=0)
        high = np.percentile(points, 100 - self.percentile, axis=0)
        # The cube is fixed at activation, so a box reaching past it scores nothing.
        self.clamped = bool(np.any(low < self.cube_min) or np.any(high > self.cube_max))
        return AABBox(np.maximum(low, self.cube_min), np.minimum(high, self.cube_max))


def save_target(out, policy, anchor, map_from_camera, score):
    """The fused target surface for core.grasping.pick --cloud, which works in odom.

    Every view went through `anchor`, so its inverse puts the cloud back into odom
    exactly; the camera is the last view's, the side pick's contact fit calls seen."""
    odom_from_map = np.linalg.inv(anchor)
    path = out / "target_cloud.npz"
    np.savez(path, points=transform_points(odom_from_map, policy.target_points()),
             camera=(odom_from_map @ map_from_camera)[:3, 3], score=score)
    return str(path)


def window_size_for(max_steps):
    """How many recent views a grasp must score in before the run may stop.

    active_grasp's window is 12, sized for a fixed-base arm sweeping tens of
    views along one trajectory. Our base drives to each view, so a run is
    `max_steps` views and 12 never fills: both of its stopping criteria -- a
    grasp voxel scored in every slot, and `len(views) > T` -- were unreachable,
    leaving "no candidate view gains anything" as the only way out. That exit
    only fires once nothing is left to reveal, which is why the robot finished
    on its most exploratory pose with the target out of frame.

    Half the budget is our number, not ETH's: it still demands agreement across
    several views, and it leaves room for `len(views) > T` to stop a run whose
    gains have gone small.
    """
    return max(3, max_steps // 2)


def better_grasp(kept, found):
    """The higher-quality of two (pose, quality) pairs; either may be None.

    `policy.best_grasp` is the best grasp on the surface fused *so far*, reset
    on entry to every predict_grasps. The base moves between steps and the IK
    filter runs from wherever it now stands, so a grasp found early disappears
    as soon as a later view cannot reach one, and the run reports no grasp at
    all. What the run found is the best any view produced.
    """
    if found is None:
        return kept
    if kept is None or found[1] > kept[1]:
        return found
    return kept


def set_torso(navigator, rise):
    """Lift the head by `rise` metres with the arm otherwise at its home pose."""
    # The head rises by half the arm-lift travel; clamp to the joint limit.
    lift = float(np.clip(2 * rise, 0.0, ARM_LIFT_MAX))
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                                   "wrist_flex_joint", "wrist_roll_joint"]
    goal.trajectory.points = [JointTrajectoryPoint(positions=[lift, *HOME], time_from_start=Duration(sec=4))]
    # The navigator's kept client: a fresh one per call lost its goal on the robot.
    outcome = navigator._run(navigator.arm, goal, 30)
    return outcome is not None and outcome.status == GoalStatus.STATUS_SUCCEEDED


def go_to_view(navigator, view, target):
    """Move the base and torso to the camera pose, then aim at the target."""
    # Read the desired camera position and viewing direction from its pose matrix.
    eye, forward = (view[:3, 3], view[:3, 2])
    yaw = math.atan2(forward[1], forward[0])
    # Set camera height, drive underneath it, and turn the head toward the target.
    lifted = set_torso(navigator, eye[2] - CAMERA_HEIGHT)
    reached = navigator.drive_to(float(eye[0]), float(eye[1]), yaw)
    aimed = navigator.look_at(target)
    return dict(lifted=lifted, reached=reached, aimed=aimed)


def explore(navigator, prompt, length=0.3, min_z_dist=0.6, max_steps=8, qual_thresh=0.9,
            grasps=True, ik=True, sim=True, out=None, rerun=False, rviz=False, blockers=(),
            return_to_grasp_view=True):
    """Observe an already visible target from new views; return the log and best grasp.

    `return_to_grasp_view=False` skips the final drive back to the view the best
    grasp came from: the mission's Park drives to the IK base for that grasp
    right after, so the return was a second drive for nothing."""
    out = out or OUTPUTS / time.strftime("explore_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    # Pin the map as it stands now; every later pose reaches it through odom.
    anchor = navigator.frame_transform("map", "odom")
    rgb, depth, k, map_from_camera = capture(anchor, sim)
    points, score = detect_points(prompt, rgb, depth, k, map_from_camera)
    bbox = AABBox.from_points(points)
    print(f"[NBV] '{prompt}' ({score:.2f}) in a {np.round(bbox.size, 2)} m box at {np.round(bbox.center, 2)}", flush=True)
    # Build camera positions that clear furniture and the costmap.
    sphere = ViewHalfSphere(bbox, min_z_dist, grid=base_placement.costmap_grid(navigator), blockers=blockers)
    policy = NextBestView(
        k, length=length, qual_thresh=qual_thresh,
        window_size=window_size_for(max_steps),
        grasp_fn=graspgenx_grasps if grasps else None,
        reachable_fn=ik_reachable(navigator, blockers=blockers) if (grasps and ik) else None,
    )
    # Start a fresh fusion volume and grasp history for this target.
    policy.activate(bbox, sphere)
    # Rebuild the box from every view's points, inside the cube fusion can reach.
    cube_min = policy.base_from_task[:3, 3]
    target = TargetBox(points, cube_min, cube_min + length)
    # Always recorded to the run folder; `rerun` also opens the viewer live.
    recording = Recording(k, depth.shape, path=out / "explore.rrd", spawn=rerun)
    # RViz gets the same calls, on the navigator's node: no second node to spin.
    if rviz:
        recording = Tee(recording, Markers(navigator, k, depth.shape))
    recording.scene(policy)
    log = dict(prompt=prompt, bbox=bbox.to_dict(), radii=sphere.radii.tolist(),
               feasible_views=len(sphere.candidates()), steps=[], best_grasp=None)
    print(f"[NBV] {log['feasible_views']} views the robot can take on radii {np.round(sphere.radii, 2)} m", flush=True)
    kept_grasp, grasp_view = (None, None)
    for step in range(1, max_steps + 1):
        # Fuse this depth image and choose whether another view is useful.
        policy.update(depth, map_from_camera)
        improved = better_grasp(kept_grasp, policy.best_grasp)
        if improved is not kept_grasp:
            # Remember where this grasp was found, not just the grasp: the IK
            # filter accepted it from this base pose.
            kept_grasp, grasp_view = (improved, map_from_camera.copy())
        recording.step(step, policy, depth, map_from_camera, image=rgb)
        record = dict(step=step, camera=map_from_camera[:3, 3].round(3).tolist(), done=policy.done,
                      grasp_quality=None if policy.best_grasp is None else round(policy.best_grasp[1], 3))
        if policy.info:
            record.update(candidates=len(policy.info["views"]),
                          gain=int(policy.info["gains"][policy.info["best"]]),
                          next_eye=policy.x_d[:3, 3].round(3).tolist())
        else:
            record["stopped"] = ("no view the robot can take" if not sphere.candidates()
                                 else "no view gains anything" if not policy.done else "policy done")
        log["steps"].append(record)
        print("[NBV] " + json.dumps(record), flush=True)
        # Stop when the policy is finished or cannot offer another useful view.
        if policy.done or not policy.info:
            break
        # Move to the chosen view before taking the next depth image.
        record["motion"] = go_to_view(navigator, policy.x_d, policy.bbox.center)
        if not record["motion"]["reached"]:
            sphere.rejected.append(policy.x_d[:3, 3].copy())  # Skip this failed camera position on later steps.
        # The costmap the candidates were vetted against was fetched before the
        # drive. On 2026-09-24 03:24 the far side of the table only entered it
        # once the robot got there, and the planner refused the next eye as an
        # obstacle. Vet the remaining candidates against what the robot has seen.
        sphere.grid = base_placement.costmap_grid(navigator)
        rgb, depth, k, map_from_camera = capture(anchor, sim)
        try:
            seen, _ = detect_points(prompt, rgb, depth, k, map_from_camera)
        except (sam3_client.ObjectNotFound, RuntimeError):
            seen = None
        if seen is not None:
            # Rebuild the box from every view so far, and re-aim the sphere at it.
            target.add(seen)
            policy.bbox = target.box()
            sphere.retarget(policy.bbox)
            print(f"[NBV] box {np.round(policy.bbox.size, 3)} m"
                  + (" (clamped to the cube)" if target.clamped else ""), flush=True)
    # A next best view is the view that reveals the most *unknown* voxels, so the
    # loop ends on the most exploratory pose it could find -- by construction the
    # one looking away from what it already knows. Run 20260924-010233 finished
    # there with the can out of frame. Go back to the view that produced the best
    # grasp: it saw the target, and the IK filter accepted a grasp from it.
    if return_to_grasp_view and grasp_view is not None and not np.allclose(grasp_view, map_from_camera, atol=1e-3):
        print("[NBV] returning to the view the best grasp was found from", flush=True)
        log["returned_to_grasp_view"] = go_to_view(navigator, grasp_view, policy.bbox.center)
    # The views raised the torso; the pick expects the homed arm at lift 0. Left
    # at 0.14 in run 20260924-021758, the folded hand sat where the head tilted
    # down at the can from the park pose, and the pick's collision check refused
    # to move: head_tilt_link <-> hand_r_distal_link on all three attempts.
    log["torso_lowered"] = set_torso(navigator, 0.0)

    # Save the fused surface in map coordinates and the run details.
    origin = policy.base_from_task[:3, 3]
    o3d.io.write_point_cloud(str(out / "scene_cloud.ply"), policy.tsdf.get_scene_cloud().translate(origin))
    np.save(out / "grid.npy", policy.tsdf.get_grid())
    log["target_cloud"] = save_target(out, policy, anchor, map_from_camera, score)
    if kept_grasp is not None:
        log["best_grasp"] = dict(quality=kept_grasp[1], palm_pose=kept_grasp[0].tolist())
    log["views_fused"] = len(policy.views)
    log["final_bbox"] = policy.bbox.to_dict()
    log["bbox_clamped"] = target.clamped
    (out / "explore.json").write_text(json.dumps(log, indent=2))
    print(f"[NBV] {len(policy.views)} views fused; outputs in {out}", flush=True)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--length", type=float, default=0.3, help="TSDF cube side, m (VGN: 0.3)")
    parser.add_argument("--min-z-dist", type=float, default=0.6, help="closest camera range, m")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--qual-thresh", type=float, default=0.9, help="grasp quality kept (ETH: 0.9)")
    parser.add_argument("--no-ik", action="store_true", help="skip the IK reachability filter on grasps")
    parser.add_argument("--no-grasps", action="store_true")
    parser.add_argument("--real", action="store_true", help="wall clock instead of sim time")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rerun", action="store_true", help="open the Rerun viewer and stream the loop live")
    parser.add_argument("--rviz", action="store_true",
                        help="also publish the loop as markers for config/rviz/nbv.rviz")
    parser.add_argument("--graph", type=Path, default=GRAPH, help="scene graph whose furniture the base keeps clear of")
    # Read the requested target and optional limits for this exploration run.
    args = parser.parse_args()

    # Start ROS and release the navigator in the finally block when the run ends.
    rclpy.init()
    navigator = Navigator(use_sim_time=not args.real)
    try:
        log = explore(navigator, args.prompt, length=args.length, min_z_dist=args.min_z_dist,
                      max_steps=args.max_steps, qual_thresh=args.qual_thresh, grasps=not args.no_grasps,
                      ik=not args.no_ik, sim=not args.real, out=args.output, rerun=args.rerun, rviz=args.rviz,
                      blockers=furniture_footprints(sg.load(args.graph)))
        if log["best_grasp"] is None:
            print("no reachable grasp on the target")
        else:
            pose = np.array(log["best_grasp"]["palm_pose"])
            print(f"best grasp quality {log['best_grasp']['quality']:.3f} at {pose[:3, 3].round(3)}")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
