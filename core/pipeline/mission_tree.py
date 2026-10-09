"""The query-to-grasp mission as a behaviour tree over core.pipeline.actions.

    python3 -m core.pipeline.mission_tree --target pringles
    python3 -m core.pipeline.mission_tree --target pringles --navigate-only true
    python3 -m core.pipeline.mission_tree --target pringles --nav false   # pick only
    python3 -m core.pipeline.mission_tree --render      # picture of the tree, no ROS

The tree decides when each action runs; core.pipeline.actions holds what each does.
Each leaf runs one existing step to its end within a single tick. The mission
is a strict sequence, and Ctrl+C still cancels a Nav2 goal inside the step.

    Mission                                   Mission (--navigate-only true)
    ├─ Home arm        first, whatever pose   ├─ Home arm
    ├─ Ready                                  ├─ Ready             no SAM3
    ├─ Find target                            ├─ Choose location   target reasoning
    ├─ Park            once                   └─ Go there          first reachable view
    ├─ Pause Nav2
    └─ Pick            once; leave the resulting hold in place

--nav false is the pick alone, for an object already in front of the camera:
Home arm, Ready (move_group and SAM3 only), Pick. No Nav2, map or scene graph,
and no Home head, which would turn the camera off the object.

Watch it live, with ROS sourced, once this is running:
    py-trees-tree-watcher      # in the terminal
    py-trees-tree-viewer       # in a window
"""

import argparse
import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import py_trees
import py_trees_ros
import rclpy
from rclpy.executors import SingleThreadedExecutor

from core.navigation import base_placement
from core.navigation.nav2_client import Navigator
from core.pipeline import actions
from core.reasoner import query
from core.scene_graph import graph as sg
from core.utils import events
from core.utils.recording import Paths

ROOT = Path(__file__).resolve().parents[2]
# Same switch core.perception.camera_ros2 reads: HSR_REAL_ROBOT=1 selects the
# robot's clock and the recording's own scene graph. launch/realrobot/
# grasp_real.launch.py and realrobot/live/grasp_preflight.py already set it.
REAL_ROBOT = os.environ.get("HSR_REAL_ROBOT", "") == "1"
# On hardware the graph follows $RECORDING like every other real-robot tool.
DEFAULT_GRAPH = (Paths().graph if REAL_ROBOT
                 else ROOT / "config/scene_graph/kitchen_objects.json")
# Map the failed mission step to a process exit code; unexpected failures use 3.
EXIT_CODES = {"Choose location": 1, "Find target": 1, "Go there": 2, "Park": 2, "Pick": 4}


class Step(py_trees.behaviour.Behaviour):
    """A leaf that runs `action()` once per visit: True is success, False failure."""

    def __init__(self, name, action):
        super().__init__(name)
        self.action = action
        self.started = False

    def initialise(self):
        self.started = False

    def update(self):
        # Show RUNNING for one tick so viewers can display the active step.
        if not self.started:
            self.started = True
            return py_trees.common.Status.RUNNING
        # For core.perception.sam3_client: which step a saved detection belongs to. The
        # pick runs in its own process and inherits it.
        os.environ["MISSION_STAGE"] = self.name
        if self.action():
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


def nbv_pregrasps(chosen):
    """The pregrasp of the grasp Explore found, as (1, 4, 4) in map, or None.

    Explore's `ik_reachable` proves a grasp reachable from *some* base pose: it
    calls base_placement.solve, which asks the IK service for base placements,
    never for the pose the camera stood at. Run 20260924-014619 kept the explore
    pose on the 1.0 m view ring and the pick found every pregrasp 0.98 m from a
    base it may move 0.25 m: GOAL_STATE_INVALID on all of them.

    So park solves the base for this pregrasp instead. It is the pose the pick's
    seeded IK plans (pick.APPROACH back along the palm), and the IK service
    returns bases on the side the grasp approaches from, so the graph's probe
    bearings that drove through the cereal boxes in run 20260924-012335 are
    not consulted.
    """
    grasp = chosen.get("nbv_grasp")
    if not grasp:
        return None
    from core.grasping.pick import APPROACH
    back_off = np.eye(4)
    back_off[2, 3] = -APPROACH
    return (np.asarray(grasp["palm_pose"], dtype=float) @ back_off)[None]


def build(steps, grasp=True, navigate_only=False, active_perception=False, nav=True):
    """The mission tree over `steps`, a dict of name -> action."""
    if not nav:
        # The pick alone: the operator has put the object in view, so the head stays.
        children = [Step("Home arm", steps["home"]), Step("Ready", steps["ready"]),
                    Step("Pick", steps["pick"])]
        return py_trees.composites.Sequence("Mission", memory=True, children=children)
    # Home first: it needs only the arm controller, and every later step assumes it.
    children = [Step("Home arm", steps["home"]), Step("Ready", steps["ready"]),
                # After Ready: get_ready is what proves the head controller is live.
                Step("Home head", steps["home_head"])]
    # Use a shorter sequence when the user only wants to drive to a likely location.
    if navigate_only:
        children.append(Step("Choose location", steps["choose"]))
        children.append(Step("Go there", steps["go"]))
        return py_trees.composites.Sequence("Mission", memory=True, children=children)
    children.append(Step("Find target", steps["find"]))
    if active_perception:
        # Still under test: fuse the target from next best views before parking.
        children.append(Step("Explore views", steps["explore"]))
    # Parking already tries every base pose the IK solver returns; once is enough.
    children.append(Step("Park", steps["park"]))
    # Parking is the handoff: once the base is on a certified pose, Nav2 must stop
    # owning it, whether the pick runs here or from another terminal afterwards.
    # PAUSE leaves map_server and AMCL up, so the map frame survives for the pick.
    children.append(Step("Pause Nav2", steps["pause"]))
    if grasp:
        children.append(Step("Pick", steps["pick"]))
    return py_trees.composites.Sequence("Mission", memory=True, children=children)


def snapshot(node):
    """A behaviour and its children with their status, as the dashboard draws them."""
    children = []
    # Copy the tree recursively so the dashboard can display every step status.
    for child in node.children:
        children.append(snapshot(child))
    return {"name": node.name, "type": type(node).__name__,
            "status": node.status.value, "children": children}


def exit_code(root):
    if root.status == py_trees.common.Status.SUCCESS:
        return 0
    return EXIT_CODES.get(root.tip().name, 3)


def mission_steps(navigator, scene, target, graph_path, top_k, bearings, timeout, perception, mode="auto",
                  rerun=False, rviz=False, nav=True, furniture=None):
    """The existing pipeline functions, as the tree's actions."""
    # Share the selected location and detected object between mission steps.
    chosen = {}

    def ready():
        actions.get_ready(navigator, target, timeout, perception=perception, nav=nav)
        return True

    def home():
        # Give the controller time to activate before sending the home pose.
        remaining, wait = actions.waiter(navigator, time.monotonic() + timeout)
        actions.home_arm(navigator, wait, remaining)
        return True

    def home_head():
        navigator.look_home()
        return True

    def choose():
        chosen["location"] = next(actions.targets(scene, target, furniture=furniture,
                                                 robot_xy=navigator.robot_xy(), top_k=top_k), None)
        # DeepSeek's picks are cached in the graph; keep them for the next query.
        sg.save(scene, graph_path)
        if chosen["location"] is None:
            print(f"[REASON] no location to search for {target!r}", flush=True)
            return False
        location = chosen["location"]
        print(f"[REASON] {target!r}: {location.label} in {location.room} ({location.source})", flush=True)
        return True

    def go():
        location = chosen["location"]
        poses, _ = actions.plan(scene, location, navigator.robot_xy(),
                                base_placement.costmap_grid(navigator),
                                base_placement.sight_grid(navigator))
        failed_drives = 0
        for pose in poses:
            if not navigator.reachable(*pose):
                continue
            if not navigator.drive_to(*pose):
                failed_drives += 1
                if failed_drives >= actions.FAILED_DRIVES:
                    return False
                continue
            # Face where the target should be, as the search does before looking.
            for aim in actions.look_points(scene, location, pose):
                if navigator.look_at(aim):
                    break
            return True
        return False

    def find():
        navigator.resume_navigation_if_paused()
        status, chosen["object"] = actions.search(scene, target, navigator, furniture=furniture,
                                                  top_k=top_k)
        if status == actions.NOT_FOUND:
            # Every place failed, DeepSeek's cached picks included: ask afresh next time.
            query.forget(scene, target)
        # Save observations and cached guesses before later motion steps can fail.
        sg.save(scene, graph_path)
        return status == actions.FOUND

    def explore():
        from core.active_perception.explore import explore as nbv_explore, furniture_footprints
        # Park drives to the grasp's IK base next, so no drive back to the view.
        log = nbv_explore(navigator, target, rerun=rerun, rviz=rviz,
                          blockers=furniture_footprints(scene), sim=not REAL_ROBOT,
                          return_to_grasp_view=False)
        # The pick plans on this fused surface instead of scanning again.
        chosen["cloud"] = log["target_cloud"]
        # Park solves the base for this grasp's pregrasp; see nbv_pregrasps.
        chosen["nbv_grasp"] = log["best_grasp"]
        print(f"[NBV] {log['views_fused']} views fused, best grasp {log['best_grasp'] and round(log['best_grasp']['quality'], 3)}", flush=True)
        return log["views_fused"] > 0

    def park():
        hand_poses = nbv_pregrasps(chosen)
        if hand_poses is not None:
            print("[PARK] parking for the grasp active perception found", flush=True)
        status = actions.make_graspable(scene, chosen["object"], navigator, bearings=bearings,
                                        target=target, hand_poses=hand_poses)
        if status != actions.READY and hand_poses is not None:
            # The pick generates its own grasps on the fused cloud, so a base the
            # graph's probes certify still serves it.
            print("[PARK] no clear base for that grasp; parking from the graph instead", flush=True)
            status = actions.make_graspable(scene, chosen["object"], navigator, bearings=bearings,
                                            target=target)
        return status == actions.READY

    def pause():
        # MoveIt moves the base during the pick; Nav2 must not fight it.
        navigator.pause_navigation()
        return True

    def pick():
        status = actions.pick_up(target, mode=mode, cloud=chosen.get("cloud"))
        print(f"[GRASP RESULT] {status}", flush=True)
        return status in (actions.PICKED, actions.GRASPED)

    return {"ready": ready, "home": home, "home_head": home_head, "choose": choose, "go": go,
            "find": find, "explore": explore, "park": park, "pause": pause, "pick": pick}


def render(grasp, navigate_only, active_perception=False, nav=True):
    """Save the tree as a picture under outputs/, without ROS or the robot."""
    # Build named placeholder actions for drawing; rendering never executes them.
    placeholders = {}
    for name in ("ready", "home", "home_head", "choose", "go", "find", "explore",
                 "park", "pause", "pick"):
        placeholders[name] = None  # never run: drawing needs names only
    root = build(placeholders, grasp=grasp, navigate_only=navigate_only, active_perception=active_perception,
                 nav=nav)
    name = "mission_tree_navigate" if navigate_only else "mission_tree" if nav else "mission_tree_pick"
    files = py_trees.display.render_dot_tree(root, name=name, target_directory=str(ROOT / "outputs"))
    print(py_trees.display.unicode_tree(root))
    for path in files.values():
        print("wrote", path)
    return 0


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--furniture", default="",
                        help="search this one piece only (node id, or a unique name or label): no "
                             "remembered places and no DeepSeek, for an object known to have moved")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bearings", type=int, default=12)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--grasp", default="true", choices=["true", "false"])
    parser.add_argument("--navigate-only", default="false", choices=["true", "false"],
                        help="reason and drive to the target's place; no SAM3 or GraspGenX")
    parser.add_argument("--nav", default="true", choices=["true", "false"],
                        help="false is the pick alone, the object already in view: no Nav2, "
                             "map or scene graph")
    parser.add_argument("--mode", default="auto", choices=["auto", "pickup", "grasp"])
    parser.add_argument("--active-perception", default="false", choices=["true", "false"],
                        help="fuse the found target from next best views before parking (under test)")
    parser.add_argument("--natural-language", default="false", choices=["true", "false"],
                        help="--target is a request such as 'fetch me my spectacles'; DeepSeek "
                             "names the object, and only that reaches the detector and the graph")
    parser.add_argument("--rerun", default="false", choices=["true", "false"],
                        help="open the Rerun viewer and stream the Explore step live; it is always saved as explore.rrd")
    parser.add_argument("--rviz", default="false", choices=["true", "false"],
                        help="publish the Explore step as markers for config/rviz/nbv.rviz")
    parser.add_argument("--render", action="store_true", help="save a picture of the tree and exit")
    args = parser.parse_args(argv)
    grasp, navigate_only = args.grasp == "true", args.navigate_only == "true"
    active_perception = args.active_perception == "true"
    nav = args.nav == "true"
    if not nav and (navigate_only or active_perception or not grasp):
        parser.error("--nav false leaves only the pick: navigate-only and active-perception "
                     "drive the base, and grasp false leaves nothing to do")
    if args.render:
        return render(grasp, navigate_only, active_perception, nav)
    if not args.target or not args.target.strip() or args.top_k < 1 or args.bearings < 1:
        parser.error("target must be nonempty; top-k and bearings must be positive")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("startup-timeout must be finite and positive")

    # Validate the graph before waiting for any robot or model service.
    scene = None
    if nav:
        scene = sg.load(args.graph)
        sg.require_map(scene)
    if args.natural_language == "true":
        # Before anything reads the target: it is SAM3's prompt, the cache key and the
        # label of the node a detection saves, and none of them should hold a sentence.
        # The graph's own objects, so 'something to drink' can come back as 'bottle'.
        known = {data["name"] or data["label"] for data in sg.objects(scene).values()} if scene else set()
        request, args.target = args.target, query.object_of(args.target, known=known)
        print(f"[REQUEST] {request!r} -> {args.target!r}", flush=True)
    # One folder of saved detections per run, named after when it ran and what it was
    # after. The dashboard passes its own run id, so both call the run the same.
    os.environ["MISSION_RUN_ID"] = (os.environ.get("MISSION_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")) \
        + "_" + re.sub("[^a-z0-9]+", "-", args.target.lower()).strip("-")
    print(f"[RUN] detections in outputs/detections/{os.environ['MISSION_RUN_ID']}", flush=True)
    rclpy.init()
    # There is no /clock on the robot: a sim-time node never sees a fresh
    # map->base TF and every step fails on it. realrobot/live/goto.py does the same.
    navigator = Navigator(use_sim_time=not REAL_ROBOT)
    steps = mission_steps(navigator, scene, args.target, args.graph, args.top_k,
                          args.bearings, args.startup_timeout, perception=not navigate_only, mode=args.mode,
                          rerun=args.rerun == "true", rviz=args.rviz == "true",
                          nav=nav, furniture=args.furniture or None)
    tree = py_trees_ros.trees.BehaviourTree(build(steps, grasp=grasp, navigate_only=navigate_only,
                                                  active_perception=active_perception, nav=nav))
    viewers = SingleThreadedExecutor()
    try:
        tree.setup(node_name="mission_tree", timeout=15.0)
        # Serve viewer requests on another thread while mission actions block.
        viewers.add_node(tree.node)
        threading.Thread(target=viewers.spin, daemon=True).start()
        # Run the sequence until a step fails or the whole mission succeeds.
        while True:
            tree.tick()
            # A step blocks inside its tick, so this lands once per status change.
            events.emit("tree", tree=snapshot(tree.root), running=tree.root.tip().name)
            if tree.root.status != py_trees.common.Status.RUNNING:
                break
            time.sleep(0.1)
        print(py_trees.display.unicode_tree(tree.root, show_status=True), flush=True)
        print(f"[RESULT] {tree.root.status.value}, last step: {tree.root.tip().name}", flush=True)
        return exit_code(tree.root)
    finally:
        # Stop the viewer thread and release tree and navigation resources on every exit.
        viewers.shutdown()
        tree.shutdown()
        navigator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        # The stack is the only thing that says which step and which call failed;
        # the one-line summary alone cost two runs of guessing. Summary last, so
        # it stays the visible line at the bottom of a long mission log.
        traceback.print_exc()
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(3)
