"""Real-robot bringup for the behaviour-tree query-to-grasp mission, PC side:
launch/search.launch.py without the simulator.

    export RECORDING=lab_20260919_wheel DEEPSEEK_API_KEY=<key>
    ros2 launch /home/ws/launch/realrobot/search_real.launch.py target:="pringles can"

Each part of the mission is a switch, so one part can be tested alone:

    nav:=false                          pick only, the object already in view: Home arm,
                                        Ready, Pick. No Nav2, IK solver, map or graph.
    grasp:=false                        find, park, pause Nav2; no pick
    navigate_only:=true                 reason and drive to the place; no SAM3
    active_perception:=true             next best views before parking (needs nav)
    furniture:=29                       search that one piece only: no remembered
                                        places, no DeepSeek

RViz, any of the three: use_rviz:=true is MoveIt's (planning scene, RX camera),
opened by move_group; use_nav_rviz:=true is Nav2's (map, scan, costmaps, plans,
2D Pose Estimate and Nav2 Goal), and only with nav; nbv_rviz:=true is the
Explore step's (TSDF, candidate views, grasp), as in simulation.

Without target:= the stack comes up and no mission runs. With it, Ready blocks on
"fresh map-to-base localization" until AMCL is seeded, which stays a step by hand
in another terminal, because the match has to be checked before the base moves:

    python3 realrobot/live/localize.py && python3 realrobot/live/localize.py --dry-run

Not started here: the robot's own stack, vision transport RX, and SAM3 and
GraspGenX on the GPU host. realrobot/live/grasp_preflight.py checks all of them.

The navigation defaults are README section 9's, not nav2_real.launch.py's: AMCL
on, which Ready waits for, and cmd_vel straight to /omni_base_controller/cmd_vel.
That bypasses Toyota's limiter: no bumper stop, no obstacle slowdown, no teleop
override. Keep the runstop in reach, or pass cmd_vel_topic:=/base_velocity while
Toyota's navigation.py is up.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.utils.recording import Paths  # noqa: E402
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as Arg
from launch.substitutions import PythonExpression

NAV_RVIZ = str(ROOT / "ros2_ws/src/hsrb_rosnav/hsrb_rosnav_config/rviz/hsr_navigation2.rviz")


def generate_launch_description():
    paths = Paths()
    arguments = [
        DeclareLaunchArgument(
            "target",
            default_value="",
            description="Object to search for, e.g. pringles can; empty starts the stack only.",
        ),
        DeclareLaunchArgument(
            "nav",
            default_value="true",
            choices=["true", "false"],
            description="false is the pick alone, the object already in view: no Nav2.",
        ),
        DeclareLaunchArgument(
            "grasp",
            default_value="true",
            choices=["true", "false"],
            description="Generate grasps and pick the object up once parked.",
        ),
        DeclareLaunchArgument(
            "active_perception",
            default_value="false",
            choices=["true", "false"],
            description="Fuse the found target from next best views before parking (under test).",
        ),
        DeclareLaunchArgument(
            "navigate_only",
            default_value="false",
            choices=["true", "false"],
            description="Reason and drive to the target; no SAM3 or GraspGenX.",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="auto",
            choices=["auto", "pickup", "grasp"],
            description="Auto lifts cylinders; grasp holds contact; pickup verifies lift.",
        ),
        DeclareLaunchArgument(
            "furniture",
            default_value="",
            description="Search this one furniture piece only (node id, or a unique name or label).",
        ),
        DeclareLaunchArgument("graph", default_value=str(paths.graph)),
        DeclareLaunchArgument(
            "map", default_value=str(paths.map_dir / "map_furniture.yaml")
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic", default_value="/omni_base_controller/cmd_vel"
        ),
        DeclareLaunchArgument(
            "localization", default_value="true", choices=["true", "false"]
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            choices=["true", "false"],
            description="MoveIt's RViz, opened by move_group.",
        ),
        DeclareLaunchArgument(
            "use_nav_rviz",
            default_value="false",
            choices=["true", "false"],
            description="Nav2's RViz (hsr_navigation2.rviz); ignored with nav:=false.",
        ),
        DeclareLaunchArgument(
            "nav_rviz_config",
            default_value=NAV_RVIZ,
            description="RViz config for use_nav_rviz; config/rviz/nav_demo.rviz is the "
                        "white-background one for filming.",
        ),
        DeclareLaunchArgument(
            "image_prefix",
            default_value="/remote",
            description="Prefix of RX's RGB and depth output topics.",
        ),
        # Two, not the simulator's three: each pick is a drive and a head sweep on the
        # robot, and a search that fails asks for a second round of as many.
        DeclareLaunchArgument("top_k", default_value="2"),
        DeclareLaunchArgument("bearings", default_value="12"),
        DeclareLaunchArgument(
            "startup_timeout",
            default_value="600",
            description="Seconds Ready may wait; long enough to seed AMCL by hand.",
        ),
        DeclareLaunchArgument(
            "rerun",
            default_value="false",
            choices=["true", "false"],
            description="Open the Rerun viewer and stream the Explore step live (needs a display).",
        ),
        DeclareLaunchArgument(
            "natural_language",
            default_value="false",
            choices=["true", "false"],
            description="target is a request ('bring me something to drink'); DeepSeek names the object.",
        ),
        DeclareLaunchArgument(
            "nbv_rviz",
            default_value="false",
            choices=["true", "false"],
            description="RViz with config/rviz/nbv.rviz: the Explore step's TSDF, views and grasp "
                        "as markers, as search.launch.py shows in simulation.",
        ),
        DeclareLaunchArgument(
            "shutdown_when_done",
            default_value="false",
            choices=["true", "false"],
            description="Shut every launched process down when the mission ends.",
        ),
    ]
    processes = []
    # Nav2 and the IK solver serve only the driving steps, so they follow nav.
    for name, default, command in [
        (
            "navigation",
            Arg("nav"),
            [
                str(ROOT / "launch/realrobot/nav2_real.launch.py"),
                ["map:=", Arg("map")],
                ["cmd_vel_topic:=", Arg("cmd_vel_topic")],
                ["localization:=", Arg("localization")],
            ],
        ),
        (
            "ik",
            Arg("nav"),
            [str(ROOT / "launch/ik_solver.launch.py"), "use_sim_time:=false"],
        ),
        (
            "move_group",
            "true",
            [
                str(ROOT / "launch/realrobot/grasp_real.launch.py"),
                ["use_rviz:=", Arg("use_rviz")],
            ],
        ),
    ]:
        arguments.append(
            DeclareLaunchArgument(
                "start_" + name, default_value=default, choices=["true", "false"]
            )
        )
        # Each is a whole launch of its own.
        processes.append(
            ExecuteProcess(
                cmd=["ros2", "launch", *command],
                name=name,
                output="both",
                condition=IfCondition(Arg("start_" + name)),
                sigterm_timeout="30",
            )
        )
    # Without nav there is no map frame for this config to show.
    processes.append(
        ExecuteProcess(
            cmd=["rviz2", "-d", Arg("nav_rviz_config")],
            name="nav_rviz",
            output="log",
            condition=IfCondition(
                PythonExpression(
                    ["'", Arg("use_nav_rviz"), "' == 'true' and '", Arg("nav"), "' == 'true'"]
                )
            ),
        )
    )
    # The robot's clock is the wall clock, so unlike the simulator's copy this
    # needs no use_sim_time for RViz to see current transforms.
    processes.append(
        ExecuteProcess(
            cmd=["rviz2", "-d", str(ROOT / "config/rviz/nbv.rviz")],
            name="nbv_rviz",
            output="log",
            condition=IfCondition(Arg("nbv_rviz")),
        )
    )
    # No settle time: Ready waits on every service the chosen steps use.
    tree = ExecuteProcess(
        cmd=[
            sys.executable,
            "-u",
            "-m",
            "core.pipeline.mission_tree",
            "--target",
            Arg("target"),
            "--furniture",
            Arg("furniture"),
            "--graph",
            Arg("graph"),
            "--top-k",
            Arg("top_k"),
            "--bearings",
            Arg("bearings"),
            "--startup-timeout",
            Arg("startup_timeout"),
            "--nav",
            Arg("nav"),
            "--grasp",
            Arg("grasp"),
            "--navigate-only",
            Arg("navigate_only"),
            "--mode",
            Arg("mode"),
            "--active-perception",
            Arg("active_perception"),
            "--rerun",
            Arg("rerun"),
            "--rviz",
            Arg("nbv_rviz"),
            "--natural-language",
            Arg("natural_language"),
        ],
        name="mission_tree",
        output="both",
        cwd=str(ROOT),
        # The pick is a child of the tree and inherits both.
        additional_env={"HSR_REAL_ROBOT": "1", "HSR_IMAGE_PREFIX": Arg("image_prefix")},
        condition=IfCondition(PythonExpression(["'", Arg("target"), "' != ''"])),
    )

    def finished(event, context):
        meanings = {
            0: "SUCCEEDED",
            1: "NOT FOUND",
            2: "FOUND but placement failed",
            3: "STARTUP / SERVICE ERROR",
            4: "PARKED but pickup failed",
            5: "PICKUP FAILED and the arm could not be stowed -- clear it by hand",
        }
        done = context.launch_configurations["shutdown_when_done"] == "true"
        after = (
            "Shutting everything down."
            if done
            else "The stack stays up; Ctrl+C to stop."
        )
        actions = [
            LogInfo(
                msg=f'[mission] {meanings.get(event.returncode, "FAILED")} '
                f"(exit {event.returncode}). {after}"
            )
        ]
        if done:
            actions.append(EmitEvent(event=Shutdown(reason="mission finished")))
        return actions

    return LaunchDescription(
        arguments
        + [
            RegisterEventHandler(OnProcessExit(target_action=tree, on_exit=finished)),
            *processes,
            tree,
        ]
    )
