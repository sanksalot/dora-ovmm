"""Simulation bringup for the behaviour-tree query-to-grasp mission."""

import subprocess
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as Arg

ROOT = Path(__file__).resolve().parents[1]


def stop_gazebo(context):
    """Signal Gazebo itself, not its launcher.

    hsrb_gazebo_launch starts Gazebo through the `ign` ruby wrapper, which does
    not pass Ctrl+C on; stopping only the wrapper leaves Gazebo running with no
    parent. Gazebo's own processes stop their server and GUI within seconds."""
    if context.launch_configurations["start_simulation"] == "true":
        # Anchored to the start of the command line, so it cannot match a shell.
        subprocess.run(["pkill", "-INT", "-f", "^ign gazebo "], check=False)
    return []


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "target",
            default_value="",
            description="Object to search for, e.g. pringles; typed in the page with web:=true",
        ),
        DeclareLaunchArgument(
            "graph", default_value=str(ROOT / "config/scene_graph/kitchen_objects.json")
        ),
        DeclareLaunchArgument(
            "world", default_value=str(ROOT / "worlds/kitchen_objects.world")
        ),
        DeclareLaunchArgument(
            "use_rviz", default_value="false", choices=["true", "false"]
        ),
        DeclareLaunchArgument("top_k", default_value="3"),
        DeclareLaunchArgument("bearings", default_value="12"),
        DeclareLaunchArgument("startup_timeout", default_value="180"),
        DeclareLaunchArgument(
            "map", default_value=str(ROOT / "config/map/kitchen_objects.yaml")
        ),
        DeclareLaunchArgument(
            "params_file", default_value=str(ROOT / "config/nav2/kitchen_objects.yaml")
        ),
        DeclareLaunchArgument(
            "grasp",
            default_value="true",
            choices=["true", "false"],
            description="Generate grasps and pick the object up once parked.",
        ),
        DeclareLaunchArgument(
            "natural_language",
            default_value="false",
            choices=["true", "false"],
            description="target is a request ('bring me something to drink'); DeepSeek names the object.",
        ),
        DeclareLaunchArgument(
            "furniture",
            default_value="",
            description="Search this one furniture piece only (node id, or a unique name or label), "
                        "as search_real.launch.py: no remembered places, no DeepSeek.",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="auto",
            choices=["auto", "pickup", "grasp"],
            description="Auto lifts cylinders; grasp holds contact; pickup verifies lift.",
        ),
        DeclareLaunchArgument(
            "navigate_only",
            default_value="false",
            choices=["true", "false"],
            description="Reason and drive to the target; no SAM3 or GraspGenX.",
        ),
        DeclareLaunchArgument(
            "active_perception",
            default_value="false",
            choices=["true", "false"],
            description="Fuse the found target from next best views before parking (under test).",
        ),
        DeclareLaunchArgument(
            "rerun",
            default_value="false",
            choices=["true", "false"],
            description="Open the Rerun viewer and stream the Explore step live (needs a display).",
        ),
        DeclareLaunchArgument(
            "nbv_rviz",
            default_value="false",
            choices=["true", "false"],
            description="Publish the Explore step as markers for config/rviz/nbv.rviz.",
        ),
        DeclareLaunchArgument(
            "shutdown_when_done",
            default_value="false",
            choices=["true", "false"],
            description="Shut every launched process down when the mission ends.",
        ),
        DeclareLaunchArgument(
            "web",
            default_value="false",
            choices=["true", "false"],
            description="Start the stack without a mission; queries come from the web dashboard.",
        ),
    ]
    processes = []
    for name, command in [
        (
            "simulation",
            [str(ROOT / "launch/apartment.launch.py"), ["world:=", Arg("world")]],
        ),
        (
            "navigation",
            [
                "hsrb_rosnav_config",
                "navigation_launch.py",
                "use_sim_time:=true",
                ["map:=", Arg("map")],
                ["params_file:=", Arg("params_file")],
            ],
        ),
        ("ik", [str(ROOT / "launch/ik_solver.launch.py"), "use_sim_time:=true"]),
        (
            "move_group",
            [
                str(ROOT / "launch/move_group.launch.py"),
                ["use_rviz:=", Arg("use_rviz")],
            ],
        ),
    ]:
        arguments.append(
            DeclareLaunchArgument(
                "start_" + name, default_value="true", choices=["true", "false"]
            )
        )
        # Each is a whole launch of its own. Gazebo takes longer than launch's
        # default 5 s to stop, and killing its launch sooner leaves it running.
        processes.append(
            ExecuteProcess(
                cmd=["ros2", "launch", *command],
                name=name,
                output="both",
                condition=IfCondition(Arg("start_" + name)),
                sigterm_timeout="30",
            )
        )
    common = [
        "--target",
        Arg("target"),
        "--graph",
        Arg("graph"),
        "--top-k",
        Arg("top_k"),
        "--bearings",
        Arg("bearings"),
        "--startup-timeout",
        Arg("startup_timeout"),
        "--grasp",
        Arg("grasp"),
        "--furniture",
        Arg("furniture"),
    ]
    # No mission with web:=true: queries come from the web dashboard instead.
    tree = ExecuteProcess(
        cmd=[
            sys.executable,
            "-u",
            "-m",
            "core.pipeline.mission_tree",
            *common,
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
        condition=UnlessCondition(Arg("web")),
    )

    # Started here rather than by hand: RViz reads TF stamped from /clock, so on
    # the wall clock every transform looks ancient and the robot model never
    # moves. The launch knows this is the simulator, so it can just say so.
    nbv_rviz = ExecuteProcess(
        cmd=[
            "ros2", "run", "rviz2", "rviz2",
            "-d", str(ROOT / "config/rviz/nbv.rviz"),
            "--ros-args", "-p", "use_sim_time:=true",
        ],
        name="nbv_rviz",
        output="log",
        condition=IfCondition(Arg("nbv_rviz")),
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
            else "Simulation remains open; Ctrl+C to stop."
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
            # This file is the simulator. .devcontainer/runtime.env exports
            # HSR_REAL_ROBOT=1 into every terminal, and mission_tree reads it to
            # pick its clock: inherited here it gave the navigator the wall
            # clock while every TF stamp came from /clock, so "fresh
            # map-to-base localization" compared 1.8e9 s against 190 s, never
            # passed, and the run died on the startup deadline. Clear it, so a
            # sim launch is a sim launch whatever the shell was set up for.
            SetEnvironmentVariable("HSR_REAL_ROBOT", ""),
            RegisterEventHandler(OnProcessExit(target_action=tree, on_exit=finished)),
            RegisterEventHandler(
                OnShutdown(on_shutdown=[OpaqueFunction(function=stop_gazebo)])
            ),
            *processes,
            nbv_rviz,
            tree,
        ]
    )
