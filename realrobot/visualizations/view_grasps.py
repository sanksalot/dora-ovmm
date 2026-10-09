"""Turn a captured cloud and its grasps round in an Open3D window, and screenshot it.

    python3 realrobot/visualizations/view_grasps.py outputs/realrobot/visualizations/<run>
    python3 realrobot/visualizations/view_grasps.py outputs/detections/<run> --name grasps_snack-can

Reads the `<name>_cloud.ply` and `<name>_hands.ply` that rgbd_frame_capture.py and
core.grasping.picture write beside their figures. Needs only Open3D, no ROS and
nothing from this repository, so it runs on the host or any machine with a display.
From the dev container the host's display is mounted but not authorised: run
`xhost +local:` in a terminal on the host once per login, then this works here too.
In the window, drag to turn, scroll to zoom, and press P to save a screenshot into
the current directory (H lists every key).
"""

import argparse
import sys
from pathlib import Path

import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="a run folder holding <name>_cloud.ply and <name>_hands.ply")
    parser.add_argument("--name", default="grasps", help="file prefix (default grasps; a live run's is grasps_<object>)")
    parser.add_argument("--point-size", type=float, default=4.0)
    args = parser.parse_args()
    cloud = o3d.io.read_point_cloud(str(args.folder / f"{args.name}_cloud.ply"))
    hands = o3d.io.read_triangle_mesh(str(args.folder / f"{args.name}_hands.ply"))
    if cloud.is_empty() or hands.is_empty():
        parser.error(f"no {args.name}_cloud.ply / {args.name}_hands.ply in {args.folder}")
    hands.compute_vertex_normals()
    print(f"{len(cloud.points)} points, hands with {len(hands.triangles)} triangles. "
          "Drag to turn, scroll to zoom, P saves a screenshot here, Q quits.", flush=True)
    viewer = o3d.visualization.Visualizer()
    if not viewer.create_window(window_name=str(args.folder), width=1280, height=960):
        sys.exit("No window: the display refused the connection. From the dev container, run "
                 "`xhost +local:` in a terminal on the host once, then retry; or run this on the host.")
    viewer.add_geometry(cloud)
    viewer.add_geometry(hands)
    viewer.get_render_option().point_size = args.point_size
    viewer.get_render_option().background_color = [1.0, 1.0, 1.0]
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    main()
