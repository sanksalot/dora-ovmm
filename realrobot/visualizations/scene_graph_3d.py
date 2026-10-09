"""The scene graph in 3D: boxes in the fused RGB-D cloud, edges between them, labels.

    python3 realrobot/visualizations/scene_graph_3d.py                  # $RECORDING -> png
    python3 realrobot/visualizations/scene_graph_3d.py --elev 30 --azim -40
    python3 realrobot/visualizations/scene_graph_3d.py --view            # Open3D window instead

Furniture is drawn as its fitted box in its room's colour, an object on or in
furniture as a small box joined to it (solid for on, dashed for in). Rooms float
above their furniture. Labels name the rooms, the objects, and the furniture they
sit on; --full labels every piece and adds the other detections as faint grey
boxes. The fused keyframe cloud (outputs/realrobot/<recording>/keyframes/scene.ply)
gives the context, cut below --ceiling so the rooms can be seen into from above.

Writes outputs/realrobot/<recording>/scene_graph_3d.png. --view opens the same
scene in Open3D, with labels, on a machine with a display (this container has
none: `xhost +local:` on the host first, see view_grasps.py).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.scene_graph import graph as sg  # noqa: E402
from core.utils.recording import Paths  # noqa: E402
from scene_graph_figure import INK, NONE_COLOUR, ROOM_COLOURS, footprint_corners  # noqa: E402

CEILING = 2.0      # m, cloud above this is dropped so the view looks into the rooms
CLOUD_POINTS = 1_000_000  # drawn in the png; Open3D gets the voxel-downsampled cloud
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(data):
    """The eight corners of a node's box: furniture turned with its footprint, objects axis-aligned."""
    lower, upper = np.asarray(data["bounds"], dtype=float)
    if data["movable"]:
        floor = np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
    else:
        floor = footprint_corners(data)
    return np.vstack([np.column_stack([floor, np.full(4, lower[2])]), np.column_stack([floor, np.full(4, upper[2])])])


def scene(graph, cloud_path, ceiling, full=False):
    """Everything the two renderers share: boxes, edges, labels and the context cloud."""
    furniture = {n: d for n, d in graph.nodes(data=True) if not d["movable"]}
    objects = {n: d for n, d in graph.nodes(data=True) if d["movable"]}
    rooms = sorted({d["room"] for d in furniture.values() if d["room"]})
    colour = {room: ROOM_COLOURS[i % len(ROOM_COLOURS)] for i, room in enumerate(rooms)}
    colour[None] = NONE_COLOUR
    support = {s: (t, e["relation"]) for s, t, e in graph.edges(data=True) if e["relation"] in ("on", "in")}
    boxes = []  # (corners, colour, width, alpha)
    for node, data in furniture.items():
        boxes.append((box_corners(data), colour[data["room"]], 1.6, 1.0))
    for node, data in objects.items():
        if node in support:
            boxes.append((box_corners(data), colour[furniture[support[node][0]]["room"]], 1.0, 1.0))
        elif full:
            boxes.append((box_corners(data), "#9E9E9E", 0.5, 0.45))
    edges = []  # (from, to, colour, relation)
    for node, (parent, relation) in support.items():
        edges.append((np.asarray(objects[node]["centroid"]), np.asarray(furniture[parent]["centroid"]),
                      colour[furniture[parent]["room"]], relation))
    labels = []  # (position, text, colour, size)
    holders = {parent for parent, _ in support.values()}
    for node, data in furniture.items():
        if not full and node not in holders:
            continue
        top = np.asarray(data["centroid"], dtype=float)
        top[2] = data["bounds"][1][2] + 0.05
        labels.append((top, data["label"], colour[data["room"]], "small"))
    for node, (parent, relation) in support.items():
        top = np.asarray(objects[node]["centroid"], dtype=float)
        top[2] = objects[node]["bounds"][1][2] + 0.03
        labels.append((top, objects[node]["label"], INK, "small"))
    for room in rooms:
        pieces = np.array([d["centroid"][:2] for d in furniture.values() if d["room"] == room])
        labels.append((np.r_[pieces.mean(0), ceiling + 1.4], room, colour[room], "large"))
    points = colours = None
    if cloud_path and Path(cloud_path).exists():
        import trimesh
        cloud = trimesh.load(str(cloud_path), process=False)
        points = np.asarray(cloud.vertices, dtype=np.float32)
        colours = np.asarray(cloud.colors)[:, :3].astype(np.float32) / 255
        keep = points[:, 2] < ceiling
        points, colours = points[keep], colours[keep]
    return dict(boxes=boxes, edges=edges, labels=labels, points=points, colours=colours, rooms=rooms, colour=colour)


def render(built, output, elev, azim):
    figure = plt.figure(figsize=(18, 12))
    axes = figure.add_subplot(projection="3d", computed_zorder=False)
    if built["points"] is not None:
        points, colours = built["points"], built["colours"]
        pick = np.random.default_rng(0).choice(len(points), min(len(points), CLOUD_POINTS), replace=False)
        axes.scatter(*points[pick].T, s=0.3, c=colours[pick], alpha=0.5, linewidths=0, zorder=1)
    for corners, tint, width, alpha in built["boxes"]:
        axes.add_collection3d(Line3DCollection(corners[EDGES], colors=tint, linewidths=width, alpha=alpha, zorder=3))
    for start, end, tint, relation in built["edges"]:
        axes.plot(*np.column_stack([start, end]), color=tint, linewidth=1.2,
                  linestyle="-" if relation == "on" else (0, (3, 2)), zorder=4)
    for position, text, tint, size in built["labels"]:
        if size == "large":
            axes.text(*position, text, color="white", fontsize=12, weight="bold", ha="center", va="center", zorder=6,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor=tint, edgecolor="none"))
        else:
            axes.text(*position, text, color=INK, fontsize=6, ha="center", va="bottom", zorder=5,
                      bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=tint, linewidth=0.6, alpha=0.9))
    everything = np.vstack([corners for corners, *_ in built["boxes"]])
    lower, upper = everything.min(0) - 0.5, everything.max(0) + 0.5
    upper[2] = max(upper[2], CEILING + 1.8)
    axes.set_xlim(lower[0], upper[0]); axes.set_ylim(lower[1], upper[1]); axes.set_zlim(lower[2], upper[2])
    axes.set_box_aspect(upper - lower, zoom=1.1)
    axes.view_init(elev, azim)
    axes.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(figure)


def view(built, title):
    """The same scene in an Open3D window, with 3D labels."""
    import open3d as o3d
    from open3d.visualization import gui, rendering
    app = gui.Application.instance
    app.initialize()
    window = o3d.visualization.O3DVisualizer(title, 1600, 1000)
    window.show_settings = False
    window.set_background((1, 1, 1, 1), None)
    if built["points"] is not None:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(built["points"].astype(float)))
        cloud.colors = o3d.utility.Vector3dVector(built["colours"].astype(float))
        window.add_geometry("scene", cloud.voxel_down_sample(0.02))
    for index, (corners, tint, width, alpha) in enumerate(built["boxes"]):
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(EDGES))
        lines.paint_uniform_color(matplotlib.colors.to_rgb(tint))
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.0 * width
        window.add_geometry(f"box{index}", lines, material)
    for index, (start, end, tint, relation) in enumerate(built["edges"]):
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector([start, end]), o3d.utility.Vector2iVector([[0, 1]]))
        lines.paint_uniform_color(matplotlib.colors.to_rgb(tint))
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.0
        window.add_geometry(f"edge{index}", lines, material)
    for position, text, tint, size in built["labels"]:
        label = window.add_3d_label(position, text)
        label.color = gui.Color(*matplotlib.colors.to_rgb(tint))
        label.scale = 1.6 if size == "large" else 0.9
    window.reset_camera_to_default()
    app.add_window(window)
    print("Drag to turn, scroll to zoom, Ctrl-drag to pan. The Actions menu exports a screenshot.", flush=True)
    app.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--cloud", type=Path, help="context point cloud (default the recording's keyframes/scene.ply)")
    parser.add_argument("--no-cloud", action="store_true")
    parser.add_argument("--ceiling", type=float, default=CEILING, help=f"drop cloud above this height (default {CEILING} m)")
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true", help="label every piece of furniture, draw proximity-only detections")
    parser.add_argument("--view", action="store_true", help="open in Open3D instead of writing the png")
    args = parser.parse_args()
    paths = Paths()
    graph = sg.load(args.graph or paths.graph)
    cloud = None if args.no_cloud else (args.cloud or paths.keyframes / "scene.ply")
    built = scene(graph, cloud, args.ceiling, args.full)
    print(f"{len(built['rooms'])} rooms, {len(built['boxes'])} boxes, {len(built['edges'])} on/in edges, "
          f"{0 if built['points'] is None else len(built['points'])} cloud points", flush=True)
    if args.view:
        view(built, paths.name)
        return
    output = args.output or ROOT / "outputs/realrobot" / paths.name / "scene_graph_3d.png"
    render(built, output, args.elev, args.azim)
    print(f"-> {output}", flush=True)


if __name__ == "__main__":
    main()
