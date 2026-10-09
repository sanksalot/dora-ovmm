"""The scene graph as stacked layers, for a paper or a slide.

    python3 realrobot/visualizations/scene_graph_figure.py                 # $RECORDING
    python3 realrobot/visualizations/scene_graph_figure.py --near          # proximity edges too
    python3 realrobot/visualizations/scene_graph_figure.py --graph G.json --map map.yaml --output fig.png

Four planes in oblique view, abstraction rising upward, as 3D scene graph papers
draw it: the laser map with each furniture footprint coloured by its room, the
objects above their places on the map, the furniture above that, and the rooms on
top. Every edge joins neighbouring layers: an object to the furniture it is on
(solid) or in (dashed), furniture to its room.

Kept simple by default: objects of one label on one piece of furniture share a
node ("monitor x4"), furniture with nothing on it shares a node per label and room
("chair x7", its footprints still drawn on the map), and detections the graph
relates only by proximity (posters, whiteboards, lights) are left out. --full
draws every piece of furniture on its own and those detections as grey dots;
--near then also draws their edges, and --label-all names them.

Writes outputs/realrobot/<recording>/scene_graph.png.
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.scene_graph import graph as sg  # noqa: E402
from core.utils.recording import Paths  # noqa: E402

# Oblique projection of a floor point (u, v) on layer z: X = u + v*SHEAR, Y = v*FLATTEN + z*RISE.
# u runs along the map's y axis, v along its x, so the lab's long side lies across the page.
SHEAR, FLATTEN, RISE = 0.45, 0.30, 6.5
MARGIN = 1.5  # m of map around the furniture
LAYERS = ["map", "objects", "furniture", "rooms"]
ROOM_COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
NONE_COLOUR = "#9E9E9E"
INK = "#2B2B2B"
FREE, OCCUPIED = 254, 0


def project(xy, layer):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    u, v = -xy[:, 1], xy[:, 0]
    return np.column_stack([u + v * SHEAR, v * FLATTEN + LAYERS.index(layer) * RISE])


def read_map(path):
    meta = yaml.safe_load(Path(path).read_text())
    image = cv2.imread(str(Path(path).parent / meta["image"]), cv2.IMREAD_UNCHANGED)
    return image, float(meta["resolution"]), np.asarray(meta["origin"][:2], dtype=float)


def footprint_corners(data):
    centre, dimensions, yaw = sg.footprint(data)
    half = np.array(dimensions[:2]) / 2
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return corners @ turn.T + centre[:2]


def draw_plane(axes, lower, upper, layer, name):
    corners = np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
    screen = project(corners, layer)
    axes.fill(*screen.T, facecolor="white", edgecolor="#D0D0D0", linewidth=0.8, alpha=0.6, zorder=0)
    left = screen[np.argmin(screen[:, 0])]
    axes.text(left[0] - 0.4, left[1], name, ha="right", va="center", fontsize=11, color="#777777", style="italic")


def repel(axes, texts, iterations=400):
    """Nudge overlapping labels apart, by the smallest move that separates each pair,
    using the extents the renderer actually gives them. Returns each label's centre."""
    axes.figure.canvas.draw()
    to_data = axes.transData.inverted()
    boxes = [text.get_window_extent().transformed(to_data) for text in texts]
    centres = np.array([[(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2] for b in boxes])
    offsets = np.array([text.get_position() for text in texts]) - centres
    half = np.array([[b.width, b.height] for b in boxes]) / 2 + 0.04
    for _ in range(iterations):
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                gap = centres[j] - centres[i]
                overlap = half[i] + half[j] - np.abs(gap)
                if (overlap > 0).all():
                    moved = True
                    axis = int(np.argmin(overlap))
                    shift = overlap[axis] / 2 * (1 if gap[axis] >= 0 else -1)
                    centres[i, axis] -= shift
                    centres[j, axis] += shift
        if not moved:
            break
    for text, centre, offset in zip(texts, centres, offsets):
        text.set_position(centre + offset)
    return centres


def draw_map(axes, image, resolution, origin, lower, upper):
    """The occupancy grid within [lower, upper] on the map plane, free white, walls dark."""
    height = image.shape[0]
    columns = np.clip(((np.array([lower[0], upper[0]]) - origin[0]) / resolution).astype(int), 0, image.shape[1])
    rows = np.clip((height - (np.array([upper[1], lower[1]]) - origin[1]) / resolution).astype(int), 0, height)
    crop = image[rows[0]:rows[1], columns[0]:columns[1]]
    # Rows are map x (flipped so x rises with the row), columns are -y: the projection's (v, u).
    grid = np.flipud(crop).T[:, ::-1]
    picture = np.full(grid.shape + (4,), 1.0)
    picture[grid <= OCCUPIED + 50] = matplotlib.colors.to_rgba("#4A4A4A")
    picture[(grid > OCCUPIED + 50) & (grid < FREE - 50)] = (1, 1, 1, 0)  # unknown
    x0, x1 = origin[0] + columns * resolution
    y0, y1 = origin[1] + (height - rows[::-1]) * resolution
    shown = axes.imshow(picture, origin="lower", extent=[-y1, -y0, x0, x1], interpolation="nearest", zorder=1)
    shown.set_transform(Affine2D(np.array([[1, SHEAR, 0], [0, FLATTEN, 0], [0, 0, 1]])) + axes.transData)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true", help="every piece of furniture on its own, proximity-only detections as dots")
    parser.add_argument("--near", action="store_true", help="with --full: draw proximity edges, faintly")
    parser.add_argument("--label-all", action="store_true", help="with --full: name the proximity-only objects too")
    args = parser.parse_args()
    paths = Paths()
    graph = sg.load(args.graph or paths.graph)
    image, resolution, origin = read_map(args.map or paths.map)
    output = args.output or ROOT / "outputs/realrobot" / paths.name / "scene_graph.png"

    furniture = {n: d for n, d in graph.nodes(data=True) if not d["movable"]}
    objects = {n: d for n, d in graph.nodes(data=True) if d["movable"]}
    rooms = sorted({d["room"] for d in furniture.values() if d["room"]})
    colour = {room: ROOM_COLOURS[i % len(ROOM_COLOURS)] for i, room in enumerate(rooms)}
    colour[None] = NONE_COLOUR
    support = {s: (t, e["relation"]) for s, t, e in graph.edges(data=True) if e["relation"] in ("on", "in")}
    near = [(s, t) for s, t, e in graph.edges(data=True) if e["relation"] == "near"]

    xy = np.array([d["centroid"][:2] for d in graph.nodes.values()])
    lower, upper = xy.min(0) - MARGIN, xy.max(0) + MARGIN
    corners = np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
    extent = np.vstack([project(corners, "map"), project(corners, "rooms")])
    span = np.ptp(extent, axis=0) + 2
    figure, axes = plt.subplots(figsize=(18, 18 * span[1] / span[0]))
    axes.set_xlim(extent[:, 0].min() - 1, extent[:, 0].max() + 1)
    axes.set_ylim(extent[:, 1].min() - 1, extent[:, 1].max() + 1)
    for layer, name in zip(LAYERS, ["Map", "Objects", "Furniture", "Rooms"]):
        draw_plane(axes, lower, upper, layer, name)
    draw_map(axes, image, resolution, origin, lower, upper)

    # Map plane: footprints in room colour.
    for node, data in furniture.items():
        corners = project(footprint_corners(data), "map")
        axes.fill(*corners.T, facecolor=colour[data["room"]], edgecolor=colour[data["room"]],
                  alpha=0.45, linewidth=0.8, zorder=2)

    # Furniture nodes: a piece with objects on it stands alone; the rest merge per label and room.
    pieces = {}  # node key -> (text, members)
    holders = {parent for parent, _ in support.values()}
    for node, data in furniture.items():
        if args.full or node in holders:
            pieces[node] = (data["label"], [node])
        else:
            key = ("group", data["label"], data["room"])
            pieces.setdefault(key, (data["label"], []))[1].append(node)
    pieces = {key: (text + (f" ×{len(members)}" if len(members) > 1 else ""), members)
              for key, (text, members) in pieces.items()}
    piece_of = {node: key for key, (_, members) in pieces.items() for node in members}
    where = {key: np.mean([furniture[m]["centroid"][:2] for m in members], axis=0) for key, (_, members) in pieces.items()}

    # Labels first, so they can be pushed apart before any edge is drawn to them.
    labels, anchors = {}, {}
    for key, (text, members) in pieces.items():
        at = project(where[key], "furniture")[0]
        labels[key] = axes.text(*at, text, fontsize=8, color=INK, ha="center", va="center", zorder=8,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                          edgecolor=colour[furniture[members[0]]["room"]], linewidth=1.4))
    grouped = collections.defaultdict(list)
    for node, data in objects.items():
        if node in support:
            grouped[(data["label"], *support[node])].append(node)
    for key, members in grouped.items():
        label, parent, relation = key
        centre = np.mean([objects[m]["centroid"][:2] for m in members], axis=0)
        anchors[key] = project(centre, "objects")[0]
        text = label + (f" ×{len(members)}" if len(members) > 1 else "")
        labels[key] = axes.text(anchors[key][0] + 0.2, anchors[key][1] + 0.02, text, fontsize=8, color=INK,
                                va="center", zorder=6,
                                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    centres = {room: [] for room in rooms}
    for node, data in furniture.items():
        if data["room"]:
            centres[data["room"]].append(data["centroid"][:2])
    for room in rooms:
        at = project(np.mean(centres[room], axis=0), "rooms")[0]
        labels[room] = axes.text(*at, room, fontsize=12, color="white", ha="center", va="center", weight="bold",
                                 zorder=9, bbox=dict(boxstyle="round,pad=0.45", facecolor=colour[room], edgecolor="none"))
    keys = list(labels)
    placed = dict(zip(keys, repel(axes, [labels[k] for k in keys])))

    # Furniture to its footprints (every member's) and to its room.
    for key, (text, members) in pieces.items():
        top = placed[key]
        tint = colour[furniture[members[0]]["room"]]
        for member in members:
            foot = project(furniture[member]["centroid"][:2], "map")[0]
            axes.plot([foot[0], top[0]], [foot[1], top[1]], color=tint, linewidth=0.5, alpha=0.3, zorder=2)
        if furniture[members[0]]["room"]:
            room = placed[furniture[members[0]]["room"]]
            axes.plot([top[0], room[0]], [top[1], room[1]], color=tint, linewidth=0.7, alpha=0.5, zorder=7)

    # Objects: a dot where each stands, its label beside it, an edge up to its furniture.
    if args.full:
        for node, data in objects.items():
            if node not in support:
                at = project(data["centroid"][:2], "objects")[0]
                axes.plot(*at, "o", markersize=3, color="#B0B0B0", markeredgewidth=0, zorder=3)
                if args.label_all:
                    axes.text(at[0] + 0.12, at[1], data["label"], fontsize=6, color="#909090", va="center", zorder=3)
    for key in grouped:
        label, parent, relation = key
        at, piece = anchors[key], placed[piece_of[parent]]
        tint = colour[furniture[parent]["room"]]
        axes.plot([at[0], piece[0]], [at[1], piece[1]], color=tint, linewidth=1.1, alpha=0.8,
                  linestyle="-" if relation == "on" else (0, (3, 2)), zorder=4)
        axes.plot(*at, "o", markersize=7, color=tint, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        anchor = np.array(labels[key].get_position())
        if np.linalg.norm(anchor - at) > 0.6:  # the label was pushed away: a leader back to its dot
            axes.plot([at[0], anchor[0] - 0.05], [at[1], anchor[1]], color="#8A8A8A", linewidth=0.7, zorder=4)
    if args.near and args.full:
        for source, target in near:
            a = project(graph.nodes[source]["centroid"][:2], "objects")[0]
            b = placed[piece_of[target]] if target in furniture else project(graph.nodes[target]["centroid"][:2], "objects")[0]
            axes.plot([a[0], b[0]], [a[1], b[1]], color="#C8C8C8", linewidth=0.5, alpha=0.7, zorder=3)

    handles = [plt.Line2D([], [], color=INK, linewidth=1.2, label="on"),
               plt.Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (3, 2)), label="in")]
    if args.full:
        handles.append(plt.Line2D([], [], marker="o", color="#B0B0B0", linestyle="", markersize=4,
                                  label=f"detected, related only by proximity ({len(objects) - len(support)})"))
    axes.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    axes.set_aspect("equal")
    axes.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    print(f"{len(rooms)} rooms, {len(furniture)} furniture as {len(pieces)} nodes, {len(support)} objects on or in "
          f"furniture, {len(objects) - len(support)} by proximity only -> {output}", flush=True)


if __name__ == "__main__":
    main()
