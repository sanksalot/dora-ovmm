#!/usr/bin/env python3
"""Render an explore.rrd as an orbiting MP4: one held shot per fused view.

    python3 realrobot/visualizations/nbv_video.py outputs/active_perception/<run>/explore.rrd
    python3 realrobot/visualizations/nbv_video.py <run>/explore.rrd --seconds-per-view 4 --deg-per-second 15

The Rerun viewer has no video export, so this reads the recording back through
rerun's dataframe API and draws each view with matplotlib: the fused surface and
voxels, the target box, the fusion cube, every candidate view coloured by gain,
the chosen next view, the camera and its path, and the best grasp. The camera
orbits the target at --deg-per-second while each view is held --seconds-per-view.
Written with OpenCV's mp4v codec, the only MP4 encoder in this image.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rerun as rr  # noqa: E402

CONTENTS = {
    "/world/tsdf/cube": ["PoseTranslation3D", "HalfSize3D"],
    "/world/tsdf/surface": ["Position3D"],
    "/world/tsdf/voxels": ["PoseTranslation3D", "Color"],
    "/world/target/box": ["PoseTranslation3D", "HalfSize3D"],
    "/world/views/candidates": ["LineStrip3D", "Color"],
    "/world/views/next_best": ["LineStrip3D"],
    "/world/camera": ["Translation3D", "TransformMat3x3", "PinholeProjection", "Resolution"],
    "/world/camera/path": ["LineStrip3D"],
    "/world/grasp": ["Translation3D", "TransformMat3x3"],
    "/plots/gain/next_best_view": ["Scalar"],
    "/plots/grasp_quality": ["Scalar"],
}
DARK = dict(bg="#161619", ink="#e6e6e6", muted="#9a9aa3", surface="#5ac878", box="#e62828", cube="#8a8a92",
            path="#4f7cf0", camera="#f0f0f0", grasp="#f5a623", next="#2ee066")
LIGHT = dict(bg="#ffffff", ink="#2b2f36", muted="#6b7280", surface="#2f9e5a", box="#d62828", cube="#b0b4bc",
             path="#2f6fd6", camera="#2b2f36", grasp="#e08a00", next="#1fa64a")


def rgba(packed):
    """Rerun packs colours as 0xRRGGBBAA."""
    packed = np.asarray(packed, dtype=np.uint32)
    return np.stack([(packed >> 24) & 255, (packed >> 16) & 255, (packed >> 8) & 255], axis=-1) / 255.0


def column(table, name, row, default=None):
    col = table.column(name)
    return col[row].as_py() if col[row].is_valid else default


def load(path):
    rec = rr.dataframe.load_recording(str(path))
    table = rec.view(index="view", contents=CONTENTS).select().read_all()
    views = []
    last = {}
    for row in range(table.num_rows):
        item = {"view": column(table, "view", row)}
        for entity, components in CONTENTS.items():
            for component in components:
                key = f"{entity}:{component}"
                value = column(table, key, row)
                # Static or unchanged entities are logged once; carry them forward.
                if value is None:
                    value = last.get(key)
                item[key] = last[key] = value
        views.append(item)
    return views


def box_edges(centre, half):
    c, h = np.asarray(centre), np.asarray(half)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return [corners[[a, b]] for a, b in pairs]


def frustum(translation, mat, projection, resolution, depth=0.2):
    """Camera pyramid from the pinhole (camera looks along +z, x right, y down)."""
    t, r = np.asarray(translation), np.asarray(mat).reshape(3, 3)
    k = np.asarray(projection).reshape(3, 3)
    w, h = resolution
    corners = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        ray = np.linalg.inv(k) @ np.array([u, v, 1.0])
        corners.append(t + r @ (ray / ray[2] * depth))
    lines = [np.array([t, c]) for c in corners]
    lines += [np.array([corners[i], corners[(i + 1) % 4]]) for i in range(4)]
    return lines


def bounds(views, cube):
    """Centre and half-width that keep the cube, every candidate view and the camera in frame."""
    points = [np.asarray(cube[0]) - cube[1], np.asarray(cube[0]) + cube[1]]
    for item in views:
        for strip in item.get("/world/views/candidates:LineStrip3D") or []:
            points.append(np.asarray(strip))
        if item.get("/world/camera:Translation3D"):
            points.append(np.asarray(item["/world/camera:Translation3D"]))
    points = np.vstack([np.atleast_2d(p) for p in points])
    centre = np.asarray(cube[0], dtype=float)
    # Turn about the target in the plane, but centre the height on the data: the
    # candidates all stand above the can, and a box centred on it wastes its lower half.
    centre[2] = (points[:, 2].min() + points[:, 2].max()) / 2
    return centre, float(np.abs(points - centre).max()) * 1.02


def draw(ax, item, theme, azim, elev, cube, show_voxels, frame, zoom):
    ax.cla()
    ax.set_facecolor(theme["bg"])
    ax.set_axis_off()
    centre, half = frame
    for lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        lim(c - half, c + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    for seg in box_edges(cube[0], cube[1]):
        ax.plot(*seg.T, color=theme["cube"], lw=0.8, alpha=0.7)
    if show_voxels and item.get("/world/tsdf/voxels:PoseTranslation3D"):
        v = np.asarray(item["/world/tsdf/voxels:PoseTranslation3D"])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=rgba(item["/world/tsdf/voxels:Color"]), s=5, marker="s",
                   alpha=0.45, linewidths=0, depthshade=False)
    if item.get("/world/tsdf/surface:Position3D"):
        p = np.asarray(item["/world/tsdf/surface:Position3D"])
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=theme["surface"], s=4, linewidths=0, depthshade=False)
    if item.get("/world/target/box:PoseTranslation3D"):
        for seg in box_edges(item["/world/target/box:PoseTranslation3D"][0], item["/world/target/box:HalfSize3D"][0]):
            ax.plot(*seg.T, color=theme["box"], lw=1.4)
    if item.get("/world/views/candidates:LineStrip3D"):
        colours = rgba(item["/world/views/candidates:Color"])
        for strip, colour in zip(item["/world/views/candidates:LineStrip3D"], colours):
            s = np.asarray(strip)
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color=colour, lw=0.7, alpha=0.9)
    if item.get("/world/views/next_best:LineStrip3D"):
        for strip in item["/world/views/next_best:LineStrip3D"]:
            s = np.asarray(strip)
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color=theme["next"], lw=1.8)
    if item.get("/world/camera/path:LineStrip3D"):
        s = np.asarray(item["/world/camera/path:LineStrip3D"][0])
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=theme["path"], lw=1.6)
    if item.get("/world/camera:Translation3D"):
        for seg in frustum(item["/world/camera:Translation3D"][0], item["/world/camera:TransformMat3x3"][0],
                           item["/world/camera:PinholeProjection"][0], item["/world/camera:Resolution"][0]):
            ax.plot(*seg.T, color=theme["camera"], lw=1.2)
    if item.get("/world/grasp:Translation3D"):
        t = np.asarray(item["/world/grasp:Translation3D"][0])
        r = np.asarray(item["/world/grasp:TransformMat3x3"][0]).reshape(3, 3)
        tip = t + r[:, 2] * -0.08  # the logged arrow points back along the approach
        ax.plot(*np.array([tip, t]).T, color=theme["grasp"], lw=2.6, solid_capstyle="round")


def caption(item, total):
    gain = item.get("/plots/gain/next_best_view:Scalar")
    grasp = item.get("/plots/grasp_quality:Scalar")
    parts = [f"view {item['view']} of {total}"]
    parts.append(f"next view gain {int(gain[0])} voxels" if gain else "no view gains anything")
    parts.append(f"best grasp {grasp[0]:.2f}" if grasp else "no grasp yet")
    return "   ·   ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rrd", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: explore.mp4 beside the .rrd")
    parser.add_argument("--seconds-per-view", type=float, default=3.0)
    parser.add_argument("--deg-per-second", type=float, default=18.0, help="orbit speed; 0 holds still")
    parser.add_argument("--elev", type=float, default=28.0, help="camera elevation, degrees")
    parser.add_argument("--start-azim", type=float, default=-60.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--zoom", type=float, default=2.2, help="how much of the picture the scene fills")
    parser.add_argument("--light", action="store_true", help="white background instead of dark")
    parser.add_argument("--no-voxels", action="store_true", help="surface points only")
    parser.add_argument("--frame", type=Path, default=None, help="also save one mid-run frame as PNG")
    args = parser.parse_args()

    views = load(args.rrd)
    if not views:
        raise SystemExit(f"{args.rrd}: no views on the 'view' timeline")
    theme = LIGHT if args.light else DARK
    cube = (views[0]["/world/tsdf/cube:PoseTranslation3D"][0], views[0]["/world/tsdf/cube:HalfSize3D"][0])
    frame = bounds(views, cube)
    width, height = (int(v) for v in args.size.lower().split("x"))
    out = args.out or args.rrd.with_name("explore.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open an mp4v writer")

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=theme["bg"])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    text = fig.text(0.03, 0.95, "", fontsize=13, color=theme["ink"], ha="left", va="top")
    frames_per_view = max(1, int(round(args.seconds_per_view * args.fps)))
    frame_index, mid_saved = 0, False
    for item in views:
        text.set_text(caption(item, len(views)))
        for _ in range(frames_per_view):
            azim = args.start_azim + args.deg_per_second * frame_index / args.fps
            draw(ax, item, theme, azim, args.elev, cube, not args.no_voxels, frame, args.zoom)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.write(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
            frame_index += 1
            if args.frame and not mid_saved and item["view"] == max(1, len(views) // 2):
                cv2.imwrite(str(args.frame), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
                mid_saved = True
    writer.release()
    print(f"{out}: {frame_index} frames, {frame_index / args.fps:.1f} s, {len(views)} views")


if __name__ == "__main__":
    main()
