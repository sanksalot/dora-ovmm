"""A picture of what the pick is about to work on: the target's point cloud, which is
what SAM3's mask became in 3D, with the best few grasps drawn on it as the open hand.

The hand is the mesh GraspGenX was registered with (docker/graspgenx/x_grippers/
hsrc_hand/vis_mesh.obj), so what is drawn is what the model placed. On a camera
image the measured depth hides the parts of the hand behind the object, and the
hidden parts are ghosted, so a finger reaching round the far side reads as such.

Saved beside the run's detections (core.perception.sam3_client.run_folder), for
showing afterwards. It is never on the path of the pick itself: DETECTIONS=0 switches
it off, and a failure to draw is reported, not raised."""

import os
from functools import lru_cache

import cv2
import numpy as np

from core.perception import sam3_client

SHOWN = 4      # grasps drawn, in the order MTC will try them
FINGER = 0.06  # m, drawn finger length; only for the picture
COLOURS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
# How much of the hand's colour shows: the first grasp solid, the rest lighter, and
# the parts the depth image hides as a ghost.
FILL, FILL_OTHERS, GHOST = 0.65, 0.45, 0.15


def hand_outline(palm, width, rotation, pad):
    """The open hand as a polyline in the cloud's frame: finger, back of the hand, finger,
    and a short stem pointing back along the approach."""
    canonical = palm @ np.linalg.inv(rotation)
    closing, approach = canonical[:3, 0], canonical[:3, 2]
    centre = canonical[:3, 3] + canonical[:3, :3] @ pad
    left, right = centre + closing * width / 2, centre - closing * width / 2
    back = -approach * FINGER
    middle = (left + right) / 2 + back
    return np.array([left, left + back, middle, middle + back * 0.8, middle, right + back, right])


@lru_cache(maxsize=None)
def gripper_mesh():
    """The registered hand, open, in GraspGenX's gripper frame: (vertices, faces)."""
    import trimesh
    from core.grasping import pick
    mesh = trimesh.load(str(pick.GRIPPER_DIR / "vis_mesh.obj"), force="mesh")
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def hand_vertices(palm):
    """The mesh's vertices placed at a palm pose, in the pose's frame."""
    from core.grasping import graspgenx_client, pick
    vertices, _ = gripper_mesh()
    model = np.asarray(palm) @ np.linalg.inv(graspgenx_client.canonical_from_palm(pick.GRIPPER))
    return vertices @ model[:3, :3].T + model[:3, 3]


def bgr(colour):
    return tuple(int(colour[i:i + 2], 16) for i in (5, 3, 1))


def draw_hands_image(picture, palms, k, camera_from_world, depth=None, labels=None):
    """Draw hands on a camera image, nearest surface winning, depth-tested against the
    measured depth when given. Returns the drawn image."""
    _, faces = gripper_mesh()
    height, width = picture.shape[:2]
    nearest = np.full((height, width), np.inf, np.float32)
    colour = np.zeros((height, width, 3), np.float32)
    owner = np.full((height, width), -1, np.int8)
    for index, palm in enumerate(palms):
        local = hand_vertices(camera_from_world @ palm)
        z = local[:, 2]
        if (z <= 0).any():
            print(f"[GRASP] hand {index + 1} crosses the image plane; not drawn", flush=True)
            continue
        pixels = np.column_stack([k[0, 0] * local[:, 0] / z + k[0, 2], k[1, 1] * local[:, 1] / z + k[1, 2]])
        triangles = local[faces]
        depths = triangles[:, :, 2].mean(1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        shade = 0.5 + 0.5 * np.abs(normals[:, 2])  # facing the camera brightest
        base = np.array(bgr(COLOURS[index % len(COLOURS)]), np.float32)
        for face in np.argsort(-depths):  # far to near; each triangle at one depth
            corners = pixels[faces[face]]
            x0, y0 = np.floor(corners.min(0)).astype(int) - 1
            x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
            if x0 >= x1 or y0 >= y1:
                continue
            patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(patch, [(corners - [x0, y0]).round().astype(np.int32)], 1)
            hit = (patch > 0) & (depths[face] < nearest[y0:y1, x0:x1])
            nearest[y0:y1, x0:x1][hit] = depths[face]
            colour[y0:y1, x0:x1][hit] = base * shade[face]
            owner[y0:y1, x0:x1][hit] = index
    drawn = np.isfinite(nearest)
    if depth is not None:
        scene = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
        visible = drawn & (nearest < scene + 0.01)
    else:
        visible = drawn
    out = picture.astype(np.float32)
    fill = np.where(owner == 0, FILL, FILL_OTHERS)[..., None]
    out = np.where(visible[..., None], (1 - fill) * out + fill * colour, out)
    hidden = drawn & ~visible
    out[hidden] = (1 - GHOST) * out[hidden] + GHOST * colour[hidden]
    out = out.round().astype(np.uint8)
    for index in range(len(palms)):
        mine = (visible & (owner == index)).astype(np.uint8)
        if not mine.any():
            continue
        outline, _ = cv2.findContours(mine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, outline, -1, bgr(COLOURS[index % len(COLOURS)]), 1, cv2.LINE_AA)
        if labels:
            rows, cols = np.nonzero(mine)
            sam3_client.tag(out, labels[index], (int(np.median(cols)), int(np.median(rows))),
                            bgr(COLOURS[index % len(COLOURS)]))
    return out


def draw_hands_3d(axes, palms):
    """Draw hands on a matplotlib 3D axes, the first solid and the rest faint; returns the
    vertices drawn, for setting the limits."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _, faces = gripper_mesh()
    everything = []
    for index, palm in enumerate(palms):
        vertices = hand_vertices(palm)
        everything.append(vertices)
        hand = Poly3DCollection(vertices[faces], alpha=0.55 if index == 0 else 0.22, linewidths=0)
        hand.set_facecolor(COLOURS[index % len(COLOURS)])
        axes.add_collection3d(hand)
    return np.vstack(everything) if everything else np.zeros((0, 3))


def export_scene(folder, name, points, grasps):
    """The cloud and the hands as two coloured PLY files, `<name>_cloud.ply` and
    `<name>_hands.ply`, for turning round in a 3D viewer
    (realrobot/visualizations/view_grasps.py). Points are coloured by height, the
    hands by rank, as in the figures."""
    import trimesh
    from matplotlib import cm
    points = np.asarray(points, dtype=float)
    height = (points[:, 2] - points[:, 2].min()) / (np.ptp(points[:, 2]) or 1.0)
    colours = (cm.get_cmap("viridis")(height) * 255).astype(np.uint8)
    trimesh.PointCloud(points, colors=colours).export(str(folder / f"{name}_cloud.ply"))
    _, faces = gripper_mesh()
    hands = []
    for index, palm in enumerate(grasps):
        rgb = [int(COLOURS[index % len(COLOURS)][i:i + 2], 16) for i in (1, 3, 5)]
        hands.append(trimesh.Trimesh(hand_vertices(palm), faces, vertex_colors=rgb + [255], process=False))
    trimesh.util.concatenate(hands).export(str(folder / f"{name}_hands.ply"))


def export_graspgenx(folder, name, points, grasps, scores, colours=None):
    """The cloud, and the grasps on it, as `graspgenx/<name>.json` in the format
    GraspGenX's own demo viewer reads (scripts/demo_object_pc.py: `pc` in metres,
    `pc_color` 0-255, `grasp_poses` in its gripper frame, `grasp_conf`). Run it with
    docker/graspgenx/run_demo.sh, see realrobot/visualizations/rgbd_frame_capture.py."""
    import json
    from core.grasping import graspgenx_client, pick
    points = np.asarray(points, dtype=float)
    if colours is None:
        from matplotlib import cm
        height = (points[:, 2] - points[:, 2].min()) / (np.ptp(points[:, 2]) or 1.0)
        colours = (cm.get_cmap("viridis")(height)[:, :3] * 255).astype(np.uint8)
    model = np.asarray(grasps) @ np.linalg.inv(graspgenx_client.canonical_from_palm(pick.GRIPPER))
    (folder / "graspgenx").mkdir(parents=True, exist_ok=True)
    (folder / "graspgenx" / f"{name}.json").write_text(json.dumps({
        "pc": points.round(5).tolist(), "pc_color": np.asarray(colours, dtype=np.uint8).tolist(),
        "grasp_poses": model.round(6).tolist(), "grasp_conf": [float(c) for c in scores]}))


def view_from(direction):
    """matplotlib (elevation, azimuth) in degrees for a viewer out along `direction`."""
    x, y, z = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return float(np.degrees(np.arcsin(np.clip(z, -1, 1)))), float(np.degrees(np.arctan2(y, x)))


def spread_view(vertices):
    """The viewpoint that lays the hands out widest on the page: look along the axis the
    vertices vary least in, from above the horizon so the floor is not in the way."""
    centred = np.asarray(vertices, dtype=float) - np.mean(vertices, axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    thinnest = axes[-1]
    if thinnest[2] < 0:
        thinnest = -thinnest
    return view_from(thinnest)


def save(prompt, points, grasps, widths, kind):
    if os.environ.get("DETECTIONS", "1") == "0":
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        folder = sam3_client.run_folder()
        folder.mkdir(parents=True, exist_ok=True)
        name = "grasps_" + sam3_client._slug(prompt)
        grasps, widths = np.asarray(grasps)[:SHOWN], np.asarray(widths)[:SHOWN]
        np.savez(folder / f"{name}.npz", points=points, grasps=grasps, widths=widths)
        export_scene(folder, name, points, grasps)
        # The order MTC tries them, as a confidence, since these carry no score.
        export_graspgenx(folder, name, points, grasps, np.linspace(1, 0.5, len(grasps)))
        figure = plt.figure(figsize=(12, 6))
        hands = np.vstack([hand_vertices(palm) for palm in grasps])
        views = [(22, -60, "from the side"), (*spread_view(np.vstack([points, hands])), "where the hands spread widest")]
        for column, (elevation, azimuth, title) in enumerate(views, 1):
            axes = figure.add_subplot(1, 2, column, projection="3d")
            axes.scatter(*points.T, s=3, c=points[:, 2], cmap="viridis", alpha=0.8)
            draw_hands_3d(axes, grasps)
            for rank, colour in enumerate(COLOURS[:len(grasps)], 1):
                axes.plot([], [], color=colour, linewidth=6, label=f"grasp {rank}, opening {widths[rank - 1] * 1000:.0f} mm")
            everything = np.vstack([points, hands])
            centre, half = everything.mean(0), np.ptp(everything, axis=0).max() / 2 * 1.05
            for axis, middle in zip("xyz", centre):
                getattr(axes, f"set_{axis}lim")(middle - half, middle + half)
                getattr(axes, f"set_{axis}label")(f"{axis} (m)")
            axes.view_init(elevation, azimuth)
            axes.set_title(title)
        axes.legend(loc="upper right", fontsize=8)
        figure.suptitle(f'"{prompt}"  |  {folder.name}\n{len(points)} points, fitted as {kind}; the best '
                        f"{len(grasps)} of GraspGenX's grasps, in the order they are tried", fontsize=11)
        figure.tight_layout()
        figure.savefig(folder / f"{name}.png", dpi=130)
        plt.close(figure)
        print(f"[GRASP] cloud and best {len(grasps)} grasps drawn in {folder / (name + '.png')}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[GRASP] picture not saved: {error}", flush=True)
