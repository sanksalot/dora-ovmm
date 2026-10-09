"""One head-camera RGB-D frame from the real HSR, and what the grasp pipeline makes of it.

    python3 realrobot/visualizations/rgbd_frame_capture.py                # the frame alone
    python3 realrobot/visualizations/rgbd_frame_capture.py "snack can"    # plus mask and grasps
    python3 realrobot/visualizations/rgbd_frame_capture.py "snack can" --show 3 --grasps 100
    python3 realrobot/visualizations/rgbd_frame_capture.py --redraw outputs/realrobot/visualizations/<run>

Without a prompt it grabs one synchronised RGB-D pair and stops there: no SAM3, no
GraspGenX, just the two frames. With a prompt it goes on the way core.grasping.pick
does, up to the model: SAM3 masks the prompt, the mask's depth becomes a cloud in
odom, and GraspGenX ranks grasps on it. The best few are drawn as the registered hand
mesh on the cloud and, projected through the camera calibration and depth-tested
against the measured depth, on the RGB frame over SAM3's mask (core.grasping.picture).
Nothing here moves the robot, and pick.py's contact refinement is not applied: the
grasps shown are GraspGenX's own, ranked by its score, with the hand open.

Writes into outputs/realrobot/visualizations/<time>_<prompt>/, one picture a file:

  rgb.png              the colour frame, untouched
  depth.png            the registered depth, turbo-coloured over 0.4-4 m, black where none
  depth_mm.png         the same depth as 16-bit millimetres, for anything downstream

and, with a prompt:

  detection.png        rgb with SAM3's mask, tagged with the prompt and its confidence
  grasps_on_image.png  detection.png with the best grasps as hands, tagged rank and score
  grasps_on_cloud.png  the mask's cloud and the same hands, from the camera's side and
                       from wherever they spread out widest; no text
  grasps_cloud.ply     the same cloud and hands as coloured 3D files, to turn round and
  grasps_hands.ply     screenshot yourself: python3 realrobot/visualizations/view_grasps.py <folder>
  graspgenx/<prompt>.json  the cloud in its true colours with every grasp asked for, in
                       the format GraspGenX's own browser viewer reads (below)
  capture.npz          every number the pictures came from; --redraw draws them again

To look at the grasps in GraspGenX's own viewer, on the GPU host:

    SAMPLE_DATA=outputs/realrobot/visualizations/<run>/graspgenx bash docker/graspgenx/run_demo.sh \
        --planner diffusion --vis-top-grasp-meshes --num-top-grasp-meshes 4

then open http://localhost:8080. The demo runs GraspGenX again on the cloud (the
stored grasps are the ones this capture got; --planner diffusion is what app.py
serves). A live run's outputs/detections/<run>/graspgenx/ works the same way.

Run it in the container with ROS_DOMAIN_ID and CYCLONEDDS_URI set for the robot,
RX up, and the SAM3 and GraspGenX servers up (realrobot/live/grasp_preflight.py
checks all three).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("HSR_REAL_ROBOT", "1")
# realrobot/visualizations/<this file>, so three levels up is the repository root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.grasping import graspgenx_client, pick, picture  # noqa: E402
from core.grasping.picture import COLOURS  # noqa: E402
from core.perception import pointcloud, sam3_client  # noqa: E402

OUTPUTS = ROOT / "outputs/realrobot/visualizations"
GRASPS = 100  # asked of GraspGenX
SHOWN = 4     # drawn, best score first
DEPTH_RANGE = (0.4, 4.0)  # m, as realrobot/dataprep/visualize.py draws keyframes


def slug(text):
    return re.sub("[^a-z0-9]+", "-", str(text).lower()).strip("-") or "unnamed"


def project(points, k, camera_from_world):
    """(N, 3) points in the cloud's frame -> (N, 2) pixels; None where behind the camera."""
    local = pointcloud.transform_points(camera_from_world, np.asarray(points, dtype=float))
    z = local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * local[:, 0] / z + k[0, 2]
        v = k[1, 1] * local[:, 1] / z + k[1, 2]
    pixels = np.column_stack([u, v])
    pixels[z <= 0] = np.nan
    return pixels


def capture(prompt, grasps, conf):
    """Robot -> frame; with a prompt, also mask, cloud and GraspGenX's ranked grasps, in odom."""
    from core.perception.camera_ros2 import grab_rgbd
    rgb, depth, k, odom_from_camera = grab_rgbd()
    print(f"[FRAME] {rgb.shape[1]}x{rgb.shape[0]}, depth on "
          f"{np.mean(np.isfinite(depth) & (depth > 0)):.0%} of pixels", flush=True)
    frame = dict(rgb=rgb, depth=depth.astype(np.float32), k=k, odom_from_camera=odom_from_camera)
    if not prompt:
        return frame
    mask, score = sam3_client.detect(rgb, prompt, conf=conf)
    box = cv2.boundingRect(mask.astype(np.uint8))  # x, y, w, h of the mask itself
    # observe() in pick.py, minus the support plane the pick needs and this does not.
    points = pointcloud.deproject(depth, k, pointcloud.object_depth_mask(depth, mask))
    if len(points) < 100:
        raise RuntimeError(f"only {len(points)} depth points on the {prompt}; back the robot off")
    world = pointcloud.transform_points(odom_from_camera, points)
    print(f"[SAM3] {prompt}: {score:.2f}, {len(world)} points", flush=True)
    # perceive() in pick.py: bounded, centred cloud in, poses back into odom, best first.
    sample = world[np.random.choice(len(world), min(len(world), pick.MAX_CLOUD_POINTS), replace=False)]
    centre = sample.mean(axis=0)
    poses, scores = graspgenx_client.generate(sample - centre, pick.GRIPPER, num_grasps=grasps)
    poses[:, :3, 3] += centre
    order = np.argsort(scores)[::-1]
    print(f"[GRASPGENX] {len(poses)} grasps, best {scores[order[0]]:.2f}, "
          f"worst {scores[order[-1]]:.2f}", flush=True)
    return dict(frame, prompt=prompt, mask=mask, box=np.array(box), score=float(score),
                points=world, grasps=poses[order], scores=scores[order])


def colour_depth(depth):
    """Depth in metres -> BGR, turbo over DEPTH_RANGE, black where there is no depth.
    Also what rgbd_recorder.py writes to its depth video."""
    valid = np.isfinite(depth) & (depth > 0)
    unit = np.clip((depth - DEPTH_RANGE[0]) / (DEPTH_RANGE[1] - DEPTH_RANGE[0]), 0, 1)
    coloured = cv2.applyColorMap((unit * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    return coloured


def draw_frames(folder, rgb, depth):
    cv2.imwrite(str(folder / "rgb.png"), rgb)
    valid = np.isfinite(depth) & (depth > 0)
    cv2.imwrite(str(folder / "depth_mm.png"), np.where(valid, depth * 1000, 0).round().astype(np.uint16))
    # Pixel for pixel with rgb.png, so the two sit side by side on a slide.
    cv2.imwrite(str(folder / "depth.png"), colour_depth(depth))


def detection_picture(rgb, mask, prompt, score):
    """The frame with the mask on it, the way sam3_client.save_detection draws it."""
    picture = rgb.copy()
    sam3_client.draw_mask(picture, mask, sam3_client.BEST, f"{prompt}  {score:.2f}")
    return picture


def draw_on_image(folder, detection, grasps, scores, k, odom_from_camera, depth):
    labels = [f"{rank}  {score:.2f}" for rank, score in enumerate(scores, 1)]
    drawn = picture.draw_hands_image(detection, grasps, k, np.linalg.inv(odom_from_camera), depth, labels)
    cv2.imwrite(str(folder / "grasps_on_image.png"), drawn)


def draw_on_cloud(folder, points, grasps, camera):
    """Two views with nothing written on them: from where the camera stood, and from
    wherever the hands spread out widest (core.grasping.picture.spread_view)."""
    hands = np.vstack([picture.hand_vertices(palm) for palm in grasps])
    everything = np.vstack([points, hands])
    centre, half = everything.mean(0), np.ptp(everything, axis=0).max() / 2 * 1.05
    views = [picture.view_from(camera - points.mean(0)), picture.spread_view(everything)]
    figure = plt.figure(figsize=(12, 6))
    for column, (elevation, azimuth) in enumerate(views, 1):
        axes = figure.add_subplot(1, 2, column, projection="3d")
        axes.scatter(*points.T, s=4, c=points[:, 2], cmap="viridis", alpha=0.9)
        picture.draw_hands_3d(axes, grasps)
        for axis, middle in zip("xyz", centre):
            getattr(axes, f"set_{axis}lim")(middle - half, middle + half)
        axes.view_init(elevation, azimuth)
        axes.set_axis_off()
        axes.set_box_aspect((1, 1, 1), zoom=1.8)  # a 3D axes leaves a wide margin otherwise
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    figure.savefig(folder / "grasps_on_cloud.png", dpi=160, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def draw(folder, data, show):
    folder.mkdir(parents=True, exist_ok=True)
    rgb, depth, k = data["rgb"], data["depth"], data["k"]
    draw_frames(folder, rgb, depth)
    if "mask" not in data:
        print(f"[SAVED] {folder}/rgb.png and depth.png", flush=True)
        return
    detection = detection_picture(rgb, data["mask"], data["prompt"], data["score"])
    cv2.imwrite(str(folder / "detection.png"), detection)
    grasps, scores = data["grasps"][:show], data["scores"][:show]
    draw_on_image(folder, detection, grasps, scores, k, data["odom_from_camera"], depth)
    draw_on_cloud(folder, data["points"], grasps, data["odom_from_camera"][:3, 3])
    picture.export_scene(folder, "grasps", data["points"], grasps)
    # Every grasp asked for, with the cloud's true colours: deproject() took the
    # pixels in np.nonzero order, so the same mask indexes the frame the same way.
    pixels = pointcloud.object_depth_mask(depth, data["mask"]) & np.isfinite(depth) & (depth > 0)
    colours = rgb[pixels][:, ::-1] if pixels.sum() == len(data["points"]) else None
    picture.export_graspgenx(folder, slug(data["prompt"]), data["points"], data["grasps"], data["scores"], colours)
    for rank, score in enumerate(data["scores"][:show], 1):
        print(f"[GRASP {rank}] {COLOURS[rank - 1]}  score {score:.2f}", flush=True)
    for name in sorted(p.name for p in folder.glob("*.png")):
        print(f"[SAVED] {folder / name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", help="what SAM3 should find, e.g. 'snack can'; "
                        "none saves the rgb and depth frames alone")
    parser.add_argument("--grasps", type=int, default=GRASPS, help=f"grasps asked of GraspGenX (default {GRASPS})")
    parser.add_argument("--show", type=int, default=SHOWN, help=f"grasps drawn, best first (default {SHOWN}, at most {len(COLOURS)})")
    parser.add_argument("--conf", type=float, default=0.5, help="SAM3 confidence threshold")
    parser.add_argument("--output", type=Path, help=f"folder to write into (default {OUTPUTS}/<time>_<prompt>)")
    parser.add_argument("--redraw", type=Path, metavar="FOLDER", help="draw a saved capture.npz again; no robot, no models")
    args = parser.parse_args()
    if not 1 <= args.show <= len(COLOURS):
        parser.error(f"--show must be between 1 and {len(COLOURS)}")
    if args.redraw:
        folder = args.output or args.redraw
        with np.load(args.redraw / "capture.npz") as saved:
            data = {key: saved[key] for key in saved.files}
        if "prompt" in data:
            data["prompt"], data["score"] = str(data["prompt"]), float(data["score"])
    else:
        data = capture(args.prompt, args.grasps, args.conf)
        folder = args.output or OUTPUTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug(args.prompt or 'frame')}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "capture.npz", **data)
    draw(folder, data, args.show)


if __name__ == "__main__":
    main()
