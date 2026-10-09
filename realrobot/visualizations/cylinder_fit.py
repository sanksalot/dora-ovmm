"""A captured cloud with pick.py's cylinder fit drawn over it, as a figure.

    python3 realrobot/visualizations/cylinder_fit.py outputs/detections/<run>
    python3 realrobot/visualizations/cylinder_fit.py outputs/detections/<run> --name grasps_pringles-can
    python3 realrobot/visualizations/cylinder_fit.py outputs/detections/<run> --azim -40

Reads the `<name>.npz` that core.grasping.picture writes beside its figures and
draws its `points` in perspective inside the fitted cylinder. Writes
`<name>_cylinder_fit.png` into the same folder.

The fit is not recomputed here. core.grasping.pick.cylinder is the one the
pipeline grasps from, so the figure shows the prediction under test rather than
a fresh fit that might flatter it. The caption carries the numbers that decide
whether pick.py accepts it: the RMS residual against its tolerance, and how much
of the circumference was visible to infer the hidden centre from.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.grasping.pick import cylinder  # noqa: E402

# Sequential, one hue light -> dark: height is a magnitude, not an identity.
CLOUD = LinearSegmentedColormap.from_list("cloud", ["#AECBE6", "#0E3654"])
FIT = "#C2402B"     # the one accent; blue against red stays separable under CVD
INK = "#4A4A4A"


def trim(path, margin=30):
    """Crop the white border off a saved figure.

    A 3D axes leaves most of the canvas empty for a shape this slender, and
    matplotlib's own zoom (`Axes3D.dist`) pushes the caps off the edge before it
    fills the frame. Cropping what was drawn is the reliable way round.
    """
    image = Image.open(path).convert("RGB")
    ink = (np.asarray(image) < 250).any(axis=2)
    rows, cols = np.where(ink)
    if not len(rows):
        return
    box = (max(int(cols.min()) - margin, 0), max(int(rows.min()) - margin, 0),
           min(int(cols.max()) + margin + 1, image.width),
           min(int(rows.max()) + margin + 1, image.height))
    image.crop(box).save(path)


def draw(points, centre, radius, bottom, top, elev, azim):
    """The cloud in perspective inside the fitted cylinder. Returns the figure."""
    # Local frame in centimetres: origin on the fitted axis, z from the base.
    # Map-frame coordinates (x~11, y~-12.6) would put five digits on every tick.
    local = np.column_stack([(points[:, 0] - centre[0]) * 100,
                             (points[:, 1] - centre[1]) * 100,
                             (points[:, 2] - bottom) * 100])
    r, h = radius * 100, (top - bottom) * 100

    # Portrait: the cylinder is three times taller than it is wide, so a square
    # canvas would be mostly margin.
    fig = plt.figure(figsize=(3.4, 8.0), dpi=260, facecolor="white")
    axes = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    axes.scatter(local[:, 0], local[:, 1], local[:, 2], c=local[:, 2], cmap=CLOUD,
                 s=1.1, linewidths=0, alpha=0.95, depthshade=False)

    theta = np.linspace(0, 2 * np.pi, 200)
    mesh_t, mesh_z = np.meshgrid(theta, np.linspace(0, h, 2))
    axes.plot_surface(r * np.cos(mesh_t), r * np.sin(mesh_t), mesh_z,
                      color=FIT, alpha=0.10, shade=False, linewidth=0)
    for z in (0, h):
        axes.plot(r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z),
                  color=FIT, lw=1.7)
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        axes.plot([r * np.cos(angle)] * 2, [r * np.sin(angle)] * 2, [0, h],
                  color=FIT, lw=0.5, alpha=0.45)
    axes.plot([0, 0], [0, 0], [0, h], color=FIT, lw=0.9, ls=(0, (4, 3)), alpha=0.8)

    axes.set_box_aspect((2 * r, 2 * r, h))   # true proportions, not a fitted box
    axes.view_init(elev=elev, azim=azim)
    # Crop to the cylinder: the default 3D margins are most of the panel here.
    axes.set_xlim(-1.05 * r, 1.05 * r)
    axes.set_ylim(-1.05 * r, 1.05 * r)
    axes.set_zlim(0, h)
    axes.set_axis_off()

    handles = [Line2D([], [], marker="o", ls="", ms=5, color="#1B4F76"),
               Line2D([], [], color=FIT, lw=1.9)]
    fig.legend(handles, [f"{len(points):,} points", "cylinder fit"],
               loc="upper left", bbox_to_anchor=(0.03, 0.985), frameon=False,
               fontsize=9, labelcolor=INK, handletextpad=0.7)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="a run folder holding <name>.npz")
    parser.add_argument("--name", default="grasps",
                        help="file prefix (default grasps; a live run's is grasps_<object>)")
    parser.add_argument("--elev", type=float, default=14.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <folder>/<name>_cylinder_fit.png")
    args = parser.parse_args()

    source = args.folder / f"{args.name}.npz"
    if not source.exists():
        parser.error(f"no {args.name}.npz in {args.folder}")
    points = np.load(source, allow_pickle=True)["points"]

    fit = cylinder(points)
    if fit is None:
        # The same rejection the pipeline makes: say so rather than drawing a
        # cylinder that pick.py would never have grasped from.
        sys.exit(f"pick.cylinder rejected this cloud ({len(points)} points): "
                 "too short, too little curvature, or an implausible radius.")
    centre, radius, bottom, top = fit

    # Fit quality, measured the way pick.py measures it: the middle height band.
    band = points[(points[:, 2] > bottom + .2 * (top - bottom)) &
                  (points[:, 2] < top - .2 * (top - bottom)), :2]
    residual = np.sqrt(np.mean((np.linalg.norm(band - centre, axis=1) - radius) ** 2))
    arc = np.degrees(np.ptp(np.arctan2(band[:, 1] - centre[1], band[:, 0] - centre[0])))
    tolerance = max(.0015, .10 * radius)

    # The fit numbers go to the terminal, not onto the figure: it is meant to be
    # dropped into a document that carries its own caption.
    fig = draw(points, centre, radius, bottom, top, args.elev, args.azim)

    output = args.output or args.folder / f"{args.name}_cylinder_fit.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    trim(output)
    print(f"{len(points)} points, diameter {2 * radius * 1000:.1f} mm, "
          f"height {(top - bottom) * 1000:.0f} mm, RMS residual {residual * 1000:.2f} mm "
          f"(tolerance {tolerance * 1000:.2f}), visible arc {arc:.0f} deg")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
