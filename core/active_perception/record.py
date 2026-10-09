"""Log the next-best-view loop to Rerun, live or to an .rrd file.

One recording shows the whole process on a "view" timeline: the depth image
and camera that were fused, the TSDF as a surface cloud and as voxels coloured
by signed distance, the target box, the fusion cube, the view sphere, every
candidate camera coloured by its information gain, the chosen next view, the
recorded camera path, the best grasp, and the gain and grasp quality as plots.

    rerun outputs/active_perception/tsdf_test/nbv_replay/nbv_replay.rrd
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

GAIN_COLORS = np.array([[59, 76, 192], [221, 221, 221], [180, 4, 38]], dtype=float)  # low, mid, high


def _ramp(values, colors=GAIN_COLORS):
    """(N,) in [0, 1] -> (N, 3) uint8 through a three-colour ramp."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    low, mid, high = colors
    # Blend low-to-middle colours below 0.5 and middle-to-high colours above it.
    first = low + (mid - low) * np.minimum(values, 0.5)[:, None] * 2
    second = mid + (high - mid) * (np.maximum(values, 0.5) - 0.5)[:, None] * 2
    return np.where(values[:, None] < 0.5, first, second).astype(np.uint8)


def view_rings(sphere, steps=97):
    """The circles the candidate cameras actually stand on.

    ViewHalfSphere samples a radius `r` at the polar angles where that sphere
    meets the robot's two camera heights, so a camera sits on a horizontal
    circle of radius `r*sin(theta)` at height `center_z + r*cos(theta)` -- not
    on a circle of radius `r` at the target's height. Drawing the latter put
    every ring 50 cm below and 21 cm outside the views it was meant to frame,
    which is what made the frusta look unmoored from the sphere.
    """
    around = np.linspace(0, 2 * np.pi, steps)
    rings = []
    for r in sphere.radii:
        for theta in sphere.thetas(r):
            radius, height = (r * np.sin(theta), sphere.center[2] + r * np.cos(theta))
            rings.append(np.column_stack([sphere.center[0] + radius * np.cos(around),
                                          sphere.center[1] + radius * np.sin(around),
                                          np.full_like(around, height)]))
    return rings


def _camera(log, path, view, intrinsic, shape, plane=0.25):
    """A real camera: its pose and pinhole, so images hang in the 3D view."""
    log(path, rr.Transform3D(translation=view[:3, 3], mat3x3=view[:3, :3]))
    log(path, rr.Pinhole(image_from_camera=intrinsic, resolution=[shape[1], shape[0]],
                            camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=plane))


def _frusta(log, path, views, intrinsic, shape, colors, depth=0.15, radii=0.003, labels=None):
    """Candidate views as coloured wireframe pyramids (Pinhole has no colour)."""
    h, w = shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    # Project the four image corners into 3D to draw each camera view.
    rays = np.c_[(corners[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
                 (corners[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1], np.ones(4)] * depth
    strips = []
    for view in views:
        eye = view[:3, 3]
        # Rotate the image-corner rays and move them to this camera position.
        pts = rays @ view[:3, :3].T + eye
        strips += [np.array([eye, p]) for p in pts] + [np.array([pts[i], pts[(i + 1) % 4]]) for i in range(4)]
    # Give all eight wireframe edges of each view the same colour.
    colors = np.repeat(np.asarray(colors, dtype=np.uint8).reshape(-1, 3), 8, axis=0)
    kwargs = {} if labels is None else dict(labels=list(labels))  # one label per batch, drawn once
    log(path, rr.LineStrips3D(strips, colors=colors, radii=radii, **kwargs))


class Recording:
    def __init__(self, intrinsic, shape, path=None, spawn=False, application="active_perception"):
        """Use separate recording streams for the file and viewer; each stream has one sink."""
        self.intrinsic, self.shape = (np.asarray(intrinsic, dtype=float), shape)
        self.streams = []
        if path is not None:
            self.streams.append(rr.new_recording(application))
            self.streams[-1].save(str(path))
        if spawn:
            # Open a local viewer, or connect to the supplied host:port.
            self.streams.append(rr.new_recording(application))
            if spawn is True:
                self.streams[-1].spawn()
            else:
                rr.connect_tcp(spawn, recording=self.streams[-1])
        self.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # Arrange the 3D scene, camera image, score plots, and step log.
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Active perception",
                                  # The rings and rejected positions crowd the scene; they
                                  # stay in the recording, one click away in the blueprint panel.
                                  contents=["+ $origin/**", "- world/camera/image", "- world/camera/depth",
                                            "- world/views/unreachable", "- world/views/sphere"]),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="world/camera", name="Head camera", contents=["+ world/camera/image"]),
                    rrb.TimeSeriesView(origin="plots", name="Information gain and grasp quality"),
                    rrb.TextLogView(origin="log", name="Steps"),
                    row_shares=[3, 2, 1]),
                column_shares=[3, 2]),
            collapse_panels=True)
        for stream in self.streams:
            stream.send_blueprint(blueprint)
        self.path = []

    def log(self, path, *archetypes, **kwargs):
        for stream in self.streams:
            stream.log(path, *archetypes, **kwargs)

    def scene(self, policy):
        """Everything that stays put once the target is boxed."""
        bbox, sphere = (policy.bbox, policy.view_sphere)
        origin, length = (policy.base_from_task[:3, 3], policy.length)
        # Draw the fixed fusion cube and the candidate viewing spheres once.
        self.log("world/tsdf/cube", rr.Boxes3D(centers=[origin + length / 2], half_sizes=[[length / 2] * 3],
                                             colors=[30, 30, 30], fill_mode="majorwireframe", labels=["TSDF cube"]), static=True)
        # One ring per camera circle, not a wireframe ball per radius: the nested
        # spheres hid the views they were meant to frame. The RViz backend draws
        # the same rings, so the two viewers agree.
        self.log("world/views/sphere", rr.LineStrips3D(view_rings(sphere),
                                                       colors=[170, 170, 170],
                                                       radii=0.004), static=True)
        # Show rejected camera positions separately from usable candidates.
        unreachable = [view for view in sphere.all_views() if not sphere.feasible(view)]
        if unreachable:
            _frusta(self.log, "world/views/unreachable", unreachable, self.intrinsic, self.shape,
                    [[190, 190, 190]] * len(unreachable), depth=0.08, radii=0.002)

    def step(self, index, policy, depth, view, image=None):
        """The state after `policy.update(depth, view)` at this step."""
        # Put every item from this step at the same point on the recording timeline.
        for stream in self.streams:
            stream.set_time_sequence("view", index)
        _camera(self.log, "world/camera", view, self.intrinsic, self.shape)
        self.log("world/camera/depth", rr.DepthImage(np.asarray(depth, dtype=np.float32), meter=1.0))
        if image is not None:
            self.log("world/camera/image", rr.Image(image[..., ::-1]))  # BGR -> RGB
        self.path.append(view[:3, 3].copy())
        self.log("world/camera/path", rr.LineStrips3D([np.array(self.path)], colors=[40, 90, 240], radii=0.005))

        bbox = policy.bbox
        self.log("world/target/box", rr.Boxes3D(centers=[bbox.center], half_sizes=[bbox.size / 2],
                                              colors=[230, 40, 40], fill_mode="majorwireframe", labels=["target"]))
        # Shift the fused surface and voxels from cube coordinates into the world.
        origin = policy.base_from_task[:3, 3]
        surface = np.asarray(policy.tsdf.get_scene_cloud().points) + origin
        self.log("world/tsdf/surface", rr.Points3D(surface, colors=[90, 200, 120], radii=policy.tsdf.voxel_size / 4))
        voxels = policy.tsdf.get_map_cloud()
        centres = np.asarray(voxels.points) + origin
        tsdf = np.asarray(voxels.colors)[:, 0]  # (tsdf + 1) / 2: 0 = behind, 1 = in front
        self.log("world/tsdf/voxels", rr.Boxes3D(centers=centres, half_sizes=[[policy.tsdf.voxel_size / 2] * 3] * len(centres),
                                               colors=_ramp(tsdf), fill_mode="solid"))

        # Colour candidate views by gain and highlight the chosen view.
        if policy.info:
            views, gains, best = (policy.info["views"], policy.info["gains"], policy.info["best"])
            _frusta(self.log, "world/views/candidates", views, self.intrinsic, self.shape, _ramp(gains / max(gains.max(), 1)))
            _frusta(self.log, "world/views/next_best", [views[best]], self.intrinsic, self.shape, [[40, 220, 60]],
                    depth=0.3, radii=0.008, labels=[f"next best view, gain {int(gains[best])}"])
            self.log("plots/gain/next_best_view", rr.Scalar(float(gains[best])))
            self.log("plots/gain/mean_over_candidates", rr.Scalar(float(gains.mean())))
        if policy.best_grasp is not None:
            pose, quality = policy.best_grasp
            self.log("world/grasp", rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
            self.log("world/grasp/approach", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, 0, -0.08]],
                                                       colors=[[250, 160, 20]], radii=0.006, labels=[f"grasp {quality:.2f}"]))
            self.log("plots/grasp_quality", rr.Scalar(quality))
        self.log("log", rr.TextLog(
            f"view {index}: {len(policy.views)} fused, "
            + (f"best gain {int(policy.info['gains'][policy.info['best']])} over {len(policy.info['views'])} candidates"
               if policy.info else "no candidate view")
            + (f", best grasp {policy.best_grasp[1]:.2f}" if policy.best_grasp else "")
            + (", done" if policy.done else "")))
