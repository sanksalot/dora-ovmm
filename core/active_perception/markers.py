"""The next-best-view loop as RViz markers: record.py's twin for ROS.

Same three calls as core.active_perception.record.Recording -- construct,
`.scene(policy)`, `.step(...)` -- so a run can drive either or both. ETH's
active_grasp visualises its policy this way; `config/rviz/nbv.rviz` is the
display side of this file.

What RViz gives that a Rerun recording cannot is everything already on the
wire: Nav2's predicted path, the costmaps, TF and the robot's own model. None
of those are published here, because they are published already -- this file
adds only what the policy knows.

Nothing here spins. Publishing is enough, so it borrows the caller's node
rather than running one of its own.

    ros2 run rviz2 rviz2 -d /home/ws/config/rviz/nbv.rviz
"""

import numpy as np
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from core.active_perception.record import _ramp, view_rings

FRAME = "map"       # the frame the policy's boxes, sphere and views are all in
# Transient local, so an RViz opened after the run started still gets the last
# message rather than an empty scene.
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
# How far down its own axis a candidate camera is drawn. At 0.15 m (record.py's
# value, right for a Rerun close-up) the pyramid is 17 x 13 cm, a sliver beside a
# 1.4 m view sphere; 0.25 m reads at room scale without the neighbours on a ring
# running into each other.
CANDIDATE_DEPTH = 0.25


def _point(xyz):
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _colour(rgb, alpha=1.0):
    r, g, b = (float(v) / 255.0 for v in rgb)
    return ColorRGBA(r=r, g=g, b=b, a=float(alpha))


def _marker(namespace, kind, stamp, scale, colour=None, alpha=1.0):
    marker = Marker()
    marker.header = Header(frame_id=FRAME, stamp=stamp)
    marker.ns = namespace
    marker.id = 0
    marker.type = kind
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
    marker.pose.orientation.w = 1.0
    marker.color = _colour(colour or (200, 200, 200), alpha)
    return marker


def _box_edges(centre, half):
    """The 12 edges of an axis-aligned box, as LINE_LIST point pairs."""
    centre, half = (np.asarray(centre, dtype=float), np.asarray(half, dtype=float))
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = centre + signs * half
    # Two corners share an edge when they differ in exactly one coordinate.
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.sum(signs[i] != signs[j]) == 1]
    return [corners[i] for pair in pairs for i in pair]


def _frustum_edges(view, rays):
    """One camera's eight wireframe edges, as LINE_LIST point pairs."""
    eye = view[:3, 3]
    corners = rays @ view[:3, :3].T + eye
    edges = []
    for corner in corners:
        edges += [eye, corner]
    for index in range(4):
        edges += [corners[index], corners[(index + 1) % 4]]
    return edges


class Markers:
    """Publishes what `Recording` logs, as MarkerArray and PointCloud2."""

    def __init__(self, node, intrinsic, shape, topic="/nbv"):
        self.node = node
        self.intrinsic, self.shape = (np.asarray(intrinsic, dtype=float), shape)
        self.markers = node.create_publisher(MarkerArray, f"{topic}/markers", LATCHED)
        self.tsdf = node.create_publisher(PointCloud2, f"{topic}/tsdf", LATCHED)
        self.surface = node.create_publisher(PointCloud2, f"{topic}/surface", LATCHED)
        self.path = []
        self.scene_markers = []     # republished every step, so they never expire

    def _rays(self, depth):
        """Image-corner rays at `depth`, for drawing a camera as a pyramid."""
        h, w = self.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        return np.c_[(corners[:, 0] - self.intrinsic[0, 2]) / self.intrinsic[0, 0],
                     (corners[:, 1] - self.intrinsic[1, 2]) / self.intrinsic[1, 1],
                     np.ones(4)] * depth

    def _cloud(self, points, stamp, intensities=None):
        header = Header(frame_id=FRAME, stamp=stamp)
        if intensities is None:
            return point_cloud2.create_cloud_xyz32(header, np.asarray(points, dtype=np.float32))
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(("x", "y", "z", "intensity"))]
        rows = np.column_stack([np.asarray(points, dtype=np.float32),
                                np.asarray(intensities, dtype=np.float32)])
        return point_cloud2.create_cloud(header, fields, rows)

    def scene(self, policy):
        """Everything that stays put once the target is boxed."""
        stamp = self.node.get_clock().now().to_msg()
        origin, length = (policy.base_from_task[:3, 3], policy.length)
        sphere = policy.view_sphere

        cube = _marker("cube", Marker.LINE_LIST, stamp, 0.004, (60, 60, 60))
        cube.points = [_point(p) for p in _box_edges(origin + length / 2, [length / 2] * 3)]

        rings = _marker("sphere", Marker.LINE_LIST, stamp, 0.003, (150, 150, 150), alpha=0.7)
        for ring in view_rings(sphere):
            for index in range(len(ring) - 1):
                rings.points += [_point(ring[index]), _point(ring[index + 1])]

        self.scene_markers = [cube, rings]

        # Views the base cannot reach, kept separate so they read as rejected
        # rather than as candidates the policy declined to pick.
        unreachable = [view for view in sphere.all_views() if not sphere.feasible(view)]
        if unreachable:
            rays = self._rays(0.08)
            rejected = _marker("unreachable", Marker.LINE_LIST, stamp, 0.002, (150, 150, 150), alpha=0.35)
            for view in unreachable:
                rejected.points += [_point(p) for p in _frustum_edges(view, rays)]
            self.scene_markers.append(rejected)

        self.markers.publish(MarkerArray(markers=self.scene_markers))

    def step(self, index, policy, depth, view, image=None):
        """The state after `policy.update(depth, view)` at this step.

        `depth` and `image` are taken for signature parity with Recording and
        are not published: RViz already has the camera streams.
        """
        stamp = self.node.get_clock().now().to_msg()
        for marker in self.scene_markers:
            marker.header.stamp = stamp
        array = list(self.scene_markers)

        self.path.append(view[:3, 3].copy())
        if len(self.path) > 1:
            trail = _marker("path", Marker.LINE_STRIP, stamp, 0.008, (40, 90, 240))
            trail.points = [_point(p) for p in self.path]
            array.append(trail)

        bbox = policy.bbox
        target = _marker("target", Marker.LINE_LIST, stamp, 0.005, (230, 40, 40))
        target.points = [_point(p) for p in _box_edges(bbox.center, bbox.size / 2)]
        array.append(target)

        if policy.info:
            views, gains, best = (policy.info["views"], policy.info["gains"], policy.info["best"])
            # Same ramp as the Rerun recording, so the two views agree on colour.
            colours = _ramp(gains / max(gains.max(), 1))
            rays = self._rays(CANDIDATE_DEPTH)
            candidates = _marker("candidates", Marker.LINE_LIST, stamp, 0.003)
            for one, colour in zip(views, colours):
                edges = _frustum_edges(one, rays)
                candidates.points += [_point(p) for p in edges]
                candidates.colors += [_colour(colour)] * len(edges)
            array.append(candidates)

            chosen = _marker("next_best", Marker.LINE_LIST, stamp, 0.008, (40, 220, 60))
            chosen.points = [_point(p) for p in _frustum_edges(views[best], self._rays(2 * CANDIDATE_DEPTH))]
            array.append(chosen)

        if policy.best_grasp is not None:
            pose, quality = policy.best_grasp
            approach = _marker("grasp", Marker.ARROW, stamp, 0.006, (250, 160, 20))
            approach.scale.y, approach.scale.z = (0.012, 0.012)
            # The palm approaches along its own -z, as in record.py.
            tip = pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, -0.08])
            approach.points = [_point(pose[:3, 3]), _point(tip)]
            label = _marker("grasp_quality", Marker.TEXT_VIEW_FACING, stamp, 0.03, (250, 160, 20))
            label.pose.position = _point(pose[:3, 3] + np.array([0.0, 0.0, 0.05]))
            label.text = f"grasp {quality:.2f}"
            array += [approach, label]

        self.markers.publish(MarkerArray(markers=array))

        # The fusion volume, in world coordinates: a surface cloud, and the
        # voxels carrying their signed distance as intensity. RViz has no TSDF
        # display, and 40^3 cubes as a CUBE_LIST is far slower than a cloud.
        origin = policy.base_from_task[:3, 3]
        surface = np.asarray(policy.tsdf.get_scene_cloud().points) + origin
        if len(surface):
            self.surface.publish(self._cloud(surface, stamp))
        voxels = policy.tsdf.get_map_cloud()
        centres = np.asarray(voxels.points) + origin
        if len(centres):
            distances = np.asarray(voxels.colors)[:, 0]  # (tsdf + 1) / 2
            self.tsdf.publish(self._cloud(centres, stamp, distances))


class Tee:
    """Drive several visualisers from one set of calls."""

    def __init__(self, *sinks):
        self.sinks = [sink for sink in sinks if sink is not None]

    def scene(self, policy):
        for sink in self.sinks:
            sink.scene(policy)

    def step(self, *args, **kwargs):
        for sink in self.sinks:
            sink.step(*args, **kwargs)
