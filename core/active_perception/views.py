"""Candidate camera views the HSR head can take: ETH's ViewHalfSphere plus
the reachability check their generate_views() gets from an IK solver.

The Panda carries its camera on a 7-DoF arm, so ETH sample a half sphere
above the target at fixed polar angles and keep the views IK can reach. The
HSR's RGB-D camera sits on a pan-tilt head over the torso lift of a mobile
base, so a view is reachable when: the camera height lies in the lift's range,
the tilt lies in the head's range, the image is level (the head has no roll),
the base stands clear of the target and of every furniture footprint, and,
given a costmap, on a free cell. The footprints matter: the laser costmap holds
a table's legs only, so Nav2 would happily drive the base under its top.
Every view is base_from_camera in the optical convention (z forward, y down).
"""

import numpy as np

from core.navigation.standoff import ROBOT_RADIUS, blocks, free

# Camera height with the torso down, and its added lift, in metres.
CAMERA_HEIGHT = 1.015
TORSO_LIFT = 0.345
# head_tilt_joint limits (hsrb_description head_v2): looking down is negative.
TILT_MIN, TILT_MAX = (-1.57, 0.52)
# The camera stands over the base centre, near enough for footprint clearance.
CLEARANCE = ROBOT_RADIUS + 0.05
# An eye refused by Nav2 rules out its neighbours too: retarget() moves every eye
# with the box centre, and on 2026-09-24 03:24 a 1 cm shift offered the refused
# (0.42, 2.82) straight back as (0.427, 2.799). Half Nav2's 0.3 m goal tolerance:
# anything closer than that is the same goal to the planner. Own number.
REJECTED_RADIUS = 0.15


def spherical_to_cartesian(r, theta, phi):
    # Turn radius, polar angle, and azimuth into an XYZ offset from the target.
    return np.r_[r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)]


def look_at(eye, center, up):
    """vgn.utils.look_at: a camera at `eye` with its optical axis on `center`."""
    eye, center = (np.asarray(eye, dtype=float), np.asarray(center, dtype=float))
    # Point the optical Z axis at the target, then build perpendicular camera axes.
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.eye(4)
    m[:3, 0] = right
    # Use the optical camera convention: X right, Y down, Z forward.
    m[:3, 1] = -up
    m[:3, 2] = forward
    m[:3, 3] = eye
    return m


class ViewHalfSphere:
    """ETH's sphere around the target, sampled where the HSR head can be."""

    def __init__(self, bbox, min_z_dist, floor_z=0.0, grid=None, blockers=(), phis=8, rings=3, ring_step=0.3):
        """Use furniture footprints (centre, dimensions, yaw) in the base frame."""
        self.bbox = bbox
        self.blockers = list(blockers)
        self.rejected = []  # eyes Nav2 refused to drive under; not offered again
        self.center = bbox.center
        self.r = 0.5 * bbox.size[2] + min_z_dist
        # Try larger viewing radii so the base can stand outside the furniture.
        self.radii = self.r + ring_step * np.arange(rings)
        self.floor_z = floor_z
        self.grid = grid
        # Space viewpoints evenly around the target in the horizontal direction.
        self.phis = np.arange(phis) * (2 * np.pi / phis)

    def retarget(self, bbox):
        """Follow the target box when new views rebuild it.

        Only the centre moves: the sampling radii come from the box height, which
        is the dimension the detector gets right, and changing them mid-run would
        silently invalidate the views already refused in `rejected`.
        """
        self.bbox = bbox
        self.center = bbox.center

    def get_view(self, theta, phi, r):
        eye = self.center + spherical_to_cartesian(r, theta, phi)
        return look_at(eye, self.center, up=np.r_[0.0, 0.0, 1.0])

    def thetas(self, r):
        """Find viewing angles at the lowest and highest camera heights."""
        heights = self.floor_z + CAMERA_HEIGHT + np.array([0.0, TORSO_LIFT])
        # Keep only torso heights that intersect this viewing sphere.
        cos = (heights - self.center[2]) / r
        return np.arccos(cos[np.abs(cos) < 1.0])

    def feasible(self, view):
        eye, forward = (view[:3, 3], view[:3, 2])
        height = eye[2] - self.floor_z
        tilt = np.arcsin(np.clip(forward[2], -1.0, 1.0))
        # Require the camera's base position to lie outside the expanded target footprint.
        clear = np.any((eye[:2] < self.bbox.min[:2] - CLEARANCE) | (eye[:2] > self.bbox.max[:2] + CLEARANCE))
        # Keep views within head limits, clear of obstacles, and not already rejected.
        return (
            CAMERA_HEIGHT - 1e-6 <= height <= CAMERA_HEIGHT + TORSO_LIFT + 1e-6
            and TILT_MIN <= tilt <= TILT_MAX
            and bool(clear)
            and not blocks(eye[:2], self.blockers)
            and not any(np.linalg.norm(eye - e) < REJECTED_RADIUS for e in self.rejected)
            and (self.grid is None or free(self.grid, eye[:2]))
        )

    def all_views(self):
        # Combine every radius, reachable camera height, and horizontal angle.
        for r in self.radii:
            for theta in self.thetas(r):
                for phi in self.phis:
                    yield self.get_view(theta, phi, r)

    def candidates(self):
        """Return the sampled camera poses that pass the robot and obstacle checks."""
        return [view for view in self.all_views() if self.feasible(view)]
