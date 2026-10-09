"""What the policy's grasp branch plugs into: GraspGenX for grasps and Toyota's
IK service for ETH's reachability check.

active_grasp runs VGN on the TSDF and trac_ik on each grasp. Here GraspGenX
reads the fused surface points, and the HSR's whole-body IK service
(launch/ik_solver.launch.py, core/navigation/base_placement.py) answers
whether some base placement and arm configuration reach the palm pose,
collision-checked against the costmap.
"""

import numpy as np

from core.grasping import graspgenx_client

GRIPPER = "hsrc_hand"
MAX_CLOUD_POINTS = 8192  # as core/grasping/pick.py


def graspgenx_grasps(points):
    """Generate palm poses and scores from surface points in map coordinates."""
    # Limit the cloud size and centre it for the grasp model.
    sample = points[np.random.default_rng(0).choice(len(points), min(len(points), MAX_CLOUD_POINTS), replace=False)]
    centre = sample.mean(axis=0)
    poses, scores = graspgenx_client.generate(sample - centre, GRIPPER)
    poses = poses.copy()
    # Move the predicted palm positions back into map coordinates.
    poses[:, :3, 3] += centre
    return poses, scores


def ik_reachable(navigator, obstacles="--costmap", top_k=5, timeout=30.0, blockers=()):
    """Build a check for base and arm solutions for the first top_k map-frame grasps.

    `blockers` are the scene graph's (centre, dimensions, yaw) footprints, as
    graspable.reposition keeps the base out of them. The costmap alone is not
    enough: the kitchen low table's top is above the laser and its footprint is
    free in the static map, so in run 20260924-020743 every base the solver
    offered for the kept grasp stood inside the table's standoff, and park
    (which does apply it) found "8 returned base poses, 0 clear of furniture".
    A grasp is reachable when at least one of its bases is clear of them.
    """
    from core.navigation import base_placement, graspable, standoff

    def reachable(poses):
        poses = np.asarray(poses, dtype=float)
        odom_from_map = navigator.frame_transform("odom", "map")
        # Check only the best few grasps to limit service calls.
        asked = poses[:top_k]
        # Transform map poses into odom, the frame expected by the IK service.
        results = base_placement.solve(odom_from_map @ asked, obstacles=obstacles,
                                       goal_frame="odom", timeout=timeout)
        map_from_odom = navigator.frame_transform("map", "odom")
        # Mark unchecked grasps as unreachable in the returned mask.
        mask = np.zeros(len(poses), dtype=bool)
        mask[:len(asked)] = [
            any(not standoff.blocks(graspable._planar(map_from_odom, base), blockers) for base in bases)
            for bases, _ in results
        ]
        return mask

    return reachable
