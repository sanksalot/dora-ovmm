# Pick up an object with MoveIt Task Constructor (MTC).
# Needs MoveIt, SAM3 and GraspGenX; pause Nav2 before picking.
# Run: python3 -m core.grasping.pick "pringles can"
# Real HSR: HSR_REAL_ROBOT=1 python3 -m core.grasping.pick "pringles can"

import sys
import tempfile
import time
import math
import os
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import rclcpp  # the C++ node MTC runs on (py_binding_tools), separate from rclpy
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped, Vector3Stamped, TwistStamped
from moveit.task_constructor import core, stages
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionIK, GetStateValidity, ChangeDriftDimensions
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from moveit_task_constructor_msgs.action import ExecuteTaskSolution
from rclpy.action import ActionClient
from rclpy.serialization import serialize_message
from rclpy.time import Time
from rclpy.parameter import Parameter as NodeParameter
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree
from sensor_msgs.msg import JointState, MultiDOFJointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Int8
from std_srvs.srv import Empty, Trigger
from octomap_msgs.msg import OctomapWithPose
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from tmc_control_msgs.action import GripperApplyEffort

from core.grasping import graspgenx_client
from core.perception import pointcloud, sam3_client
from core.perception.camera_ros2 import BASE_FRAME, USE_SIM_TIME, grab_rgbd, grab_hand_rgb, RobotTransforms
from core.utils.transforms import matrix_from_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "launch"))
import planning_model  # noqa: E402

# The 5-DOF arm alone cannot reach most grasp orientations; the base adds three.
GROUP = "whole_body"
HAND = "hand_palm_link"
FINGERS = [
    f"hand_{side}_{part}_link"
    for side in "lr"
    for part in ("proximal", "spring_proximal", "mimic_distal", "distal")
]
TARGET = "target"
GRIPPER = "hsrc_hand"
CANDIDATES = 20  # best-scored GraspGenX grasps offered to MTC
APPROACH = 0.08  # m, straight final approach along the palm's z axis
TEST_LIFT = 0.03  # verify physical motion before accepting a hold
MAX_CLOUD_POINTS = 8192  # bounds GraspGenX GPU memory
OPEN_HAND = 1.1  # hand_motor_joint; wider breaks the distal finger limit
GRASP_LIFT_MAX = 0.65  # reserve travel for approach and the verification lift
# Nm, negative closes. The lowest torque tried on a hand-held can (effort_test,
# 2026-09-19) and judged 'held', springs 0.41 / 0.49 above free air; -0.05 read
# the same. Was -0.3, never run on hardware. At -0.02 on the robot (2026-09-22,
# office table) the fingers did not wrap the can and it wobbled through the test
# lift, nearly out of the hand. -0.15, a little under half the allowed -0.3
# (about 2 N at the pads), held the can firmly through the lift on the next run
# the same night, no dent. HSR_GRASP_EFFORT_NM overrides.
CLOSE_EFFORT = -0.15
# The share of a mask's pixels that must carry depth for the view to be a
# measurement of the object. The camera returns nothing under ~0.5 m: on
# 2026-09-19 masks at 0.53 m and beyond had depth on 78-97% of their pixels, at
# 0.45-0.48 m on 0-29%; and after the lift on 2026-09-22 the can in the hand had
# 2%, 198 points of background at 2.9 m that then failed the lift check with the
# can held. Own choice, under every working view seen so far (30% and up).
MIN_DEPTH_COVERAGE = 0.10
# A lifted object is checked by the camera only when it can see the whole of it.
# Held, the can sits 0.49-0.50 m from the head camera, the near limit: on
# 2026-09-22 (23:58) the check view had depth on 15% of the mask, a 5 cm slice of
# the side between the hand and that limit, and its top read as the can's top,
# so a can in the hand was judged not to have risen. Views that measured the
# whole can ran at 0.51 m and beyond with 78% or more; own thresholds between.
LIFT_CHECK_RANGE = 0.50    # m, median depth on the mask
LIFT_CHECK_COVERAGE = 0.30
# How far a lifted can may sit from where the hand's own motion says it should be
# and still be the same can: sideways as the pads centre it (9.6 mm on 2026-09-19,
# 16.6 mm on 2026-09-23), upward as it rides up in the closing pads (10 mm on
# 2026-09-23). A can 25 mm off is not the one lifted (test). Own choices.
LIFT_SETTLE = 0.02  # m, sideways
LIFT_RIDE = 0.02    # m, above the hand's rise
# Head tilts tried, top to bottom, when the first view of a pick shows no target:
# the pick alone leaves the head where it was (mission_tree --nav false), and on
# 2026-09-23 (01:21) that was level, with the box on the table right below the
# frame. The head tilts down to -1.57; these cover a tabletop 0.4-1 m ahead.
SEARCH_TILTS = [-0.35, -0.6, -0.85]
# Pad gap left around the object by the fast pre-close: 15 mm a side for the
# width and placement error of the grasp. Own choice, not calibrated. The press
# test of 2026-09-19 met a ~75 mm can at command 0.58-0.60 (profile gap 74-76 mm),
# so the closing profile itself needs no allowance; perception gave that can as
# 69-73 mm, and it sat visibly off centre in the hand.
PRECLOSE_MARGIN = 0.03  # m
# Pad gap around the planned width inside which close() takes its small steps.
# Own choice. Perception gave a ~75 mm can as 68.7-73.0 mm (09-19), so contact
# comes up to 6 mm of gap early; and a stiff can took the springs from 0.04 to
# 0.24 in one 0.02 step (run 20260919_182003), past the whole 0.06-0.20 window.
FINE_MARGIN = 0.012  # m
# Use fingertip spacing to detect an empty grasp; motor angle alone is unreliable.
EMPTY_GAP = 0.01
# A motor that stops following its command is on the object, springs or not. On
# 2026-09-22 the fingers met the can and the reading stayed at 0.627 while the
# command went on to 0.617; the springs sat flat, so the spring test never fired
# and the closure ran on until a step to the blocked joint timed out. In free air
# the reading trails the command by the mean spring (free_air_reading), and while
# moving it lags that by up to 0.05 rad over a 0.02 step. Beyond this margin
# something is holding the fingers open. Own choice, from that run's samples.
STALL_LAG = 0.08
# The whole_body group's active joints, for joint-space goals.
WHOLE_BODY_JOINTS = ["odom_x", "odom_y", "odom_t", "arm_lift_joint", "arm_flex_joint",
                     "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
HEAD_LINKS = ["head_rgbd_sensor_link", "head_tilt_link"]
BASE_SLACK = {
    "odom_x": 0.25,
    "odom_y": 0.25,
    "odom_t": 0.2,
}  # m, m, rad around parked base
JOINT_STATES = "/whole_body_moveit/joint_states"  # includes the base's odom joints


# Contact geometry

GRIPPER_DIR = Path(__file__).resolve().parents[2] / 'docker/graspgenx/x_grippers/hsrc_hand'

# Choose approaches from measured shape and floor height in odom metres.
# MoveIt still checks reachability and collisions.
# Front and side only. Overhead reaches are hard to plan for this arm over furniture,
# so objects are set out with a graspable side towards the robot; one too wide from
# there is refused, not taken from above. 'top_down' (with 'wide_approach' and
# 'top_down_max_z') is still understood below, should a shape need it again.
SHAPE_POLICIES = {
    'cylinder': {'approach': 'front'},
    'sphere': {'approach': 'front'},
    'cube': {'approach': 'front'},
    'cuboid': {'approach': 'front'},
}


def cloud(points):
    # Require at least 100 finite XYZ points within the allowed grasp workspace.
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise RuntimeError('Insufficient finite target geometry')
    if np.ptp(points, axis=0).max() > .6:
        raise RuntimeError('Target geometry exceeds the grasp workspace')
    return points


def pad_for_width(width):
    # Look up the calibrated pad centre for this opening width.
    profile = json.loads((GRIPPER_DIR / 'closing_profile.json').read_text())
    gaps = np.array([s['gap'][0] for s in profile])
    if not gaps[0] <= width <= gaps[-1] - .012:
        raise RuntimeError(f'Contact width {width:.3f} m outside calibrated aperture')
    # Interpolate the pad centre between neighbouring calibrated finger openings.
    return np.array([np.interp(width, gaps, [s['center'][i] for s in profile]) for i in range(3)])


def calibrated_palm_pose(rotation,contact,width):
    pose=np.eye(4)
    pose[:3,:3]=rotation
    # Rotate the calibrated pad offset and subtract it to locate the palm.
    pose[:3,3]=np.asarray(contact)-np.asarray(rotation)@pad_for_width(width)
    return pose


def cylinder(points):
    # Use the middle height band to fit the cylinder without its top and bottom.
    bottom, top = np.quantile(points[:, 2], [.01, .99])
    side = points[(points[:, 2] > bottom + .2*(top-bottom)) &
                  (points[:, 2] < top - .2*(top-bottom)), :2]
    if len(side) < 100 or top-bottom < .04:
        return None
    origin = side.mean(0)
    xy = side-origin
    # Fit a circle to the visible side points in the horizontal plane.
    a = np.column_stack([2*xy, np.ones(len(xy))])
    if np.linalg.matrix_rank(a) < 3:
        return None
    fit = np.linalg.lstsq(a, np.sum(xy*xy, axis=1), rcond=None)[0]
    radius2 = fit[2] + fit[:2]@fit[:2]
    if radius2 <= 0:
        return None
    radius = np.sqrt(radius2)
    centre = origin+fit[:2]
    # Allow depth noise proportional to object size, retaining the small-object
    # tolerance. Resolved rectangular tops take priority in contact_candidates.
    # Reject poor fits, implausible sizes, or too little visible curvature.
    residual = np.sqrt(np.mean((np.linalg.norm(side-centre, axis=1)-radius)**2))
    if not .008 <= radius <= .15 or residual > max(.0015, .10*radius):
        return None
    # Require enough of the cylinder arc to estimate its hidden centre.
    if np.linalg.norm(((side-centre)/radius).mean(0)) > .95:
        return None  # too little visible arc to infer the hidden centre
    return centre, radius, bottom, top


def top_rectangle(points):
    top = np.quantile(points[:, 2], .98)
    face = points[np.abs(points[:, 2]-top) < .003, :2]
    if len(face) < max(100, .08*len(points)):
        return None
    # Fit a rectangle around the top surface and check how fully points cover it.
    rect = cv2.minAreaRect(face.astype(np.float32))
    corners = cv2.boxPoints(rect).astype(float)
    edges = np.roll(corners, -1, axis=0)-corners
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < .012 or lengths.max() > .45:
        return None
    area = ConvexHull(face).volume
    if area / (lengths[0]*lengths[1]) < .9:
        return None  # circular/irregular top, rather than a resolved rectangle
    # Return the short top edge as the preferred finger-closing direction.
    index = int(np.argmin(lengths))
    return corners, edges[index]/lengths[index], lengths[index], top


def sphere(points):
    points = cloud(points)
    if len(points) < 100:
        return None
    origin = points.mean(0)
    local = points-origin
    if np.linalg.svd(local, compute_uv=False)[-1]/np.sqrt(len(points)) < .004:
        return None
    # Refit the sphere three times, trimming the largest surface errors each time.
    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        a = np.column_stack([2*local[keep], np.ones(keep.sum())])
        fit = np.linalg.lstsq(a, np.sum(local[keep]**2, axis=1), rcond=None)[0]
        squared = fit[3]+fit[:3]@fit[:3]
        if squared <= 0:
            return None
        radius = np.sqrt(squared); centre = origin+fit[:3]
        error = np.abs(np.linalg.norm(points-centre, axis=1)-radius)
        # Keep the best-fitting 85 percent of points for the next sphere fit.
        keep = error <= np.quantile(error, .85)
    if not .015 <= radius <= .055 or np.sqrt(np.mean(error[keep]**2)) > .0025:
        return None
    if np.linalg.norm(((points-centre)/radius).mean(0)) > .85:
        return None
    return centre, radius


def rectangular_candidates(rect, bottom, camera_position):
    corners, short_axis, short_width, top = rect
    height = top-bottom
    if height < .015:
        raise RuntimeError('Only a flat top is visible; object height is unresolved')
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=1)
    dimensions = np.r_[lengths[:2], height]
    # Choose the box policy from the measured proportions.
    shape = 'cube' if dimensions.max()/dimensions.min() < 1.15 else 'cuboid'
    policy = SHAPE_POLICIES[shape]
    centre = corners.mean(0)
    toward = centre-np.asarray(camera_position)[:2]
    if np.linalg.norm(toward) < .01:
        raise RuntimeError('Front approach direction is unresolved')
    toward /= np.linalg.norm(toward)
    screen_horizontal = np.array([-toward[1], toward[0]])
    horizontal_width = np.ptp(corners@screen_horizontal)
    profile = json.loads((GRIPPER_DIR/'closing_profile.json').read_text())
    max_width = profile[-1]['gap'][0]-.012
    mode = policy['approach']
    # Use an overhead approach if the front is too wide, unless the object is too high.
    if horizontal_width > max_width:
        mode = policy.get('wide_approach', mode)
    if top > policy.get('top_down_max_z', -np.inf):
        mode = 'front'

    poses, widths = [], []
    axes = [short_axis, np.array([-short_axis[1], short_axis[0]])]
    # Try closing along each box axis, keeping only calibrated opening widths.
    for axis in axes:
        width = float(np.ptp(corners@axis))
        if mode == 'top_down' and width > short_width*1.1:
            continue
        try:
            pad_for_width(width)
        except RuntimeError:
            continue
        for sign in [1., -1.]:
            if mode == 'top_down':
                closing = np.r_[sign*axis, 0.]
                approach = np.array([0., 0., -1.])
                contact = np.r_[centre, top-min(.025, .35*width, .45*height)]
            else:
                approach = np.r_[sign*np.array([-axis[1], axis[0]]), 0.]
                # Approach only from the observed/front half of the object.
                if approach[:2]@toward < .5:
                    continue
                closing = -np.cross([0., 0., 1.], approach)
                depth = float(np.ptp(corners@approach[:2]))
                # Place contact slightly inside the observed front face, bounded by object size.
                penetration = front_penetration(width, depth)
                contact = np.r_[centre, (bottom+top)/2]
                contact -= approach*(depth/2-penetration)
            orientation = np.column_stack([closing, np.cross(approach, closing), approach])
            poses.append(calibrated_palm_pose(orientation, contact, width))
            widths.append(width)
            if mode == 'front' and shape == 'cuboid':
                # Try another wrist roll and a higher contact to clear low tables.
                raised = contact + np.array([0., 0., min(.04, .2*height)])
                flipped = orientation @ np.diag([-1., -1., 1.])
                for rotation, point in [(flipped, contact), (orientation, raised), (flipped, raised)]:
                    poses.append(calibrated_palm_pose(rotation, point, width))
                    widths.append(width)
    if not poses:
        raise RuntimeError(f'No aperture-compatible {shape} {mode} grasp; '
                           f'top z={top:.3f} m, horizontal width={horizontal_width:.3f} m')
    return poses, widths, f'{shape} {mode}'


def flat_face_at_grasp_height(points, half=.03):
    """The points within `half` of the object's mid-height and that slab's normal, when
    the slab is one vertical plane; otherwise None."""
    bottom, top = np.quantile(points[:, 2], [.01, .99])
    slab = points[np.abs(points[:, 2]-(bottom+top)/2) < half]
    if len(slab) < 100:
        return None
    singular, axes = np.linalg.svd(slab-slab.mean(0), full_matrices=False)[1:]
    if singular[-1]/np.sqrt(len(slab)) >= .003 or abs(axes[-1][2]) > .35:
        return None
    return slab, axes[-1]


# How far past an object's front surface the pads close, for a front grasp: half the
# closing width, at most FRONT_REACH, and never past the middle of a known depth.
# Was a third of the width and 25 mm at most; on the box of 2026-09-23 (01:30) that
# held, but at the box's front edge. Own choice, 20 mm on a 40 mm box.
FRONT_REACH = 0.035


def front_penetration(width, depth=None):
    reach = min(FRONT_REACH, .5*width)
    return reach if depth is None else min(reach, .45*depth)


def front_face_candidates(points, normal, camera_position):
    """Front grasps on one vertical face seen square on: what a slim box shows with its
    narrow side to the robot (2026-09-23 00:49, cloud 26 x 42 x 147 mm, refused as a
    flat surface). The face's width and height are measured; its depth is not, and a
    front grasp needs none beyond the pads' reach. rectangular_candidates' front
    mode, on a face instead of a resolved top; own extension."""
    approach = np.r_[normal[:2], 0.]
    approach /= np.linalg.norm(approach)
    if approach @ (points.mean(0)-np.asarray(camera_position)) < 0:
        approach = -approach  # into the face, away from the camera
    closing = -np.cross([0., 0., 1.], approach)
    across = np.quantile(points@closing, [.01, .99])
    bottom, top = np.quantile(points[:, 2], [.01, .99])
    width, height = float(across[1]-across[0]), top-bottom
    if height < .015:
        raise RuntimeError('Only a flat surface is visible; object height is unresolved')
    pad_for_width(width)
    centre = np.r_[(points@approach).mean()*approach[:2] + across.mean()*closing[:2], (bottom+top)/2]
    contact = centre + approach*front_penetration(width)
    orientation = np.column_stack([closing, np.cross(approach, closing), approach])
    flipped = orientation @ np.diag([-1., -1., 1.])
    raised = contact + np.array([0., 0., min(.04, .2*height)])
    poses = [calibrated_palm_pose(rotation, point, width)
             for rotation, point in [(orientation, contact), (flipped, contact), (orientation, raised), (flipped, raised)]]
    return poses, [width]*len(poses), 'front face'


def contact_candidates(points, palms, camera_position):
    points = cloud(points)
    rotation = np.array(json.loads((GRIPPER_DIR/'config.json').read_text())['base_rotation'])
    # Convert palm poses into the model's gripper frame for contact calibration.
    canonical = np.asarray(palms) @ np.linalg.inv(rotation)
    cyl, rect = cylinder(points), top_rectangle(points)
    results, widths = [], []
    envelope = points
    kind = 'observed front section'
    # Two visible box faces can also fit a noisy circle; prefer a resolved top.
    if cyl is not None and rect is None:
        centre, radius, bottom, top = cyl
        width = 2*radius
        pad = pad_for_width(width)
        # Build a full cylinder outline for collision checking, including its hidden side.
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        envelope = np.array([[centre[0]+radius*np.cos(t), centre[1]+radius*np.sin(t), z]
                             for z in [bottom, top] for t in theta])
        kind = 'cylinder'
        for pose in canonical:
            if abs(pose[2, 2]) > .35 or abs(pose[2, 0]) > .35:
                continue
            approach = pose[:3, 2].copy(); approach[2] = 0
            approach /= np.linalg.norm(approach)
            toward = np.r_[centre, (bottom+top)/2]-camera_position
            if approach@toward / np.linalg.norm(toward[:2]) < .5:
                continue
            closing = np.cross([0., 0., 1.], approach)
            # Keep palm +X upward so the forearm is less likely to hide the target.
            closing *= -1
            corrected = np.eye(4)
            corrected[:3, :3] = np.column_stack([closing, np.cross(approach, closing), approach])
            # Place side contact slightly above centre, staying below the rim.
            contact = np.r_[centre, (bottom+top)/2 + min(.03, .25*(top-bottom))]
            corrected = calibrated_palm_pose(corrected[:3,:3],contact,width)
            # Reject a corrected cylinder grasp that moves too far from the model proposal.
            if np.linalg.norm(corrected[:3, 3]-pose[:3, 3]) > .10:
                continue
            results.append(corrected@rotation); widths.append(width)
    elif rect is not None:
        corners, short_axis, width, top = rect
        bottom = np.quantile(points[:, 2], .01)
        envelope = np.array([[x, y, z] for z in [bottom, top] for x, y in corners])
        # Use the resolved top rectangle to generate calibrated box grasps.
        poses, widths, kind = rectangular_candidates(rect, bottom, camera_position)
        results = [pose@rotation for pose in poses]
    elif sphere(points) is not None:
        centre, radius = sphere(points)
        width = 2*radius
        pad_for_width(width)
        envelope = centre + radius*np.array([[x,y,z] for x in [-1.,1.]
                                             for y in [-1.,1.] for z in [-1.,1.]])
        toward = centre[:2]-np.asarray(camera_position)[:2]
        heading = np.arctan2(toward[1], toward[0])
        kind = 'sphere'
        # Offer top-down sphere grasps only when the object is low enough.
        if (SHAPE_POLICIES['sphere']['approach'] == 'top_down'
                and centre[2]+radius <= SHAPE_POLICIES['sphere']['top_down_max_z']):
            for angle in heading+np.array([0., np.pi/2, np.pi, 3*np.pi/2]):
                closing = np.array([np.cos(angle), np.sin(angle), 0.])
                approach = np.array([0., 0., -1.])
                orientation = np.column_stack([closing, np.cross(approach, closing), approach])
                results.append(calibrated_palm_pose(orientation, centre, width)@rotation)
                widths.append(width)
        # Also try side approaches around the direction of the observing camera.
        for angle in heading+np.array([0., -.26, .26, -.52, .52]):
            approach = np.array([np.cos(angle), np.sin(angle), 0.])
            closing = -np.cross([0.,0.,1.], approach)
            orientation = np.column_stack([closing, np.cross(approach, closing), approach])
            for roll in [np.eye(3), np.diag([-1.,-1.,1.])]:
                results.append(calibrated_palm_pose(orientation@roll, centre, width)@rotation)
                widths.append(width)
    elif np.linalg.svd(points-points.mean(0),compute_uv=False)[-1]/np.sqrt(len(points)) < .003:
        normal = np.linalg.svd(points-points.mean(0), full_matrices=False)[2][-1]
        if abs(normal[2]) > .35:
            raise RuntimeError('A single flat surface does not constrain a grasp; another view is needed')
        poses, widths, kind = front_face_candidates(points, normal, camera_position)
        results = [pose@rotation for pose in poses]
    elif (face := flat_face_at_grasp_height(points)) is not None:
        # A box seen narrow side on with its top in view (2026-09-23 01:25): the cloud is
        # not flat, but the slice where the pads close is one vertical face. Grasp it
        # square on. GraspGenX's proposal ran 19 deg off the face there, and a pad met
        # the front corner and pushed the box away instead of closing on its side.
        slab, normal = face
        poses, widths, kind = front_face_candidates(slab, normal, camera_position)
        results = [pose@rotation for pose in poses]
    else:
        for pose in canonical:
            approach = pose[:3, 2]
            toward = points.mean(0)-camera_position
            if abs(approach[2]) > .35 or approach@toward/np.linalg.norm(toward) < .65:
                continue
            local = (points-pose[:3, 3])@pose[:3, :3]
            # Measure the visible surface in the narrow band where the pads will close.
            section = local[np.abs(local[:,1]-.01276) < .0175]
            if len(section) < 100:
                continue
            lo, hi = np.quantile(section,[.01,.99],axis=0)
            width = hi[0]-lo[0]
            try:
                pad = pad_for_width(width)
            except RuntimeError:
                continue
            # Require resolved depth as well as width. Do not invent a back face.
            if hi[2]-lo[2] < .01:
                continue
            # Pads at least front_penetration in, as the box branches do: the middle of
            # the visible depth is 7 mm behind a face seen square on (box, 2026-09-23
            # 01:05), and fingers there close in front of the object.
            depth = max((lo[2]+hi[2])/2, lo[2]+front_penetration(width))
            contact = np.array([(lo[0]+hi[0])/2,pad[1],depth])
            corrected = calibrated_palm_pose(pose[:3,:3],pose[:3,3]+pose[:3,:3]@contact,width)
            if np.linalg.norm(corrected[:3,3]-pose[:3,3]) > .06:
                continue
            results.append(corrected@rotation); widths.append(width)
    # Fail if no candidate fits both the observed geometry and the hand opening.
    if not results:
        raise RuntimeError(f'No aperture-compatible {kind} grasp in the observed geometry')
    return envelope, np.asarray(results), np.asarray(widths), kind


def verify_object_lift(before, after, expected):
    before, after = cloud(before), cloud(after)
    expected = np.asarray(expected, dtype=float)
    if expected[2] < .015:
        raise RuntimeError('Test lift was too small to verify')
    # Compare cylinder radius, centre, and top; fingers may hide the bottom.
    first, second = cylinder(before), cylinder(after)
    if first is not None and second is not None:
        centre0, radius0, _, top0 = first
        centre1, radius1, _, top1 = second
        rise = top1-top0
        # Undo the can's own fitted motion, not the hand's, and compare the remaining
        # surface with the original cloud: the can settles in the closing pads, 15 mm
        # sideways and 10 mm further up than the hand on 2026-09-23 (kitchen, 00:14),
        # with only its top in view. Further than LIFT_SETTLE sideways is another
        # object; LIFT_RIDE up is as far as the pads carry it.
        motion = np.r_[centre1-centre0, rise]
        distances = cKDTree(before).query(after-motion)[0]
        coverage = float(np.mean(distances < .01))
        if (abs(radius1-radius0) <= .003 and
                np.linalg.norm(centre1-centre0-expected[:2]) <= LIFT_SETTLE and
                rise >= .6*expected[2] and rise-expected[2] <= LIFT_RIDE and
                coverage >= .7):
            return {'rise_m': float(rise), 'surface_coverage': coverage}
    # Keep the original sphere radius when fitting motion from the visible surface.
    first, second = sphere(before), sphere(after)
    if first is not None and second is not None:
        centre, radius = first
        _, observed_radius = second
        sample = after[::max(1, len(after)//3000)]
        # Estimate sphere translation with a robust fit that tolerates small surface errors.
        fit = least_squares(lambda c: np.linalg.norm(sample-c, axis=1)-radius,
                            centre, loss='soft_l1', f_scale=.002, max_nfev=50)
        motion = fit.x-centre
        coverage = float(np.mean(cKDTree(before).query(after-expected)[0] < .01))
        top_rise = np.quantile(after[:,2], .95)-np.quantile(before[:,2], .95)
        if (fit.success and abs(observed_radius-radius) <= min(.005, max(.003, .1*radius))
                and np.linalg.norm(motion[:2]-expected[:2]) <= .008
                and motion[2] >= .6*expected[2] and abs(motion[2]-expected[2]) <= .01
                and top_rise >= .6*expected[2] and abs(top_rise-expected[2]) <= .01
                and coverage >= .7):
            return {'rise_m': float(motion[2]), 'surface_coverage': coverage}
    # For other shapes, compare height changes and surface overlap in both directions.
    dz = np.quantile(after[:,2],[.05,.5,.95])-np.quantile(before[:,2],[.05,.5,.95])
    if dz[-1] < .6*expected[2] or np.max(np.abs(dz-expected[2])) > .012:
        raise RuntimeError('Object did not rise with the hand (or became too occluded to verify)')
    shifted = after-expected
    distances = cKDTree(before[::max(1,len(before)//3000)]).query(shifted)[0]
    reverse = cKDTree(shifted[::max(1,len(shifted)//3000)]).query(before)[0]
    if np.mean(distances < .01) < .7 or np.mean(reverse < .012) < .6:
        raise RuntimeError('Lifted target surface does not match the original object')
    return {'rise_m':float(np.median(dz)), 'surface_coverage':float(np.mean(distances < .01))}

# Pregrasp routes

def side_waypoint(overhead, base_xy, distance=.12):
    # Create a side waypoint only for a nearly vertical downward approach.
    if overhead[2, 2] > -.9:
        return None
    toward = np.asarray(base_xy)-overhead[:2, 3]
    norm = np.linalg.norm(toward)
    if norm < .01:
        return None
    side = overhead.copy()
    # Shift toward the base without moving more than halfway to it.
    side[:2, 3] += min(distance, norm/2)*toward/norm
    return side


def pregrasp_routes(planner, overhead, parked, base_xy, group, hand, frame, ik=None):
    def move(name, matrix, timeout=3.):
        goal = PoseStamped()
        goal.header.frame_id = frame
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = matrix[:3, 3].tolist()
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = q.tolist()
        stage = stages.MoveTo(name, planner)
        stage.group = group
        ik_frame = PoseStamped(); ik_frame.header.frame_id = hand; ik_frame.pose.orientation.w = 1.
        stage.ik_frame = ik_frame
        # Prefer IK seeded from the current robot pose to keep the base nearby.
        # Fall back to a pose goal if that joint solution is unavailable.
        goal_state = ik(matrix) if ik is not None else None
        if goal_state is not None:
            print(f'[IK] {name}: seeded whole-body solution inside the base window', flush=True)
            stage.setGoal(goal_state)
            # A valid joint goal deserves the time to route around the table.
            timeout = max(timeout, 15.)
        else:
            stage.setGoal(goal)
        # Keep the base within its parking window along the whole planned route.
        stage.path_constraints = parked
        stage.timeout = timeout
        return stage

    # For overhead grasps, try a robot-facing waypoint before the direct route.
    side = side_waypoint(overhead, base_xy)
    if side is None:
        return move('pregrasp', overhead)
    routes = core.Fallbacks('overhead pregrasp routes')
    staged = core.SerialContainer('via robot-facing side')
    staged.insert(move('side staging', side, timeout=10.))
    staged.insert(move('move above target', overhead, timeout=10.))
    routes.insert(staged)
    routes.insert(move('direct overhead pregrasp', overhead))
    return routes

# Visual servo

def mask_feature(mask):
    v,u=np.nonzero(mask)
    if len(u)<100:
        raise RuntimeError('Too few hand-camera target pixels')
    # Reject clipped masks because their visible centre may not be the target centre.
    if u.min()==0 or v.min()==0 or u.max()==mask.shape[1]-1 or v.max()==mask.shape[0]-1:
        raise RuntimeError('Target is clipped by the hand camera image boundary')
    return np.array([u.mean(),v.mean()]),len(u)


def project_feature(point,k):
    point=np.asarray(point,float)
    if not np.isfinite(point).all() or point[2]<.05:
        raise RuntimeError('Invalid hand-camera feature depth')
    # Project the 3D point with camera intrinsics, then divide by depth to get pixels.
    pixel=k@point
    return pixel[:2]/pixel[2]


def hand_image_velocity(uv,depth,desired_uv,k,basis_camera):
    point=np.array([(uv[0]-k[0,2])/k[0,0],(uv[1]-k[1,2])/k[1,1],1.])*depth
    velocity,error=image_velocity(point,desired_uv,k,basis_camera,min_depth=.05)
    # Reverse the sign because moving the camera moves the image in the opposite direction.
    return -velocity,error


def orientation_velocity(desired,current):
    # Turn orientation error into angular velocity and cap its speed.
    error=Rotation.from_matrix(desired@current.T).as_rotvec()
    velocity=1.5*error
    velocity*=min(1.,.08/max(np.linalg.norm(velocity),1e-12))
    return velocity


def approach_velocity(start,current,distance):
    axis=start[:3,2]
    delta=current[:3,3]-start[:3,3]
    # Separate forward progress from sideways drift along the planned approach.
    progress=float(delta@axis)
    lateral=delta-progress*axis
    angle=Rotation.from_matrix(start[:3,:3].T@current[:3,:3]).magnitude()
    if progress < -.004 or progress > distance+.003:
        raise RuntimeError('Servo approach left its axial travel bounds')
    if np.linalg.norm(lateral)>.004 or angle>.04:
        raise RuntimeError('Servo approach left its alignment corridor')
    remaining=distance-progress
    # Finish only when forward distance, sideways drift, and rotation are all small.
    finished=abs(remaining)<.002 and np.linalg.norm(lateral)<.002 and angle<.02
    velocity=axis*np.clip(.8*remaining,0.,.006)-.8*lateral
    velocity*=min(1.,.006/max(np.linalg.norm(velocity),1e-12))
    return velocity,finished,progress


def feature(depth, k, mask):
    valid = pointcloud.object_depth_mask(depth, mask)
    v, u = np.nonzero(valid)
    uv = np.array([u.mean(), v.mean()])
    # Estimate target depth near the mask centre, with the full mask as a fallback.
    near = (u-uv[0])**2+(v-uv[1])**2 < 25
    z = np.median(depth[v[near],u[near]]) if near.any() else np.median(depth[valid])
    xyz = np.array([(uv[0]-k[0,2])*z/k[0,0],(uv[1]-k[1,2])*z/k[1,1],z])
    return uv, xyz, int(valid.sum())


def image_velocity(tool_point_camera, target_uv, k, basis_camera, gain=.6, max_speed=.012, min_depth=.15):
    x,y,z = np.asarray(tool_point_camera,float)
    if not np.isfinite([x,y,z]).all() or z < min_depth:
        raise RuntimeError('Control feature is behind/too close to the camera')
    predicted = np.array([k[0,0]*x/z+k[0,2],k[1,1]*y/z+k[1,2]])
    error = np.asarray(target_uv)-predicted
    # Relate small 3D movements to image-pixel movement using camera intrinsics.
    interaction = np.array([[k[0,0]/z,0,-k[0,0]*x/z**2],
                            [0,k[1,1]/z,-k[1,1]*y/z**2]])
    jacobian = interaction@basis_camera
    if not np.isfinite(jacobian).all() or np.linalg.cond(jacobian) > 50:
        raise RuntimeError('Image Jacobian cannot resolve transverse alignment from this view')
    # Solve for the correction in the allowed motion directions and limit speed.
    velocity = gain*np.linalg.solve(jacobian,error)
    velocity *= min(1.,max_speed/max(np.linalg.norm(velocity),1e-12))
    return velocity, error


class VisualServo:
    def __init__(self,node):
        self.node = node
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer,node)
        # Send Cartesian velocity commands and listen for Servo stop or error status.
        self.publisher = node.create_publisher(TwistStamped,'/grasp_servo/delta_twist_cmds',10)
        self.status = None
        self.status_time = 0.
        self.subscription = node.create_subscription(Int8,'/grasp_servo/status',self._status,10)

    def _status(self,message):
        self.status = message.data
        self.status_time = time.monotonic()

    def transform(self,link='hand_palm_link',stamp=None):
        stamp = Time() if stamp is None else stamp
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            rclpy.spin_once(self.node,timeout_sec=.02)
            if self.buffer.can_transform('odom',link,stamp):
                transform = self.buffer.lookup_transform('odom',link,stamp)
                age = (self.node.get_clock().now()-Time.from_msg(transform.header.stamp)).nanoseconds/1e9
                # Ignore transforms older than half a second during motion feedback.
                if age > .5:
                    continue
                return matrix_from_transform(transform.transform)
        raise RuntimeError(f'No fresh odom-to-{link} transform')

    def service(self,name,kind=Trigger,request=None):
        # Call the named Servo control service and require a successful reply.
        client=self.node.create_client(kind,'/grasp_servo/'+name)
        try:
            if not client.wait_for_service(timeout_sec=5.):
                raise RuntimeError(f'MoveIt Servo {name} unavailable')
            future=client.call_async(request or kind.Request())
            rclpy.spin_until_future_complete(self.node,future,timeout_sec=5.)
            if not future.done() or not future.result().success:
                raise RuntimeError(f'MoveIt Servo {name} failed')
        finally:
            self.node.destroy_client(client)

    def command(self,linear,angular=None):
        msg=TwistStamped()
        msg.header.stamp=self.node.get_clock().now().to_msg()
        # Express the velocity command in odom so it matches the feedback poses.
        msg.header.frame_id='odom'
        msg.twist.linear.x,msg.twist.linear.y,msg.twist.linear.z=map(float,linear)
        if angular is not None:
            msg.twist.angular.x,msg.twist.angular.y,msg.twist.angular.z=map(float,angular)
        self.publisher.publish(msg)

    def stop(self):
        # Send zero velocity before pausing so smoothing does not leave motion active.
        end=time.monotonic()+.3
        while time.monotonic()<end:
            self.command(np.zeros(3));rclpy.spin_once(self.node,timeout_sec=.02)
        self.service('pause_servo')
        # Wait for measured hand motion to settle before returning control to MTC.
        first=self.transform()
        end=time.monotonic()+3.
        while time.monotonic()<end:
            rclpy.spin_once(self.node,timeout_sec=.1)
            second=self.transform()
            if np.linalg.norm(first[:3,3]-second[:3,3]) < .0005:
                return
            first=second
        raise RuntimeError('Arm did not settle after visual servoing')

    def align(self,prompt,pregrasp,reference_feature,reference_area):
        start=self.transform();base=self.transform('base_footprint')
        desired_uv=None; hand_area=None
        basis=pregrasp[:3,:2]  # centring only; preserve axial pregrasp clearance
        history=[];converged=0
        try:
            self.service('start_servo')
            # Hold orientation so image correction comes from translation.
            self.service('change_drift_dimensions',ChangeDriftDimensions,
                         ChangeDriftDimensions.Request())
            self.service('unpause_servo')
            deadline=time.monotonic()+180.
            while time.monotonic()<deadline:
                self.command(np.zeros(3))
                rgb,k,odom_camera,image_hand=grab_hand_rgb(timeout=8.)
                captured=time.monotonic()
                mask,_=sam3_client.detect(rgb,prompt,timeout=5.)
                uv,area=mask_feature(mask)
                camera_odom=np.linalg.inv(odom_camera)
                target_camera=(camera_odom@np.r_[reference_feature,1.])[:3]
                if target_camera[2] < .05:
                    raise RuntimeError('Target too close to hand camera')
                observed=np.array([(uv[0]-k[0,2])/k[0,0],
                                   (uv[1]-k[1,2])/k[1,1],1.])*target_camera[2]
                # Project the target from the planned pregrasp to set the desired image position.
                if desired_uv is None:
                    # Find the fixed hand-to-camera transform from poses captured with this image.
                    hand_camera=np.linalg.inv(image_hand)@odom_camera
                    desired_point=(np.linalg.inv(pregrasp@hand_camera)@np.r_[reference_feature,1.])[:3]
                    desired_uv=project_feature(desired_point,k)
                    if not (0<=desired_uv[0]<rgb.shape[1] and 0<=desired_uv[1]<rgb.shape[0]):
                        raise RuntimeError('Planned target feature is outside the hand camera view')
                    hand_area=area
                if time.monotonic()-captured > 1.:
                    raise RuntimeError('Visual observation became stale during detection')
                # Reject large mask-size changes that can indicate occlusion or another object.
                if not .6 <= area/hand_area <= 1.5:
                    raise RuntimeError('Target became occluded or changed identity during IBVS')
                target_world=odom_camera[:3,:3]@observed+odom_camera[:3,3]
                # Reject target changes or robot drift before issuing another correction.
                if np.linalg.norm(target_world-reference_feature) > .035:
                    raise RuntimeError('Target association moved beyond the bounded alignment region')
                hand=self.transform();current_base=self.transform('base_footprint')
                if np.linalg.norm(current_base[:3,3]-base[:3,3]) > .003:
                    raise RuntimeError('Base moved during arm-only visual servoing')
                if Rotation.from_matrix(base[:3,:3].T@current_base[:3,:3]).magnitude() > .01:
                    raise RuntimeError('Base rotated during arm-only visual servoing')
                if np.linalg.norm(hand[:3,3]-pregrasp[:3,3]) > .035:
                    raise RuntimeError('IBVS correction exceeded 35 mm')
                if Rotation.from_matrix(pregrasp[:3,:3].T@hand[:3,:3]).magnitude() > .06:
                    raise RuntimeError('Arm orientation drift exceeded the IBVS bound')
                lateral,error=hand_image_velocity(uv,target_camera[2],desired_uv,k,camera_odom[:3,:3]@basis)
                angular=orientation_velocity(pregrasp[:3,:3],hand[:3,:3])
                norm=float(np.linalg.norm(error));history.append(norm)
                print(f'[IBVS] image error {norm:.2f} px',flush=True)
                # Require three consecutive aligned observations before accepting convergence.
                converged=converged+1 if norm < 2. else 0
                if converged >= 3:
                    return history
                if len(history)>8 and min(history[-4:]) > min(history[:-4])+2.:
                    raise RuntimeError('IBVS image error is increasing')
                # Move briefly using this observation, then stop for the next image.
                until=time.monotonic()+.15
                while time.monotonic()<until:
                    rclpy.spin_once(self.node,timeout_sec=.02)
                    if self.status in (-1,2,4,5) or time.monotonic()-self.status_time > .5:
                        raise RuntimeError(f'MoveIt Servo halted or status stale ({self.status})')
                    self.command(basis@lateral if norm>=2. else np.zeros(3),angular)
                self.command(np.zeros(3))
            raise RuntimeError('IBVS did not converge before timeout')
        finally:
            self.stop()

    def approach(self,prompt,reference_feature,distance=.08,start_pose=None):
        if not 0.<distance<=.08:
            raise ValueError('Final approach must be within 80 mm')
        start=self.transform() if start_pose is None else np.asarray(start_pose,float)
        approach_velocity(start,self.transform(),distance)
        base=self.transform('base_footprint')
        history=[];settled=0;initial_area=None;best=0.;last_progress=time.monotonic()
        try:
            self.service('start_servo')
            self.service('change_drift_dimensions',ChangeDriftDimensions,ChangeDriftDimensions.Request())
            self.service('unpause_servo')
            deadline=time.monotonic()+300.
            while time.monotonic()<deadline:
                self.command(np.zeros(3))
                rgb,k,camera,_=grab_hand_rgb(timeout=8.)
                captured=time.monotonic()
                mask,_=sam3_client.detect(rgb,prompt,timeout=5.)
                if time.monotonic()-captured>1.:
                    raise RuntimeError('Approach target observation became stale')
                # Check that the detected target still contains its expected image position.
                point=(np.linalg.inv(camera)@np.r_[reference_feature,1.])[:3]
                pixel=project_feature(point,k)
                u,v=np.rint(pixel).astype(int)
                if not (0<=u<rgb.shape[1] and 0<=v<rgb.shape[0]):
                    raise RuntimeError('Target reference left the hand-camera view')
                if mask.sum()<100 or not cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))[v,u]:
                    raise RuntimeError('Target detection no longer contains the expected approach feature')
                if initial_area is None:initial_area=int(mask.sum())
                if mask.sum()<.6*initial_area:
                    raise RuntimeError('Approach target became substantially occluded')
                hand=self.transform();current_base=self.transform('base_footprint')
                if np.linalg.norm(current_base[:3,3]-base[:3,3])>.003 or Rotation.from_matrix(base[:3,:3].T@current_base[:3,:3]).magnitude()>.01:
                    raise RuntimeError('Base moved during Servo-only approach')
                velocity,finished,progress=approach_velocity(start,hand,distance)
                history.append(progress)
                print(f'[SERVO APPROACH] {progress*1000:.1f}/{distance*1000:.1f} mm',flush=True)
                settled=settled+1 if finished else 0
                if settled>=3:break
                if progress>best+.0005:
                    best=progress;last_progress=time.monotonic()
                # Abort if the approach stops making measurable forward progress.
                if not finished and time.monotonic()-last_progress>20.:
                    raise RuntimeError('Servo approach made no measurable progress')
                angular=orientation_velocity(start[:3,:3],hand[:3,:3])
                # Keep checking pose and Servo status while sending this motion burst.
                until=time.monotonic()+1.
                while time.monotonic()<until:
                    rclpy.spin_once(self.node,timeout_sec=.02)
                    if self.status in (-1,2,4,5) or time.monotonic()-self.status_time>.5:
                        raise RuntimeError(f'Servo approach halted or status stale ({self.status})')
                    hand=self.transform()
                    velocity,finished,_=approach_velocity(start,hand,distance)
                    angular=orientation_velocity(start[:3,:3],hand[:3,:3])
                    self.command(np.zeros(3) if finished else velocity,angular)
                self.command(np.zeros(3))
            else:
                raise RuntimeError('Servo approach did not reach its endpoint before timeout')
        finally:
            self.stop()
        # Check the settled endpoint, not just the last sample before stopping.
        _,finished,_=approach_velocity(start,self.transform(),distance)
        if not finished:
            raise RuntimeError('Servo stopped short of the final endpoint; refusing closure')
        return history

# Scene payload

# Empty maps mean no change in a scene diff, but replace the map in a full scene.


def compact(solution):
    # Track the last map so repeated map data can be omitted from scene diffs.
    previous = serialize_message(solution.start_scene.world.octomap)
    for step in solution.sub_trajectory:
        scene = step.scene_diff
        encoded = serialize_message(scene.world.octomap)
        # Remove only repeated nonempty diff maps; full scene replacements keep their meaning.
        if scene.is_diff and scene.world.octomap.octomap.data and encoded == previous:
            scene.world.octomap = OctomapWithPose()
        # Update the remembered map when a stage supplies new data or replaces the full scene.
        elif scene.world.octomap.octomap.data or not scene.is_diff:
            previous = encoded
        # Keep the stage's attached objects but drop the joint positions it
        # captured at planning time. A stage that replaces the whole scene
        # carries a full robot state, and applying that mid-execution pins the
        # scene to where the robot was planned to be rather than where it is.
        # After an interrupted run the arm then starts a motion from a pose it
        # never reached, which is the jerk on the next attempt (2026-09-18).
        scene.robot_state.is_diff = True
        scene.robot_state.joint_state = JointState()
        scene.robot_state.multi_dof_joint_state = MultiDOFJointState()
    return solution

# Contact closure

def wait_for_hold(read_contact, clock, seconds=5., timeout=90.):
    started = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        left, right = read_contact()
        # No ceiling: the torque hold reads 0.41 / 0.49 on a can (effort_test, 2026-09-19).
        if not contact_state(left, right, .06, ceiling=math.inf):
            raise RuntimeError('Object slipped: bilateral contact lost after lift')
        # Measure the hold with the supplied robot clock and detect clock resets.
        now = clock()
        if started is None:
            started = now
        if now < started:
            raise RuntimeError('Simulation clock reset during hold verification')
        if now - started >= seconds:
            return
        time.sleep(.2)
    raise RuntimeError('Timed out waiting for sustained grasp contact')


def free_air_reading(command):
    """What the motor reads in free air at this command: the command less the mean
    spring deflection, which is how the joint reports (spring_rise's note)."""
    rows = json.loads((Path(__file__).parent / 'free_air_springs.json').read_text())[::-1]
    at = [r['command'] for r in rows]
    return command - np.interp(command, at, [(r['left'] + r['right']) / 2 for r in rows])


def spring_rise(q, command=None):
    springs = [q[f'hand_{s}_spring_proximal_joint'] for s in 'lr']
    if USE_SIM_TIME:
        return tuple(springs)
    # Under the torque grasp nothing is commanded; the reading is the command
    # minus the mean spring (to 0.002 rad over both sweeps), so add it back.
    if command is None:
        command = q['hand_motor_joint'] + sum(springs)/2
    rows = json.loads((Path(__file__).parent / 'free_air_springs.json').read_text())[::-1]
    at = [r['command'] for r in rows]
    return tuple(spring - np.interp(command, at, [r[name] for r in rows])
                 for spring, name in zip(springs, ('left', 'right')))


def contact_state(left, right, threshold, ceiling=.40):
    if not all(math.isfinite(x) for x in (left, right, threshold)):
        raise ValueError('Non-finite finger feedback')
    if not .06 <= threshold <= .18:
        raise ValueError('Contact threshold outside calibrated range')
    # Stop excessive finger deflection before deciding whether both fingers touched.
    # The ceiling sits at what the torque hold applies anyway (0.41 / 0.49 on a can,
    # judged held and undamaged, 2026-09-19). The simulation-era 0.20 left a 0.10-0.20
    # window that an object loading one finger first cannot hit (run 20260919_183338).
    if max(left, right) > ceiling:
        raise RuntimeError('Excessive or asymmetric finger contact; closure stopped')
    return min(left, right) > threshold


def command_for_gap(gap):
    profile = json.loads((GRIPPER_DIR / 'closing_profile.json').read_text())
    return float(np.interp(gap, [s['gap'][0] for s in profile], [s['motor'] for s in profile]))


def preclose(node, width):
    target = command_for_gap(width + PRECLOSE_MARGIN)
    travel = joint_positions(node)['hand_motor_joint'] - target
    # Not worth a move when the object nearly fills the open hand.
    if travel < .05:
        return None
    # hand_motor_joint's velocity limit is 1.0 rad/s; stay under it on long moves.
    seconds = max(.5, travel/.8)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ['hand_motor_joint']
    goal.trajectory.points = [JointTrajectoryPoint(
        positions=[target], time_from_start=Duration(sec=int(seconds), nanosec=int(seconds % 1*1e9)))]
    result = run_action(node, FollowJointTrajectory,
                         '/gripper_controller/follow_joint_trajectory', goal, 10.)
    if result.result.error_code != 0:
        raise RuntimeError('Finger pre-close failed: '+result.result.error_string)
    print(f'[CONTACT] pre-closed to motor {target:.3f} for a {width*1000:.1f} mm contact', flush=True)
    return target


def close(node, threshold=.06, motor=None, fine_below=0.):
    # Step from the last commanded angle when there is one. The joint reads lower
    # than its command by the mean spring deflection (0.06 rad after a pre-close),
    # so restarting from the reading turns the first step into a jump.
    if motor is None:
        motor = joint_positions(node)['hand_motor_joint']
    seen_contact = False
    samples = []
    diagnostics = os.environ.get('HSR_GRASP_DIAGNOSTICS')
    zero = None
    for _ in range(100):
        q = joint_positions(node)
        left, right = spring_rise(q, motor)
        # Zero both springs on the first reading, which is free air (the pre-close
        # stops 15 mm a side short): their zeros moved 0.05 over one afternoon of
        # torque holds. A first reading already at touch level is left alone.
        if zero is None:
            zero = (left, right) if max(abs(left), abs(right)) < .06 else (0., 0.)
        left, right = left - zero[0], right - zero[1]
        if diagnostics:
            samples.append(dict(time=time.time(), command=motor, motor=q['hand_motor_joint'], left=left, right=right,
                                **{f'raw_{s}': q[f'hand_{s}_spring_proximal_joint'] for s in 'lr'}))
            Path(diagnostics).mkdir(parents=True, exist_ok=True)
            (Path(diagnostics)/'closure_feedback.json').write_text(json.dumps(samples, indent=2))
        # Stop if contact disappears after the fingers have already touched the object.
        if seen_contact and min(left, right) < .02:
            raise RuntimeError('Object contact lost during closure; refusing further squeezing')
        seen_contact |= min(left, right) > .06
        if not USE_SIM_TIME and q['hand_motor_joint'] - free_air_reading(motor) > STALL_LAG:
            # The joint is held open by the object. Confirm it is not a slow controller.
            for _ in range(3):
                time.sleep(.2)
                again = joint_positions(node)['hand_motor_joint']
                if abs(again - q['hand_motor_joint']) > .01 or again - free_air_reading(motor) <= STALL_LAG:
                    break
            else:
                print(f"[CONTACT] motor stopped at {q['hand_motor_joint']:.3f} with the command at {motor:.3f}: "
                      f"on the object, springs {left:.3f}, {right:.3f}", flush=True)
                return {'left': left, 'right': right, 'motor': q['hand_motor_joint']}
        if contact_state(left, right, threshold):
            # Confirm contact persists across several feedback samples before holding position.
            for _ in range(3):
                time.sleep(.2)
                l, r = (rise - z for rise, z in zip(spring_rise(joint_positions(node), motor), zero))
                if not contact_state(l, r, .06):
                    raise RuntimeError('Bilateral contact was lost during position hold')
            print(f'[CONTACT] position hold, springs {left:.3f}, {right:.3f}', flush=True)
            return {'left': left, 'right': right, 'motor': q['hand_motor_joint']}
        # Close in smaller steps near the planned contact (fine_below) or once either
        # finger touches: one 0.02 step past contact adds 0.13-0.20 rad of spring on a
        # stiff can (runs 20260919_180259, _182003), more than the window below allows.
        motor -= .005 if max(left, right) > .06 or motor <= fine_below else .02
        if motor < .10:
            raise RuntimeError('Minimum closure reached without bilateral contact')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['hand_motor_joint']
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(motor)], time_from_start=Duration(nanosec=500000000))]
        result = run_action(node, FollowJointTrajectory,
                             '/gripper_controller/follow_joint_trajectory', goal, 10.)
        if result.result.error_code != 0:
            raise RuntimeError('Finger position step failed: '+result.result.error_string)
    raise RuntimeError('Closure exhausted without bilateral contact')


class DepthTooSparse(RuntimeError):
    """The mask is there and the depth under it is not: usually closer than ~0.5 m."""


class GeometryOccluded(RuntimeError):
    def __init__(self,message,centre):
        super().__init__(message)
        self.centre=centre


def aim_head(node,pan,tilt):
    goal=FollowJointTrajectory.Goal()
    # Send pan and tilt together so the head reaches one observation pose.
    goal.trajectory.joint_names=['head_pan_joint','head_tilt_joint']
    goal.trajectory.points=[JointTrajectoryPoint(positions=[float(pan),float(tilt)],
                                                time_from_start=Duration(sec=2))]
    result=run_action(node,FollowJointTrajectory,'/head_trajectory_controller/follow_joint_trajectory',goal,30.)
    if result.status!=GoalStatus.STATUS_SUCCEEDED or result.result.error_code!=0:
        raise RuntimeError('Head did not reach its observation pose')


def reframe_target(node,poses,centre):
    pivot=poses.transform('head_pan_link')[:3,3]
    base=poses.transform('base_footprint')
    yaw=math.atan2(base[1,0],base[0,0])
    offset=centre-pivot
    pan=math.atan2(offset[1],offset[0])-yaw
    # Wrap the pan angle before checking the head's joint limits.
    pan=math.atan2(math.sin(pan),math.cos(pan))
    # Same HSRC camera-height approximation used by navigation.look_at.
    tilt=math.atan2(offset[2]-.248,np.linalg.norm(offset[:2]))
    if not (-3.84<=pan<=1.75 and -1.57<=tilt<=.52):
        raise RuntimeError('Target is outside the head observation workspace')
    aim_head(node,pan,tilt)


def foreground_occlusion(depth, mask, support_mask=None):
    mask=np.asarray(mask,bool)
    outer=cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~mask
    # Compare against interior depth to avoid mixed pixels at the object boundary.
    interior=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    if not interior.any():
        interior=mask
    inside=depth[interior&np.isfinite(depth)&(depth>0)]
    if not inside.size:
        return 0.
    # Measure against the target's own near surface, not against the nearest
    # interior pixel of each neighbour. The head looks down, so the top face of
    # an upright object is nearer than the body just inside the silhouette, and
    # the per-neighbour comparison read the can's own lid as a blocker: 0.20 on
    # a table holding nothing but the can (2026-09-17). A real blocker stands in
    # front of the whole object, which this still catches.
    near=float(np.percentile(inside,2))
    valid=outer&np.isfinite(depth)&(depth>0)
    # Exclude the supporting surface from the foreground-blocker test.
    if support_mask is not None:
        valid &= ~support_mask
    if not valid.any():
        return 0.
    return float(np.mean(depth[valid] < near-.03))


def support_pixels(depth,k,transform,target):
    return support_plane(depth,k,transform,target)[0]


def support_plane(depth,k,transform,target):
    v,u=np.indices(depth.shape)
    x=(u-k[0,2])*depth/k[0,0];y=(v-k[1,2])*depth/k[1,1]
    xyz=np.stack([x,y,depth],axis=-1)@transform[:3,:3].T+transform[:3,3]
    # Subsample the depth cloud to reduce the support-surface search cost.
    sample=xyz[::4,::4].reshape(-1,3)
    centre=np.median(target,axis=0);bottom=np.quantile(target[:,2],.01)
    valid=np.isfinite(sample).all(axis=1)&(np.linalg.norm(sample[:,:2]-centre[:2],axis=1)<.35)
    valid &= (sample[:,2]>=bottom-.25)&(sample[:,2]<=bottom+.015)
    sample=sample[valid]
    if len(sample)<100:
        return np.zeros(depth.shape,bool), None
    # Group nearby points by height to find a broad support surface.
    bins=np.floor(sample[:,2]/.01).astype(int)
    values=np.unique(bins)
    # A broad horizontal edge is enough when the camera cannot see the full tabletop.
    for value in values[::-1]:
        plane=sample[bins==value]
        if len(plane)<100 or np.max(np.ptp(plane[:,:2],axis=0))<.30:
            continue
        height=float(np.median(plane[:,2]))
        return np.abs(xyz[:,:,2]-height)<.015, height
    return np.zeros(depth.shape,bool), None


_VIEW_COUNT = [0]


def save_view(prompt, rgb, depth, k, odom_from_camera, mask, score, **extra):
    index = _VIEW_COUNT[0]
    _VIEW_COUNT[0] += 1
    root = os.environ.get('HSR_GRASP_DIAGNOSTICS')
    if not root:
        return index
    folder = Path(root)/'views'
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder/f'{index:02d}.jpg'), rgb)
    np.savez_compressed(folder/f'{index:02d}.npz', depth=np.asarray(depth, np.float32),
                        mask=np.asarray(mask, bool), k=np.asarray(k, float),
                        odom_from_camera=np.asarray(odom_from_camera, float),
                        score=float(score), prompt=str(prompt), **extra)
    return index


def observe_lifted(prompt):
    """observe(), or None when the lifted object is inside the camera's blind range or
    only a fragment of it is measured there (LIFT_CHECK_RANGE, LIFT_CHECK_COVERAGE)."""
    try:
        view = observe(prompt)
    except (DepthTooSparse, pointcloud.InvalidTargetDepth) as error:
        print(f'[LIFT] no camera check: {error}', flush=True)
        return None
    if view['median_range'] < LIFT_CHECK_RANGE or view['coverage'] < LIFT_CHECK_COVERAGE:
        print(f"[LIFT] no camera check: at {view['median_range']:.2f} m with depth on "
              f"{view['coverage']:.0%} of the mask, only a fragment of the {prompt} is measured", flush=True)
        return None
    return view


def observe(prompt):
    rgb, depth, k, odom_from_camera = grab_rgbd()
    try:
        mask, score = sam3_client.detect(rgb, prompt)
    except sam3_client.ObjectNotFound:
        # Keep the frame anyway: an empty detection usually means the head is
        # not looking at the target, and the image shows where it is looking.
        index=save_view(prompt,rgb,depth,k,odom_from_camera,np.zeros(depth.shape,bool),0.)
        print(f'[VIEW {index}] {prompt}: SAM3 found nothing; frame saved', flush=True)
        raise
    v,u=np.nonzero(mask)
    # Report depth coverage and range on the mask before any rejection, so a
    # view lost to missing depth still says what the camera saw.
    valid=np.isfinite(depth)&(depth>0)&mask
    coverage=float(valid.sum()/max(int(mask.sum()),1))
    median_range=float(np.median(depth[valid])) if valid.any() else float('nan')
    index=save_view(prompt,rgb,depth,k,odom_from_camera,mask,score,
                    coverage=coverage,median_range=median_range)
    print(f'[VIEW {index}] {prompt}: SAM3 {score:.2f}, mask rows {v.min()}-{v.max()} '
          f'cols {u.min()}-{u.max()}, depth on {coverage:.0%} of mask, '
          f'median range {median_range:.2f} m', flush=True)
    if coverage < MIN_DEPTH_COVERAGE:
        # What depth there is under such a mask is the background seen past the
        # object, not the object: it is closer than the camera can measure.
        raise DepthTooSparse(f"depth on only {coverage:.0%} of the {prompt}'s mask; "
                             "it is closer than the depth camera can measure")
    points = pointcloud.deproject(depth, k, pointcloud.object_depth_mask(depth, mask))
    if len(points) < 100:
        # The head camera returns little below about 0.5 m (MIN_DEPTH_COVERAGE).
        near = (f"; at {median_range:.2f} m it is closer than the depth camera can measure, "
                f"back the robot off" if median_range < .5 else "")
        raise DepthTooSparse(f"only {len(points)} depth points on the {prompt}{near}")
    _, image_point, area = feature(depth, k, mask)
    clipped=bool(u.min()<3 or v.min()<3 or u.max()>=mask.shape[1]-3 or v.max()>=mask.shape[0]-3)
    # Put the target points and image feature into odom coordinates.
    world_points=pointcloud.transform_points(odom_from_camera, points)
    support,support_height=support_plane(depth,k,odom_from_camera,world_points)
    return dict(points=world_points, support=support_height, view=index,
                camera=odom_from_camera[:3, 3], score=score, area=area,
                clipped=clipped, occlusion=foreground_occlusion(depth,mask,support),
                coverage=coverage, median_range=median_range,
                feature=odom_from_camera[:3, :3] @ image_point + odom_from_camera[:3, 3])


def merge_target_views(views):
    merged = views[0]['points'].copy()
    for view in views[1:]:
        points = view['points']
        # Require overlap in both directions before merging another target view.
        forward = float(np.mean(cKDTree(merged).query(points)[0] < .012))
        reverse = float(np.mean(cKDTree(points).query(merged)[0] < .012))
        if min(forward, reverse) < .15:
            raise RuntimeError('Head views do not overlap on the same static target '
                               f'({forward:.0%} of the new view and {reverse:.0%} of the '
                               f'merged views lie within 12 mm of the other)')
        merged = np.vstack([merged, points])
    # Equal spatial weighting prevents a close/dense view dominating the fit.
    _, indices = np.unique(np.floor(merged/.002).astype(np.int64), axis=0, return_index=True)
    result = dict(views[0])
    result['points'] = merged[indices]
    # A clipped view may miss the tabletop; keep the highest support any view found.
    heights = [view.get('support') for view in views if view.get('support') is not None]
    result['support'] = max(heights) if heights else None
    return result


def observe_geometry(node, prompt, poses=None):
    try:
        initial = observe(prompt)
    except sam3_client.ObjectNotFound:
        # Look down in steps before giving up: what the operator put in front of
        # the robot is below a level head more often than anywhere else.
        joints = joint_positions(node)
        for angle in SEARCH_TILTS:
            if angle >= joints['head_tilt_joint'] - .05:
                continue
            print(f"[GEOMETRY] nothing at tilt {joints['head_tilt_joint']:.2f}; looking down to {angle:.2f}", flush=True)
            aim_head(node, joints['head_pan_joint'], angle)
            try:
                initial = observe(prompt)
                break
            except sam3_client.ObjectNotFound:
                continue
        else:
            raise
    if initial['clipped']:
        # A level/reset head may see only the top of a tabletop object. Centre
        # that detection before choosing scan angles, then acquire fresh depth.
        centre = (initial['points'].min(0)+initial['points'].max(0))/2
        print('[GEOMETRY] initial target clipped; re-aiming head before scan', flush=True)
        reframe_target(node, poses if poses is not None else RobotTransforms(node), centre)
        initial = observe(prompt)
    joints = joint_positions(node)
    pan, tilt = joints['head_pan_joint'], joints['head_tilt_joint']
    def describe(view, label):
        points = view['points']
        print(f"[GEOMETRY] {label}: view {view.get('view', '?')}, {len(points)} points, "
              f"z {points[:, 2].min():.3f}-{points[:, 2].max():.3f} m, "
              f"extent {np.round(np.ptp(points, axis=0)*1000, 1)} mm, "
              f"clipped {view.get('clipped')}, support {view.get('support')}", flush=True)
    describe(initial, f'initial view at pan {pan:.2f} tilt {tilt:.2f}')
    views = [initial]
    def aim(value):
        aim_head(node,pan,value)
    try:
        # Look slightly above and below the original head angle for more target geometry.
        for offset in [-.20, .20]:
            angle = float(np.clip(tilt+offset, -1.57, .52))
            if abs(angle-tilt) < .01:
                continue
            aim(angle)
            try:
                view = observe(prompt)
            except (sam3_client.ObjectNotFound, pointcloud.InvalidTargetDepth) as error:
                print(f'[GEOMETRY] skipping head tilt {angle:.2f}: {error}', flush=True)
                continue
            # Reject a wrong instance immediately rather than expanding the hull.
            try:
                merge_target_views(views+[view])
            except RuntimeError as error:
                describe(view, f'rejected view at tilt {angle:.2f}')
                print(f'[GEOMETRY] skipping head tilt {angle:.2f}: {error}', flush=True)
                continue
            describe(view, f'accepted view at tilt {angle:.2f}')
            views.append(view)
    finally:
        aim(tilt)
    if len(views)<2:
        raise RuntimeError('Could not obtain complementary target depth views')
    result = merge_target_views(views)
    # Require at least one unclipped view and no strongly foreground-blocked view.
    if all(view['clipped'] for view in views) or any(view['occlusion']>.08 for view in views):
        raise GeometryOccluded('Head scan remains clipped or contains foreground-occluded geometry',
                               (result['points'].min(0)+result['points'].max(0))/2)
    extent = np.ptp(result['points'],axis=0)
    print(f'[GEOMETRY] {len(views)} head views, extent {np.round(extent*1000,1)} mm',flush=True)
    return result


def perceive(prompt, node=None, poses=None, cloud=None):
    if cloud is not None:
        # Active perception's fused TSDF surface, already in odom, instead of a head scan.
        saved = np.load(cloud)
        observation = dict(points=saved['points'], camera=saved['camera'],
                           score=float(saved['score']), support=None)
        print(f'[GEOMETRY] using the fused cloud {cloud}', flush=True)
    else:
        observation = observe_geometry(node,prompt,poses) if node is not None else observe(prompt)
    points = observation['points']
    # Limit and centre the object cloud before asking the grasp model.
    sample = points[np.random.choice(len(points), min(len(points), MAX_CLOUD_POINTS), replace=False)]
    # Remember the cloud centre so predicted grasps can be returned to odom.
    centre = sample.mean(axis=0)
    poses, scores = graspgenx_client.generate(sample - centre, GRIPPER)
    poses[:, :3, 3] += centre
    best = np.argsort(scores)[::-1]
    # Refine the ranked grasps using measured shape and calibrated finger contact.
    envelope, grasps, widths, kind = contact_candidates(points, poses[best], observation['camera'])
    observation['geometry_kind'] = kind
    print(f"{prompt}: SAM3 {observation['score']:.2f}, {len(points)} points, "
          f"{kind}, {len(grasps)} aperture-compatible grasps", flush=True)
    return envelope, grasps[:CANDIDATES], widths[:CANDIDATES], observation


def wait(node, future, what, seconds=30.0):
    rclpy.spin_until_future_complete(node, future, timeout_sec=seconds)
    if not future.done():
        raise RuntimeError(f"{what} timed out")
    return future.result()


def call(node, kind, name, request):
    client = node.create_client(kind, name)
    if not client.wait_for_service(timeout_sec=10.0):
        raise RuntimeError(f"{name} unavailable; is move_group running?")
    return wait(node, client.call_async(request), name)


# One subscription per node, kept for the life of the run. wait_for_message
# creates a fresh subscriber for every call and throws it away; on the real
# robot's link a new subscriber's first message is regularly lost, so each call
# was a 5 s gamble, and close() makes up to a hundred of them (2026-09-18).
_JOINT_STATES = {}


def joint_positions(node, timeout=5.0):
    entry = _JOINT_STATES.get(id(node))
    if entry is None:
        latest = {}
        entry = _JOINT_STATES[id(node)] = (
            node.create_subscription(JointState, JOINT_STATES,
                                     lambda message: latest.__setitem__('message', message), 10),
            latest,
        )
    _, latest = entry
    # Require a sample newer than this call: closure reads contact from it.
    latest.pop('message', None)
    deadline = time.monotonic() + timeout
    while 'message' not in latest:
        if time.monotonic() > deadline:
            raise RuntimeError(f"no {JOINT_STATES}")
        rclpy.spin_once(node, timeout_sec=0.1)
    message = latest['message']
    # Pair each reported joint name with its measured position.
    return dict(zip(message.name, message.position))


def fingertip_gap(node):
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa: F841, fills buffer while spun
    # Measure the relative transform between the left and right fingertips.
    tips = ("hand_l_finger_tip_frame", "hand_r_finger_tip_frame", Time())
    deadline = time.monotonic() + 5.0
    while not buffer.can_transform(*tips):
        if time.monotonic() > deadline:
            raise RuntimeError("no fingertip TF")
        rclpy.spin_once(node, timeout_sec=0.1)
    gap = buffer.lookup_transform(*tips).transform.translation
    # Use the translation length as the physical gap between fingertips.
    return float(np.linalg.norm([gap.x, gap.y, gap.z]))


def target_box(points, support=None):
    lower, upper = points.min(axis=0).astype(float), points.max(axis=0).astype(float)
    if support is not None and np.isfinite(support) and support < lower[2]:
        gap = lower[2]-support
        if gap > .01:
            print(f'[GEOMETRY] lowest observed target point is {gap*1000:.0f} mm above the '
                  f'support plane; extending the collision box down to it', flush=True)
        lower[2] = support
    return lower, upper


def model_target(node, points, support=None):
    # Represent the target by its axis-aligned box in the planning scene.
    lower, upper = target_box(points, support)
    box = CollisionObject(id=TARGET, operation=CollisionObject.ADD)
    box.header.frame_id = BASE_FRAME
    box.pose.position.x, box.pose.position.y, box.pose.position.z = (
        (lower + upper) / 2
    ).tolist()
    box.primitives = [
        SolidPrimitive(type=SolidPrimitive.BOX, dimensions=(upper - lower).tolist())
    ]
    box.pose.orientation.w = 1.0
    primitive_pose = Pose()
    primitive_pose.orientation.w = 1.0
    box.primitive_poses = [primitive_pose]
    scene = PlanningScene(is_diff=True)
    scene.robot_state.is_diff = True
    # An empty id detaches whatever a previous pick left in the hand.
    scene.robot_state.attached_collision_objects = [
        AttachedCollisionObject(
            object=CollisionObject(operation=CollisionObject.REMOVE)
        )
    ]
    # Replace the old target with the newly measured collision box.
    scene.world.collision_objects = [box]
    # Clear the old map before publishing the scene to stay within DDS size limits.
    # Rebuild it below with the target box excluded from depth updates.
    call(node, Empty, "/clear_octomap", Empty.Request())
    if not call(
        node,
        ApplyPlanningScene,
        "/apply_planning_scene",
        ApplyPlanningScene.Request(scene=scene),
    ).success:
        raise RuntimeError("move_group rejected the target box")
    depth_relay(node, True)
    # Retry early depth maps that incorrectly overlap the robot.
    for attempt in range(1, 4):
        call(node, Empty, "/clear_octomap", Empty.Request())
        # Wait for actual map data, then check that the current robot state is clear.
        deadline = time.monotonic() + 30.0
        while True:
            snapshot = call(node, GetPlanningScene, '/get_planning_scene',
                            GetPlanningScene.Request(components=PlanningSceneComponents(
                                components=PlanningSceneComponents.ROBOT_STATE |
                                PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
                                PlanningSceneComponents.OCTOMAP))).scene
            if snapshot.world.octomap.octomap.data:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('No fresh depth map after clearing OctoMap')
            time.sleep(.25)
        valid = call(node, GetStateValidity, '/check_state_validity',
                     GetStateValidity.Request(robot_state=snapshot.robot_state,
                                              group_name=GROUP))
        if valid.valid:
            break
        bodies = sorted({f'{c.contact_body_1} <-> {c.contact_body_2}' for c in valid.contacts})
        print(f"[MAP] rebuilt map overlaps the robot (attempt {attempt}/3): "
              f"{', '.join(bodies) or 'no contact details'}", flush=True)
        time.sleep(2.)
    else:
        raise RuntimeError('Robot overlaps the rebuilt collision scene; refusing motion')
    # Freeze the map so new depth voxels do not interrupt the planned motion.
    depth_relay(node, False)


def depth_relay(node, enabled):
    value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=enabled)
    request = SetParameters.Request(parameters=[Parameter(name="enabled", value=value)])
    call(node, SetParameters, "/octomap_depth_camera/set_parameters", request)


# Action clients are kept for the life of the run, for the same reason as the
# joint-state subscription above: it is a client's first goal that the link
# loses, so a fresh client per call made every closure step pay the 5 s resend.
_ACTION_CLIENTS = {}


def action_client(node, kind, name):
    client = _ACTION_CLIENTS.get((id(node), name))
    if client is None:
        client = _ACTION_CLIENTS[(id(node), name)] = ActionClient(node, kind, name)
    return client


def run_action(node, kind, name, goal, seconds):
    client = action_client(node, kind, name)
    # Keep waiting rather than giving up on one attempt. Discovery of the
    # robot's endpoints arrives in part and late over this link: a fresh client
    # has taken 6.8 s to see a server that another client saw instantly, and a
    # recovery run failed outright with "gripper_controller unavailable" while
    # the robot was up and reachable (2026-09-18).
    for attempt in range(1, 4):
        if client.wait_for_server(timeout_sec=10.0):
            break
        print(f"[ACTION] {name}: no server after 10 s "
              f"(attempt {attempt}/3), still waiting", flush=True)
    else:
        raise RuntimeError(f"{name} unavailable after 30 s")
    # Retry unanswered goals: the link drops a request now and then, and it is
    # a client's first goal that goes missing most often.
    for attempt in range(1, 4):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if future.done():
            handle = future.result()
            break
        print(f"[ACTION] {name}: no goal response in 5 s (attempt {attempt}/3), resending", flush=True)
    else:
        raise RuntimeError(f"{name} goal timed out")
    # Proceed to result waiting only after the action server accepts a goal.
    if not handle.accepted:
        raise RuntimeError(f"{name} rejected the goal")
    try:
        # Ask again every 5 s: the result request is lost like any other (run
        # 20260919_175927 failed on it), and repeating it commands nothing.
        deadline = time.monotonic() + seconds
        while True:
            future = handle.get_result_async()
            rclpy.spin_until_future_complete(
                node, future, timeout_sec=max(0., min(5., deadline - time.monotonic())))
            if future.done():
                return future.result()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{name} timed out")
    except BaseException:
        # On errors or Ctrl+C, stop the motion rather than leaving it running.
        cancellation = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(node, cancellation, timeout_sec=5.0)
        raise


def open_hand(node):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ['hand_motor_joint']
    goal.trajectory.points = [JointTrajectoryPoint(
        positions=[OPEN_HAND], time_from_start=Duration(sec=1, nanosec=500000000))]
    result = run_action(node, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory', goal, 15.)
    if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
        raise RuntimeError('the hand did not open')
    # Wait briefly for the measured hand opening and passive springs to settle.
    deadline = time.monotonic() + 5.
    while time.monotonic() < deadline:
        q = joint_positions(node)
        if abs(q['hand_motor_joint'] - OPEN_HAND) < .05 and max(
                q['hand_l_spring_proximal_joint'], q['hand_r_spring_proximal_joint']) < .3:
            return
    print('[GRASP] finger springs still deflected after opening; planning anyway', flush=True)


def close_hand(node):
    # Read the optional closing effort and require it to stay within the calibrated range.
    effort = float(os.environ.get('HSR_GRASP_EFFORT_NM', str(CLOSE_EFFORT)))
    if not np.isfinite(effort) or not -.3 <= effort <= -.012:
        raise ValueError('HSR_GRASP_EFFORT_NM must be between -0.3 and -0.012 Nm')
    print(f'[GRASP] closing effort {effort:.3f} Nm', flush=True)
    goal = GripperApplyEffort.Goal(effort=effort)
    closed = run_action(node, GripperApplyEffort, "/gripper_controller/grasp", goal, 30.0)
    if closed.status != GoalStatus.STATUS_SUCCEEDED:
        raise RuntimeError("the hand did not close")


def mtc_node():
    # Give the C++ MTC node the same robot model, planner settings, and clock mode.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as params:
        yaml.safe_dump(
            {"/**": {"ros__parameters": {**planning_model.moveit_params(), "use_sim_time": USE_SIM_TIME}}}, params
        )
        params.flush()
        options = rclcpp.NodeOptions()
        options.arguments = ["--ros-args", "--params-file", params.name]
        options.automatically_declare_parameters_from_overrides = True
        return rclcpp.Node("mtc_pick", options)


def stamped(frame, matrix=np.eye(4)):
    pose = PoseStamped()
    pose.header.frame_id = frame
    # Copy matrix translation and rotation into a frame-labelled ROS pose.
    position, orientation = pose.pose.position, pose.pose.orientation
    position.x, position.y, position.z = matrix[:3, 3].tolist()
    orientation.x, orientation.y, orientation.z, orientation.w = (
        Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()
    )
    return pose


def move_hand(name, planner, direction, frame, min_distance, max_distance):
    stage = stages.MoveRelative(name, planner)
    stage.group = GROUP
    stage.ik_frame = stamped(HAND)
    stage.min_distance = min_distance
    stage.max_distance = max_distance
    # Label the motion direction with the frame in which it is expressed.
    vector = Vector3Stamped()
    vector.header.frame_id = frame
    vector.vector.x, vector.vector.y, vector.vector.z = direction
    stage.setDirection(vector)
    return stage


def seeded_ik(node, matrix, base, frame=BASE_FRAME):
    scene = call(node, GetPlanningScene, "/get_planning_scene", GetPlanningScene.Request(
        components=PlanningSceneComponents(components=PlanningSceneComponents.ROBOT_STATE))).scene
    request = GetPositionIK.Request()
    request.ik_request.group_name = GROUP
    request.ik_request.robot_state = scene.robot_state
    request.ik_request.ik_link_name = HAND
    request.ik_request.pose_stamped = stamped(frame, matrix)
    # Preserve the current-state seed; check collisions separately below.
    # The service's collision retry would otherwise switch to random seeds.
    request.ik_request.avoid_collisions = False
    request.ik_request.timeout = Duration(sec=1)
    response = call(node, GetPositionIK, "/compute_ik", request)
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        print('[IK] no seeded whole-body solution', flush=True)
        return None
    solution = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
    # IK answers a heading within +-pi while odom_t keeps counting turns: parked at
    # +3.31, the same heading came back as -2.97 and every pregrasp was refused as a
    # -6.28 rad turn (2026-09-21). Put it on the base's turn, or the goal is that spin.
    solution['odom_t'] = base['odom_t'] + (solution['odom_t']-base['odom_t']+np.pi) % (2*np.pi) - np.pi
    # Say how far the base must go: the base controller, not the arm, is what
    # fails these reaches (README §10), so this number decides the outcome.
    travel = np.hypot(solution['odom_x']-base['odom_x'], solution['odom_y']-base['odom_y'])
    turn = solution['odom_t']-base['odom_t']
    # Keep the solved base position and heading within the allowed parking slack.
    if any(abs(solution[joint] - base[joint]) > slack for joint, slack in BASE_SLACK.items()):
        print(f'[IK] solution needs {travel*1000:.0f} mm of base travel and a {turn:+.2f} rad turn; '
              'outside the base window', flush=True)
        return None
    # Reject the IK solution if it collides with the current scene.
    valid = call(node, GetStateValidity, "/check_state_validity",
                 GetStateValidity.Request(robot_state=response.solution, group_name=GROUP))
    if not valid.valid:
        print(f'[IK] solution with {travel*1000:.0f} mm of base travel is in collision', flush=True)
        return None
    print(f'[IK] seeded solution moves the base {travel*1000:.0f} mm and turns {turn:+.2f} rad', flush=True)
    goal = RobotState(is_diff=True)
    goal.joint_state.name = list(WHOLE_BODY_JOINTS)
    goal.joint_state.position = [float(solution[joint]) for joint in WHOLE_BODY_JOINTS]
    return goal


class Pick:
    def __init__(self, node, ik=None):
        self.node = node
        self.ik = ik
        self.ompl = None
        # Use short, slow Cartesian steps for the final approach and lift.
        self.cartesian = core.CartesianPath()
        self.cartesian.step_size = 0.005  # enough steps for MoveIt's joint-jump check
        for planner in (self.cartesian,):
            planner.max_velocity_scaling_factor = 0.15
            planner.max_acceleration_scaling_factor = 0.15

    def task(self, name):
        task = core.Task()
        task.name = name
        task.loadRobotModel(self.node)
        # Make a fresh planner for the robot model loaded by this task.
        self.ompl = core.PipelinePlanner(self.node, 'move_group')
        self.ompl.max_velocity_scaling_factor = .15
        self.ompl.max_acceleration_scaling_factor = .15
        task.add(stages.CurrentState("current state"))
        return task

    def reach(self, grasps, base):
        task = self.task("reach")
        # Restore target collision checks before opening or moving to pregrasp.
        protect = stages.ModifyPlanningScene("check target during pregrasp")
        protect.allowCollisions(TARGET, FINGERS, False)
        # The last lift's exemption outlives its task in move_group, as the one above does.
        protect.allowCollisions("<octomap>", FINGERS, False)
        protect.allowCollisions("<octomap>", [TARGET], False)
        # Allow head-link overlap with the camera-origin voxel seen in real depth maps.
        protect.allowCollisions("<octomap>", HEAD_LINKS, True)
        task.add(protect)
        if abs(base.get("hand_motor_joint", 0.0) - OPEN_HAND) > 0.005:
            open_hand = stages.MoveTo("open hand before pregrasp", self.ompl)
            open_hand.group = "gripper"
            open_hand.setGoal({"hand_motor_joint": OPEN_HAND})
            task.add(open_hand)
        # Keep the base near where Nav2 parked it.
        parked = Constraints(
            joint_constraints=[
                JointConstraint(
                    joint_name=joint,
                    position=base[joint],
                    tolerance_above=slack,
                    tolerance_below=slack,
                    weight=1.0,
                )
                for joint, slack in BASE_SLACK.items()
            ]
        )
        # Reserve arm-lift travel for the approach and verification lift.
        parked.joint_constraints.append(JointConstraint(
            joint_name='arm_lift_joint',position=GRASP_LIFT_MAX/2,
            tolerance_above=GRASP_LIFT_MAX/2,tolerance_below=GRASP_LIFT_MAX/2,
            weight=1.))
        # Move backward along palm Z to get each pregrasp pose.
        back_off = np.eye(4)
        back_off[2, 3] = -APPROACH
        # Best-scored first; stop at the first grasp that works.
        # Try each grasp as a complete pregrasp-and-approach sequence.
        candidates = core.Fallbacks("grasp candidates")
        for index, grasp in enumerate(grasps):
            candidate = core.SerialContainer(f"grasp {index}")
            # Unfold the arm first when the grasp approaches from directly above.
            if grasp[2, 2] < -.9:
                clearance = stages.MoveTo('unfold arm for overhead reach', self.ompl)
                clearance.group = 'arm'
                clearance.setGoal(dict(zip(
                    ['arm_lift_joint','arm_flex_joint','arm_roll_joint',
                     'wrist_flex_joint','wrist_roll_joint'], [.35,-.5,-1.57,-1.07,0.])))
                clearance.timeout = 10.
                candidate.insert(clearance)
            pregrasp = pregrasp_routes(
                self.ompl, grasp @ back_off, parked,
                [base['odom_x'], base['odom_y']], GROUP, HAND, BASE_FRAME,
                ik=(lambda matrix: self.ik(matrix, base)) if self.ik else None)
            candidate.insert(pregrasp)
            # Allow finger contact only after reaching pregrasp, then approach straight in.
            allow = stages.ModifyPlanningScene("allow finger contact")
            allow.allowCollisions(TARGET, FINGERS, True)
            candidate.insert(allow)
            candidate.insert(move_hand("complete approach", self.cartesian,
                                       (0., 0., 1.), HAND, APPROACH, APPROACH))
            candidates.insert(candidate)
        task.add(candidates)
        return task

    def approach(self):
        task = self.task("approach")
        task.add(move_hand("complete approach", self.cartesian, (0., 0., 1.),
                           HAND, APPROACH, APPROACH))
        return task

    def correct_endpoint(self, grasp):
        task = self.task('correct grasp endpoint')
        move = stages.MoveTo('bounded endpoint correction', self.cartesian)
        move.group = GROUP
        move.ik_frame = stamped(HAND)
        # Use the selected calibrated grasp as the small endpoint-correction goal.
        move.setGoal(stamped(BASE_FRAME, grasp))
        task.add(move)
        return task

    def clear_view(self):
        task=self.task('clear arm from target observation')
        home=stages.MoveTo('observation home',self.ompl)
        home.group='arm'
        # Fold the arm into the observation pose so the head can see the target again.
        home.setGoal(dict(zip(['arm_lift_joint','arm_flex_joint','arm_roll_joint',
                              'wrist_flex_joint','wrist_roll_joint'],[0.,0.,-1.57,-1.57,0.])))
        home.timeout=10.
        task.add(home)
        return task

    def lift(self):
        task = self.task("lift")
        attach = stages.ModifyPlanningScene("attach target")
        # The map was frozen before the hand closed: voxels at the can's edge now sit
        # inside the closed fingers and would reject the start state.
        attach.allowCollisions("<octomap>", FINGERS, True)
        # It also holds the can's own voxels, which count against the can only once it
        # is attached: "<octomap> colliding with target" with the can held. The lift is
        # TEST_LIFT straight up, so the can meets nothing else the map knows of.
        attach.allowCollisions("<octomap>", [TARGET], True)
        # Attach the target to the palm in the planning scene before moving upward.
        attach.attachObject(TARGET, HAND)
        task.add(attach)
        task.add(
            move_hand("lift", self.cartesian, (0.0, 0.0, 1.0), BASE_FRAME, TEST_LIFT, TEST_LIFT)
        )
        return task


START_TOLERANCE = .05  # rad; refuse to execute a plan starting elsewhere


def check_start(node, solution, tolerance=START_TOLERANCE):
    measured = joint_positions(node)
    for step in solution.sub_trajectory:
        trajectory = step.trajectory.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            continue
        errors = [(abs(measured[name]-position), name)
                  for name, position in zip(trajectory.joint_names,
                                            trajectory.points[0].positions)
                  if name in measured]
        if errors and max(errors)[0] > tolerance:
            error, name = max(errors)
            raise RuntimeError(
                f"the plan starts {error:.3f} rad away from the robot at {name} "
                f"(limit {tolerance:.3f}); stow the arm with "
                f"realrobot/live/recover_home.py and run again")
        return


def execute(node, task):
    if not task.plan(max_solutions=1):
        raise RuntimeError(f"MTC found no {task.name} plan")
    solution = task.solutions[0].toMsg()
    compact(solution)
    check_start(node, solution)
    # Check payload size while preserving the collision scene for later stages.
    size = len(serialize_message(solution))
    print(f'[MTC] execution payload {size/1024/1024:.2f} MiB', flush=True)
    if size > 32*1024*1024:
        raise RuntimeError('Task collision snapshot exceeds the 32 MiB execution budget')
    goal = ExecuteTaskSolution.Goal(solution=solution)
    done = run_action(node, ExecuteTaskSolution, "/execute_task_solution", goal, 300.0)
    if done.result.error_code.val != 1:
        raise RuntimeError(
            f"executing {task.name} failed: MoveIt error {done.result.error_code.val}"
        )


def clear_observation_pose(node, planner):
    # Freeze depth updates during the clearance motion, then restart them afterward.
    depth_relay(node, False)
    try:
        execute(node, planner.clear_view())
    finally:
        depth_relay(node, True)


def hold_nav2():
    """Pause Nav2 when it is up and able to drive, as the mission does before its pick.

    The reach moves the base through MoveIt, and only a paused Nav2 is certain not to
    move it as well: a leftover goal, a recovery or an RViz click would. Run from the
    mission it is paused already, and a second PAUSE is refused, so only an active one
    is touched. With the grasp stack alone there is no Nav2, and nothing to do."""
    from lifecycle_msgs.msg import State
    from core.navigation import nav2_client
    navigator = nav2_client.Navigator(use_sim_time=USE_SIM_TIME)
    try:
        if not navigator._is_running("controller_server"):
            return
        if navigator.navigation_state("controller_server") == State.PRIMARY_STATE_ACTIVE:
            navigator.pause_navigation()
    finally:
        navigator.destroy_node()


def pick(prompt, mode='auto', cloud=None):
    if mode not in ('auto', 'pickup', 'grasp'):
        raise ValueError('Unknown grasp mode')
    rclpy.init()
    rclcpp.init()
    node = rclpy.create_node("pick", parameter_overrides=[NodeParameter("use_sim_time", value=USE_SIM_TIME)])
    try:
        # Before anything moves. Refused, it raises, and the pick ends here.
        hold_nav2()
        poses = RobotTransforms(node)
        planner = Pick(mtc_node(), ik=lambda matrix, base: seeded_ik(node, matrix, base))
        # Restore lift travel before observing if the torso is already too high.
        if joint_positions(node)['arm_lift_joint'] > GRASP_LIFT_MAX:
            previous=observe(prompt)['points']
            print('[GRASP] restoring arm-lift margin before replanning',flush=True)
            clear_observation_pose(node,planner)
            reframe_target(node,poses,(previous.min(0)+previous.max(0))/2)
        try:
            points, grasps, widths, observation = perceive(prompt,node,poses,cloud)
        # Try one camera-clearance motion and a fresh observation when geometry is hidden.
        except GeometryOccluded as blocked:
            print('[GEOMETRY] foreground blocks target; planning one return to observation home',flush=True)
            clear_observation_pose(node,planner)
            reframe_target(node,poses,blocked.centre)
            points, grasps, widths, observation = perceive(prompt,node,poses)
        # For showing afterwards: the measured cloud with the first few grasps on it.
        from core.grasping import picture
        picture.save(prompt, observation['points'], grasps, widths, observation.get('geometry_kind'))
        # Auto mode test-lifts cylinders; other shapes stop at a contact hold.
        is_can = observation.get('geometry_kind') == 'cylinder'
        lift_requested = mode == 'pickup' or (mode == 'auto' and is_can)
        # Allow a bounded insertion adjustment along the approach; cylinders default to 1 cm.
        insertion = float(os.environ.get('HSR_GRASP_INSERTION_M', '.01' if is_can else '0'))
        if not np.isfinite(insertion) or not 0 <= insertion <= .01:
            raise ValueError('HSR_GRASP_INSERTION_M must be between 0 and 0.01 m')
        if insertion:
            grasps = grasps.copy()
            grasps[:, :3, 3] += np.minimum(insertion, .2*widths)[:, None]*grasps[:, :3, 2]
            print(f'[GRASP] trial forward insertion up to {insertion*1000:.1f} mm', flush=True)
        diagnostics = os.environ.get('HSR_GRASP_DIAGNOSTICS')
        if diagnostics:
            Path(diagnostics).mkdir(parents=True, exist_ok=True)
            support_height = observation.get('support')
            np.savez(Path(diagnostics)/'planned_grasps.npz', points=points, grasps=grasps, widths=widths,
                     support=np.nan if support_height is None else support_height)
        if joint_positions(node)['hand_motor_joint'] < OPEN_HAND - .05:
            open_hand(node)
        # State where the robot is parked relative to the target. The base, not
        # the arm, is what fails these reaches, and how far it must travel is
        # set by this distance (README section 10).
        parked = joint_positions(node)
        print(f"[GRASP] parked at x {parked['odom_x']:+.3f} y {parked['odom_y']:+.3f} "
              f"t {parked['odom_t']:+.3f}, target centre "
              f"{np.hypot(points[:, 0].mean()-parked['odom_x'], points[:, 1].mean()-parked['odom_y'])*1000:.0f} mm away",
              flush=True)
        # Build the collision scene and execute the first complete reachable grasp plan.
        model_target(node, points, observation.get('support'))
        execute(node, planner.reach(grasps, joint_positions(node)))
        reached = poses.transform()
        # Match the measured palm pose to the candidate the planner reached.
        distances = np.linalg.norm(grasps[:, :3, 3]-reached[:3, 3], axis=1)
        angles = Rotation.from_matrix(
            grasps[:, :3, :3].transpose(0, 2, 1) @ reached[:3, :3]).magnitude()
        index = int(np.argmin(distances + .1*angles))
        # Correct a small position error only when orientation is already close.
        if .004 < distances[index] <= .01 and angles[index] <= .04:
            print(f'[GRASP] correcting {distances[index]*1000:.1f} mm endpoint error', flush=True)
            execute(node, planner.correct_endpoint(grasps[index]))
            reached = poses.transform()
            distances[index] = np.linalg.norm(grasps[index,:3,3]-reached[:3,3])
            angles[index] = Rotation.from_matrix(grasps[index,:3,:3].T@reached[:3,:3]).magnitude()
        # Refuse to close unless the measured palm position and orientation match the grasp.
        if distances[index] > .004 or angles[index] > .04:
            raise RuntimeError(f'Palm did not reach the calibrated grasp pose '
                               f'({distances[index]*1000:.1f} mm, {angles[index]:.3f} rad); refusing closure')
        print(f"[GRASP] reached candidate {index}, contact width {widths[index]*1000:.1f} mm", flush=True)
        if diagnostics:
            # The head still looks at the target and the hand is in view: where the
            # pads sit against the object, for the next time the fingers miss it.
            try:
                cv2.imwrite(str(Path(diagnostics)/'reached.jpg'), grab_rgbd(timeout=10.)[0])
            except (RuntimeError, ValueError) as error:
                print(f'[GRASP] no frame of the reached pose: {error}', flush=True)
        # Close until both finger springs show contact, then rule out an empty grasp.
        width = float(widths[index])
        # The firmer 0.14 is for simulation, where this position hold is the whole grip.
        # On the robot the torque grasp below holds, and on a stiff can each small step
        # adds about 0.08 of rise (0.039, 0.121, run 20260919_182411): 0.14 would
        # overshoot the 0.20 ceiling.
        close(node, threshold=.14 if lift_requested and USE_SIM_TIME else .10, motor=preclose(node, width),
              fine_below=command_for_gap(width + FINE_MARGIN))
        # close() proves both pads touch but barely squeezes: a can stopped at rises
        # near 0.2 was loose in the hand (run 20260919_180259). Hold with the robot's
        # own torque grasp. Real robot only; simulation was validated on the position hold.
        if not USE_SIM_TIME:
            close_hand(node)
        gap = fingertip_gap(node)
        if gap < EMPTY_GAP:
            raise RuntimeError("the hand closed on nothing")
        # Return a contact-only hold when this mode does not request a test lift.
        if not lift_requested:
            print('[GRASPED] bilateral contact; holding without lift', flush=True)
            return 'grasped'
        # Test-lift the object and use fresh depth to verify that it moved with the hand.
        before_lift = poses.transform()
        execute(node, planner.lift())
        after_lift = poses.transform()
        def read_contact():
            q = joint_positions(node)
            rclpy.spin_once(node, timeout_sec=0.)  # update the /clock subscription
            return spring_rise(q)
        measured = observe_lifted(prompt)
        if measured is None:
            # The lift ran and the camera cannot see it. Both fingers still loaded is
            # what 'grasped' already means; 'picked' stays reserved for a seen rise.
            wait_for_hold(read_contact, lambda: node.get_clock().now().nanoseconds / 1e9)
            print('[GRASPED] lifted with sustained bilateral contact; rise not seen', flush=True)
            return 'grasped'
        if diagnostics:
            np.savez(Path(diagnostics)/'lift_observation.npz', before=observation['points'],
                     after=measured['points'], expected=after_lift[:3,3]-before_lift[:3,3])
        evidence = verify_object_lift(observation['points'], measured['points'],
                                      after_lift[:3,3]-before_lift[:3,3])
        print(f"[LIFT] {prompt}: object rose {evidence['rise_m']*1000:.1f} mm; checking hold", flush=True)
        # Require sustained bilateral contact before taking the final verification image.
        wait_for_hold(read_contact, lambda: node.get_clock().now().nanoseconds / 1e9)
        # After the contact dwell, check again that the object remains elevated.
        held = observe(prompt)
        if diagnostics:
            np.savez(Path(diagnostics)/'hold_observation.npz', before=observation['points'],
                     after=held['points'], expected=after_lift[:3,3]-before_lift[:3,3])
        evidence = verify_object_lift(observation['points'], held['points'],
                                      after_lift[:3,3]-before_lift[:3,3])
        if not contact_state(*read_contact(), .06, ceiling=math.inf):
            raise RuntimeError('Object slipped during final camera verification')
        print(f"[VERIFIED] {prompt}: sustained contact and {evidence['rise_m']*1000:.1f} mm lift", flush=True)
        return 'picked'
    except (RuntimeError, ValueError) as error:
        print(f"pick failed: {error}")
        return False
    finally:
        try:
            depth_relay(node, False)  # idle until the next pick rebuilds the octomap
        except RuntimeError as error:
            print(f"could not stop the depth relay: {error}")
        node.destroy_node()
        rclpy.shutdown()
        rclcpp.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pick up an object with MoveIt Task Constructor (MTC).')
    parser.add_argument('target')
    parser.add_argument('--mode', choices=['auto', 'pickup', 'grasp'], default='auto')
    parser.add_argument('--cloud', help="active perception's target_cloud.npz, used instead of a head scan")
    args = parser.parse_args()
    outcome = pick(args.target, args.mode, args.cloud)
    # Distinguish contact-only success from a verified pickup for the mission.
    raise SystemExit(3 if outcome == 'grasped' else 0 if outcome else 1)