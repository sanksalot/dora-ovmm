"""Capture synchronized RGB-D and transform it at the depth image timestamp."""

import os
import time
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from core.utils.transforms import matrix_from_transform
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

# Select the real robot clock and image topics with HSR_REAL_ROBOT=1.
# HSR_IMAGE_PREFIX overrides /remote; camera calibration arrives directly over DDS.
# launch/realrobot/grasp_real.launch.py sets both.
REAL_ROBOT = os.environ.get("HSR_REAL_ROBOT", "") == "1"
USE_SIM_TIME = not REAL_ROBOT
IMAGE_PREFIX = os.environ.get("HSR_IMAGE_PREFIX", "/remote" if REAL_ROBOT else "")
RGB_TOPIC = IMAGE_PREFIX + "/head_rgbd_sensor/rgb/image_rect_color"
DEPTH_TOPIC = IMAGE_PREFIX + "/head_rgbd_sensor/depth_registered/image_rect_raw"
CAMERA_INFO_TOPIC = "/head_rgbd_sensor/rgb/camera_info"
# Reliability belongs to the topics, not to the clock: vision-transport
# republishes best-effort, Gazebo's bridge publishes reliable. Deciding it from
# use_sim_time instead let a real-robot run started without --real ask /remote
# for RELIABLE, which it never offers; DDS matched nothing and the capture
# blocked to its timeout behind one QoS warning.
IMAGE_QOS = (qos_profile_sensor_data if REAL_ROBOT
             else QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE))
BASE_FRAME = "odom"
# Allow extra time to discover real-robot image publishers and transforms.
CAPTURE_TIMEOUT = 45. if REAL_ROBOT else 15.


class RobotTransforms:
    """Read-only pose feedback; creates no robot command publishers."""
    def __init__(self,node):
        self.node=node
        self.buffer=Buffer()
        self.listener=TransformListener(self.buffer,node)

    def transform(self,link='hand_palm_link'):
        deadline=time.monotonic()+3.
        # Process incoming transforms until a fresh pose is available or time runs out.
        while time.monotonic()<deadline:
            rclpy.spin_once(self.node,timeout_sec=.02)
            if not self.buffer.can_transform(BASE_FRAME,link,Time()):
                continue
            pose=self.buffer.lookup_transform(BASE_FRAME,link,Time())
            age=(self.node.get_clock().now()-Time.from_msg(pose.header.stamp)).nanoseconds/1e9
            # Accept only a recent pose so feedback does not use an old robot position.
            if age<=.5:
                return matrix_from_transform(pose.transform)
        raise RuntimeError(f'No fresh {BASE_FRAME}-to-{link} transform')


def grab_hand_rgb(timeout=8., use_sim_time=USE_SIM_TIME):
    """Return rectified hand-camera BGR, intrinsics, and odom camera/palm poses at image time."""
    import cv2
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = Node('grasp_hand_camera', parameter_overrides=[Parameter('use_sim_time', value=use_sim_time)])
    try:
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        # Store the most recent hand image and calibration while waiting for a usable pair.
        frames = {}
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE) if use_sim_time else qos_profile_sensor_data
        node.create_subscription(Image, '/hand_camera/image_raw', lambda m: frames.update(image=m), qos)
        node.create_subscription(CameraInfo, '/hand_camera/camera_info', lambda m: frames.update(info=m), qos_profile_sensor_data)
        start = node.get_clock().now().nanoseconds
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if not {'image','info'} <= frames.keys():
                continue
            msg, info = frames['image'], frames['info']
            stamp = Time.from_msg(msg.header.stamp)
            # Wait for a new image and reject images more than half a second old.
            if stamp.nanoseconds <= start:
                continue
            if (node.get_clock().now()-stamp).nanoseconds > 500_000_000:
                continue
            if msg.header.frame_id != info.header.frame_id or (msg.width,msg.height)!=(info.width,info.height):
                raise ValueError('Hand image and calibration do not match')
            if info.distortion_model not in ('plumb_bob','rational_polynomial'):
                raise ValueError('Unsupported hand camera distortion model')
            # Look up the camera and palm poses at the same image timestamp.
            links = [msg.header.frame_id, 'hand_palm_link']
            if not all(buffer.can_transform(BASE_FRAME, link, stamp) for link in links):
                continue
            k = np.array(info.k).reshape(3,3)
            if not np.isfinite(k).all() or min(k[0,0],k[1,1]) <= 0:
                raise ValueError('Invalid hand camera calibration')
            rgb = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Remove lens distortion when the calibration includes distortion coefficients.
            if np.any(info.d):
                rgb = cv2.undistort(rgb,k,np.array(info.d),None,k)
            poses = [matrix_from_transform(buffer.lookup_transform(BASE_FRAME,link,stamp).transform) for link in links]
            return rgb,k,*poses
        raise RuntimeError('No fresh calibrated hand camera image with timestamped TF')
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()


def grab_rgbd(target_frame=BASE_FRAME, timeout=CAPTURE_TIMEOUT, use_sim_time=USE_SIM_TIME):
    """Return BGR, depth in metres, intrinsics, and T_target_camera."""
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = Node(
        "grasp_pipeline_camera",
        parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
    )
    try:
        bridge = CvBridge()
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        frames = {}
        capture_start = node.get_clock().now().nanoseconds
        rgb_sub = message_filters.Subscriber(
            node, Image, RGB_TOPIC, qos_profile=IMAGE_QOS
        )
        depth_sub = message_filters.Subscriber(
            node, Image, DEPTH_TOPIC, qos_profile=IMAGE_QOS
        )
        # Pair RGB and depth images captured within 50 milliseconds.
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 10, 0.05
        )

        # Store synchronized RGB and depth together so their timestamps stay paired.
        def receive_images(rgb, depth):
            frames["rgb"] = rgb
            frames["depth"] = depth

        def receive_info(message):
            frames["info"] = message

        sync.registerCallback(receive_images)
        info_sub = node.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, receive_info, qos_profile_sensor_data
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            # Wait until both images and the camera calibration have arrived.
            if not {"rgb", "depth", "info"} <= frames.keys():
                continue
            rgb, depth, info = (frames["rgb"], frames["depth"], frames["info"])
            # Reject queued images from before this capture began.
            rgb_time = Time.from_msg(rgb.header.stamp).nanoseconds
            depth_time = Time.from_msg(depth.header.stamp).nanoseconds
            if min(rgb_time, depth_time) <= capture_start:
                continue
            # Use the camera pose when the image was captured, not its pose now.
            stamp = Time.from_msg(depth.header.stamp)
            frame = depth.header.frame_id
            if not buffer.can_transform(target_frame, frame, stamp):
                continue
            if rgb.header.frame_id != frame or info.header.frame_id != frame:
                raise ValueError("Expected depth registered into the RGB optical frame")
            image = bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
            depth_values = bridge.imgmsg_to_cv2(
                depth, desired_encoding="passthrough"
            ).astype(np.float32)
            if depth.encoding not in ("16UC1", "32FC1"):
                raise ValueError(f"Unsupported depth encoding: {depth.encoding}")
            if depth_values.shape != image.shape[:2]:
                raise ValueError("RGB and registered depth dimensions differ")
            # Integer depth is millimetres; floating-point depth is metres.
            if depth.encoding == "16UC1":
                depth_meters = depth_values / 1000
            else:
                depth_meters = depth_values
            # Return the calibration and camera pose alongside the converted images.
            intrinsics = np.array(info.k).reshape(3, 3)
            transform = buffer.lookup_transform(target_frame, frame, stamp).transform
            return (image, depth_meters, intrinsics, matrix_from_transform(transform))
        raise RuntimeError(
            f"No synchronized RGB-D with TF to {target_frame} within {timeout}s"
        )
    finally:
        node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()
