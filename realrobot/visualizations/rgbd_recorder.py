"""Record the head camera's RGB and depth streams, each to its own mp4, for showing.

    python3 realrobot/visualizations/rgbd_recorder.py                 # until Ctrl-C
    python3 realrobot/visualizations/rgbd_recorder.py --seconds 30    # then stops itself
    HSR_IMAGE_PREFIX= python3 realrobot/visualizations/rgbd_recorder.py   # a bag or the sim

Writes outputs/realrobot/visualizations/<time>_video/rgb.mp4 and depth.mp4. Depth is
coloured the way rgbd_frame_capture.py colours its depth.png (turbo over 0.4-4 m,
black where the sensor returned nothing), so stills and video match.

Each file runs at a fixed --fps: every 1/fps the newest frame of that stream is
written, again if nothing new arrived, so playback keeps real time whatever rate
the stream actually runs at. Nothing here commands the robot: subscribing is what
makes RX stream, so the first frames take a moment. Run it in the container with
ROS_DOMAIN_ID and CYCLONEDDS_URI set for the robot, and RX up.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("HSR_REAL_ROBOT", "1")
# realrobot/visualizations/<this file>, so three levels up is the repository root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rgbd_frame_capture import OUTPUTS, colour_depth  # noqa: E402

FPS = 15.0


class Stream:
    """One topic to one mp4: keeps the newest frame, and tick() writes it."""

    def __init__(self, path, fps, convert):
        self.path, self.fps, self.convert = Path(path), fps, convert
        self.latest = None
        self.writer = None
        self.received = self.written = 0

    def push(self, message):
        self.latest = self.convert(message)
        self.received += 1

    def tick(self):
        if self.latest is None:
            return
        if self.writer is None:
            height, width = self.latest.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open {self.path} for writing")
            print(f"[REC] {self.path.name}: {width}x{height} at {self.fps:g} fps", flush=True)
        self.writer.write(self.latest)
        self.written += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()


def depth_metres(message, bridge):
    values = bridge.imgmsg_to_cv2(message, desired_encoding="passthrough").astype(np.float32)
    if message.encoding == "16UC1":
        return values / 1000  # millimetres
    if message.encoding == "32FC1":
        return values
    raise ValueError(f"Unsupported depth encoding: {message.encoding}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=FPS, help=f"frames per second written (default {FPS:g})")
    parser.add_argument("--seconds", type=float, help="stop after this long (default: until Ctrl-C)")
    parser.add_argument("--output", type=Path, help=f"folder to write into (default {OUTPUTS}/<time>_video)")
    args = parser.parse_args()

    import rclpy
    from cv_bridge import CvBridge
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from core.perception.camera_ros2 import DEPTH_TOPIC, RGB_TOPIC

    folder = args.output or OUTPUTS / f"{time.strftime('%Y%m%d-%H%M%S')}_video"
    folder.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge()
    streams = {
        RGB_TOPIC: Stream(folder / "rgb.mp4", args.fps, lambda m: bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")),
        DEPTH_TOPIC: Stream(folder / "depth.mp4", args.fps, lambda m: colour_depth(depth_metres(m, bridge))),
    }
    rclpy.init()
    node = rclpy.create_node("rgbd_recorder", parameter_overrides=[Parameter("use_sim_time", value=False)])
    for topic, stream in streams.items():
        node.create_subscription(Image, topic, stream.push, qos_profile_sensor_data)
        print(f"[REC] waiting on {topic}", flush=True)
    node.create_timer(1 / args.fps, lambda: [stream.tick() for stream in streams.values()])
    started = time.monotonic()
    deadline = started + args.seconds if args.seconds else None
    try:
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for stream in streams.values():
            stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    elapsed = time.monotonic() - started
    for stream in streams.values():
        rate = stream.received / elapsed if elapsed else 0.0
        print(f"[REC] {stream.path}: {stream.written} frames written ({stream.written / args.fps:.1f} s) "
              f"from {stream.received} received at {rate:.1f} Hz", flush=True)
    if not any(stream.written for stream in streams.values()):
        sys.exit("nothing recorded: is RX up, and are ROS_DOMAIN_ID and CYCLONEDDS_URI set for the robot?")


if __name__ == "__main__":
    main()
