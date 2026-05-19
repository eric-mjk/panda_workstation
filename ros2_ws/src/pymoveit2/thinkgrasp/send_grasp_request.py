#!/usr/bin/env python3

"""Snapshot Isaac ROS image topics and request a ThinkGrasp pose on key press."""

from __future__ import annotations

import json
from pathlib import Path
import select
import sys
import termios
from threading import Thread
import time
import tty
from typing import Optional

import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg

try:
    from client import GraspPoseClient, ThinkGraspClientError
except ImportError:
    from thinkgrasp.client import GraspPoseClient, ThinkGraspClientError


DEFAULT_OUTPUT_DIR = "/tmp/thinkgrasp"
DEFAULT_TARGET_WIDTH = 640
DEFAULT_TARGET_HEIGHT = 480
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def read_key(timeout_sec: float = 0.05) -> Optional[str]:
    if select.select([sys.stdin], [], [], timeout_sec)[0]:
        return sys.stdin.read(1).lower()
    return None


def image_msg_to_array(msg: ImageMsg) -> np.ndarray:
    dtype, channels = encoding_to_dtype_and_channels(msg.encoding)
    array = np.frombuffer(msg.data, dtype=dtype)

    if msg.is_bigendian != (array.dtype.byteorder == ">"):
        array = array.byteswap().newbyteorder()

    if channels == 1:
        array = array.reshape((msg.height, msg.step // array.dtype.itemsize))
        return array[:, : msg.width].copy()

    row_items = msg.step // array.dtype.itemsize
    array = array.reshape((msg.height, row_items))
    array = array[:, : msg.width * channels]
    return array.reshape((msg.height, msg.width, channels)).copy()


def encoding_to_dtype_and_channels(encoding: str):
    normalized = encoding.lower()
    if normalized in ("rgb8", "bgr8", "rgba8", "bgra8"):
        return np.uint8, 4 if "a" in normalized else 3
    if normalized in ("mono8", "8uc1"):
        return np.uint8, 1
    if normalized in ("mono16", "16uc1"):
        return np.uint16, 1
    if normalized == "32fc1":
        return np.float32, 1

    raise ValueError(f"Unsupported image encoding '{encoding}'")


def save_rgb_image(msg: ImageMsg, path: Path, target_size) -> None:
    array = image_msg_to_array(msg)
    encoding = msg.encoding.lower()

    if encoding == "bgr8":
        array = array[:, :, ::-1]
    elif encoding == "bgra8":
        array = array[:, :, [2, 1, 0, 3]]
    elif encoding in ("mono8", "8uc1"):
        resize_image(Image.fromarray(array), target_size, RESAMPLE_BILINEAR).save(path)
        return
    elif encoding not in ("rgb8", "rgba8"):
        raise ValueError(f"RGB topic must be rgb8/bgr8/rgba8/bgra8/mono8, got {msg.encoding}")

    resize_image(Image.fromarray(array), target_size, RESAMPLE_BILINEAR).save(path)


def save_depth_image(msg: ImageMsg, path: Path, target_size) -> None:
    array = image_msg_to_array(msg)
    encoding = msg.encoding.lower()

    if encoding == "32fc1":
        finite_depth = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(finite_depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
        resize_image(Image.fromarray(depth_mm), target_size, RESAMPLE_NEAREST).save(path)
        return

    if encoding in ("mono16", "16uc1"):
        resize_image(
            Image.fromarray(array.astype(np.uint16)),
            target_size,
            RESAMPLE_NEAREST,
        ).save(path)
        return

    if encoding in ("mono8", "8uc1"):
        resize_image(Image.fromarray(array), target_size, RESAMPLE_NEAREST).save(path)
        return

    raise ValueError(f"Depth topic must be 32FC1/16UC1/mono16/mono8, got {msg.encoding}")


def resize_image(image: Image.Image, target_size, resample) -> Image.Image:
    if image.size == target_size:
        return image
    return image.resize(target_size, resample=resample)


class ThinkGraspSnapshotClient(Node):
    def __init__(self):
        super().__init__("thinkgrasp_snapshot_client")

        self.declare_parameter("instruction", "")
        self.declare_parameter("server_url", "http://127.0.0.1:5000")
        self.declare_parameter("timeout_sec", 300.0)
        self.declare_parameter("rgb_topic", "/isaac_rgb")
        self.declare_parameter("depth_topic", "/isaac_depth")
        self.declare_parameter("output_dir", DEFAULT_OUTPUT_DIR)
        self.declare_parameter("rgb_path", "")
        self.declare_parameter("depth_path", "")
        self.declare_parameter("text_path", "")
        self.declare_parameter("trigger_key", "g")
        self.declare_parameter("target_width", DEFAULT_TARGET_WIDTH)
        self.declare_parameter("target_height", DEFAULT_TARGET_HEIGHT)

        self._latest_rgb: Optional[ImageMsg] = None
        self._latest_depth: Optional[ImageMsg] = None
        self._request_in_progress = False

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self.create_subscription(ImageMsg, rgb_topic, self._rgb_callback, 10)
        self.create_subscription(ImageMsg, depth_topic, self._depth_callback, 10)

        self._timer = self.create_timer(0.05, self._keyboard_tick)
        trigger_key = self.get_parameter("trigger_key").value
        self.get_logger().info(
            f"Listening to {rgb_topic} and {depth_topic}. Press '{trigger_key}' "
            "to request a grasp, or 'q' to quit."
        )

    def _rgb_callback(self, msg: ImageMsg) -> None:
        self._latest_rgb = msg

    def _depth_callback(self, msg: ImageMsg) -> None:
        self._latest_depth = msg

    def _keyboard_tick(self) -> None:
        key = read_key()
        if key is None:
            return

        if key == "q":
            self.get_logger().info("Quit requested.")
            rclpy.shutdown()
            return

        trigger_key = self.get_parameter("trigger_key").value.lower()
        if key == trigger_key:
            self._start_request()

    def _start_request(self) -> None:
        if self._request_in_progress:
            self.get_logger().warn("A ThinkGrasp request is already running.")
            return
        if self._latest_rgb is None:
            self.get_logger().warn("No RGB image received yet.")
            return
        if self._latest_depth is None:
            self.get_logger().warn("No depth image received yet.")
            return

        instruction = self.get_parameter("instruction").value
        if not instruction:
            self.get_logger().error(
                'No instruction set. Run with --ros-args -p instruction:="pick up the red cup"'
            )
            return

        self._request_in_progress = True
        Thread(target=self._save_and_request, args=(instruction,), daemon=True).start()

    def _save_and_request(self, instruction: str) -> None:
        try:
            rgb_path, depth_path, text_path = self._snapshot_paths()
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)

            target_size = (
                int(self.get_parameter("target_width").value),
                int(self.get_parameter("target_height").value),
            )
            self.get_logger().info(
                f"Saving RGB {self._latest_rgb.width}x{self._latest_rgb.height} "
                f"and depth {self._latest_depth.width}x{self._latest_depth.height} "
                f"as {target_size[0]}x{target_size[1]} for ThinkGrasp."
            )
            save_rgb_image(self._latest_rgb, rgb_path, target_size)
            save_depth_image(self._latest_depth, depth_path, target_size)
            text_path.write_text(instruction + "\n", encoding="utf-8")

            self.get_logger().info(
                f"Saved snapshot: rgb={rgb_path}, depth={depth_path}, text={text_path}"
            )

            client = GraspPoseClient(
                self.get_parameter("server_url").value,
                timeout_sec=float(self.get_parameter("timeout_sec").value),
            )
            result = client.get_grasp_pose(rgb_path, depth_path, text_path)
        except (OSError, ValueError, ThinkGraspClientError) as exc:
            self.get_logger().error(f"ThinkGrasp request failed: {exc}")
        else:
            self.get_logger().info("ThinkGrasp result:")
            print(json.dumps(result.raw, indent=2), flush=True)
        finally:
            self._request_in_progress = False

    def _snapshot_paths(self):
        output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_rgb = output_dir / f"rgb_{timestamp}.png"
        default_depth = output_dir / f"depth_{timestamp}.png"
        default_text = output_dir / f"instruction_{timestamp}.txt"

        rgb_path = self._path_parameter_or_default("rgb_path", default_rgb)
        depth_path = self._path_parameter_or_default("depth_path", default_depth)
        text_path = self._path_parameter_or_default("text_path", default_text)
        return rgb_path, depth_path, text_path

    def _path_parameter_or_default(self, parameter_name: str, default_path: Path) -> Path:
        configured = self.get_parameter(parameter_name).value
        if configured:
            return Path(configured).expanduser()
        return default_path


def main() -> int:
    rclpy.init()
    old_terminal_settings = termios.tcgetattr(sys.stdin)
    node = ThinkGraspSnapshotClient()

    try:
        tty.setcbreak(sys.stdin.fileno())
        rclpy.spin(node)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
