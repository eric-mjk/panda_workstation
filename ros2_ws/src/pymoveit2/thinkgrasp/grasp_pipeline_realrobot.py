#!/usr/bin/env python3

"""Snapshot RealSense topics, call ThinkGrasp, and execute grasp on the real Panda."""

from __future__ import annotations

from math import pi
from pathlib import Path
import select
import sys
import termios
from threading import Event, Thread
import time
import tty
from typing import Optional

from control_msgs.action import GripperCommand
import numpy as np
from PIL import Image
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image as ImageMsg, JointState
import tf2_geometry_msgs
import tf2_ros

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot

try:
    from client_realrobot import GraspPoseClient, ThinkGraspClientError
except ImportError:
    from thinkgrasp.client_realrobot import GraspPoseClient, ThinkGraspClientError


DEFAULT_OUTPUT_DIR = "/tmp/thinkgrasp"
DEFAULT_TARGET_WIDTH = 640
DEFAULT_TARGET_HEIGHT = 480
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST

PANDA_READY_JOINTS = [0.0, -pi / 4.0, 0.0, -3.0 * pi / 4.0, 0.0, pi / 2.0, pi / 4.0]

BASE_LINK = "panda_link0"
END_EFFECTOR_LINK = "panda_hand_tcp"
MOVE_GROUP = "panda_arm"
GRASP_POSE_TO_MOVE_POSE = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def transform_from_pose(pose: PoseStamped, child_frame: str) -> TransformStamped:
    transform = TransformStamped()
    transform.header = pose.header
    transform.child_frame_id = child_frame
    transform.transform.translation.x = pose.pose.position.x
    transform.transform.translation.y = pose.pose.position.y
    transform.transform.translation.z = pose.pose.position.z
    transform.transform.rotation = pose.pose.orientation
    return transform


def pose_to_matrix(pose: PoseStamped) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(
        [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ]
    ).as_matrix()
    matrix[:3, 3] = [
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z,
    ]
    return matrix


def pose_from_matrix(matrix: np.ndarray, header) -> PoseStamped:
    quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()

    pose = PoseStamped()
    pose.header = header
    pose.pose.position.x = float(matrix[0, 3])
    pose.pose.position.y = float(matrix[1, 3])
    pose.pose.position.z = float(matrix[2, 3])
    pose.pose.orientation.x = float(quat_xyzw[0])
    pose.pose.orientation.y = float(quat_xyzw[1])
    pose.pose.orientation.z = float(quat_xyzw[2])
    pose.pose.orientation.w = float(quat_xyzw[3])
    return pose


def apply_pose_post_transform(pose: PoseStamped, post_transform: np.ndarray) -> PoseStamped:
    return pose_from_matrix(pose_to_matrix(pose) @ post_transform, pose.header)


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
            Image.fromarray(array.astype(np.uint16)), target_size, RESAMPLE_NEAREST
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


class GraspPipelineNode(Node):
    def __init__(self):
        super().__init__("thinkgrasp_grasp_pipeline")

        self.declare_parameter("instruction", "")
        self.declare_parameter("server_url", "http://127.0.0.1:5000")
        self.declare_parameter("timeout_sec", 300.0)
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("output_dir", DEFAULT_OUTPUT_DIR)
        self.declare_parameter("rgb_path", "")
        self.declare_parameter("depth_path", "")
        self.declare_parameter("text_path", "")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("tf_lookup_timeout_sec", 3.0)
        self.declare_parameter("target_width", DEFAULT_TARGET_WIDTH)
        self.declare_parameter("target_height", DEFAULT_TARGET_HEIGHT)
        self.declare_parameter("trigger_key", "g")
        self.declare_parameter("confirm_key", "e")
        self.declare_parameter("approach_dist_m", 0.10)
        self.declare_parameter("lift_height_m", 0.15)
        self.declare_parameter("max_velocity", 0.3)
        self.declare_parameter("max_acceleration", 0.3)
        self.declare_parameter("gripper_action_name", "panda_gripper/gripper_action")
        self.declare_parameter("gripper_joint_name", robot.gripper_joint_names()[0])
        self.declare_parameter("gripper_open_position", robot.OPEN_GRIPPER_JOINT_POSITIONS[0])
        self.declare_parameter("gripper_closed_position", robot.CLOSED_GRIPPER_JOINT_POSITIONS[0])
        self.declare_parameter("gripper_close_action", "toggle")
        self.declare_parameter("gripper_position_tolerance", 0.003)
        self.declare_parameter("gripper_motion_timeout_sec", 3.0)
        self.declare_parameter("gripper_assume_close_success_on_timeout", True)
        self.declare_parameter("gripper_close_fire_and_forget", True)
        self.declare_parameter("gripper_close_settle_sec", 1.5)
        self.declare_parameter("gripper_max_effort", 20.0)
        self.declare_parameter("gripper_server_timeout_sec", 10.0)
        self.declare_parameter("gripper_result_timeout_sec", 10.0)

        self._callback_group = ReentrantCallbackGroup()

        self._moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR_LINK,
            group_name=MOVE_GROUP,
            callback_group=self._callback_group,
        )
        self._moveit2.max_velocity = float(self.get_parameter("max_velocity").value)
        self._moveit2.max_acceleration = float(self.get_parameter("max_acceleration").value)

        self._gripper_action_client = ActionClient(
            self,
            GripperCommand,
            self.get_parameter("gripper_action_name").value,
            callback_group=self._callback_group,
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self.create_subscription(ImageMsg, rgb_topic, self._rgb_callback, 10)
        self.create_subscription(ImageMsg, depth_topic, self._depth_callback, 10)
        self.create_subscription(JointState, "joint_states", self._joint_state_callback, 10)

        self._latest_rgb: Optional[ImageMsg] = None
        self._latest_depth: Optional[ImageMsg] = None
        self._gripper_joint_position: Optional[float] = None
        self._request_in_progress = False

        self._confirm_event = Event()
        self._cancel_event = Event()
        self.waiting_confirmation = False

        self.get_logger().info(f"Subscribed to {rgb_topic} and {depth_topic}.")

    def _rgb_callback(self, msg: ImageMsg) -> None:
        self._latest_rgb = msg

    def _depth_callback(self, msg: ImageMsg) -> None:
        self._latest_depth = msg

    def _joint_state_callback(self, msg: JointState) -> None:
        gripper_joint_name = self.get_parameter("gripper_joint_name").value
        if gripper_joint_name not in msg.name:
            return
        self._gripper_joint_position = msg.position[msg.name.index(gripper_joint_name)]

    def confirm_grasp(self) -> None:
        if self.waiting_confirmation:
            self._confirm_event.set()

    def cancel_grasp(self) -> None:
        if self.waiting_confirmation:
            self._cancel_event.set()

    def trigger_grasp(self) -> None:
        if self._request_in_progress:
            self.get_logger().warn("A grasp is already in progress.")
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

        rgb_msg = self._latest_rgb
        depth_msg = self._latest_depth

        self._request_in_progress = True
        Thread(
            target=self._pipeline_worker,
            args=(instruction, rgb_msg, depth_msg),
            daemon=True,
        ).start()

    def _pipeline_worker(
        self,
        instruction: str,
        rgb_msg: ImageMsg,
        depth_msg: ImageMsg,
    ) -> None:
        try:
            rgb_path, depth_path, text_path = self._snapshot_paths()
            for p in (rgb_path, depth_path, text_path):
                p.parent.mkdir(parents=True, exist_ok=True)

            target_size = (
                int(self.get_parameter("target_width").value),
                int(self.get_parameter("target_height").value),
            )
            self.get_logger().info(
                f"Saving RGB {rgb_msg.width}x{rgb_msg.height} "
                f"and depth {depth_msg.width}x{depth_msg.height} "
                f"as {target_size[0]}x{target_size[1]} for ThinkGrasp."
            )
            save_rgb_image(rgb_msg, rgb_path, target_size)
            save_depth_image(depth_msg, depth_path, target_size)
            text_path.write_text(instruction + "\n", encoding="utf-8")
            self.get_logger().info(
                f"Saved: rgb={rgb_path}, depth={depth_path}, text={text_path}"
            )

            client = GraspPoseClient(
                self.get_parameter("server_url").value,
                timeout_sec=float(self.get_parameter("timeout_sec").value),
            )
            result = client.get_grasp_pose(rgb_path, depth_path, text_path)
            self.get_logger().info(f"ThinkGrasp xyz: {result.xyz}")

            self._execute_grasp(result)

        except (OSError, ValueError, ThinkGraspClientError) as exc:
            self.get_logger().error(f"Pipeline failed: {exc}")
        finally:
            self._request_in_progress = False
            self.waiting_confirmation = False

    def _execute_grasp(self, result) -> None:
        camera_frame = self.get_parameter("camera_frame").value
        base_to_camera = self._lookup_base_to_camera(camera_frame)
        if base_to_camera is None:
            return

        camera_grasp_pose = self._thinkgrasp_result_to_pose(result, camera_frame)
        base_grasp_pose = tf2_geometry_msgs.do_transform_pose_stamped(
            camera_grasp_pose, base_to_camera
        )
        base_grasp_pose = apply_pose_post_transform(
            base_grasp_pose, GRASP_POSE_TO_MOVE_POSE
        )

        approach_dist = float(self.get_parameter("approach_dist_m").value)

        # grasp approach direction (EE z-axis)
        quat_grasp = [
            base_grasp_pose.pose.orientation.x,
            base_grasp_pose.pose.orientation.y,
            base_grasp_pose.pose.orientation.z,
            base_grasp_pose.pose.orientation.w,
        ]
        approach_vec = R.from_quat(quat_grasp).apply([0.0, 0.0, 1.0])

        base_pregrasp_pose = PoseStamped()
        base_pregrasp_pose.header = base_grasp_pose.header
        base_pregrasp_pose.pose.orientation = base_grasp_pose.pose.orientation
        base_pregrasp_pose.pose.position.x = (
            base_grasp_pose.pose.position.x - approach_dist * approach_vec[0]
        )
        base_pregrasp_pose.pose.position.y = (
            base_grasp_pose.pose.position.y - approach_dist * approach_vec[1]
        )
        base_pregrasp_pose.pose.position.z = (
            base_grasp_pose.pose.position.z - approach_dist * approach_vec[2]
        )

        pos_grasp = [
            base_grasp_pose.pose.position.x,
            base_grasp_pose.pose.position.y,
            base_grasp_pose.pose.position.z,
        ]
        pos_pregrasp = [
            base_pregrasp_pose.pose.position.x,
            base_pregrasp_pose.pose.position.y,
            base_pregrasp_pose.pose.position.z,
        ]
        quat_pregrasp = quat_grasp

        stamp = self.get_clock().now().to_msg()
        base_grasp_pose.header.stamp = stamp
        base_pregrasp_pose.header.stamp = stamp
        self._static_tf_broadcaster.sendTransform([
            transform_from_pose(base_pregrasp_pose, "thinkgrasp_pregrasp"),
            transform_from_pose(base_grasp_pose, "thinkgrasp_grasp"),
        ])

        confirm_key = self.get_parameter("confirm_key").value.lower()
        camera_xyz = base_to_camera.transform.translation
        self.get_logger().info(
            "Published predicted grasp TFs. "
            f"Camera xyz={[round(camera_xyz.x, 4), round(camera_xyz.y, 4), round(camera_xyz.z, 4)]}. "
            f"Press '{confirm_key}' to execute or any other key to cancel."
        )

        self._confirm_event.clear()
        self._cancel_event.clear()
        self.waiting_confirmation = True

        while not self._confirm_event.is_set() and not self._cancel_event.is_set():
            time.sleep(0.05)

        self.waiting_confirmation = False

        if self._cancel_event.is_set():
            self.get_logger().info("Grasp cancelled.")
            return

        self.get_logger().info(
            f"Grasp pose - xyz: {[round(v, 4) for v in pos_grasp]}, "
            f"quat_xyzw: {[round(v, 4) for v in quat_grasp]}"
        )

        self.get_logger().info("Returning to ready position.")
        self._moveit2.move_to_configuration(joint_positions=PANDA_READY_JOINTS)
        if not self._moveit2.wait_until_executed():
            self.get_logger().error("Failed to reach ready position.")
            return

        self.get_logger().info("Moving to pre-grasp.")
        self._moveit2.move_to_pose(position=pos_pregrasp, quat_xyzw=quat_pregrasp)
        if not self._moveit2.wait_until_executed():
            self.get_logger().error("Pre-grasp motion failed.")
            return

        self.get_logger().info("Opening gripper.")
        if not self._command_gripper(
            float(self.get_parameter("gripper_open_position").value),
            skip_if_at_target=True,
        ):
            self.get_logger().error("Failed to open gripper.")
            return

        self.get_logger().info("Approaching grasp.")
        self._moveit2.move_to_pose(
            position=pos_grasp,
            quat_xyzw=quat_grasp,
            cartesian=True,
            cartesian_max_step=0.002,
        )
        if not self._moveit2.wait_until_executed():
            self.get_logger().error("Grasp approach failed.")
            return

        self.get_logger().info("Closing gripper.")
        close_position = self._target_gripper_position_for_close()
        if close_position is None:
            self.get_logger().error("Failed to choose a close gripper target.")
            return
        if bool(self.get_parameter("gripper_close_fire_and_forget").value):
            close_sent = self._send_gripper_goal_fire_and_forget(close_position)
        else:
            close_sent = self._command_gripper(
                close_position,
                assume_success_on_timeout=bool(
                    self.get_parameter("gripper_assume_close_success_on_timeout").value
                ),
            )
        if not close_sent:
            self.get_logger().error("Failed to close gripper.")
            return

        lift_height = float(self.get_parameter("lift_height_m").value)
        pos_lift = [pos_grasp[0], pos_grasp[1], pos_grasp[2] + lift_height]
        self.get_logger().info(f"Lifting {lift_height} m.")
        self._moveit2.move_to_pose(
            position=pos_lift,
            quat_xyzw=quat_grasp,
            cartesian=True,
            cartesian_max_step=0.002,
        )
        if not self._moveit2.wait_until_executed():
            self.get_logger().error("Lift failed.")
            return

        self.get_logger().info("Grasp pipeline complete.")

    def _command_gripper(
        self,
        position: float,
        skip_if_at_target: bool = False,
        assume_success_on_timeout: bool = False,
    ) -> bool:
        server_timeout_sec = float(self.get_parameter("gripper_server_timeout_sec").value)
        result_timeout_sec = float(self.get_parameter("gripper_result_timeout_sec").value)
        max_effort = float(self.get_parameter("gripper_max_effort").value)

        if skip_if_at_target and self._is_gripper_at_position(position):
            self.get_logger().info(
                f"Gripper already at position={position:.4f}; skipping command."
            )
            return True

        if not self._gripper_action_client.wait_for_server(timeout_sec=server_timeout_sec):
            self.get_logger().error(
                f"Gripper action server '{self._gripper_action_client._action_name}' is not available."
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.get_logger().info(
            f"Commanding gripper position={position:.4f}, max_effort={max_effort:.1f}"
        )
        start_position = self._gripper_joint_position
        send_future = self._gripper_action_client.send_goal_async(goal)
        if not self._wait_for_future_or_gripper_motion(
            send_future,
            server_timeout_sec,
            position,
            start_position,
        ):
            if assume_success_on_timeout:
                self.get_logger().warn(
                    "Timed out waiting for gripper goal acknowledgement; "
                    "assuming command was accepted and continuing."
                )
                return True
            self.get_logger().error("Timed out sending gripper goal.")
            return False
        if not send_future.done():
            self.get_logger().warn(
                "Gripper action acknowledgement timed out, but joint state shows motion; continuing."
            )
            return True

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal was rejected.")
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future_or_gripper_motion(
            result_future,
            result_timeout_sec,
            position,
            start_position,
        ):
            if assume_success_on_timeout:
                self.get_logger().warn(
                    "Timed out waiting for gripper result; assuming command succeeded and continuing."
                )
                return True
            self.get_logger().error("Timed out waiting for gripper result.")
            return False
        if not result_future.done():
            self.get_logger().warn(
                "Gripper action result timed out, but joint state shows motion; continuing."
            )
            return True

        result = result_future.result().result
        self.get_logger().info(
            "Gripper result: "
            f"position={result.position:.4f}, effort={result.effort:.4f}, "
            f"reached_goal={result.reached_goal}, stalled={result.stalled}"
        )
        return result.reached_goal or result.stalled

    def _send_gripper_goal_fire_and_forget(self, position: float) -> bool:
        server_timeout_sec = float(self.get_parameter("gripper_server_timeout_sec").value)
        max_effort = float(self.get_parameter("gripper_max_effort").value)
        settle_sec = float(self.get_parameter("gripper_close_settle_sec").value)

        if not self._gripper_action_client.wait_for_server(timeout_sec=server_timeout_sec):
            self.get_logger().error(
                f"Gripper action server '{self._gripper_action_client._action_name}' is not available."
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.get_logger().info(
            f"Sending gripper command position={position:.4f}, max_effort={max_effort:.1f} "
            "without waiting for action acknowledgement."
        )
        self._gripper_action_client.send_goal_async(goal)
        if settle_sec > 0.0:
            time.sleep(settle_sec)
        return True

    def _target_gripper_position_for_close(self) -> Optional[float]:
        action = str(self.get_parameter("gripper_close_action").value).lower()
        open_position = float(self.get_parameter("gripper_open_position").value)
        closed_position = float(self.get_parameter("gripper_closed_position").value)

        if action == "close":
            return closed_position
        if action == "open":
            return open_position
        if action == "toggle":
            if self._gripper_joint_position is None:
                self.get_logger().warn(
                    "No gripper joint state received; assuming gripper is open and closing it."
                )
                return closed_position

            midpoint = (open_position + closed_position) / 2.0
            if self._gripper_joint_position > midpoint:
                self.get_logger().info(
                    f"Toggle close: current gripper position={self._gripper_joint_position:.4f} "
                    f"> midpoint={midpoint:.4f}; closing."
                )
                return closed_position

            self.get_logger().info(
                f"Toggle close: current gripper position={self._gripper_joint_position:.4f} "
                f"<= midpoint={midpoint:.4f}; opening."
            )
            return open_position

        self.get_logger().error(
            f"Unknown gripper_close_action '{action}'. Use toggle, close, or open."
        )
        return None

    def _is_gripper_at_position(self, position: float) -> bool:
        if self._gripper_joint_position is None:
            return False
        tolerance = float(self.get_parameter("gripper_position_tolerance").value)
        return abs(self._gripper_joint_position - position) <= tolerance

    def _gripper_moved_toward_position(
        self,
        position: float,
        start_position: Optional[float],
    ) -> bool:
        if self._gripper_joint_position is None or start_position is None:
            return False
        tolerance = float(self.get_parameter("gripper_position_tolerance").value)
        if self._is_gripper_at_position(position):
            return True
        if position < start_position:
            return self._gripper_joint_position < start_position - tolerance
        if position > start_position:
            return self._gripper_joint_position > start_position + tolerance
        return False

    def _wait_for_future(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return future.done()

    def _wait_for_future_or_gripper_motion(
        self,
        future,
        timeout_sec: float,
        position: float,
        start_position: Optional[float],
    ) -> bool:
        motion_timeout_sec = float(self.get_parameter("gripper_motion_timeout_sec").value)
        deadline = time.monotonic() + timeout_sec
        motion_deadline = time.monotonic() + motion_timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= motion_deadline and self._gripper_moved_toward_position(
                position,
                start_position,
            ):
                return True
            if time.monotonic() >= deadline:
                return self._gripper_moved_toward_position(position, start_position)
            time.sleep(0.01)
        return future.done()

    def _lookup_base_to_camera(self, camera_frame: str) -> Optional[TransformStamped]:
        timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)
        try:
            return self._tf_buffer.lookup_transform(
                BASE_LINK,
                camera_frame,
                Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().error(
                f"Could not transform {camera_frame} into {BASE_LINK}: {exc}"
            )
            return None

    def _thinkgrasp_result_to_pose(self, result, camera_frame: str) -> PoseStamped:
        xyz = np.asarray(result.xyz, dtype=float)
        quat_xyzw = R.from_matrix(np.asarray(result.rot, dtype=float)).as_quat()

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = camera_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.x = float(quat_xyzw[0])
        pose.pose.orientation.y = float(quat_xyzw[1])
        pose.pose.orientation.z = float(quat_xyzw[2])
        pose.pose.orientation.w = float(quat_xyzw[3])
        return pose

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
    node = GraspPipelineNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    node.create_rate(1.0).sleep()  # allow MoveIt2 to connect

    trigger_key = node.get_parameter("trigger_key").value.lower()
    confirm_key = node.get_parameter("confirm_key").value.lower()
    old_terminal_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        node.get_logger().info(
            f"Press '{trigger_key}' to snapshot + call ThinkGrasp, "
            f"'{confirm_key}' to confirm and execute, any other key to cancel. "
            f"'q' to quit."
        )
        while rclpy.ok():
            key = read_key()
            if key is None:
                continue
            if key == "q":
                node.get_logger().info("Quit requested.")
                node.cancel_grasp()
                break
            if node.waiting_confirmation:
                if key == confirm_key:
                    node.confirm_grasp()
                else:
                    node.cancel_grasp()
            elif key == trigger_key:
                node.trigger_grasp()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; shutting down.")
        node.cancel_grasp()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        executor_thread.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
