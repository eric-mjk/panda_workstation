#!/usr/bin/env python3

import math
import os
from dataclasses import dataclass
from threading import Thread
from typing import List, Optional, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot


DEFAULT_BASE_LINK = "panda_link0"
DEFAULT_END_EFFECTOR_LINK = "panda_hand_tcp"
DEFAULT_MOVE_GROUP = "panda_manipulator"
DEFAULT_PLANNER_ID = "RRTConnectkConfigDefault"
DEFAULT_CONFIG_BASENAME = "retrieval_config.yaml"
DEFAULT_FINAL_POSE = [
    0.0,
    -math.pi / 4.0,
    0.0,
    -3.0 * math.pi / 4.0,
    0.0,
    math.pi / 2.0,
    math.pi / 4.0,
]


@dataclass
class RetrievalConfig:
    length: float
    final_pose: List[float]


@dataclass
class MoveItContext:
    moveit2: MoveIt2
    joint_names: List[str]
    executor: MultiThreadedExecutor
    executor_thread: Thread


def default_config_file() -> str:
    try:
        package_share = get_package_share_directory("pymoveit2")
        return os.path.join(package_share, "heuristics", DEFAULT_CONFIG_BASENAME)
    except Exception:
        return os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_BASENAME)


def declare_common_parameters(node: Node) -> None:
    node.declare_parameter("config_file", "")
    node.declare_parameter("planner_id", DEFAULT_PLANNER_ID)
    node.declare_parameter("base_link_name", DEFAULT_BASE_LINK)
    node.declare_parameter("end_effector_name", DEFAULT_END_EFFECTOR_LINK)
    node.declare_parameter("group_name", DEFAULT_MOVE_GROUP)
    node.declare_parameter("joint_prefix", "panda_")
    node.declare_parameter("max_velocity", 0.5)
    node.declare_parameter("max_acceleration", 0.5)
    node.declare_parameter("cartesian_max_step", 0.0025)
    node.declare_parameter("cartesian_fraction_threshold", 0.0)
    node.declare_parameter("cartesian_jump_threshold", 0.0)
    node.declare_parameter("cartesian_avoid_collisions", False)
    node.declare_parameter("tf_lookup_timeout_sec", 3.0)


def load_retrieval_config(node: Node) -> Optional[RetrievalConfig]:
    config_file = str(node.get_parameter("config_file").value or default_config_file())
    if not os.path.exists(config_file):
        node.get_logger().error(f"Retrieval config file does not exist: {config_file}")
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as exc:
        node.get_logger().error(f"Failed to load retrieval config '{config_file}': {exc}")
        return None

    length = raw_config.get("length", 0.10)
    final_pose = raw_config.get("final_pose", DEFAULT_FINAL_POSE)

    try:
        length = float(length)
        final_pose = [float(position) for position in final_pose]
    except (TypeError, ValueError) as exc:
        node.get_logger().error(f"Invalid retrieval config values in '{config_file}': {exc}")
        return None

    if length <= 0.0:
        node.get_logger().error(f"Retrieval length must be positive, got {length}.")
        return None
    if len(final_pose) != 7:
        node.get_logger().error(
            f"final_pose must contain exactly 7 joint values, got {len(final_pose)}."
        )
        return None

    return RetrievalConfig(length=length, final_pose=final_pose)


def create_moveit_context(node: Node, callback_group: ReentrantCallbackGroup) -> MoveItContext:
    joint_prefix = node.get_parameter("joint_prefix").value
    joint_names = robot.joint_names(prefix=joint_prefix)
    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name=node.get_parameter("base_link_name").value,
        end_effector_name=node.get_parameter("end_effector_name").value,
        group_name=node.get_parameter("group_name").value,
        callback_group=callback_group,
    )

    moveit2.planner_id = node.get_parameter("planner_id").value
    moveit2.max_velocity = node.get_parameter("max_velocity").value
    moveit2.max_acceleration = node.get_parameter("max_acceleration").value
    moveit2.cartesian_jump_threshold = node.get_parameter("cartesian_jump_threshold").value
    moveit2.cartesian_avoid_collisions = node.get_parameter(
        "cartesian_avoid_collisions"
    ).value

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    node.create_rate(1.0).sleep()

    return MoveItContext(
        moveit2=moveit2,
        joint_names=joint_names,
        executor=executor,
        executor_thread=executor_thread,
    )


def lookup_current_tcp_pose(
    node: Node,
    tf_buffer: tf2_ros.Buffer,
) -> Optional[Tuple[List[float], List[float]]]:
    base_link_name = node.get_parameter("base_link_name").value
    end_effector_name = node.get_parameter("end_effector_name").value
    timeout_sec = float(node.get_parameter("tf_lookup_timeout_sec").value)

    try:
        transform = tf_buffer.lookup_transform(
            base_link_name,
            end_effector_name,
            Time(),
            timeout=Duration(seconds=timeout_sec),
        )
    except tf2_ros.TransformException as exc:
        node.get_logger().error(
            f"Could not transform {end_effector_name} into {base_link_name}: {exc}"
        )
        return None

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    position = [translation.x, translation.y, translation.z]
    quat_xyzw = [rotation.x, rotation.y, rotation.z, rotation.w]
    return position, quat_xyzw


def tcp_negative_z_in_base(quat_xyzw: List[float]) -> List[float]:
    x, y, z, w = quat_xyzw
    rotation_matrix_z_axis = [
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
    negative_z = [
        -rotation_matrix_z_axis[0],
        -rotation_matrix_z_axis[1],
        -rotation_matrix_z_axis[2],
    ]
    norm = math.sqrt(sum(component * component for component in negative_z))
    return [component / norm for component in negative_z]


def shutdown_context(context: Optional[MoveItContext]) -> None:
    rclpy.shutdown()
    if context is not None:
        context.executor_thread.join()
