#!/usr/bin/env python3

import json
import math
import subprocess
from pathlib import Path
from threading import Thread

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros


def _result_json_path(node: Node) -> Path:
    explicit_path = str(node.get_parameter("result_json").value or "")
    if explicit_path:
        return Path(explicit_path)

    output_root = Path(str(node.get_parameter("output_root").value))
    experiment_name = str(node.get_parameter("experiment_name").value)
    new_path = output_root / experiment_name / "directions" / "final_3d_direction.json"
    if new_path.is_file():
        return new_path
    return output_root / experiment_name / "offline" / "final_3d_direction.json"


def _load_direction(path: Path, direction_key: str) -> list[float] | None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    direction = data.get(direction_key)
    if not isinstance(direction, list) or len(direction) != 3:
        return None

    direction = [float(v) for v in direction]
    norm = math.sqrt(sum(v * v for v in direction))
    if norm <= 1e-9:
        return None
    return [v / norm for v in direction]


def _lookup_current_tcp_pose(
    node: Node,
    tf_buffer: tf2_ros.Buffer,
) -> tuple[list[float], list[float]] | None:
    base_link_name = str(node.get_parameter("base_link_name").value)
    end_effector_name = str(node.get_parameter("end_effector_name").value)
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


def main() -> None:
    rclpy.init()
    node = Node("fetchbench_pull_best_direction")

    node.declare_parameter("output_root", "/workspace/ros2_ws/ours_experiment")
    node.declare_parameter("experiment_name", "ex1")
    node.declare_parameter("result_json", "")
    node.declare_parameter("direction_key", "aggregate_direction")
    node.declare_parameter("pull_distance_m", 0.15)
    node.declare_parameter("base_link_name", "panda_link0")
    node.declare_parameter("end_effector_name", "panda_hand_tcp")
    node.declare_parameter("tf_lookup_timeout_sec", 3.0)

    executor_thread = None
    try:
        direction_path = _result_json_path(node)
        if not direction_path.is_file():
            node.get_logger().error(f"Direction JSON does not exist: {direction_path}")
            return

        direction_key = str(node.get_parameter("direction_key").value)
        direction = _load_direction(direction_path, direction_key)
        if direction is None:
            node.get_logger().error(
                f"Could not read a valid 3D direction from '{direction_key}' in {direction_path}"
            )
            return

        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer, node)

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def _spin() -> None:
            try:
                executor.spin()
            except rclpy.executors.ExternalShutdownException:
                pass

        executor_thread = Thread(target=_spin, daemon=True)
        executor_thread.start()

        current_pose = _lookup_current_tcp_pose(node, tf_buffer)
        if current_pose is None:
            return

        position, quat_xyzw = current_pose
        distance = float(node.get_parameter("pull_distance_m").value)
        target_position = [position[i] + distance * direction[i] for i in range(3)]

        node.get_logger().info(f"Direction file: {direction_path}")
        node.get_logger().info(
            f"Pull direction: x={direction[0]:.4f}  y={direction[1]:.4f}  z={direction[2]:.4f}"
        )
        node.get_logger().info(
            f"Initial position: x={position[0]:.4f}  y={position[1]:.4f}  z={position[2]:.4f}"
        )
        node.get_logger().info(
            f"Target  position: x={target_position[0]:.4f}  y={target_position[1]:.4f}  z={target_position[2]:.4f}"
        )

        pos_str = f"[{target_position[0]}, {target_position[1]}, {target_position[2]}]"
        quat_str = f"[{quat_xyzw[0]}, {quat_xyzw[1]}, {quat_xyzw[2]}, {quat_xyzw[3]}]"
        cmd = [
            "ros2",
            "run",
            "pymoveit2",
            "panda_pose_goal.py",
            "--ros-args",
            "-p",
            f"position:={pos_str}",
            "-p",
            f"quat_xyzw:={quat_str}",
            "-p",
            "cartesian:=true",
        ]
        node.get_logger().info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            node.get_logger().error("FetchBench pull motion failed.")
            return

        node.get_logger().info("FetchBench pull motion complete.")
    finally:
        rclpy.shutdown()
        if executor_thread is not None:
            executor_thread.join()


if __name__ == "__main__":
    main()
