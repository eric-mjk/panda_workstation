#!/usr/bin/env python3

import subprocess
from threading import Thread

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
import tf2_ros

from heuristics.common import (
    declare_common_parameters,
    load_retrieval_config,
    lookup_current_tcp_pose,
)


def main():
    rclpy.init()
    node = Node("vertical_retrieval")
    declare_common_parameters(node)

    config = load_retrieval_config(node)
    if config is None:
        rclpy.shutdown()
        return

    executor_thread = None
    try:
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer, node)

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def _spin():
            try:
                executor.spin()
            except rclpy.executors.ExternalShutdownException:
                pass

        executor_thread = Thread(target=_spin, daemon=True)
        executor_thread.start()

        current_pose = lookup_current_tcp_pose(node, tf_buffer)
        if current_pose is None:
            return

        position, quat_xyzw = current_pose
        target_position = [position[0], position[1], position[2] + config.length]

        node.get_logger().info(
            f"Initial position: x={position[0]:.4f}  y={position[1]:.4f}  z={position[2]:.4f}"
        )
        node.get_logger().info(
            f"Target  position: x={target_position[0]:.4f}  y={target_position[1]:.4f}  z={target_position[2]:.4f}"
        )
        node.get_logger().info(
            f"Moving {config.length:.3f} m along +Z in "
            f"{node.get_parameter('base_link_name').value}."
        )

        pos_str = f"[{target_position[0]}, {target_position[1]}, {target_position[2]}]"
        quat_str = f"[{quat_xyzw[0]}, {quat_xyzw[1]}, {quat_xyzw[2]}, {quat_xyzw[3]}]"
        cmd = [
            "ros2", "run", "pymoveit2", "panda_pose_goal.py",
            "--ros-args",
            "-p", f"position:={pos_str}",
            "-p", f"quat_xyzw:={quat_str}",
            "-p", "cartesian:=true",
        ]
        node.get_logger().info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            node.get_logger().error("Vertical retrieval motion failed.")
            return

        node.get_logger().info("Vertical retrieval complete.")
    finally:
        rclpy.shutdown()
        if executor_thread is not None:
            executor_thread.join()


if __name__ == "__main__":
    main()
