#!/usr/bin/env python3

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
import tf2_ros

from heuristics.common import (
    create_moveit_context,
    declare_common_parameters,
    load_retrieval_config,
    lookup_current_tcp_pose,
    shutdown_context,
    tcp_negative_z_in_base,
)
from heuristics.default_mover import move_to_final_pose


def main():
    rclpy.init()
    node = Node("approach_direction_retrieval")
    declare_common_parameters(node)

    config = load_retrieval_config(node)
    context = None
    if config is None:
        rclpy.shutdown()
        return

    try:
        callback_group = ReentrantCallbackGroup()
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer, node)
        context = create_moveit_context(node, callback_group)

        current_pose = lookup_current_tcp_pose(node, tf_buffer)
        if current_pose is None:
            return

        position, quat_xyzw = current_pose
        direction = tcp_negative_z_in_base(quat_xyzw)
        target_position = [
            position[index] + config.length * direction[index] for index in range(3)
        ]
        node.get_logger().info(
            f"Moving {config.length:.3f} m along "
            f"-Z of {node.get_parameter('end_effector_name').value}."
        )
        context.moveit2.move_to_pose(
            position=target_position,
            quat_xyzw=quat_xyzw,
            cartesian=True,
            cartesian_max_step=node.get_parameter("cartesian_max_step").value,
            cartesian_fraction_threshold=node.get_parameter(
                "cartesian_fraction_threshold"
            ).value,
        )
        if not context.moveit2.wait_until_executed():
            node.get_logger().error("Approach-direction retrieval motion failed.")
            return

        move_to_final_pose(
            node=node,
            moveit2=context.moveit2,
            joint_names=context.joint_names,
            final_pose=config.final_pose,
        )
    finally:
        shutdown_context(context)


if __name__ == "__main__":
    main()
