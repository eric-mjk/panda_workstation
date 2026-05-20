#!/usr/bin/env python3

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from heuristics.common import (
    create_moveit_context,
    declare_common_parameters,
    load_retrieval_config,
    shutdown_context,
)


def move_to_final_pose(node: Node, moveit2, joint_names, final_pose) -> bool:
    node.get_logger().info(f"Moving to configured final pose: {final_pose}")
    moveit2.move_to_configuration(
        joint_positions=final_pose,
        joint_names=joint_names,
    )
    if not moveit2.wait_until_executed():
        node.get_logger().error("Failed to reach configured final pose.")
        return False
    return True


def main():
    rclpy.init()
    node = Node("default_mover")
    declare_common_parameters(node)

    config = load_retrieval_config(node)
    context = None
    if config is None:
        rclpy.shutdown()
        return

    try:
        callback_group = ReentrantCallbackGroup()
        context = create_moveit_context(node, callback_group)
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
