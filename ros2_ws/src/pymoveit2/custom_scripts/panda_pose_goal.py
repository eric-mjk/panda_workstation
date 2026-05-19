#!/usr/bin/env python3

from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot


DEFAULT_BASE_LINK = "panda_link0"
DEFAULT_END_EFFECTOR_LINK = "panda_link8"
DEFAULT_MOVE_GROUP = "panda_arm"


def main():
    rclpy.init()
    node = Node("custom_panda_pose_goal")

    node.declare_parameter("position", [0.45, 0.0, 0.35])
    node.declare_parameter("quat_xyzw", [1.0, 0.0, 0.0, 0.0])
    node.declare_parameter("cartesian", False)
    node.declare_parameter("cartesian_max_step", 0.0025)
    node.declare_parameter("cartesian_fraction_threshold", 0.0)
    node.declare_parameter("cartesian_jump_threshold", 0.0)
    node.declare_parameter("cartesian_avoid_collisions", False)
    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")
    node.declare_parameter("base_link_name", DEFAULT_BASE_LINK)
    node.declare_parameter("end_effector_name", DEFAULT_END_EFFECTOR_LINK)
    node.declare_parameter("group_name", DEFAULT_MOVE_GROUP)
    node.declare_parameter("max_velocity", 0.5)
    node.declare_parameter("max_acceleration", 0.5)

    callback_group = ReentrantCallbackGroup()
    group_name = node.get_parameter("group_name").value
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=node.get_parameter("base_link_name").value,
        end_effector_name=node.get_parameter("end_effector_name").value,
        group_name=group_name,
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

    position = node.get_parameter("position").value
    quat_xyzw = node.get_parameter("quat_xyzw").value
    cartesian = node.get_parameter("cartesian").value
    cartesian_max_step = node.get_parameter("cartesian_max_step").value
    cartesian_fraction_threshold = node.get_parameter(
        "cartesian_fraction_threshold"
    ).value

    node.get_logger().info(
        f"Moving Panda group '{group_name}' to pose: position={position}, quat_xyzw={quat_xyzw}"
    )
    moveit2.move_to_pose(
        position=position,
        quat_xyzw=quat_xyzw,
        cartesian=cartesian,
        cartesian_max_step=cartesian_max_step,
        cartesian_fraction_threshold=cartesian_fraction_threshold,
    )
    moveit2.wait_until_executed()

    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
