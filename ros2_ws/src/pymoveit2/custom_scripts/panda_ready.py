#!/usr/bin/env python3

from math import pi
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

PANDA_NAMED_STATES = {
    # Mirrors franka_moveit_config/srdf/group_definition.xacro.
    "ready": [0.0, -pi / 4.0, 0.0, -3.0 * pi / 4.0, 0.0, pi / 2.0, pi / 4.0],
    "extended": [0.0, 0.0, 0.0, -0.1, 0.0, pi / 2.0, pi / 4.0],
}


def main():
    rclpy.init()
    node = Node("custom_panda_named_state")

    node.declare_parameter("named_state", "ready")
    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")
    node.declare_parameter("base_link_name", DEFAULT_BASE_LINK)
    node.declare_parameter("end_effector_name", DEFAULT_END_EFFECTOR_LINK)
    node.declare_parameter("group_name", DEFAULT_MOVE_GROUP)
    node.declare_parameter("joint_prefix", "panda_")
    node.declare_parameter("max_velocity", 0.5)
    node.declare_parameter("max_acceleration", 0.5)

    named_state = node.get_parameter("named_state").value
    if named_state not in PANDA_NAMED_STATES:
        available_states = ", ".join(sorted(PANDA_NAMED_STATES))
        node.get_logger().error(
            f"Unknown named_state '{named_state}'. Available states: {available_states}"
        )
        rclpy.shutdown()
        return

    callback_group = ReentrantCallbackGroup()
    group_name = node.get_parameter("group_name").value
    joint_prefix = node.get_parameter("joint_prefix").value
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(prefix=joint_prefix),
        base_link_name=node.get_parameter("base_link_name").value,
        end_effector_name=node.get_parameter("end_effector_name").value,
        group_name=group_name,
        callback_group=callback_group,
    )

    moveit2.planner_id = node.get_parameter("planner_id").value
    moveit2.max_velocity = node.get_parameter("max_velocity").value
    moveit2.max_acceleration = node.get_parameter("max_acceleration").value

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    node.create_rate(1.0).sleep()

    joint_positions = PANDA_NAMED_STATES[named_state]
    joint_names = robot.joint_names(prefix=joint_prefix)
    node.get_logger().info(
        f"Moving Panda group '{group_name}' to named state '{named_state}': "
        f"{dict(zip(joint_names, joint_positions))}"
    )
    moveit2.move_to_configuration(
        joint_positions=joint_positions,
        joint_names=joint_names,
    )
    moveit2.wait_until_executed()

    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
