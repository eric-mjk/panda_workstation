#!/usr/bin/env python3

import sys
from typing import Optional

from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_ACTION_NAME = "panda_gripper/gripper_action"
DEFAULT_GRIPPER_JOINT = "panda_finger_joint1"
DEFAULT_OPEN_POSITION = 0.04
DEFAULT_CLOSED_POSITION = 0.0


class PandaGripperControl(Node):
    def __init__(self):
        super().__init__("custom_panda_gripper_control")

        self.declare_parameter("action", "toggle")
        self.declare_parameter("action_name", DEFAULT_ACTION_NAME)
        self.declare_parameter("gripper_joint_name", DEFAULT_GRIPPER_JOINT)
        self.declare_parameter("open_position", DEFAULT_OPEN_POSITION)
        self.declare_parameter("closed_position", DEFAULT_CLOSED_POSITION)
        self.declare_parameter("position", DEFAULT_OPEN_POSITION)
        self.declare_parameter("max_effort", 20.0)
        self.declare_parameter("server_timeout_sec", 3.0)
        self.declare_parameter("result_timeout_sec", 10.0)
        self.declare_parameter("joint_state_timeout_sec", 1.0)

        self._gripper_joint_name = self.get_parameter("gripper_joint_name").value
        self._joint_position: Optional[float] = None
        self.create_subscription(JointState, "joint_states", self._joint_state_callback, 10)

        action_name = self.get_parameter("action_name").value
        self._client = ActionClient(self, GripperCommand, action_name)

    def _joint_state_callback(self, msg: JointState):
        if self._gripper_joint_name not in msg.name:
            return
        index = msg.name.index(self._gripper_joint_name)
        self._joint_position = msg.position[index]

    def wait_for_joint_state(self, timeout_sec: float):
        start_time = self.get_clock().now()
        while rclpy.ok() and self._joint_position is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
            if elapsed >= timeout_sec:
                return False
        return True

    def target_position_for_action(self, action: str) -> Optional[float]:
        open_position = float(self.get_parameter("open_position").value)
        closed_position = float(self.get_parameter("closed_position").value)

        if action == "open":
            return open_position
        if action == "close":
            return closed_position
        if action == "position":
            return float(self.get_parameter("position").value)
        if action == "toggle":
            timeout_sec = float(self.get_parameter("joint_state_timeout_sec").value)
            if not self.wait_for_joint_state(timeout_sec):
                self.get_logger().warn(
                    "No gripper joint state received; assuming gripper is open and closing it."
                )
                return closed_position

            midpoint = (open_position + closed_position) / 2.0
            if self._joint_position > midpoint:
                return closed_position
            return open_position

        self.get_logger().error(
            f"Unknown action '{action}'. Use open, close, toggle, or position."
        )
        return None

    def command(self, position: float) -> bool:
        server_timeout_sec = float(self.get_parameter("server_timeout_sec").value)
        result_timeout_sec = float(self.get_parameter("result_timeout_sec").value)
        max_effort = float(self.get_parameter("max_effort").value)

        if not self._client.wait_for_server(timeout_sec=server_timeout_sec):
            self.get_logger().error(
                f"Gripper action server '{self._client._action_name}' is not available."
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.get_logger().info(
            f"Commanding gripper position={position:.4f}, max_effort={max_effort:.1f}"
        )
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=server_timeout_sec)
        if not send_future.done():
            self.get_logger().error("Timed out sending gripper goal.")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal was rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=result_timeout_sec)
        if not result_future.done():
            self.get_logger().error("Timed out waiting for gripper result.")
            return False

        result = result_future.result().result
        self.get_logger().info(
            "Gripper result: "
            f"position={result.position:.4f}, effort={result.effort:.4f}, "
            f"reached_goal={result.reached_goal}, stalled={result.stalled}"
        )
        return result.reached_goal or result.stalled


def main():
    rclpy.init()
    node = PandaGripperControl()

    action = str(node.get_parameter("action").value).lower()
    target_position = node.target_position_for_action(action)
    if target_position is None:
        rclpy.shutdown()
        sys.exit(2)

    success = node.command(target_position)
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
