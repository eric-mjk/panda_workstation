#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

PANDA_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

class JointReader(Node):
    def __init__(self):
        super().__init__("joint_reader")
        self.done = False
        self.create_subscription(JointState, "joint_states", self._cb, 10)

    def _cb(self, msg: JointState):
        if self.done:
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            positions = [round(name_to_pos[j], 6) for j in PANDA_JOINTS]
            print(positions)
            self.done = True
        except KeyError:
            pass  # message may not have all joints yet

def main():
    rclpy.init()
    node = JointReader()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
