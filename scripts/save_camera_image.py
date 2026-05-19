#!/usr/bin/env python3

import os
import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SaveOneCameraPNG(Node):
    def __init__(self):
        super().__init__("save_one_camera_png")

        self.bridge = CvBridge()
        self.save_path = "/workspace/camera_snapshot.png"
        self.saved = False

        self.sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.callback,
            10
        )

        self.get_logger().info("Waiting for one RGB image...")

    def callback(self, msg):
        if self.saved:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(self.save_path, img)

            self.saved = True
            self.get_logger().info(f"Saved image to {self.save_path}")
            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = SaveOneCameraPNG()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()


if __name__ == "__main__":
    main()