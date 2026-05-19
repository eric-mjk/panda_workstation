#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import numpy as np

from realsese_bringup.transforms import quaternion_xyzw_from_matrix


T_HAND2CAMERA = np.array([
  [-0.00655526, -0.999961, 0.00593217, 0.092329],
  [0.999813, -0.006662, -0.0181578, 0.00308126],
  [0.0181966, 0.00581203, 0.999818, 0.0622155],
  [0.0, 0.0, 0.0, 1.0]
])


def matrix_to_transform_stamped(T, parent_frame, child_frame, stamp):
    """
    T is the transform from parent_frame to child_frame.

    Meaning:
        T_parent_child
        child_frame pose expressed in parent_frame
    """

    tf_msg = TransformStamped()

    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame

    # translation
    tf_msg.transform.translation.x = float(T[0, 3])
    tf_msg.transform.translation.y = float(T[1, 3])
    tf_msg.transform.translation.z = float(T[2, 3])

    # rotation
    qx, qy, qz, qw = quaternion_xyzw_from_matrix(T)

    tf_msg.transform.rotation.x = float(qx)
    tf_msg.transform.rotation.y = float(qy)
    tf_msg.transform.rotation.z = float(qz)
    tf_msg.transform.rotation.w = float(qw)

    return tf_msg

    
class CameraTFPublisher(Node):
    def __init__(self):
        super().__init__("camera_tf_publisher")


        self.static_broadcaster = StaticTransformBroadcaster(self)

        self.timer = self.create_timer(0.5, self.publish_static_tf)

        self.publish_static_tf()


    def publish_static_tf(self):
        tf_msg = matrix_to_transform_stamped(
            T_HAND2CAMERA,
            "panda_hand",
            "camera_link",
            self.get_clock().now().to_msg(),
        )

        self.static_broadcaster.sendTransform(tf_msg)

def main():
    rclpy.init()
    node = CameraTFPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
