#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import numpy as np

from realsese_bringup.transforms import quaternion_xyzw_from_matrix


SIM_T_HAND2L515_CAMERA_ISAAC = np.array([
    [
        -0.0000000298023224,
        0.9999999999999996,
        -0.0000000000000026,
        0.10000000149011634,
    ],
    [
        0.9999999999999996,
        0.0000000298023224,
        0.0000000000000021,
        -0.00000000022271273714125073,
    ],
    [
        0.0000000000000021,
        -0.0000000000000026,
        -0.9999999999999998,
        0.0534,
    ],
    [0.0, 0.0, 0.0, 1.0],
])

SIM_T_ISAAC_CAMERA2MOVEIT_CAMERA = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

SIM_T_HAND2L515_CAMERA = (
    SIM_T_HAND2L515_CAMERA_ISAAC @ SIM_T_ISAAC_CAMERA2MOVEIT_CAMERA
)


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

    
class SimCameraTF(Node):
    def __init__(self):
        super().__init__("sim_camera_tf_publisher")


        self.static_broadcaster = StaticTransformBroadcaster(self)

        self.timer = self.create_timer(0.5, self.publish_static_tf)

        self.publish_static_tf()


    def publish_static_tf(self):
        tf_msg = matrix_to_transform_stamped(
            SIM_T_HAND2L515_CAMERA,
            "panda_hand",
            "l515_camera",
            self.get_clock().now().to_msg(),
        )

        self.static_broadcaster.sendTransform(tf_msg)

def main():
    rclpy.init()
    node = SimCameraTF()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
