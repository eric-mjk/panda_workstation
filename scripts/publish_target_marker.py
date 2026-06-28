#!/usr/bin/env python3

import argparse

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker


class TargetMarker(Node):
    def __init__(self, frame_id: str, point_xyz: list[float], radius: float) -> None:
        super().__init__("fetchbench_target_marker")
        self.publisher = self.create_publisher(Marker, "/fetchbench_target_marker", 10)
        self.frame_id = str(frame_id)
        self.point_xyz = [float(v) for v in point_xyz]
        self.radius = float(radius)
        self.timer = self.create_timer(0.2, self.publish_marker)

    def publish_marker(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fetchbench_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.point_xyz[0]
        marker.pose.position.y = self.point_xyz[1]
        marker.pose.position.z = self.point_xyz[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.radius
        marker.scale.y = self.radius
        marker.scale.z = self.radius
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.publisher.publish(marker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a target point marker in RViz.")
    parser.add_argument("--frame", default="panda_link0", help="RViz/global frame for the point.")
    parser.add_argument("--point", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--radius", type=float, default=0.04, help="Marker sphere diameter in meters.")
    args = parser.parse_args()

    rclpy.init()
    node = TargetMarker(args.frame, args.point, args.radius)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
