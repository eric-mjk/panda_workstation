#!/usr/bin/env python3

import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer


INTRINSIC = np.array(
    [
        [906.4620361328125, 0.0, 645.7659912109375],
        [0.0, 906.65283203125, 375.2723388671875],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


class RGBDToPointCloud(Node):
    def __init__(self):
        super().__init__("rgbd_to_pointcloud")

        # -----------------------------
        # ROS topic parameters
        # -----------------------------
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("points_topic", "/camera/rgbd/points")

        self.declare_parameter("frame_id", "")

        # Depth handling
        # 32FC1 depth in meters  -> depth_scale = 1.0
        # 16UC1 RealSense L515   -> depth_scale = 4000.0 for 0.25 mm units
        # 16UC1 depth in mm      -> depth_scale = 1000.0
        self.declare_parameter("depth_scale", 4000.0)
        self.declare_parameter("max_depth", 3.0)

        # Use stride > 1 to make the cloud lighter
        self.declare_parameter("stride", 4)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.points_topic = self.get_parameter("points_topic").value
        self.frame_id_override = self.get_parameter("frame_id").value

        self.fx = float(INTRINSIC[0, 0])
        self.fy = float(INTRINSIC[1, 1])
        self.cx = float(INTRINSIC[0, 2])
        self.cy = float(INTRINSIC[1, 2])

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.stride = int(self.get_parameter("stride").value)

        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.rgb_sub = Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=sensor_qos,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=sensor_qos,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.callback)

        self.pc_pub = self.create_publisher(
            PointCloud2,
            self.points_topic,
            10,
        )

        self.get_logger().info("RGBD to PointCloud node started")
        self.get_logger().info(f"RGB topic:         {self.rgb_topic}")
        self.get_logger().info(f"Depth topic:       {self.depth_topic}")
        self.get_logger().info(f"Points topic:      {self.points_topic}")
        self.get_logger().info(
            f"Intrinsics:        fx={self.fx:.3f}, fy={self.fy:.3f}, "
            f"cx={self.cx:.3f}, cy={self.cy:.3f}"
        )

    def callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        if depth.ndim != 2:
            self.get_logger().warn("Depth image is not single-channel")
            return

        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth.shape[:2]

        if (rgb_h, rgb_w) != (depth_h, depth_w):
            self.get_logger().warn(
                f"RGB/depth size mismatch: "
                f"RGB={rgb_w}x{rgb_h}, depth={depth_w}x{depth_h}. "
                "This script assumes aligned RGB and depth."
            )
            return

        if depth.dtype == np.uint16:
            z = depth.astype(np.float32) / self.depth_scale
        else:
            z = depth.astype(np.float32)

        h, w = z.shape
        stride = max(1, self.stride)

        us, vs = np.meshgrid(
            np.arange(0, w, stride),
            np.arange(0, h, stride),
        )

        z_sampled = z[vs, us]

        valid = np.isfinite(z_sampled)
        valid &= z_sampled > 0.0
        valid &= z_sampled < self.max_depth

        us_valid = us[valid].astype(np.float32)
        vs_valid = vs[valid].astype(np.float32)
        zs = z_sampled[valid].astype(np.float32)

        xs = (us_valid - self.cx) * zs / self.fx
        ys = (vs_valid - self.cy) * zs / self.fy

        colors = rgb[vs_valid.astype(np.int32), us_valid.astype(np.int32)]

        points = []

        for x, y, z_val, color in zip(xs, ys, zs, colors):
            r = int(color[0])
            g = int(color[1])
            b = int(color[2])

            rgb_uint32 = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack("f", struct.pack("I", rgb_uint32))[0]

            points.append([float(x), float(y), float(z_val), rgb_float])

        frame_id = self.frame_id_override
        if frame_id == "":
            frame_id = rgb_msg.header.frame_id

        cloud_msg = self.create_cloud_xyzrgb(
            points,
            stamp=rgb_msg.header.stamp,
            frame_id=frame_id,
        )

        self.pc_pub.publish(cloud_msg)

    def create_cloud_xyzrgb(self, points, stamp, frame_id):
        fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="rgb",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        cloud_data = []
        for p in points:
            cloud_data.append(struct.pack("ffff", p[0], p[1], p[2], p[3]))

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * len(points)
        cloud.is_dense = False
        cloud.data = b"".join(cloud_data)

        return cloud


def main(args=None):
    rclpy.init(args=args)
    node = RGBDToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
