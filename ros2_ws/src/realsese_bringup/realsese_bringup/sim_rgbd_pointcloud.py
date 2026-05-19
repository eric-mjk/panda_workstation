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


SIM_INTRINSIC = np.array(
    [
        [1399.0697520372776, 0.0, 960.0],
        [0.0, 1399.0697520372776, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


class SimRGBDToPointCloud(Node):
    def __init__(self):
        super().__init__("sim_rgbd_to_pointcloud")

        # -----------------------------
        # ROS topic parameters
        # -----------------------------
        self.declare_parameter("rgb_topic", "/isaac_rgb")
        self.declare_parameter("depth_topic", "/isaac_depth")
        self.declare_parameter("points_topic", "/camera/rgbd/points")

        self.declare_parameter("frame_id", "l515_camera")
        self.declare_parameter("use_message_timestamp", False)

        # Depth handling
        # 32FC1 depth in meters  -> depth_scale = 1.0
        # 16UC1 RealSense L515   -> depth_scale = 4000.0 for 0.25 mm units
        # 16UC1 depth in mm      -> depth_scale = 1000.0
        self.declare_parameter("depth_scale", 4000.0)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("sync_slop", 0.5)

        # Use stride > 1 to make the cloud lighter
        self.declare_parameter("stride", 4)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.points_topic = self.get_parameter("points_topic").value
        self.frame_id_override = self.get_parameter("frame_id").value
        self.use_message_timestamp = bool(
            self.get_parameter("use_message_timestamp").value
        )

        self.fx = float(SIM_INTRINSIC[0, 0])
        self.fy = float(SIM_INTRINSIC[1, 1])
        self.cx = float(SIM_INTRINSIC[0, 2])
        self.cy = float(SIM_INTRINSIC[1, 2])

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)
        self.stride = int(self.get_parameter("stride").value)
        self.rgb_count = 0
        self.depth_count = 0
        self.synced_count = 0
        self.last_point_count = 0

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

        self.create_subscription(
            Image,
            self.rgb_topic,
            self.debug_rgb_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self.debug_depth_callback,
            sensor_qos,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.callback)

        self.pc_pub = self.create_publisher(
            PointCloud2,
            self.points_topic,
            10,
        )

        self.get_logger().info("Sim RGBD to PointCloud node started")
        self.get_logger().info(f"RGB topic:         {self.rgb_topic}")
        self.get_logger().info(f"Depth topic:       {self.depth_topic}")
        self.get_logger().info(f"Points topic:      {self.points_topic}")
        self.get_logger().info(f"Point cloud frame: {self.frame_id_override}")
        self.get_logger().info(
            f"Use input stamp:   {self.use_message_timestamp}"
        )
        self.get_logger().info(
            f"Intrinsics:        fx={self.fx:.3f}, fy={self.fy:.3f}, "
            f"cx={self.cx:.3f}, cy={self.cy:.3f}"
        )
        self.get_logger().info(f"Sync slop:         {self.sync_slop:.3f}s")

        self.create_timer(2.0, self.log_diagnostics)

    def debug_rgb_callback(self, _msg: Image):
        self.rgb_count += 1

    def debug_depth_callback(self, _msg: Image):
        self.depth_count += 1

    def log_diagnostics(self):
        if self.synced_count == 0:
            self.get_logger().warn(
                "Waiting for synchronized RGBD frames. "
                f"RGB received={self.rgb_count}, depth received={self.depth_count}. "
                f"Check topic names and timestamps if both counts are increasing."
            )
            return

        self.get_logger().info(
            f"RGBD syncs={self.synced_count}, last cloud points={self.last_point_count}, "
            f"RGB received={self.rgb_count}, depth received={self.depth_count}"
        )

    def callback(self, rgb_msg: Image, depth_msg: Image):
        self.synced_count += 1

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

        if not np.any(valid):
            finite_depth = z_sampled[np.isfinite(z_sampled)]
            if finite_depth.size == 0:
                self.get_logger().warn("No finite depth values in sampled depth image.")
            else:
                self.get_logger().warn(
                    "No valid depth points after filtering. "
                    f"sampled depth min={float(np.min(finite_depth)):.4f}, "
                    f"max={float(np.max(finite_depth)):.4f}, "
                    f"max_depth={self.max_depth:.4f}, depth_scale={self.depth_scale:.4f}"
                )

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

        stamp = self.get_clock().now().to_msg()
        if self.use_message_timestamp:
            stamp = rgb_msg.header.stamp

        cloud_msg = self.create_cloud_xyzrgb(
            points,
            stamp=stamp,
            frame_id=frame_id,
        )

        self.last_point_count = len(points)
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
    node = SimRGBDToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
