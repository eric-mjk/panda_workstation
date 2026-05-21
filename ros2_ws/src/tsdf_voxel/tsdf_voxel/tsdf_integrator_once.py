#!/usr/bin/env python3
"""Interactive TSDF integration debug node."""

import atexit
import os
import select
import sys
import termios
import tty

import numpy as np
import open3d as o3d
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

DEFAULT_PARAMETERS = {
    'rgb_topic': '',
    'depth_topic': '',
    'camera_config_file': '',
    'tsdf_config_file': '',
    'depth_scale': 1.0,
    'max_depth': 3.0,
    'voxel_length': 0.005,
    'sdf_trunc': 0.04,
    'sync_slop': 0.1,
    'output_dir': '/workspace/ros2_ws/tsdf_meshes',
    'world_frame': '',
    'mesh_marker_topic': '/tsdf_mesh_marker',
    'keyboard_poll_period': 0.05,
    'status_log_period': 2.0,
}


class TSDFIntegratorOnce(Node):
    """Wait for keyboard commands to integrate and publish TSDF debug meshes."""

    def __init__(self):
        super().__init__('tsdf_integrator_once')

        for name, default_value in DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, default_value)

        self._camera_config_file = self.get_parameter('camera_config_file').value
        if self._camera_config_file == '':
            raise ValueError('camera_config_file is required')

        self._camera_config = self._load_yaml(self._camera_config_file)
        self._tsdf_config_file = self.get_parameter('tsdf_config_file').value
        if self._tsdf_config_file == '':
            raise ValueError('tsdf_config_file is required')

        self._tsdf_config = self._load_yaml(self._tsdf_config_file)
        source_params = self._get_camera_source_parameters()
        debug_params = self._camera_config.get('tsdf_debug', {}).get(
            'ros__parameters', {}
        )
        tsdf_params = self._tsdf_config.get('tsdf_integrator', {}).get(
            'ros__parameters', {}
        )

        self._rgb_topic = self._get_config_parameter('rgb_topic', source_params)
        self._depth_topic = self._get_config_parameter('depth_topic', source_params)
        self._depth_scale = float(
            self._get_config_parameter('depth_scale', source_params)
        )
        self._max_depth = float(
            self._get_config_parameter('max_depth', source_params)
        )
        self._sync_slop = float(
            self._get_config_parameter('sync_slop', source_params)
        )

        self._voxel_length = float(
            self._get_config_parameter('voxel_length', tsdf_params)
        )
        self._sdf_trunc = float(
            self._get_config_parameter('sdf_trunc', tsdf_params)
        )
        self._voxel_bounds = tsdf_params.get('voxel_bounds')

        self._output_dir = self._get_config_parameter('output_dir', debug_params)
        self._world_frame = self._get_config_parameter('world_frame', debug_params)
        self._mesh_marker_topic = self._get_config_parameter(
            'mesh_marker_topic', debug_params
        )
        self._keyboard_poll_period = float(
            self._get_config_parameter('keyboard_poll_period', debug_params)
        )
        self._status_log_period = float(
            self._get_config_parameter('status_log_period', debug_params)
        )

        self._bridge = CvBridge()
        self._intrinsic = self._load_camera_model()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._volume = self._make_volume()
        self._rgb_count = 0
        self._depth_count = 0
        self._synced_count = 0
        self._integrated_count = 0
        self._latest_rgb_msg = None
        self._latest_depth_msg = None
        self._terminal_settings = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._rgb_debug_sub = self.create_subscription(
            Image, self._rgb_topic, self._rgb_debug_callback, sensor_qos
        )
        self._depth_debug_sub = self.create_subscription(
            Image, self._depth_topic, self._depth_debug_callback, sensor_qos
        )

        self._rgb_sub = Subscriber(self, Image, self._rgb_topic, qos_profile=sensor_qos)
        self._depth_sub = Subscriber(
            self, Image, self._depth_topic, qos_profile=sensor_qos
        )
        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=10,
            slop=self._sync_slop,
        )
        self._sync.registerCallback(self._rgbd_callback)
        self.create_timer(self._status_log_period, self._log_waiting_status)
        self.create_timer(self._keyboard_poll_period, self._poll_keyboard)

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._mesh_pub = self.create_publisher(
            Marker, self._mesh_marker_topic, marker_qos
        )
        collision_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', collision_qos
        )
        self._collision_object_id = 'tsdf_mesh'
        self._setup_keyboard()

        self.get_logger().info('Interactive TSDF debug node started')
        self.get_logger().info(f'Camera config: {self._camera_config_file}')
        self.get_logger().info(f'TSDF config: {self._tsdf_config_file}')
        self.get_logger().info(f'RGB topic: {self._rgb_topic}')
        self.get_logger().info(f'Depth topic: {self._depth_topic}')
        self.get_logger().info(f'TF lookup: {self._world_frame} -> {self._camera_frame}')
        self.get_logger().info(f'Sync slop: {self._sync_slop:.3f}s')
        self.get_logger().info(f'Output dir: {self._output_dir}')
        self.get_logger().info(f'Voxel bounds in {self._world_frame}: {self._voxel_bounds}')
        self.get_logger().info(f'Marker topic: {self._mesh_marker_topic}')
        self.get_logger().info('Press p to integrate one RGBD pair')
        self.get_logger().info(
            f'Press m to extract/publish {self._mesh_marker_topic} and /collision_object'
        )
        self.get_logger().info('Press r to remove collision object from planning scene')

    def _load_yaml(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            return yaml.safe_load(config_file)

    def _get_camera_source_parameters(self) -> dict:
        for node_name in ('sim_rgbd_to_pointcloud', 'rgbd_to_pointcloud'):
            params = self._camera_config.get(node_name, {}).get('ros__parameters')
            if params is not None:
                return params
        raise ValueError(
            'camera_config_file must contain sim_rgbd_to_pointcloud or '
            'rgbd_to_pointcloud ros__parameters'
        )

    def _get_config_parameter(self, name: str, config_parameters: dict):
        ros_value = self.get_parameter(name).value
        if ros_value != DEFAULT_PARAMETERS[name]:
            return ros_value
        return config_parameters.get(name, ros_value)

    def _load_camera_model(self):
        camera_info = self._camera_config['/camera/color/camera_info']
        k = camera_info['k']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(camera_info['width']),
            height=int(camera_info['height']),
            fx=float(k[0]),
            fy=float(k[4]),
            cx=float(k[2]),
            cy=float(k[5]),
        )

        self._camera_frame = camera_info['header']['frame_id']

        if self._world_frame == '':
            raise ValueError('camera config tsdf_debug.ros__parameters.world_frame is required')
        if self._camera_frame == '':
            raise ValueError('/camera/color/camera_info.header.frame_id is required')

        self.get_logger().info(
            f'Intrinsics: {camera_info["width"]}x{camera_info["height"]}, '
            f'fx={k[0]:.2f}, fy={k[4]:.2f}, cx={k[2]:.2f}, cy={k[5]:.2f}'
        )
        self.get_logger().info(f'Camera frame from config: {self._camera_frame}')

        return intrinsic

    def _make_volume(self):
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self._voxel_length,
            sdf_trunc=self._sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def _rgb_debug_callback(self, _msg: Image) -> None:
        self._rgb_count += 1

    def _depth_debug_callback(self, _msg: Image) -> None:
        self._depth_count += 1

    def _log_waiting_status(self) -> None:
        self.get_logger().info(
            'Status: '
            f'RGB received={self._rgb_count}, depth received={self._depth_count}, '
            f'synced={self._synced_count}, integrated={self._integrated_count}'
        )

    def _rgbd_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        self._latest_rgb_msg = rgb_msg
        self._latest_depth_msg = depth_msg
        self._synced_count += 1

        if self._synced_count == 1:
            self.get_logger().info('Received first synchronized RGBD pair')
            self.get_logger().info(
                f'RGB msg: {rgb_msg.width}x{rgb_msg.height}, '
                f'encoding={rgb_msg.encoding}'
            )
            self.get_logger().info(
                f'Depth msg: {depth_msg.width}x{depth_msg.height}, '
                f'encoding={depth_msg.encoding}'
            )

    def _poll_keyboard(self) -> None:
        if not sys.stdin or not select.select([sys.stdin], [], [], 0.0)[0]:
            return

        key = sys.stdin.read(1)
        if key == 'p':
            self._integrate_latest_rgbd()
        elif key == 'm':
            self._extract_and_publish_mesh()
        elif key == 'r':
            self._remove_collision_object()
        elif key == 'q':
            self.get_logger().info('Quit requested')
            self._request_shutdown()

    def _integrate_latest_rgbd(self) -> None:
        if self._latest_rgb_msg is None or self._latest_depth_msg is None:
            self.get_logger().warn('No synchronized RGBD pair available yet')
            return

        rgb_msg = self._latest_rgb_msg
        depth_msg = self._latest_depth_msg

        self.get_logger().info('Integrating one synchronized RGBD pair')
        self.get_logger().info(
            f'RGB msg: {rgb_msg.width}x{rgb_msg.height}, encoding={rgb_msg.encoding}'
        )
        self.get_logger().info(
            f'Depth msg: {depth_msg.width}x{depth_msg.height}, '
            f'encoding={depth_msg.encoding}'
        )

        try:
            rgb_np = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_raw = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        if depth_raw.ndim != 2:
            self.get_logger().error(f'Depth image is not single-channel: {depth_raw.shape}')
            return

        self._log_array_stats('RGB array', rgb_np)
        self._log_array_stats('Depth raw', depth_raw)

        rgb_np = np.ascontiguousarray(rgb_np, dtype=np.uint8)
        depth_np = _to_open3d_depth(depth_raw)
        self._log_array_stats('Depth Open3D', depth_np)

        if rgb_np.shape[:2] != depth_np.shape[:2]:
            self.get_logger().error(
                f'RGB/depth size mismatch: RGB={rgb_np.shape[:2]}, '
                f'depth={depth_np.shape[:2]}'
            )
            return

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            o3d.geometry.Image(depth_np),
            depth_scale=self._depth_scale,
            depth_trunc=self._max_depth,
            convert_rgb_to_intensity=False,
        )

        camera_pose = self._lookup_camera_pose()
        if camera_pose is None:
            return

        self._volume.integrate(rgbd, self._intrinsic, np.linalg.inv(camera_pose))
        self._integrated_count += 1
        self.get_logger().info(
            f'Integrated frame {self._integrated_count} using latest TF pose'
        )

    def _extract_and_publish_mesh(self) -> None:
        if self._integrated_count == 0:
            self.get_logger().warn('No TSDF frames integrated yet. Press p first.')
            return

        self.get_logger().info('Extracting TSDF mesh with marching cubes')
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh = self._crop_mesh_to_bounds(mesh)

        vertex_count = len(mesh.vertices)
        triangle_count = len(mesh.triangles)
        self.get_logger().info(
            f'Mesh: vertices={vertex_count}, triangles={triangle_count}'
        )

        if vertex_count == 0 or triangle_count == 0:
            self.get_logger().warn('Mesh is empty; nothing to publish')
            return

        os.makedirs(self._output_dir, exist_ok=True)
        timestamp = self.get_clock().now().nanoseconds
        mesh_path = os.path.join(self._output_dir, f'debug_tsdf_mesh_{timestamp}.ply')
        success = o3d.io.write_triangle_mesh(mesh_path, mesh)
        self.get_logger().info(f'Mesh write success={success}: {mesh_path}')

        marker = self._mesh_to_marker(mesh)
        self._mesh_pub.publish(marker)
        self.get_logger().info(f'Published mesh marker on {self._mesh_marker_topic}')

        collision_obj = self._mesh_to_collision_object(mesh)
        self._collision_pub.publish(collision_obj)
        self.get_logger().info(
            f'Published collision object "{self._collision_object_id}" to /collision_object'
        )

    def _crop_mesh_to_bounds(self, mesh):
        if self._voxel_bounds is None:
            return mesh

        bounds = self._voxel_bounds
        min_bound = [
            float(bounds['x'][0]),
            float(bounds['y'][0]),
            float(bounds['z'][0]),
        ]
        max_bound = [
            float(bounds['x'][1]),
            float(bounds['y'][1]),
            float(bounds['z'][1]),
        ]
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.asarray(min_bound, dtype=np.float64),
            max_bound=np.asarray(max_bound, dtype=np.float64),
        )
        cropped = mesh.crop(bbox)
        self.get_logger().info(
            f'Cropped mesh to {self._world_frame} bounds: '
            f'x={bounds["x"]}, y={bounds["y"]}, z={bounds["z"]}'
        )
        return cropped

    def _lookup_camera_pose(self):
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._world_frame,
                self._camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception as e:
            self.get_logger().error(
                f'TF lookup failed for {self._world_frame} -> '
                f'{self._camera_frame}: {e}'
            )
            return None

        self.get_logger().info(
            f'Using TF: {tf_msg.header.frame_id} -> {tf_msg.child_frame_id}'
        )
        return _transform_msg_to_matrix(tf_msg.transform)

    def _log_array_stats(self, name: str, array: np.ndarray) -> None:
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        if finite.size == 0:
            self.get_logger().info(
                f'{name}: shape={array.shape}, dtype={array.dtype}, no finite values'
            )
            return
        self.get_logger().info(
            f'{name}: shape={array.shape}, dtype={array.dtype}, '
            f'min={float(np.min(finite)):.6f}, max={float(np.max(finite)):.6f}'
        )

    def _mesh_to_marker(self, mesh):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

        marker = Marker()
        marker.header.frame_id = self._world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tsdf_mesh'
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.7
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        flat_idx = triangles.reshape(-1)
        flat_vertices = vertices[flat_idx]
        marker.points = [
            Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
            for v in flat_vertices
        ]

        if colors is not None and len(colors) == len(vertices):
            flat_colors = colors[flat_idx]
            marker.colors = [
                ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=1.0)
                for c in flat_colors
            ]

        return marker

    def _mesh_to_collision_object(self, mesh) -> CollisionObject:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        shape_mesh = Mesh()
        shape_mesh.vertices = [
            Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in vertices
        ]
        shape_mesh.triangles = [
            MeshTriangle(vertex_indices=[int(t[0]), int(t[1]), int(t[2])])
            for t in triangles
        ]

        identity_pose = Pose()
        identity_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        obj = CollisionObject()
        obj.header.frame_id = self._world_frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = self._collision_object_id
        obj.operation = CollisionObject.ADD
        obj.meshes = [shape_mesh]
        obj.mesh_poses = [identity_pose]
        return obj

    def _remove_collision_object(self) -> None:
        obj = CollisionObject()
        obj.header.frame_id = self._world_frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = self._collision_object_id
        obj.operation = CollisionObject.REMOVE
        self._collision_pub.publish(obj)
        self.get_logger().info(
            f'Removed collision object "{self._collision_object_id}" from planning scene'
        )

    def _setup_keyboard(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().warn(
                'stdin is not a TTY; keyboard commands may require pressing Enter'
            )
            return
        self._terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        atexit.register(self._restore_keyboard)

    def _restore_keyboard(self) -> None:
        if self._terminal_settings is None:
            return
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal_settings)
        self._terminal_settings = None

    def _request_shutdown(self) -> None:
        self.create_timer(0.1, self._shutdown)

    def _shutdown(self) -> None:
        self._restore_keyboard()
        if rclpy.ok():
            rclpy.shutdown()


def _transform_msg_to_matrix(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    qx = float(q.x)
    qy = float(q.y)
    qz = float(q.z)
    qw = float(q.w)

    mat = np.eye(4)
    mat[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    mat[0, 1] = 2.0 * (qx * qy - qw * qz)
    mat[0, 2] = 2.0 * (qx * qz + qw * qy)
    mat[1, 0] = 2.0 * (qx * qy + qw * qz)
    mat[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    mat[1, 2] = 2.0 * (qy * qz - qw * qx)
    mat[2, 0] = 2.0 * (qx * qz - qw * qy)
    mat[2, 1] = 2.0 * (qy * qz + qw * qx)
    mat[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    mat[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return mat


def _to_open3d_depth(depth: np.ndarray) -> np.ndarray:
    if np.issubdtype(depth.dtype, np.floating):
        return np.ascontiguousarray(depth, dtype=np.float32)

    # Cast integer depth to float32 — Open3D's Image constructor rejects uint16
    # with newer numpy versions due to buffer protocol incompatibility.
    # Values are preserved exactly; depth_scale still handles unit conversion.
    return np.ascontiguousarray(depth, dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = TSDFIntegratorOnce()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()


if __name__ == '__main__':
    main()
