#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def _read_ascii_ply(path: Path) -> tuple[list[str], list[list[float]], list[list[int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = 0
    edge_count = 0
    vertex_properties: list[str] = []
    current_element = None
    header_end = None

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            current_element = parts[1]
            if parts[1] == "vertex":
                vertex_count = int(parts[2])
            elif parts[1] == "edge":
                edge_count = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and current_element == "vertex":
            vertex_properties.append(parts[-1])
        elif line.strip() == "end_header":
            header_end = i + 1
            break

    if header_end is None:
        raise ValueError(f"PLY header has no end_header: {path}")

    vertex_rows = [
        [float(v) for v in row.split()]
        for row in lines[header_end : header_end + vertex_count]
        if row.strip()
    ]
    edge_start = header_end + vertex_count
    edge_rows = [
        [int(v) for v in row.split()[:2]]
        for row in lines[edge_start : edge_start + edge_count]
        if row.strip()
    ]
    return vertex_properties, vertex_rows, edge_rows


def _field(row: list[float], properties: list[str], name: str, default: float = 0.0) -> float:
    try:
        return row[properties.index(name)]
    except ValueError:
        return default


def _color(r: float, g: float, b: float, a: float) -> ColorRGBA:
    msg = ColorRGBA()
    msg.r = float(r) / 255.0
    msg.g = float(g) / 255.0
    msg.b = float(b) / 255.0
    msg.a = float(a)
    return msg


def _point(x: float, y: float, z: float) -> Point:
    msg = Point()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    return msg


class DebugGeometryPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fetchbench_publish_debug_geometry")

        self.declare_parameter("output_root", "/workspace/ros2_ws/ours_experiment")
        self.declare_parameter("experiment_name", "ex1")
        self.declare_parameter("frame_id", "panda_link0")
        self.declare_parameter("voxel_size_m", 0.02)
        self.declare_parameter("debug_point_size_m", 0.012)
        self.declare_parameter("direction_line_width_m", 0.012)
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("tsdf_ply", "")
        self.declare_parameter("target_points_ply", "")
        self.declare_parameter("scoring_points_ply", "")
        self.declare_parameter("result_json", "")

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._tsdf_pub = self.create_publisher(Marker, "/fetchbench_debug/tsdf_without_target", qos)
        self._target_points_pub = self.create_publisher(Marker, "/fetchbench_debug/target_surface_points", qos)
        self._scoring_points_pub = self.create_publisher(Marker, "/fetchbench_debug/scoring_points_100", qos)
        self._direction_pub = self.create_publisher(MarkerArray, "/fetchbench_execute/best_direction_set", qos)

        self._markers = self._build_markers()
        period = max(0.1, float(self.get_parameter("publish_period_sec").value))
        self.create_timer(period, self._publish)
        self._publish()

    def _exp_dir(self) -> Path:
        return Path(str(self.get_parameter("output_root").value)) / str(self.get_parameter("experiment_name").value)

    def _path_param(self, name: str, default_suffix: str) -> Path:
        explicit = str(self.get_parameter(name).value or "")
        if explicit:
            return Path(explicit)
        return self._exp_dir() / default_suffix

    def _stamp(self, marker: Marker) -> None:
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(self.get_parameter("frame_id").value)

    def _delete_marker(self, ns: str, marker_id: int) -> Marker:
        marker = Marker()
        self._stamp(marker)
        marker.ns = ns
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def _point_marker(
        self,
        *,
        path: Path,
        ns: str,
        marker_id: int,
        color: tuple[int, int, int, float],
        scale: float,
        use_vertex_colors: bool = True,
        states: set[int] | None = None,
    ) -> Marker:
        if not path.is_file():
            self.get_logger().warn(f"PLY does not exist: {path}")
            return self._delete_marker(ns, marker_id)
        properties, rows, _ = _read_ascii_ply(path)
        marker = Marker()
        self._stamp(marker)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(scale)
        marker.scale.y = float(scale)
        marker.scale.z = float(scale)
        marker.lifetime = Duration(sec=0, nanosec=0)

        fixed_color = _color(*color)
        for row in rows:
            if states is not None:
                state = int(_field(row, properties, "state", 1.0))
                if state not in states:
                    continue
            marker.points.append(_point(_field(row, properties, "x"), _field(row, properties, "y"), _field(row, properties, "z")))
            if use_vertex_colors:
                marker.colors.append(
                    _color(
                        _field(row, properties, "red", float(color[0])),
                        _field(row, properties, "green", float(color[1])),
                        _field(row, properties, "blue", float(color[2])),
                        float(color[3]),
                    )
                )
            else:
                marker.colors.append(fixed_color)
        self.get_logger().info(f"Loaded {ns}: {path} points={len(marker.points)}")
        return marker

    def _arrow_marker(
        self,
        *,
        ns: str,
        marker_id: int,
        start: Point,
        direction: list[float],
        color: tuple[int, int, int, float],
        length_m: float = 0.20,
    ) -> Marker:
        norm = sum(float(v) * float(v) for v in direction) ** 0.5
        if norm <= 1e-12:
            return self._delete_marker(ns, marker_id)
        end = _point(
            start.x + float(length_m) * float(direction[0]) / norm,
            start.y + float(length_m) * float(direction[1]) / norm,
            start.z + float(length_m) * float(direction[2]) / norm,
        )
        marker = Marker()
        self._stamp(marker)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.points.append(start)
        marker.points.append(end)
        width = float(self.get_parameter("direction_line_width_m").value)
        marker.scale.x = width
        marker.scale.y = width * 2.8
        marker.scale.z = width * 2.2
        marker.color = _color(*color)
        marker.lifetime = Duration(sec=0, nanosec=0)
        return marker

    def _direction_marker_array(self, result_json: Path) -> MarkerArray:
        array = MarkerArray()
        delete_specs = (
            ("fetchbench_geometry_direction", 10),
            ("fetchbench_vlm_direction", 11),
            ("fetchbench_aggregate_direction", 12),
        )
        array.markers.append(self._delete_marker("fetchbench_best_direction_ply", 0))
        if not result_json.is_file():
            self.get_logger().warn(f"Direction JSON does not exist: {result_json}")
            for ns, marker_id in delete_specs:
                array.markers.append(self._delete_marker(ns, marker_id))
            array.markers.append(self._delete_marker("fetchbench_object_point", 1))
            return array

        try:
            data: dict[str, Any] = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"Could not load direction JSON: {exc}")
            for ns, marker_id in delete_specs:
                array.markers.append(self._delete_marker(ns, marker_id))
            array.markers.append(self._delete_marker("fetchbench_object_point", 1))
            return array

        value = data.get("grasp_position_world")
        object_point = None
        if isinstance(value, list) and len(value) == 3:
            object_point = _point(float(value[0]), float(value[1]), float(value[2]))

        if object_point is not None:
            sphere = Marker()
            self._stamp(sphere)
            sphere.ns = "fetchbench_object_point"
            sphere.id = 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = object_point
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.035
            sphere.scale.y = 0.035
            sphere.scale.z = 0.035
            sphere.color = _color(255, 255, 0, 1.0)
            sphere.lifetime = Duration(sec=0, nanosec=0)
            array.markers.append(sphere)

        if object_point is None:
            self.get_logger().warn(f"No grasp_position_world in {result_json}")
            return array

        specs = (
            ("fetchbench_geometry_direction", 10, "geometry_only_direction", (0, 220, 80, 1.0)),
            ("fetchbench_vlm_direction", 11, "vlm_only_direction", (0, 120, 255, 1.0)),
            ("fetchbench_aggregate_direction", 12, "aggregate_direction", (255, 40, 40, 1.0)),
        )
        for ns, marker_id, key, color in specs:
            direction = data.get(key)
            if isinstance(direction, list) and len(direction) == 3:
                array.markers.append(
                    self._arrow_marker(
                        ns=ns,
                        marker_id=marker_id,
                        start=object_point,
                        direction=[float(v) for v in direction],
                        color=color,
                    )
                )
            else:
                array.markers.append(self._delete_marker(ns, marker_id))

        self.get_logger().info(f"Loaded output directions: {result_json}")
        return array

    def _build_markers(self) -> dict[str, Marker | MarkerArray]:
        point_size = float(self.get_parameter("debug_point_size_m").value)
        return {
            "tsdf": self._point_marker(
                path=self._path_param("tsdf_ply", "directions/debug_geometry/tsdf_without_target.ply"),
                ns="fetchbench_tsdf_without_target",
                marker_id=0,
                color=(80, 180, 255, 0.45),
                scale=point_size,
                use_vertex_colors=False,
            ),
            "target": self._point_marker(
                path=self._path_param("target_points_ply", "directions/debug_geometry/target_surface_points.ply"),
                ns="fetchbench_target_surface_points",
                marker_id=0,
                color=(0, 220, 80, 1.0),
                scale=point_size,
                use_vertex_colors=True,
            ),
            "scoring": self._point_marker(
                path=self._path_param("scoring_points_ply", "directions/debug_geometry/scoring_points_100.ply"),
                ns="fetchbench_scoring_points_100",
                marker_id=0,
                color=(255, 0, 255, 1.0),
                scale=point_size * 1.8,
                use_vertex_colors=True,
            ),
            # Sweep-point visualization is intentionally disabled here; it is too dense for routine RViz debugging.
            # To inspect sweep PLYs, use an offline PLY viewer directly on directions/debug_geometry/sweep_*.ply.
            "direction": self._direction_marker_array(
                self._path_param("result_json", "directions/final_3d_direction.json"),
            ),
        }

    def _publish(self) -> None:
        for key in ("tsdf", "target", "scoring"):
            marker = self._markers[key]
            if isinstance(marker, Marker):
                self._stamp(marker)
        direction = self._markers["direction"]
        if isinstance(direction, MarkerArray):
            for marker in direction.markers:
                self._stamp(marker)
            self._direction_pub.publish(direction)
        self._tsdf_pub.publish(self._markers["tsdf"])
        self._target_points_pub.publish(self._markers["target"])
        self._scoring_points_pub.publish(self._markers["scoring"])


def main() -> None:
    rclpy.init()
    node = DebugGeometryPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
