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


STATE_FILTERS = {
    "all": {-1, 0, 1},
    "free": {-1},
    "unknown": {0},
    "occupied": {1},
    "known": {-1, 1},
}


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
        self.declare_parameter("occupancy_states", "occupied")
        self.declare_parameter("voxel_size_m", 0.02)
        self.declare_parameter("debug_point_size_m", 0.012)
        self.declare_parameter("direction_line_width_m", 0.012)
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("occupancy_ply", "")
        self.declare_parameter("tsdf_ply", "")
        self.declare_parameter("target_points_ply", "")
        self.declare_parameter("scoring_points_ply", "")
        self.declare_parameter("sweep_ply", "")
        self.declare_parameter("direction_ply", "")
        self.declare_parameter("result_json", "")

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._occupancy_pub = self.create_publisher(Marker, "/fetchbench_debug/original_occupancy", qos)
        self._tsdf_pub = self.create_publisher(Marker, "/fetchbench_debug/tsdf_without_target", qos)
        self._target_points_pub = self.create_publisher(Marker, "/fetchbench_debug/target_surface_points", qos)
        self._scoring_points_pub = self.create_publisher(Marker, "/fetchbench_debug/scoring_points_100", qos)
        self._sweep_pub = self.create_publisher(Marker, "/fetchbench_debug/sweep_points", qos)
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

    def _direction_marker_array(self, path: Path, result_json: Path) -> MarkerArray:
        array = MarkerArray()
        if not path.is_file():
            self.get_logger().warn(f"Direction PLY does not exist: {path}")
            array.markers.append(self._delete_marker("fetchbench_best_direction_ply", 0))
            array.markers.append(self._delete_marker("fetchbench_object_point", 1))
            return array

        properties, rows, edges = _read_ascii_ply(path)
        vertices = [_point(_field(row, properties, "x"), _field(row, properties, "y"), _field(row, properties, "z")) for row in rows]

        line = Marker()
        self._stamp(line)
        line.ns = "fetchbench_best_direction_ply"
        line.id = 0
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = float(self.get_parameter("direction_line_width_m").value)
        line.color = _color(255, 0, 0, 1.0)
        line.lifetime = Duration(sec=0, nanosec=0)
        for edge in edges:
            if len(edge) < 2:
                continue
            if 0 <= edge[0] < len(vertices) and 0 <= edge[1] < len(vertices):
                line.points.append(vertices[edge[0]])
                line.points.append(vertices[edge[1]])

        object_point = vertices[0] if vertices else None
        if result_json.is_file():
            try:
                data: dict[str, Any] = json.loads(result_json.read_text(encoding="utf-8"))
                value = data.get("grasp_position_world")
                if isinstance(value, list) and len(value) == 3:
                    object_point = _point(float(value[0]), float(value[1]), float(value[2]))
            except (OSError, TypeError, ValueError) as exc:
                self.get_logger().warn(f"Could not load object point: {exc}")

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

        array.markers.append(line)
        self.get_logger().info(f"Loaded output direction: {path}")
        return array

    def _build_markers(self) -> dict[str, Marker | MarkerArray]:
        exp = self._exp_dir()
        debug = exp / "directions" / "debug_geometry"
        state_filter = STATE_FILTERS.get(str(self.get_parameter("occupancy_states").value), {1})
        point_size = float(self.get_parameter("debug_point_size_m").value)
        return {
            "occupancy": self._point_marker(
                path=self._path_param("occupancy_ply", "occupancy_grid.ply"),
                ns="fetchbench_original_occupancy",
                marker_id=0,
                color=(255, 160, 40, 0.8),
                scale=float(self.get_parameter("voxel_size_m").value),
                use_vertex_colors=True,
                states=state_filter,
            ),
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
            "sweep": self._point_marker(
                path=self._path_param("sweep_ply", "directions/debug_geometry/sweep_aggregate_direction.ply"),
                ns="fetchbench_sweep_points",
                marker_id=0,
                color=(255, 255, 255, 1.0),
                scale=point_size,
                use_vertex_colors=True,
            ),
            "direction": self._direction_marker_array(
                self._path_param("direction_ply", "directions/aggregate_direction.ply"),
                self._path_param("result_json", "directions/final_3d_direction.json"),
            ),
        }

    def _publish(self) -> None:
        for key in ("occupancy", "tsdf", "target", "scoring", "sweep"):
            marker = self._markers[key]
            if isinstance(marker, Marker):
                self._stamp(marker)
        direction = self._markers["direction"]
        if isinstance(direction, MarkerArray):
            for marker in direction.markers:
                self._stamp(marker)
            self._direction_pub.publish(direction)
        self._occupancy_pub.publish(self._markers["occupancy"])
        self._tsdf_pub.publish(self._markers["tsdf"])
        self._target_points_pub.publish(self._markers["target"])
        self._scoring_points_pub.publish(self._markers["scoring"])
        self._sweep_pub.publish(self._markers["sweep"])


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
