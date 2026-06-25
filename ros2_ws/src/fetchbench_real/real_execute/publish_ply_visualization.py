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


def _experiment_path(node: Node, experiment_param: str, suffix: str) -> Path:
    output_root = Path(str(node.get_parameter("output_root").value))
    experiment_name = str(node.get_parameter(experiment_param).value)
    return output_root / experiment_name / suffix


class PlyVisualizationPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fetchbench_publish_ply_visualization")

        self.declare_parameter("output_root", "/workspace/ros2_ws/ours_experiment")
        self.declare_parameter("experiment_name", "ex1")
        self.declare_parameter("occupancy_ply", "")
        self.declare_parameter("direction_ply", "")
        self.declare_parameter("result_json", "")
        self.declare_parameter("publish_occupancy", True)
        self.declare_parameter("publish_direction_set", True)
        self.declare_parameter("occupancy_states", "all")
        self.declare_parameter("frame_id", "panda_link0")
        self.declare_parameter("voxel_size_m", 0.02)
        self.declare_parameter("object_point_size_m", 0.035)
        self.declare_parameter("direction_line_width_m", 0.012)
        self.declare_parameter("free_alpha", 0.04)
        self.declare_parameter("unknown_alpha", 0.06)
        self.declare_parameter("occupied_alpha", 0.90)
        self.declare_parameter("publish_period_sec", 1.0)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._occupancy_pub = self.create_publisher(
            Marker,
            "/fetchbench_execute/occupancy_ply",
            qos,
        )
        self._direction_pub = self.create_publisher(
            MarkerArray,
            "/fetchbench_execute/best_direction_set",
            qos,
        )

        self._occupancy_marker = self._build_occupancy_marker()
        self._direction_markers = self._build_direction_marker_array()

        period = max(0.1, float(self.get_parameter("publish_period_sec").value))
        self.create_timer(period, self._publish)
        self._publish()

    def _resolve_occupancy_ply(self) -> Path:
        explicit = str(self.get_parameter("occupancy_ply").value or "")
        if explicit:
            return Path(explicit)
        return _experiment_path(self, "experiment_name", "occupancy_final.ply")

    def _resolve_direction_ply(self) -> Path:
        explicit = str(self.get_parameter("direction_ply").value or "")
        if explicit:
            return Path(explicit)
        return _experiment_path(self, "experiment_name", "offline/best_direction.ply")

    def _resolve_result_json(self) -> Path:
        explicit = str(self.get_parameter("result_json").value or "")
        if explicit:
            return Path(explicit)
        return _experiment_path(self, "experiment_name", "offline/final_3d_direction.json")

    def _stamp_and_frame(self, marker: Marker) -> None:
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(self.get_parameter("frame_id").value)

    def _delete_marker(self, ns: str, marker_id: int) -> Marker:
        marker = Marker()
        self._stamp_and_frame(marker)
        marker.ns = ns
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def _build_occupancy_marker(self) -> Marker:
        if not bool(self.get_parameter("publish_occupancy").value):
            return self._delete_marker("fetchbench_occupancy_ply", 0)

        path = self._resolve_occupancy_ply()
        if not path.is_file():
            self.get_logger().error(f"Occupancy PLY does not exist: {path}")
            return self._delete_marker("fetchbench_occupancy_ply", 0)

        state_filter_name = str(self.get_parameter("occupancy_states").value)
        states = STATE_FILTERS.get(state_filter_name)
        if states is None:
            self.get_logger().error(
                f"Invalid occupancy_states='{state_filter_name}'. Use one of {sorted(STATE_FILTERS)}"
            )
            return self._delete_marker("fetchbench_occupancy_ply", 0)

        properties, rows, _ = _read_ascii_ply(path)
        marker = Marker()
        self._stamp_and_frame(marker)
        marker.ns = "fetchbench_occupancy_ply"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(sec=0, nanosec=0)
        voxel_size = float(self.get_parameter("voxel_size_m").value)
        marker.scale.x = voxel_size
        marker.scale.y = voxel_size
        marker.scale.z = voxel_size

        alpha_by_state = {
            -1: float(self.get_parameter("free_alpha").value),
            0: float(self.get_parameter("unknown_alpha").value),
            1: float(self.get_parameter("occupied_alpha").value),
        }

        for row in rows:
            state = int(_field(row, properties, "state", 1.0))
            if state not in states:
                continue
            marker.points.append(
                _point(
                    _field(row, properties, "x"),
                    _field(row, properties, "y"),
                    _field(row, properties, "z"),
                )
            )
            marker.colors.append(
                _color(
                    _field(row, properties, "red", 255.0),
                    _field(row, properties, "green", 255.0),
                    _field(row, properties, "blue", 255.0),
                    alpha_by_state.get(state, 1.0),
                )
            )

        self.get_logger().info(
            f"Loaded occupancy PLY: {path}  points={len(marker.points)}  states={state_filter_name}"
        )
        return marker

    def _load_object_point(self, direction_vertices: list[Point]) -> Point | None:
        path = self._resolve_result_json()
        if path.is_file():
            try:
                data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
                value = data.get("grasp_position_world")
                if isinstance(value, list) and len(value) == 3:
                    return _point(float(value[0]), float(value[1]), float(value[2]))
            except (OSError, ValueError, TypeError) as exc:
                self.get_logger().warn(f"Could not read object point from {path}: {exc}")
        if direction_vertices:
            return direction_vertices[0]
        return None

    def _build_direction_marker_array(self) -> MarkerArray:
        array = MarkerArray()
        if not bool(self.get_parameter("publish_direction_set").value):
            array.markers.append(self._delete_marker("fetchbench_best_direction_ply", 0))
            array.markers.append(self._delete_marker("fetchbench_object_point", 1))
            return array

        path = self._resolve_direction_ply()
        if not path.is_file():
            self.get_logger().error(f"Direction PLY does not exist: {path}")
            array.markers.append(self._delete_marker("fetchbench_best_direction_ply", 0))
            array.markers.append(self._delete_marker("fetchbench_object_point", 1))
            return array

        properties, rows, edges = _read_ascii_ply(path)
        vertices = [
            _point(
                _field(row, properties, "x"),
                _field(row, properties, "y"),
                _field(row, properties, "z"),
            )
            for row in rows
        ]

        line = Marker()
        self._stamp_and_frame(line)
        line.ns = "fetchbench_best_direction_ply"
        line.id = 0
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = float(self.get_parameter("direction_line_width_m").value)
        line.color = _color(255.0, 0.0, 0.0, 1.0)
        line.lifetime = Duration(sec=0, nanosec=0)

        for edge in edges:
            if len(edge) < 2:
                continue
            start_idx, end_idx = edge[0], edge[1]
            if 0 <= start_idx < len(vertices) and 0 <= end_idx < len(vertices):
                line.points.append(vertices[start_idx])
                line.points.append(vertices[end_idx])

        object_point = self._load_object_point(vertices)
        if object_point is not None:
            sphere = Marker()
            self._stamp_and_frame(sphere)
            sphere.ns = "fetchbench_object_point"
            sphere.id = 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = object_point
            sphere.pose.orientation.w = 1.0
            size = float(self.get_parameter("object_point_size_m").value)
            sphere.scale.x = size
            sphere.scale.y = size
            sphere.scale.z = size
            sphere.color = _color(255.0, 255.0, 0.0, 1.0)
            sphere.lifetime = Duration(sec=0, nanosec=0)
            array.markers.append(sphere)

        array.markers.append(line)
        self.get_logger().info(
            f"Loaded direction PLY: {path}  vertices={len(vertices)}  edges={len(edges)}"
        )
        return array

    def _publish(self) -> None:
        self._stamp_and_frame(self._occupancy_marker)
        for marker in self._direction_markers.markers:
            self._stamp_and_frame(marker)
        self._occupancy_pub.publish(self._occupancy_marker)
        self._direction_pub.publish(self._direction_markers)


def main() -> None:
    rclpy.init()
    node = PlyVisualizationPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
