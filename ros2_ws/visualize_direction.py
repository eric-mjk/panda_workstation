#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def read_ply_header(path: Path) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8") as f:
        header_lines = 0
        vertex_count = 0
        edge_count = 0
        for line in f:
            header_lines += 1
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif len(parts) == 3 and parts[:2] == ["element", "edge"]:
                edge_count = int(parts[2])
            elif line.strip() == "end_header":
                break
    return header_lines, vertex_count, edge_count


def read_direction_ply(path: Path) -> o3d.geometry.LineSet:
    header_lines, vertex_count, edge_count = read_ply_header(path)
    rows = path.read_text(encoding="utf-8").splitlines()[header_lines:]

    vertices = np.array(
        [[float(v) for v in row.split()[:6]] for row in rows[:vertex_count]],
        dtype=np.float64,
    )
    edges = np.array(
        [[int(v) for v in row.split()[:2]] for row in rows[vertex_count : vertex_count + edge_count]],
        dtype=np.int32,
    )
    if vertex_count == 0 or edge_count == 0 or vertices.shape[1] < 6 or edges.shape[1] < 2:
        raise ValueError("Expected direction PLY with colored vertices and edge indices")

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(vertices[:, :3])
    lines.lines = o3d.utility.Vector2iVector(edges[:, :2])
    line_color = vertices[0, 3:6] / 255.0
    lines.colors = o3d.utility.Vector3dVector(np.tile(line_color, (edge_count, 1)))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Path to best_direction.ply")
    parser.add_argument("--line-width", type=float, default=5.0)
    args = parser.parse_args()

    direction = read_direction_ply(Path(args.ply))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Direction: {Path(args.ply).name}")
    vis.add_geometry(direction)
    vis.add_geometry(axis)
    render = vis.get_render_option()
    render.line_width = float(args.line_width)
    render.background_color = np.array([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
