#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


STATE_FILTERS = {
    "all": {-1, 0, 1},
    "free": {-1},
    "unknown": {0},
    "occupied": {1},
    "known": {-1, 1},
}


def read_ply_header(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8") as f:
        header = []
        header_lines = 0
        for line in f:
            header_lines += 1
            header.append(line.strip())
            if line.strip() == "end_header":
                break
    return header, header_lines


def read_fetchbench_occupancy_ply(path: Path, states: set[int]) -> o3d.geometry.PointCloud:
    header, header_lines = read_ply_header(path)
    if "property int state" not in header:
        raise ValueError("Expected occupancy PLY with columns: x y z state red green blue")
    data = np.loadtxt(path, skiprows=header_lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError("Expected PLY columns: x y z state red green blue")

    mask = np.isin(data[:, 3].astype(int), list(states))
    data = data[mask]
    if data.size == 0:
        raise ValueError("No points matched the selected state filter")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(data[:, 4:7] / 255.0)
    return cloud


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Path to occupancy_final.ply")
    parser.add_argument(
        "--states",
        choices=sorted(STATE_FILTERS),
        default="all",
        help="Voxel states to show: free=-1, unknown=0, occupied=1",
    )
    parser.add_argument("--point-size", type=float, default=5.0)
    args = parser.parse_args()

    cloud = read_fetchbench_occupancy_ply(Path(args.ply), STATE_FILTERS[args.states])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Occupancy: {args.states}")
    vis.add_geometry(cloud)
    vis.add_geometry(axis)
    render = vis.get_render_option()
    render.point_size = float(args.point_size)
    render.background_color = np.array([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
