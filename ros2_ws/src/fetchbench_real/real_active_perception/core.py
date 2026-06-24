from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROI_X_RANGE = (0.20, 0.75)
DEFAULT_ROI_Y_RANGE = (-0.35, 0.35)
DEFAULT_ROI_Z_RANGE = (-0.05, 0.35)
DEFAULT_VOXEL_SIZE_M = 0.02
DEFAULT_Z_PLANE_M = 0.0
DEFAULT_MIN_DEPTH_M = 0.10
DEFAULT_MAX_DEPTH_M = 1.50
DEFAULT_UNKNOWN_SCORE_MAX_DISTANCE_M = DEFAULT_VOXEL_SIZE_M * 10
DEFAULT_UNRESOLVED_TARGET_PENALTY = 0.3
DEFAULT_MAX_SCORING_VOXELS = 1200


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def default_candidates_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "view_candidates"
        / "view_candidates.json"
    )


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray | list[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_msg_to_matrix(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_xyzw_to_rotmat([q.x, q.y, q.z, q.w])
    mat[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return mat


def intrinsics_matrix_from_camera_info(camera_info) -> np.ndarray:
    k = camera_info.k
    return np.asarray(
        [
            [float(k[0]), 0.0, float(k[2])],
            [0.0, float(k[4]), float(k[5])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def candidate_pose(candidate: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    pos = candidate.get("cam_position")
    quat = candidate.get("cam_quaternion_xyzw")
    if pos is None or quat is None:
        return None, None
    pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
    if pos_arr.shape[0] != 3 or quat_arr.shape[0] != 4:
        return None, None
    return pos_arr, quat_arr


def candidate_has_joint_angles(candidate: dict[str, Any], joint_dof_count: int | None = 7) -> bool:
    joints = np.asarray(candidate.get("joint_angles", []), dtype=np.float64).reshape(-1)
    return joints.shape[0] > 0 if joint_dof_count is None else joints.shape[0] == int(joint_dof_count)


def load_candidates(path: str | Path, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = load_json(path)
    raw_candidates = doc.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"No candidates found in {path}")

    candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        pos, quat = candidate_pose(candidate)
        if pos is None or quat is None:
            continue
        candidate = dict(candidate)
        candidate["cam_position"] = [float(v) for v in pos]
        candidate["cam_quaternion_xyzw"] = [float(v) for v in quat]
        candidates.append(candidate)
        if len(candidates) >= max(1, int(top_k)):
            break

    if not candidates:
        raise ValueError(f"No valid candidates after schema check in {path}")
    return candidates, doc


def _raycast_unblocked_mask_python(
    grid: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    voxel_size: float,
    camera_xyz: np.ndarray,
    target_points_xyz: np.ndarray,
    step_len: float,
    end_margin: float,
) -> np.ndarray:
    mask = np.ones((target_points_xyz.shape[0],), dtype=bool)
    nz, ny, nx = grid.shape
    for i, target in enumerate(target_points_xyz):
        vec = target - camera_xyz
        dist = float(np.linalg.norm(vec))
        if dist <= end_margin:
            continue
        direction = vec / max(dist, 1e-12)
        n_steps = int(max(1, math.floor((dist - end_margin) / max(step_len, 1e-6))))
        for step in range(1, n_steps + 1):
            point = camera_xyz + direction * (step * step_len)
            ix = int(math.floor((float(point[0]) - x0) / voxel_size))
            iy = int(math.floor((float(point[1]) - y0) / voxel_size))
            iz = int(math.floor((float(point[2]) - z0) / voxel_size))
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and int(grid[iz, iy, ix]) == 1:
                mask[i] = False
                break
    return mask


class VoxelAccumulator:
    """Depth-integrated occupancy grid.

    Grid state matches the original active-perception code:
    -1 free, 0 unknown, 1 occupied.
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        voxel_size: float,
        z_plane: float,
        min_depth_m: float,
        max_depth_m: float,
        pixel_stride: int,
        min_component_voxels: int,
    ) -> None:
        self.x_range = tuple(float(v) for v in x_range)
        self.y_range = tuple(float(v) for v in y_range)
        self.z_range = tuple(float(v) for v in z_range)
        self.voxel_size = float(voxel_size)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.pixel_stride = max(1, int(pixel_stride))
        self.min_component_voxels = max(1, int(min_component_voxels))

        self.nx = int(np.ceil((self.x_range[1] - self.x_range[0]) / self.voxel_size))
        self.ny = int(np.ceil((self.y_range[1] - self.y_range[0]) / self.voxel_size))
        self.nz = int(np.ceil((self.z_range[1] - self.z_range[0]) / self.voxel_size))
        self.grid = np.zeros((self.nz, self.ny, self.nx), dtype=np.int8)
        self.log_odds = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.checked_region = np.zeros((self.nz, self.ny, self.nx), dtype=bool)

        self.l_hit = 2.0
        self.l_free = 0.40
        self.l_occ_thresh = 1.0
        self.l_clamp = 10.0
        self.update_count = 0
        self.last_camera_position: np.ndarray | None = None
        self.last_update_stats: dict[str, int] = {}
        self.ground_layer_index: int | None = None
        if self.z_range[0] <= float(z_plane) < self.z_range[1]:
            idx = int(np.floor((float(z_plane) - self.z_range[0]) / self.voxel_size))
            if 0 <= idx < self.nz:
                self.ground_layer_index = idx
        self._apply_ground_plane()

    def _apply_ground_plane(self) -> None:
        if self.ground_layer_index is not None:
            self.grid[: self.ground_layer_index + 1, :, :] = 1

    def world_to_voxel(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        ix = np.floor((pts[:, 0] - self.x_range[0]) / self.voxel_size).astype(np.int32)
        iy = np.floor((pts[:, 1] - self.y_range[0]) / self.voxel_size).astype(np.int32)
        iz = np.floor((pts[:, 2] - self.z_range[0]) / self.voxel_size).astype(np.int32)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (iz >= 0) & (iz < self.nz)
        return np.stack((ix, iy, iz), axis=1), valid

    def voxel_center(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return np.asarray(
            [
                self.x_range[0] + (float(ix) + 0.5) * self.voxel_size,
                self.y_range[0] + (float(iy) + 0.5) * self.voxel_size,
                self.z_range[0] + (float(iz) + 0.5) * self.voxel_size,
            ],
            dtype=np.float32,
        )

    def iter_world_points(self, state_value: int, stride: int = 1) -> np.ndarray:
        indices = np.argwhere(self.grid == int(state_value))
        if self.ground_layer_index is not None:
            indices = indices[indices[:, 0] > self.ground_layer_index]
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        stride = max(1, int(stride))
        if stride > 1:
            indices = indices[
                (indices[:, 0] % stride == 0)
                & (indices[:, 1] % stride == 0)
                & (indices[:, 2] % stride == 0)
            ]
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = np.empty((indices.shape[0], 3), dtype=np.float32)
        pts[:, 0] = self.x_range[0] + (indices[:, 2] + 0.5) * self.voxel_size
        pts[:, 1] = self.y_range[0] + (indices[:, 1] + 0.5) * self.voxel_size
        pts[:, 2] = self.z_range[0] + (indices[:, 0] + 0.5) * self.voxel_size
        return pts

    def _mark_free_ray(self, camera_xyz: np.ndarray, endpoint_xyz: np.ndarray) -> None:
        vec = endpoint_xyz - camera_xyz
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return
        steps = max(1, int(np.ceil(dist / (self.voxel_size * 0.75))))
        for step in range(steps):
            point = camera_xyz + (step / steps) * vec
            idxs, valid = self.world_to_voxel(point.reshape(1, 3))
            if not valid[0]:
                continue
            ix, iy, iz = [int(v) for v in idxs[0]]
            self.checked_region[iz, iy, ix] = True
            self.log_odds[iz, iy, ix] = max(-self.l_clamp, float(self.log_odds[iz, iy, ix]) - self.l_free)

    def raycast_unblocked_mask(self, camera_xyz: np.ndarray, target_points_xyz: np.ndarray) -> np.ndarray:
        points = np.ascontiguousarray(np.asarray(target_points_xyz, dtype=np.float64).reshape(-1, 3))
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        camera = np.ascontiguousarray(np.asarray(camera_xyz, dtype=np.float64).reshape(3))
        return _raycast_unblocked_mask_python(
            self.grid,
            float(self.x_range[0]),
            float(self.y_range[0]),
            float(self.z_range[0]),
            float(self.voxel_size),
            camera,
            points,
            float(max(self.voxel_size * 0.5, 1e-4)),
            float(self.voxel_size * 0.5),
        )

    def _clear_camera_sphere(self, camera_xyz: np.ndarray, radius_m: float = 0.05) -> None:
        idxs, valid = self.world_to_voxel(camera_xyz.reshape(1, 3))
        if not valid[0]:
            return
        cix, ciy, ciz = [int(v) for v in idxs[0]]
        r_vox = int(np.ceil(radius_m / self.voxel_size))
        for iz in range(max(0, ciz - r_vox), min(self.nz, ciz + r_vox + 1)):
            for iy in range(max(0, ciy - r_vox), min(self.ny, ciy + r_vox + 1)):
                for ix in range(max(0, cix - r_vox), min(self.nx, cix + r_vox + 1)):
                    if (ix - cix) ** 2 + (iy - ciy) ** 2 + (iz - ciz) ** 2 <= r_vox**2:
                        self.log_odds[iz, iy, ix] = min(float(self.log_odds[iz, iy, ix]), -self.l_free)
                        self.grid[iz, iy, ix] = -1

    def _prune_occupied_components(self, occupied: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import label
        except Exception:
            return occupied
        labeled, num_features = label(occupied, structure=np.ones((3, 3, 3), dtype=bool))
        if num_features <= 0:
            return np.zeros_like(occupied, dtype=bool)
        counts = np.bincount(labeled.ravel())
        keep_labels = np.where(counts >= self.min_component_voxels)[0]
        keep_labels = keep_labels[keep_labels > 0]
        if keep_labels.size == 0:
            return np.zeros_like(occupied, dtype=bool)
        return np.isin(labeled, keep_labels)

    def _apply_log_odds_to_grid(self) -> None:
        occupied = self._prune_occupied_components(self.log_odds >= self.l_occ_thresh)
        free = self.log_odds < 0.0
        self.grid[:] = 0
        self.grid[free] = -1
        self.grid[occupied] = 1
        self._apply_ground_plane()

    def update(self, depth_m: np.ndarray, cam_to_world: np.ndarray, intrinsics: np.ndarray) -> dict[str, int]:
        self.update_count += 1
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            self.last_update_stats = {"valid_depth_samples": 0, "inside_roi_samples": 0, "unique_endpoint_voxels": 0}
            return self.counts()

        height, width = depth.shape
        u_coords = np.arange(0, width, self.pixel_stride, dtype=np.int32)
        v_coords = np.arange(0, height, self.pixel_stride, dtype=np.int32)
        uu, vv = np.meshgrid(u_coords, v_coords)
        z = depth[vv, uu]
        valid = np.isfinite(z) & (z >= self.min_depth_m) & (z <= self.max_depth_m)
        valid_depth_samples = int(np.count_nonzero(valid))
        self.last_camera_position = cam_to_world[:3, 3].astype(np.float32)
        if not np.any(valid):
            self.last_update_stats = {"valid_depth_samples": 0, "inside_roi_samples": 0, "unique_endpoint_voxels": 0}
            return self.counts()

        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        x = (uu.astype(np.float32) - cx) * z / max(fx, 1e-6)
        y = (vv.astype(np.float32) - cy) * z / max(fy, 1e-6)
        points_cam = np.stack((x[valid], y[valid], z[valid], np.ones(np.count_nonzero(valid), dtype=np.float32)), axis=0)
        points_world = (cam_to_world @ points_cam).T[:, :3]

        idxs, inside = self.world_to_voxel(points_world)
        if not np.any(inside):
            self.last_update_stats = {
                "valid_depth_samples": valid_depth_samples,
                "inside_roi_samples": 0,
                "unique_endpoint_voxels": 0,
            }
            return self.counts()

        camera_xyz = cam_to_world[:3, 3].astype(np.float32)
        endpoint_indices = np.unique(idxs[inside], axis=0)
        self.last_update_stats = {
            "valid_depth_samples": valid_depth_samples,
            "inside_roi_samples": int(np.count_nonzero(inside)),
            "unique_endpoint_voxels": int(endpoint_indices.shape[0]),
        }
        camera_idx_arr, camera_inside = self.world_to_voxel(camera_xyz.reshape(1, 3))
        camera_idx = tuple(int(v) for v in camera_idx_arr[0]) if camera_inside[0] else None

        endpoint_voxels: list[tuple[int, int, int]] = []
        for ix, iy, iz in endpoint_indices:
            ix, iy, iz = int(ix), int(iy), int(iz)
            if camera_idx is not None and (ix, iy, iz) == camera_idx:
                continue
            endpoint = self.voxel_center(ix, iy, iz)
            self._mark_free_ray(camera_xyz, endpoint)
            endpoint_voxels.append((ix, iy, iz))

        for ix, iy, iz in endpoint_voxels:
            self.checked_region[iz, iy, ix] = True
            self.log_odds[iz, iy, ix] = min(self.l_clamp, float(self.log_odds[iz, iy, ix]) + self.l_hit)

        self._clear_camera_sphere(camera_xyz)
        self._apply_log_odds_to_grid()
        self._clear_camera_sphere(camera_xyz)
        return self.counts()

    def counts(self) -> dict[str, int]:
        return {
            "free": int(np.count_nonzero(self.grid == -1)),
            "unknown": int(np.count_nonzero(self.grid == 0)),
            "occupied": int(np.count_nonzero(self.grid == 1)),
            "checked": int(np.count_nonzero(self.checked_region)),
            "total": int(self.grid.size),
        }

    def write_ply(self, output_path: str | Path, include_unknown: bool = False) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = []
        colors = []
        states = []
        for state, color in [(-1, (70, 150, 255)), (1, (255, 160, 40))]:
            pts = self.iter_world_points(state, stride=1)
            if pts.shape[0] > 0:
                chunks.append(pts)
                colors.append(np.tile(np.asarray(color, dtype=np.uint8), (pts.shape[0], 1)))
                states.append(np.full((pts.shape[0],), state, dtype=np.int32))
        if include_unknown:
            pts = self.iter_world_points(0, stride=1)
            if pts.shape[0] > 0:
                chunks.append(pts)
                colors.append(np.tile(np.asarray((90, 90, 90), dtype=np.uint8), (pts.shape[0], 1)))
                states.append(np.zeros((pts.shape[0],), dtype=np.int32))
        if chunks:
            points = np.concatenate(chunks, axis=0)
            rgb = np.concatenate(colors, axis=0)
            state_values = np.concatenate(states, axis=0)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            rgb = np.empty((0, 3), dtype=np.uint8)
            state_values = np.empty((0,), dtype=np.int32)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property int state\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c, s in zip(points, rgb, state_values):
                f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(s)} {int(c[0])} {int(c[1])} {int(c[2])}\n")


class UnknownVoxelScorer:
    def __init__(
        self,
        min_threshold: float = DEFAULT_VOXEL_SIZE_M,
        max_threshold: float = DEFAULT_UNKNOWN_SCORE_MAX_DISTANCE_M,
        unresolved_target_penalty: float = DEFAULT_UNRESOLVED_TARGET_PENALTY,
    ) -> None:
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.unresolved_target_penalty = float(unresolved_target_penalty)
        self.penalty_multiplier: np.ndarray | None = None
        self.pending_targeted_mask: np.ndarray | None = None
        self.pending_targeted_weights: np.ndarray | None = None
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.latest_scores = np.empty((0,), dtype=np.float32)

    def record_targeted_voxels(
        self,
        accumulator: VoxelAccumulator,
        visible_unknown_points: np.ndarray,
        target_weights: np.ndarray | None = None,
    ) -> None:
        if self.penalty_multiplier is None or self.penalty_multiplier.shape != accumulator.grid.shape:
            self.penalty_multiplier = np.ones(accumulator.grid.shape, dtype=np.float32)
        if visible_unknown_points.shape[0] == 0:
            self.pending_targeted_mask = None
            self.pending_targeted_weights = None
            return
        idxs, valid = accumulator.world_to_voxel(visible_unknown_points)
        mask = np.zeros(accumulator.grid.shape, dtype=bool)
        weights_grid = np.zeros(accumulator.grid.shape, dtype=np.float32)
        idxs = idxs[valid]
        if target_weights is None:
            weights = np.ones((visible_unknown_points.shape[0],), dtype=np.float32)
        else:
            weights = np.asarray(target_weights, dtype=np.float32).reshape(-1)
            if weights.shape[0] != visible_unknown_points.shape[0]:
                weights = np.ones((visible_unknown_points.shape[0],), dtype=np.float32)
        weights = np.clip(weights[valid], 0.0, 1.0)
        for (ix, iy, iz), weight in zip(idxs, weights):
            ix, iy, iz = int(ix), int(iy), int(iz)
            mask[iz, iy, ix] = True
            weights_grid[iz, iy, ix] = max(float(weights_grid[iz, iy, ix]), float(weight))
        self.pending_targeted_mask = mask
        self.pending_targeted_weights = weights_grid

    def _apply_pending_penalty(self, accumulator: VoxelAccumulator) -> None:
        if self.pending_targeted_mask is None or self.penalty_multiplier is None:
            return
        target_weights = (
            self.pending_targeted_mask.astype(np.float32)
            if self.pending_targeted_weights is None
            else np.clip(self.pending_targeted_weights, 0.0, 1.0)
        )
        still_unknown = (accumulator.grid == 0) & self.pending_targeted_mask
        if np.any(still_unknown):
            penalty = 1.0 - target_weights[still_unknown] * (1.0 - self.unresolved_target_penalty)
            self.penalty_multiplier[still_unknown] *= penalty
        resolved = self.pending_targeted_mask & (accumulator.grid != 0)
        if np.any(resolved):
            self.penalty_multiplier[resolved] = 1.0
        self.pending_targeted_mask = None
        self.pending_targeted_weights = None

    def _query_nearest_distance(self, occupied_points: np.ndarray, unknown_points: np.ndarray) -> np.ndarray:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(occupied_points)
            dists, _ = tree.query(unknown_points, workers=-1)
            return np.asarray(dists, dtype=np.float32)
        except Exception:
            out = np.empty((unknown_points.shape[0],), dtype=np.float32)
            batch = 2048
            for start in range(0, unknown_points.shape[0], batch):
                pts = unknown_points[start : start + batch]
                d2 = np.sum((pts[:, None, :] - occupied_points[None, :, :]) ** 2, axis=2)
                out[start : start + batch] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
            return out

    def compute(self, accumulator: VoxelAccumulator) -> tuple[np.ndarray, np.ndarray]:
        self._apply_pending_penalty(accumulator)
        occupied_points = accumulator.iter_world_points(1, stride=1)
        unknown_points = accumulator.iter_world_points(0, stride=1)
        if occupied_points.shape[0] == 0 or unknown_points.shape[0] == 0:
            self.latest_points = np.empty((0, 3), dtype=np.float32)
            self.latest_scores = np.empty((0,), dtype=np.float32)
            return self.latest_points, self.latest_scores
        dists = self._query_nearest_distance(occupied_points, unknown_points)
        span = max(self.max_threshold - self.min_threshold, 1e-9)
        scores = np.clip(1.0 - (dists - self.min_threshold) / span, 0.0, 1.0).astype(np.float32)
        if self.penalty_multiplier is not None and self.penalty_multiplier.shape == accumulator.grid.shape:
            idxs, valid = accumulator.world_to_voxel(unknown_points)
            idxs = idxs[valid]
            if idxs.shape[0] == unknown_points.shape[0]:
                scores *= self.penalty_multiplier[idxs[:, 2], idxs[:, 1], idxs[:, 0]]
        nonzero = scores > 0.0
        self.latest_points = unknown_points[nonzero].astype(np.float32, copy=True)
        self.latest_scores = scores[nonzero].astype(np.float32, copy=True)
        return self.latest_points, self.latest_scores


class CandidateScoreManager:
    def __init__(
        self,
        intrinsics: np.ndarray,
        width: int,
        height: int,
        min_depth_m: float,
        max_depth_m: float,
        ig_with_raycast: bool = False,
        max_scoring_voxels: int = DEFAULT_MAX_SCORING_VOXELS,
    ) -> None:
        self.fx = float(intrinsics[0, 0])
        self.fy = float(intrinsics[1, 1])
        self.cx = float(intrinsics[0, 2])
        self.cy = float(intrinsics[1, 2])
        self.width = int(width)
        self.height = int(height)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.ig_with_raycast = bool(ig_with_raycast)
        self.max_scoring_voxels = max(1, int(max_scoring_voxels))

    def project_points(
        self,
        points_world: np.ndarray,
        cam_position: np.ndarray,
        cam_quat_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if points_world.shape[0] == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return empty, empty, empty
        r_world_cam = quat_xyzw_to_rotmat(cam_quat_xyzw)
        points_cam = (r_world_cam.T @ (points_world - cam_position.reshape(1, 3)).T).T
        z = points_cam[:, 2]
        u = self.fx * (points_cam[:, 0] / np.maximum(z, 1e-6)) + self.cx
        v = self.fy * (points_cam[:, 1] / np.maximum(z, 1e-6)) + self.cy
        return u, v, z

    def visible_mask(self, points_world: np.ndarray, cam_position: np.ndarray, cam_quat_xyzw: np.ndarray) -> np.ndarray:
        if points_world.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        u, v, z = self.project_points(points_world, cam_position, cam_quat_xyzw)
        valid_depth = (z >= self.min_depth_m) & (z <= self.max_depth_m)
        inside = (u >= 0.0) & (u < self.width) & (v >= 0.0) & (v < self.height)
        return valid_depth & inside

    def image_center_weights(self, points_world: np.ndarray, cam_position: np.ndarray, cam_quat_xyzw: np.ndarray) -> np.ndarray:
        if points_world.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        u, v, _ = self.project_points(points_world.astype(np.float64), cam_position, cam_quat_xyzw)
        center_u = 0.5 * float(max(self.width - 1, 1))
        center_v = 0.5 * float(max(self.height - 1, 1))
        max_radius = math.hypot(max(center_u, self.width - 1 - center_u), max(center_v, self.height - 1 - center_v))
        radius = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
        return np.clip(1.0 - radius / max(max_radius, 1e-6), 0.0, 1.0).astype(np.float32)

    def candidate_visible_mask(
        self,
        candidate: dict[str, Any],
        points_world: np.ndarray,
        accumulator: VoxelAccumulator | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        cam_pos, cam_quat = candidate_pose(candidate)
        if cam_pos is None or cam_quat is None or points_world.shape[0] == 0:
            return np.zeros((points_world.shape[0],), dtype=bool), cam_pos, cam_quat
        visible = self.visible_mask(points_world.astype(np.float64), cam_pos, cam_quat)
        if self.ig_with_raycast and accumulator is not None and np.any(visible):
            raycast_visible = accumulator.raycast_unblocked_mask(cam_pos, points_world[visible])
            out = visible.copy()
            out[visible] = raycast_visible
            visible = out
        return visible, cam_pos, cam_quat

    def score_candidate(
        self,
        candidate: dict[str, Any],
        unknown_points: np.ndarray,
        unknown_scores: np.ndarray,
        accumulator: VoxelAccumulator | None = None,
    ) -> tuple[float, int, float]:
        if unknown_points.shape[0] == 0:
            return 0.0, 0, 0.0
        visible, cam_pos, _ = self.candidate_visible_mask(candidate, unknown_points.astype(np.float64), accumulator)
        if cam_pos is None or not np.any(visible):
            return 0.0, 0, 0.0
        pts = unknown_points[visible]
        scores = unknown_scores[visible]
        dists = np.linalg.norm(pts - cam_pos.reshape(1, 3), axis=1)
        weights = 1.0 / (dists + 1e-3)
        return float(np.sum(scores * weights)), int(pts.shape[0]), float(np.mean(dists))

    def select_best(
        self,
        candidates: list[dict[str, Any]],
        unknown_points: np.ndarray,
        unknown_scores: np.ndarray,
        used_indices: set[int],
        accumulator: VoxelAccumulator | None = None,
        require_joint_angles: bool = False,
        joint_dof_count: int | None = 7,
    ) -> tuple[int | None, dict[str, Any] | None]:
        if unknown_points.shape[0] > self.max_scoring_voxels:
            keep = np.argpartition(unknown_scores, -self.max_scoring_voxels)[-self.max_scoring_voxels :]
            unknown_points = unknown_points[keep]
            unknown_scores = unknown_scores[keep]
        best_idx = None
        best_meta = None
        best_score = -1.0
        for idx, candidate in enumerate(candidates):
            if idx in used_indices:
                continue
            if require_joint_angles and not candidate_has_joint_angles(candidate, joint_dof_count):
                continue
            score, visible_count, mean_dist = self.score_candidate(candidate, unknown_points, unknown_scores, accumulator)
            if score <= best_score:
                continue
            best_score = score
            best_idx = idx
            best_meta = {
                "weighted_sum": float(score),
                "visible_unknown_voxels": int(visible_count),
                "mean_distance": float(mean_dist),
            }
        return best_idx, best_meta
