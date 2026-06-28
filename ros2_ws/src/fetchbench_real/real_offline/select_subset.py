from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


NUM_TARGET_LINE_POINTS = 10
DEPTH_OCCLUSION_THRESHOLD_M = 0.05
DEPTH_PATCH_RADIUS = 1
CENTER_SCORE_WEIGHT = 1.0
PROJECTED_SIZE_WEIGHT = 0.3
OCCLUSION_PENALTY_WEIGHT = 1.0
OUT_OF_FRAME_PENALTY_WEIGHT = 1.0
PROJECTION_SCORE_MARGIN_RATIO = 0.03
MIN_RAY_GROUND_ANGLE_DEG = 30.0
CENTER_REGION_RATIO = 0.60
MIN_CENTER_REGION_POINTS = 7
DEFAULT_SIM_WEIGHT = 1.0
DEFAULT_NUM_STARTS = 16


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_dir:
        return Path(args.experiment_dir).expanduser().resolve()
    return Path(args.output_root).expanduser().resolve() / str(args.experiment_name)


def _resolve_views_dir(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    if (exp_dir / "pose.json").is_file() and (exp_dir / "intrinsics.json").is_file():
        return exp_dir
    legacy = exp_dir / "views"
    if (legacy / "pose.json").is_file() and (legacy / "intrinsics.json").is_file():
        return legacy
    return exp_dir


def _parse_vec3(values: list[str] | None, name: str) -> np.ndarray | None:
    if values is None:
        return None
    text = " ".join(str(v) for v in values)
    for ch in "[](),":
        text = text.replace(ch, " ")
    parts = [p for p in text.split() if p]
    if len(parts) != 3:
        raise ValueError(f"{name} expects 3 numbers")
    out = np.asarray([float(v) for v in parts], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _parse_indices(raw: str | None, available: list[int]) -> list[int]:
    if raw is None or raw.strip().lower() in ("", "all"):
        return available
    out = [int(item) for item in raw.replace(",", " ").split()]
    missing = sorted(set(out) - set(available))
    if missing:
        raise ValueError(f"Requested view indices are not available in pose.json: {missing}")
    return sorted(dict.fromkeys(out))


def _load_intrinsics(views_dir: Path) -> dict[str, Any]:
    data = _load_json(views_dir / "intrinsics.json")
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
    }


def _scaled_intrinsics(intr: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    width = int(width)
    height = int(height)
    sx = float(width) / max(float(intr["width"]), 1.0)
    sy = float(height) / max(float(intr["height"]), 1.0)
    out = dict(intr)
    out["width"] = width
    out["height"] = height
    out["fx"] = float(intr["fx"]) * sx
    out["fy"] = float(intr["fy"]) * sy
    out["cx"] = float(intr["cx"]) * sx
    out["cy"] = float(intr["cy"]) * sy
    return out


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return int(img.size[0]), int(img.size[1])


def _load_pose_map(views_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    pose_doc = _load_json(views_dir / "pose.json")
    pose_map: dict[int, dict[str, Any]] = {}
    for pose in pose_doc.get("poses", []):
        idx = int(pose["index"])
        c2w = np.asarray(pose["cam_matrix"], dtype=np.float64)
        pose_map[idx] = {
            "pose": pose,
            "c2w": c2w,
            "w2c": np.linalg.inv(c2w),
            "cam_position": np.asarray(pose.get("cam_position", c2w[:3, 3]), dtype=np.float64),
        }
    if not pose_map:
        raise ValueError(f"No poses found in {views_dir / 'pose.json'}")
    return pose_map, pose_doc


def _read_depth_m(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path))
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def _sample_depth_m(depth_m: np.ndarray, u: float, v: float, radius: int = DEPTH_PATCH_RADIUS) -> float:
    ui = int(round(u))
    vi = int(round(v))
    h, w = depth_m.shape[:2]
    x0 = max(0, ui - radius)
    x1 = min(w, ui + radius + 1)
    y0 = max(0, vi - radius)
    y1 = min(h, vi + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def _project_point_to_image(
    point_world: np.ndarray,
    c2w: np.ndarray,
    intr: dict[str, Any],
) -> tuple[float, float, float, bool]:
    p_world = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    p_cam = np.linalg.inv(c2w) @ p_world
    z = float(p_cam[2])
    if z <= 1e-6:
        return 0.0, 0.0, 0.0, False
    u = float(intr["fx"] * p_cam[0] / z + intr["cx"])
    v = float(intr["fy"] * p_cam[1] / z + intr["cy"])
    valid = 0.0 <= u < float(intr["width"]) and 0.0 <= v < float(intr["height"])
    return u, v, z, valid


def _is_inside_score_margin(u: float, v: float, width: int, height: int) -> bool:
    margin_u = PROJECTION_SCORE_MARGIN_RATIO * float(width)
    margin_v = PROJECTION_SCORE_MARGIN_RATIO * float(height)
    return margin_u <= u < float(width) - margin_u and margin_v <= v < float(height) - margin_v


def _is_inside_center_region(u: float, v: float, width: int, height: int) -> bool:
    x0 = 0.5 * (1.0 - CENTER_REGION_RATIO) * float(width)
    x1 = 0.5 * (1.0 + CENTER_REGION_RATIO) * float(width)
    y0 = 0.5 * (1.0 - CENTER_REGION_RATIO) * float(height)
    y1 = 0.5 * (1.0 + CENTER_REGION_RATIO) * float(height)
    return x0 <= u <= x1 and y0 <= v <= y1


def _compute_centering_score(u: float, v: float, center_u: float, center_v: float, max_dist: float) -> float:
    dist = math.hypot(float(u) - center_u, float(v) - center_v)
    return float(np.clip(1.0 - dist / max(max_dist, 1e-6), 0.0, 1.0))


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def _objective(indices: list[int], view_scores: np.ndarray, cos_mat: np.ndarray, sim_weight: float) -> float:
    score = float(np.sum(view_scores[indices]))
    if len(indices) <= 1:
        return score
    penalty = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            penalty += float(cos_mat[indices[i], indices[j]])
    return score - float(sim_weight) * penalty


def _greedy_select(
    candidate_count: int,
    k: int,
    view_scores: np.ndarray,
    cos_mat: np.ndarray,
    sim_weight: float,
    start_idx: int,
) -> list[int]:
    selected = [int(start_idx)]
    remaining = set(range(candidate_count))
    remaining.remove(int(start_idx))
    while len(selected) < int(k) and remaining:
        best_i = None
        best_gain = -1e18
        for i in remaining:
            sim_sum = float(np.sum(cos_mat[i, selected]))
            gain = float(view_scores[i]) - float(sim_weight) * sim_sum
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected.append(int(best_i))
        remaining.remove(int(best_i))
    return selected


def _local_swap_improve(
    selected: list[int],
    candidate_count: int,
    view_scores: np.ndarray,
    cos_mat: np.ndarray,
    sim_weight: float,
) -> list[int]:
    current = selected[:]
    current_obj = _objective(current, view_scores, cos_mat, sim_weight)
    improved = True
    while improved:
        improved = False
        selected_set = set(current)
        for out_pos, _out_idx in enumerate(current):
            for in_idx in range(candidate_count):
                if in_idx in selected_set:
                    continue
                trial = current[:]
                trial[out_pos] = int(in_idx)
                trial_obj = _objective(trial, view_scores, cos_mat, sim_weight)
                if trial_obj > current_obj + 1e-9:
                    current = trial
                    current_obj = trial_obj
                    improved = True
                    break
            if improved:
                break
    return current


def _target_world_from_args(args: argparse.Namespace, exp_dir: Path) -> np.ndarray:
    manual = _parse_vec3(args.grasp_world, "--grasp-world")
    if manual is not None:
        return manual
    for target_path in (exp_dir / "target_point.json", exp_dir / "offline" / "target_point.json"):
        if target_path.is_file():
            data = _load_json(target_path)
            return np.asarray(data["grasp_position_world"], dtype=np.float64)
    raise ValueError("Provide --grasp-world x y z or run prep once to create target_point.json")


def select_best_subset_indices(
    exp_dir: Path,
    view_indices: list[int],
    grasp_world: np.ndarray,
    num_views: int,
    sim_weight: float = DEFAULT_SIM_WEIGHT,
    num_starts: int = DEFAULT_NUM_STARTS,
    output_dir: Path | None = None,
    label: str = "",
) -> tuple[list[int], dict[str, Any]]:
    exp_dir = Path(exp_dir)
    views_dir = _resolve_views_dir(exp_dir)
    intr = _load_intrinsics(views_dir)
    pose_map, pose_doc = _load_pose_map(views_dir)
    reached = set(int(i) for i in pose_doc.get("reached_indices", []))
    pool = [int(i) for i in view_indices if int(i) in pose_map]
    if reached:
        pool = [i for i in pool if i in reached]
    if not pool:
        raise RuntimeError("No available reached views for subset selection.")

    target_world = np.asarray(grasp_world, dtype=np.float64)
    target_world_z0 = target_world.copy()
    target_world_z0[2] = 0.0
    target_line_points = [
        target_world + t * (target_world_z0 - target_world)
        for t in np.linspace(0.0, 1.0, NUM_TARGET_LINE_POINTS)
    ]

    candidates: list[dict[str, Any]] = []
    filtered_low_angle = 0
    filtered_center_region = 0
    for idx in pool:
        rgb_path = views_dir / "rgb" / f"{idx:04d}.png"
        depth_path = views_dir / "depth" / f"{idx:04d}.png"
        if not rgb_path.is_file() or not depth_path.is_file():
            continue
        width, height = _image_size(rgb_path)
        rgb_intr = _scaled_intrinsics(intr, width, height)
        center_u = float(width) / 2.0
        center_v = float(height) / 2.0
        max_dist = math.hypot(center_u, center_v)
        pose = pose_map[idx]
        cam_pos = np.asarray(pose["cam_position"], dtype=np.float64)
        ray = target_world - cam_pos
        ray_n = _normalize(ray)
        if float(np.linalg.norm(ray_n)) < 1e-8:
            continue
        cos_theta = float(np.linalg.norm(ray_n[:2]))
        ray_ground_angle_deg = float(math.degrees(math.atan2(abs(float(ray_n[2])), max(cos_theta, 1e-12))))
        if ray_ground_angle_deg < MIN_RAY_GROUND_ANGLE_DEG:
            filtered_low_angle += 1
            continue
        occlusion_threshold_m = DEPTH_OCCLUSION_THRESHOLD_M / cos_theta if cos_theta > 1e-12 else float("inf")

        grasp_u, grasp_v, grasp_depth_m, grasp_valid = _project_point_to_image(target_world, pose["c2w"], rgb_intr)
        if not grasp_valid:
            continue
        if not _is_inside_score_margin(grasp_u, grasp_v, width, height):
            continue

        depth_m = _read_depth_m(depth_path)
        depth_h, depth_w = depth_m.shape[:2]
        center_score_sum = 0.0
        size_score_sum = 0.0
        occlusion_penalty_sum = 0.0
        out_of_frame_penalty_sum = 0.0
        center_region_points = 0
        valid_count = 0
        occluded_count = 0
        out_of_frame_count = 0
        projection_points = []

        for target_p in target_line_points:
            u, v, expected_depth_m, proj_valid = _project_point_to_image(target_p, pose["c2w"], rgb_intr)
            if proj_valid and _is_inside_center_region(u, v, width, height):
                center_region_points += 1
            score_valid = proj_valid and _is_inside_score_margin(u, v, width, height)
            if not proj_valid:
                out_of_frame_count += 1
                out_of_frame_penalty_sum += 1.0
                projection_points.append({"u": float(u), "v": float(v), "status": "invalid"})
                continue
            if not score_valid:
                projection_points.append({"u": float(u), "v": float(v), "status": "margin"})
                continue
            valid_count += 1
            depth_u = u * float(depth_w) / max(float(width), 1.0)
            depth_v = v * float(depth_h) / max(float(height), 1.0)
            observed_depth_m = _sample_depth_m(depth_m, depth_u, depth_v)
            is_occluded = observed_depth_m > 0.0 and (expected_depth_m - observed_depth_m) > occlusion_threshold_m
            if is_occluded:
                occluded_count += 1
                occlusion_penalty_sum += 1.0
                projection_points.append({"u": float(u), "v": float(v), "status": "occluded"})
                continue
            center_score_sum += _compute_centering_score(u, v, center_u, center_v, max_dist)
            dist_to_target = float(np.linalg.norm(target_p - cam_pos))
            if dist_to_target < 0.2:
                size_score_sum += 0.5
            elif dist_to_target > 1.5:
                size_score_sum += 0.3
            else:
                size_score_sum += 1.0 / (1.0 + (dist_to_target - 0.5) ** 2)
            projection_points.append({"u": float(u), "v": float(v), "status": "valid"})

        if center_region_points < MIN_CENTER_REGION_POINTS:
            filtered_center_region += 1
            continue
        if valid_count == 0:
            continue

        center_score = center_score_sum / float(NUM_TARGET_LINE_POINTS)
        size_score = size_score_sum / float(NUM_TARGET_LINE_POINTS)
        occlusion_penalty = occlusion_penalty_sum / float(NUM_TARGET_LINE_POINTS)
        out_of_frame_penalty = out_of_frame_penalty_sum / float(NUM_TARGET_LINE_POINTS)
        view_score = (
            CENTER_SCORE_WEIGHT * center_score
            + PROJECTED_SIZE_WEIGHT * size_score
            - OCCLUSION_PENALTY_WEIGHT * occlusion_penalty
            - OUT_OF_FRAME_PENALTY_WEIGHT * out_of_frame_penalty
        )
        candidates.append(
            {
                "idx": int(idx),
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "cam_pos": cam_pos,
                "ray": ray_n,
                "view_score": float(view_score),
                "components": {
                    "center_framing": float(center_score),
                    "projected_size": float(size_score),
                    "occlusion_penalty": float(occlusion_penalty),
                    "out_of_frame_penalty": float(out_of_frame_penalty),
                    "line_points_inside_center_region": int(center_region_points),
                    "line_points_visible": int(valid_count),
                    "line_points_occluded": int(occluded_count),
                    "line_points_out_of_frame": int(out_of_frame_count),
                    "grasp_projection_valid": bool(grasp_valid),
                    "cos_theta_ray_to_z0": float(cos_theta),
                    "ray_ground_angle_deg": float(ray_ground_angle_deg),
                    "occlusion_threshold_m": float(occlusion_threshold_m),
                },
                "projection_points": projection_points,
                "grasp_point_projection_uv": [float(grasp_u), float(grasp_v)],
                "grasp_expected_depth_m": float(grasp_depth_m),
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No valid subset candidates. Filtered low-angle={filtered_low_angle}, "
            f"center-region={filtered_center_region}."
        )

    n = len(candidates)
    k = min(max(1, int(num_views)), n)
    view_scores = np.asarray([c["view_score"] for c in candidates], dtype=np.float64)
    rays = np.asarray([c["ray"] for c in candidates], dtype=np.float64)
    cos_mat = np.clip(rays @ rays.T, 0.0, 1.0)
    np.fill_diagonal(cos_mat, 0.0)

    starts = np.argsort(-view_scores)[: min(max(1, int(num_starts)), n)]
    best_sel: list[int] | None = None
    best_obj = -1e18
    for start_idx in starts:
        selected = _greedy_select(n, k, view_scores, cos_mat, float(sim_weight), int(start_idx))
        if len(selected) < k:
            continue
        selected = _local_swap_improve(selected, n, view_scores, cos_mat, float(sim_weight))
        obj = _objective(selected, view_scores, cos_mat, float(sim_weight))
        if obj > best_obj:
            best_obj = obj
            best_sel = selected
    if best_sel is None:
        raise RuntimeError("Failed to compute a valid subset.")

    selected_candidates = [candidates[i] for i in best_sel]
    selected_sorted = sorted(selected_candidates, key=lambda c: int(c["idx"]))
    selected_indices = [int(c["idx"]) for c in selected_sorted]
    idx_to_candidate_pos = {int(c["idx"]): i for i, c in enumerate(candidates)}

    pair_penalty = 0.0
    selected_pair_similarities = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            idx_i = selected_indices[i]
            idx_j = selected_indices[j]
            pos_i = idx_to_candidate_pos[idx_i]
            pos_j = idx_to_candidate_pos[idx_j]
            sim = float(cos_mat[pos_i, pos_j])
            pair_penalty += sim
            selected_pair_similarities.append({"pair": [idx_i, idx_j], "cosine_similarity": sim})

    result = {
        "format": "fetchbench_real_subset_v1",
        "label": label,
        "experiment_dir": str(exp_dir),
        "num_views_requested": int(num_views),
        "num_views_selected": int(k),
        "num_candidates_after_filter": int(n),
        "num_views_filtered_low_ray_angle": int(filtered_low_angle),
        "num_views_filtered_center_region": int(filtered_center_region),
        "min_ray_ground_angle_deg": float(MIN_RAY_GROUND_ANGLE_DEG),
        "center_region_ratio": float(CENTER_REGION_RATIO),
        "min_center_region_points": int(MIN_CENTER_REGION_POINTS),
        "input_view_indices": [int(v) for v in pool],
        "selected_indices": selected_indices,
        "selected_pair_similarities": selected_pair_similarities,
        "objective": {
            "sum_view_scores": float(np.sum([c["view_score"] for c in selected_candidates])),
            "sum_pair_cosine": float(pair_penalty),
            "value": float(best_obj),
            "formula": "sum(view_score_i) - sim_weight * sum(cos(ray_i, ray_j), i<j)",
            "sim_weight": float(sim_weight),
        },
        "target_grasp_position_world": [float(v) for v in target_world.tolist()],
        "selected_views": [
            {
                "index": int(c["idx"]),
                "rgb_path": str(c["rgb_path"]),
                "depth_path": str(c["depth_path"]),
                "view_score": float(c["view_score"]),
                "components": c["components"],
                "ray_to_target": [float(v) for v in c["ray"].tolist()],
                "camera_position": [float(v) for v in c["cam_pos"].tolist()],
                "grasp_point_projection_uv": c["grasp_point_projection_uv"],
                "grasp_expected_depth_m": float(c["grasp_expected_depth_m"]),
            }
            for c in selected_sorted
        ],
    }

    if output_dir is not None:
        suffix = f"_{label}" if label else ""
        _write_json(Path(output_dir) / f"best_subset_{k}{suffix}.json", result)
    return selected_indices, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select FetchBench-style subsets from real ROS AP views")
    parser.add_argument("--output-root", default="/workspace/ros2_ws/ours_experiment")
    parser.add_argument("--experiment-name", default="ex2")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--view-indices", default="all", help="'all' or comma/space separated view indices")
    parser.add_argument("--num-views", type=int, required=True)
    parser.add_argument("--grasp-world", nargs="+", default=None, help="Manual grasp point in world frame: x y z")
    parser.add_argument("--sim-weight", type=float, default=DEFAULT_SIM_WEIGHT)
    parser.add_argument("--num-starts", type=int, default=DEFAULT_NUM_STARTS)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    exp_dir = _resolve_experiment_dir(args)
    views_dir = _resolve_views_dir(exp_dir)
    pose_map, _pose_doc = _load_pose_map(views_dir)
    available = sorted(pose_map.keys())
    view_indices = _parse_indices(args.view_indices, available)
    grasp_world = _target_world_from_args(args, exp_dir)
    subset_dir = exp_dir / "offline" / "subset"
    selected, result = select_best_subset_indices(
        exp_dir=exp_dir,
        view_indices=view_indices,
        grasp_world=grasp_world,
        num_views=int(args.num_views),
        sim_weight=float(args.sim_weight),
        num_starts=int(args.num_starts),
        output_dir=subset_dir,
        label=str(args.label),
    )
    suffix = f"_{args.label}" if args.label else ""
    print(f"Selected indices: {selected}")
    print(f"Objective: {result['objective']['value']:.6f}")
    print(f"Saved: {subset_dir / f'best_subset_{len(selected)}{suffix}.json'}")


if __name__ == "__main__":
    main()
