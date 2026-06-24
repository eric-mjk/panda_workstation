#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MV_VLM_DIR = SCRIPT_DIR / "mv_vlm"
if str(MV_VLM_DIR) not in sys.path:
    sys.path.insert(0, str(MV_VLM_DIR))

from data_utils import (  # noqa: E402
    build_image_path,
    collect_points_near_center,
    get_background_class_id,
    get_pose_index_to_transforms,
    get_pose_reached_indices,
    get_target_class_id,
    random_downsample_points,
    rotate_direction_scores_180,
    should_rotate_image_180,
)
from data_utils import draw_yellow_circle  # noqa: E402
from gemini_utils import gemini_version, get_gemini_response, parse_single_view_response, process_single_index  # noqa: E402
from prompt import prompt_single  # noqa: E402
from tsdf_utils import (  # noqa: E402
    build_scene_visualization_plys,
    build_tsdf_volume,
    collect_target_object_points,
    compute_distilled_direction_scores,
    compute_geometry_scores,
    compute_volume_bounds,
    fibonacci_upper_hemisphere,
    interpolate_clock_score,
)
from viz_utils import draw_direction_arrows_on_image, plot_best_directions_comparison, plot_candidate_scores_contour_like_vis, plot_candidate_scores_like_vis  # noqa: E402


DATASET_ROOT = REPO_ROOT / "dataset"
DISTILL_NUM_DIRECTIONS = 1000
LATITUDE_UP_AXIS = 2
DISTILL_RHO = 0.1  # step size in world units (metres)
MIN_LATITUDE_DEG = 10.0  # exclude near-horizontal dirs to avoid IK friction
DISTILL_WEIGHT_POWER = 1.5  # exponent on (d_i/L_i): >1 suppresses near-axial views more aggressively
DISTILL_WEIGHT_MIN_RATIO = 0.05  # hard-zero views whose d_i/L_i is below this (nearly axis-parallel)
PLY_LOCAL_RADIUS_M = 0.30
PLY_LOCAL_MAX_POINTS = 30000

# TSDF / geometry scoring constants (mirrors test_multi_view.py)
TSDF_VOXEL_SIZE_M = 0.01
TSDF_TRUNCATION_M = 0.04
TSDF_VOLUME_MARGIN_M = 0.20
TSDF_INTEGRATION_STRIDE = 1
TSDF_TARGET_MASK_DILATE_PX = 0
GEO_TARGET_POINT_STRIDE = 4
GEO_TARGET_POINT_MAX_POINTS = 2500
GEO_SWEEP_START_M = 0.00
GEO_SWEEP_END_M = 0.15
GEO_SWEEP_NUM_STEPS = 16
GEO_COLLISION_CLEARANCE_M = 0.01
GEO_SAFE_CLEARANCE_M = 0.10
GEO_ADAPTIVE_THRESHOLDS = False
GEO_ADAPTIVE_LOW_Q = 0.20
GEO_ADAPTIVE_HIGH_Q = 0.80
GEO_LAMBDA = 0.5  # weight of geometry in final fusion: final = lambda*distill + (1-lambda)*geometry


def _project_point(
    w2c: np.ndarray, K: np.ndarray, point_world_h: np.ndarray
) -> tuple[float, float, float] | None:
    """Project homogeneous world point [4] through w2c and K.
    Returns (u, v, z_cam) or None when the point is behind the camera.
    """
    p_cam = w2c @ point_world_h
    z = float(p_cam[2])
    if abs(z) < 1e-4:
        return None
    u = float(K[0, 0] * p_cam[0] / z + K[0, 2])
    v = float(K[1, 1] * p_cam[1] / z + K[1, 2])
    return u, v, z


def _compute_weighted_direction_scores(
    candidate_dirs: np.ndarray,
    grasp_world: np.ndarray,
    views_info: list[dict],
    rho: float = DISTILL_RHO,
    weight_power: float = DISTILL_WEIGHT_POWER,
    weight_min_ratio: float = DISTILL_WEIGHT_MIN_RATIO,
) -> np.ndarray:
    """Compute S_3D(v) per candidate direction using the paper's per-view
    projection weighting:

        S_3D(v) = sum_i  w_i * s_i(v)  /  (sum_i w_i + eps)

    where
        (dx_i, dy_i) = Pi_i(x + rho*v) - Pi_i(x)
        d_i          = ||(dx_i, dy_i)||   (2D displacement length, pixels)
        L_i          = f_i * rho / z_i    (max in-plane projected length for view i)
        ratio_i      = d_i / L_i          (≈ sin of angle from optical axis)
        w_i          = ratio_i^weight_power  if ratio_i >= weight_min_ratio else 0
        s_i(v)       = clock_interpolate(q_i, (dx_i, dy_i))

    weight_power > 1 makes the weighting much more peaked around views where
    the direction is nearly perpendicular to the optical axis (lies in the image
    plane).  E.g. power=3: ratio=0.5 → w=0.125 vs. linear w=0.5.
    weight_min_ratio hard-zeros views that are nearly axis-parallel.

    Each entry of `views_info` must have:
        w2c          – (4,4) world-to-camera matrix
        K            – (3,3) intrinsic matrix
        clock_scores – (12,) scores in the *original* (un-rotated) camera frame
    """
    grasp_h = np.array([*grasp_world, 1.0], dtype=np.float64)
    out = np.zeros(len(candidate_dirs), dtype=np.float32)

    for ci, dir_v in enumerate(candidate_dirs):
        displaced_h = np.array([*(grasp_world + rho * np.asarray(dir_v, dtype=np.float64)), 1.0], dtype=np.float64)
        total_w = 0.0
        total_ws = 0.0

        for vinfo in views_info:
            w2c = vinfo["w2c"]
            K = vinfo["K"]
            clock_scores = vinfo["clock_scores"]

            p0 = _project_point(w2c, K, grasp_h)
            if p0 is None:
                continue
            p1 = _project_point(w2c, K, displaced_h)
            if p1 is None:
                continue

            u0, v0, z0 = p0
            u1, v1, _ = p1
            dx = u1 - u0
            dy = v1 - v0
            d_i = float(np.hypot(dx, dy))

            # L_i = focal * rho / depth  (max in-plane projected displacement)
            f_avg = float((K[0, 0] + K[1, 1]) * 0.5)
            L_i = f_avg * rho / abs(z0)
            ratio_i = d_i / L_i

            # Hard-zero near-axial views; raise ratio to power for sharper selectivity
            if ratio_i < weight_min_ratio:
                continue
            w_i = ratio_i ** weight_power

            s_i = _interpolate_clock_score_from_image_delta(clock_scores, dx, dy)

            total_w += w_i
            total_ws += w_i * s_i

        out[ci] = total_ws / (total_w + 1e-8)

    return out


def _interpolate_clock_score_from_image_delta(clock_scores: np.ndarray, dx: float, dy: float) -> float:
    """Clock-score interpolation for image displacement (dx, dy).

    Clock convention follows the prompt:
    - 12 o'clock: image up (dy < 0)
    - 3 o'clock : image right (dx > 0)
    - clockwise increases by 30 deg
    - score indices [0..11] map to [1..12] o'clock
    """
    scores = np.asarray(clock_scores, dtype=np.float32).reshape(-1)
    if scores.shape[0] != 12:
        raise ValueError("clock_scores must have length 12")

    # 0 deg at 12 o'clock, clockwise positive.
    # Mirror on y-axis for clock interpolation so x-direction matches intended convention.
    angle_clock_deg = float(np.degrees(np.arctan2(float(-dx), float(-dy))) % 360.0)
    clock_pos = angle_clock_deg / 30.0

    # Convert to zero-based score index space where 0->1 o'clock, 11->12 o'clock.
    pos = (clock_pos - 1.0) % 12.0
    lo = int(np.floor(pos)) % 12
    hi = (lo + 1) % 12
    t = float(pos - np.floor(pos))
    return float((1.0 - t) * scores[lo] + t * scores[hi])


def _compute_single_direction_debug_rows(
    direction: np.ndarray,
    grasp_world: np.ndarray,
    views_info: list[dict],
    rho: float = DISTILL_RHO,
    weight_power: float = DISTILL_WEIGHT_POWER,
    weight_min_ratio: float = DISTILL_WEIGHT_MIN_RATIO,
) -> tuple[list[dict], float]:
    """Compute per-view interpolation/weight terms for one direction for debugging."""
    dir_v = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(dir_v))
    if norm < 1e-12:
        return [], 0.0
    dir_v = dir_v / norm

    grasp_h = np.array([*grasp_world, 1.0], dtype=np.float64)
    displaced_h = np.array([*(grasp_world + rho * dir_v), 1.0], dtype=np.float64)
    rows: list[dict] = []
    total_w = 0.0
    total_ws = 0.0

    for vinfo in views_info:
        w2c = vinfo["w2c"]
        K = vinfo["K"]
        clock_scores = vinfo["clock_scores"]
        idx = vinfo.get("idx")

        p0 = _project_point(w2c, K, grasp_h)
        p1 = _project_point(w2c, K, displaced_h)
        if p0 is None or p1 is None:
            rows.append(
                {
                    "idx": idx,
                    "dx": None,
                    "dy": None,
                    "interpolated_vlm_score": None,
                    "weight": 0.0,
                    "reason": "projection_invalid",
                }
            )
            continue

        u0, v0, z0 = p0
        u1, v1, _ = p1
        dx = float(u1 - u0)
        dy = float(v1 - v0)
        d_i = float(np.hypot(dx, dy))

        f_avg = float((K[0, 0] + K[1, 1]) * 0.5)
        L_i = f_avg * rho / max(abs(z0), 1e-8)
        ratio_i = d_i / max(L_i, 1e-8)

        s_i = _interpolate_clock_score_from_image_delta(clock_scores, dx, dy)
        if ratio_i < weight_min_ratio:
            w_i = 0.0
            reason = "below_min_ratio"
        else:
            w_i = float(ratio_i ** weight_power)
            reason = "ok"

        total_w += w_i
        total_ws += w_i * s_i
        rows.append(
            {
                "idx": idx,
                "dx": dx,
                "dy": dy,
                "interpolated_vlm_score": s_i,
                "weight": w_i,
                "ratio": float(ratio_i),
                "reason": reason,
            }
        )

    final_score = float(total_ws / (total_w + 1e-8))
    return rows, final_score


def _parse_debug_direction_arg(raw_dir: str | list[str] | None) -> np.ndarray | None:
    """Parse --dir input supporting formats like:
    - 0.4,0.5,0.3
    - (0.4, 0.5, 0.3)
    - 0.4 0.5 0.3
    """
    if raw_dir is None:
        return None
    if isinstance(raw_dir, str):
        tokens = [raw_dir]
    else:
        tokens = [str(x) for x in raw_dir]

    text = " ".join(tokens).strip()
    for ch in "()[]{}":
        text = text.replace(ch, " ")
    text = text.replace(",", " ")
    parts = [p for p in text.split() if p]
    if len(parts) != 3:
        raise ValueError(
            "--dir must provide 3 numbers, e.g. --dir 0.4,0.5,0.3 or --dir (-0.5, 0.5, 0.2)"
        )
    vals = np.asarray([float(p) for p in parts], dtype=np.float64)
    norm = float(np.linalg.norm(vals))
    if norm < 1e-12:
        raise ValueError("--dir must be non-zero")
    return vals / norm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-view Gemini aggregation from subset selection")
    parser.add_argument("--scene", required=True, help="Scene id, e.g. 01")
    parser.add_argument("--scene-num", required=True, help="Scene number, e.g. 000")
    parser.add_argument("--num-views", type=int, required=True, help="Number of views to load from best_subset_{num_views}.json")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker count (0 => min(num_views, cpu_count))")
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Reuse cached per-view JSON if present; only call Gemini when cache is missing.",
    )
    parser.add_argument(
        "--ply",
        action="store_true",
        help="Export PLY visualization with local point cloud around grasp point and direction-score sphere.",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip TSDF geometry scoring; use distill scores only as final scores.",
    )
    parser.add_argument(
        "--add_bev",
        action="store_true",
        help="Include BEV view in VLM score aggregation. If omitted, BEV is used only for visualization/context.",
    )
    parser.add_argument(
        "--ap-only",
        action="store_true",
        help="Read subset from active_perception/subset and write outputs to active_perception/vlm_ours.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed per-view interpolation/weight terms for a manual direction when --dir is provided.",
    )
    parser.add_argument(
        "--dir",
        nargs="+",
        type=str,
        default=None,
        help="Manual direction vector for debug, e.g. --dir 0.4,0.5,0.3 or --dir (-0.5, 0.5, 0.2)",
    )
    return parser.parse_args()


def _concat_images_grid(images: list[np.ndarray], save_path: Path, cols: int = 2, padding: int = 10) -> str:
    if not images:
        raise ValueError("No images to concatenate")
    rows = int(np.ceil(len(images) / float(cols)))
    cell_h = max(img.shape[0] for img in images)
    cell_w = max(img.shape[1] for img in images)
    canvas_h = rows * cell_h + (rows + 1) * padding
    canvas_w = cols * cell_w + (cols + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), 0, dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y0 = padding + r * (cell_h + padding)
        x0 = padding + c * (cell_w + padding)
        h, w = img.shape[:2]
        yoff = (cell_h - h) // 2
        xoff = (cell_w - w) // 2
        canvas[y0 + yoff : y0 + yoff + h, x0 + xoff : x0 + xoff + w] = img

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    return str(save_path)


def _draw_debug_direction_arrow(
    image: np.ndarray,
    center_uv: tuple[int, int],
    dx: float,
    dy: float,
    score: float,
    weight: float,
) -> np.ndarray:
    out = image.copy()
    cx, cy = int(center_uv[0]), int(center_uv[1])
    end_x = int(round(cx + dx))
    end_y = int(round(cy + dy))
    cv2.arrowedLine(out, (cx, cy), (end_x, end_y), (255, 0, 0), 3, cv2.LINE_AA, tipLength=0.24)
    cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)
    txt = f"dbg s={score:.3f} w={weight:.3f}"
    font_scale = 0.68
    text_thickness = 2
    pad_x = 6
    pad_y = 6
    (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    x0 = max(0, cx + 8)
    y0 = max(0, cy - th - baseline - (pad_y + 6))
    x1 = min(out.shape[1] - 1, x0 + tw + pad_x * 2)
    y1 = min(out.shape[0] - 1, y0 + th + baseline + pad_y * 2)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), 1)
    cv2.putText(
        out,
        txt,
        (x0 + pad_x, y1 - baseline - pad_y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )
    return out


def _bev_arrow_endpoint_from_world_xy(center: tuple[int, int], dir_xy: np.ndarray, length_px: int) -> tuple[int, int]:
    """Project world XY direction onto BEV image pixels.

    Convention used in experiment code:
      12 o'clock -> +x, 3 o'clock -> -y, 9 o'clock -> +y
    So in image coordinates: right is -y, up is +x.
    """
    x, y = float(dir_xy[0]), float(dir_xy[1])
    norm = float(np.hypot(x, y))
    if norm < 1e-8:
        return center
    x /= norm
    y /= norm
    dx_px = -y * float(length_px)
    dy_px = -x * float(length_px)
    end_x = int(round(center[0] + dx_px))
    end_y = int(round(center[1] + dy_px))
    return end_x, end_y


def _score_to_bgr(score: float) -> tuple[int, int, int]:
    s = float(np.clip(score, 0.0, 1.0))
    blue = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    red = np.array([0.0, 0.0, 255.0], dtype=np.float32)
    c = (1.0 - s) * blue + s * red
    return int(c[0]), int(c[1]), int(c[2])


def _build_clock_bin_scores_from_candidates(candidate_dirs: np.ndarray, candidate_scores: np.ndarray) -> np.ndarray:
    dirs = np.asarray(candidate_dirs, dtype=np.float32)
    scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("candidate_dirs must have shape (N,3)")
    if scores.shape[0] != dirs.shape[0]:
        raise ValueError("candidate_scores must have shape (N,)")

    xy = dirs[:, :2]
    xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
    xy_unit = xy / np.maximum(xy_norm, 1e-8)
    bins = np.zeros(12, dtype=np.float32)
    for i in range(12):
        lon_rad = np.deg2rad(float(i * 30))
        target_xy = np.array([np.cos(lon_rad), np.sin(lon_rad)], dtype=np.float32)
        best_idx = int(np.argmax(xy_unit @ target_xy))
        bins[i] = float(scores[best_idx])
    return bins


def _draw_clock_scores_world_aligned(
    image: np.ndarray,
    center: tuple[int, int],
    vlm_clock_scores: np.ndarray,
    geo_clock_scores: np.ndarray | None,
    total_clock_scores: np.ndarray,
    arrow_len: int = 210,
    box_radius: int = 140,
    lon_radius: int = 190,
) -> np.ndarray:
    """Draw 12 clock scores using world-longitude bins.

    lon = 0,30,...,330 with direction vector [cos(lon), sin(lon)].
    """
    out = image.copy()
    cx, cy = center
    vlm_scores = np.asarray(vlm_clock_scores, dtype=np.float32).reshape(-1)
    total_scores = np.asarray(total_clock_scores, dtype=np.float32).reshape(-1)
    if vlm_scores.shape[0] != 12 or total_scores.shape[0] != 12:
        return out
    geo_scores = None
    if geo_clock_scores is not None:
        geo_scores = np.asarray(geo_clock_scores, dtype=np.float32).reshape(-1)
        if geo_scores.shape[0] != 12:
            geo_scores = None

    for i in range(12):
        lon_deg = i * 30
        lon_rad = np.deg2rad(float(lon_deg))
        dir_xy = np.array([np.cos(lon_rad), np.sin(lon_rad)], dtype=np.float32)
        end_x, end_y = _bev_arrow_endpoint_from_world_xy((cx, cy), dir_xy, arrow_len)
        color = _score_to_bgr(float(vlm_scores[i]))

        cv2.arrowedLine(out, (cx, cy), (end_x, end_y), color=color, thickness=4, line_type=cv2.LINE_AA, tipLength=0.22)

        tx, ty = _bev_arrow_endpoint_from_world_xy((cx, cy), dir_xy, box_radius)
        if geo_scores is None:
            lines = [
                f"vlm {float(vlm_scores[i]):.2f}",
                "geo  N/A",
                f"total {float(total_scores[i]):.2f}",
            ]
        else:
            lines = [
                f"vlm {float(vlm_scores[i]):.2f}",
                f"geo {float(geo_scores[i]):.2f}",
                f"total {float(total_scores[i]):.2f}",
            ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.30
        thick = 1
        pad = 3
        line_h = 11
        text_w = 0
        for line in lines:
            (lw, _), _ = cv2.getTextSize(line, font, fs, thick)
            text_w = max(text_w, lw)
        box_w = text_w + pad * 2
        box_h = line_h * len(lines) + pad * 2
        x0 = tx - box_w // 2
        y0 = ty - box_h // 2
        x1 = x0 + box_w
        y1 = y0 + box_h
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (20, 20, 20), 1)
        for li, line in enumerate(lines):
            yy = y0 + pad + (li + 1) * line_h - 2
            cv2.putText(out, line, (x0 + pad, yy), font, fs, (20, 20, 20), thick, cv2.LINE_AA)

        lx, ly = _bev_arrow_endpoint_from_world_xy((cx, cy), dir_xy, lon_radius)
        lon_txt = f"lon {lon_deg}"
        cv2.putText(out, lon_txt, (lx - 20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)

    return out


def _draw_best_arrows_on_bev(
    image: np.ndarray,
    center: tuple[int, int],
    candidate_dirs: np.ndarray,
    distill_best_idx: int,
    total_best_idx: int,
    geometry_best_idx: int | None,
    arrow_len: int = 140,
) -> np.ndarray:
    out = image.copy()
    arrow_specs = [
        ("vlm", int(distill_best_idx), (255, 255, 0)),  # cyan (BGR)
        ("total", int(total_best_idx), (0, 0, 255)),    # red
    ]
    if geometry_best_idx is not None:
        arrow_specs.insert(1, ("geo", int(geometry_best_idx), (0, 255, 0)))

    for label, idx, color in arrow_specs:
        dir_xy = candidate_dirs[idx][:2]
        end_x, end_y = _bev_arrow_endpoint_from_world_xy(center, dir_xy, arrow_len)
        cv2.arrowedLine(out, center, (end_x, end_y), color=color, thickness=4, line_type=cv2.LINE_AA, tipLength=0.22)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        tx = end_x + 6
        ty = end_y - 6
        cv2.rectangle(out, (tx - 4, ty - th - 3), (tx + tw + 4, ty + baseline + 3), (255, 255, 255), -1)
        cv2.rectangle(out, (tx - 4, ty - th - 3), (tx + tw + 4, ty + baseline + 3), color, 1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    cv2.circle(out, center, 8, (0, 0, 255), -1)
    return out


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _process_bev(
    bev_rgb_path: Path,
    grasp_pixel_uv: list,
    out_dir: Path,
    idx_output_dir: Path,
    skip_gemini: bool,
    prompt_text: str,
) -> dict:
    """Run Gemini on the BEV image; query point from target_point.json grasp_pixel_uv."""
    query_point = (int(round(grasp_pixel_uv[0])), int(round(grasp_pixel_uv[1])))
    marked_image = draw_yellow_circle(str(bev_rgb_path), query_point)
    # BEV is always top-down; no 180-degree rotation.

    idx_prefix = "idx_bev"
    input_vis_path = idx_output_dir / f"{idx_prefix}_input_with_query.png"
    raw_text_path = idx_output_dir / f"{idx_prefix}_gemini_response_raw.txt"
    response_json_path = idx_output_dir / f"{idx_prefix}_gemini_response.json"
    dir_vis_path = out_dir / f"{idx_prefix}_direction_scores_vis.png"

    cv2.imwrite(str(input_vis_path), marked_image)

    used_cached_response = False
    if skip_gemini:
        try:
            with response_json_path.open("r", encoding="utf-8") as f:
                response_dict = json.load(f)
            if "Direction scores" not in response_dict and "direction_scores" in response_dict:
                response_dict["Direction scores"] = response_dict["direction_scores"]
            if not isinstance(response_dict.get("Direction scores"), list):
                raise ValueError("Cached BEV JSON has no valid Direction scores.")
            used_cached_response = True
        except Exception as e:
            raise RuntimeError(
                f"--skip-gemini is enabled but BEV cached response is unavailable/invalid: {e}"
            ) from e

    if not used_cached_response:
        print("[idx bev] Getting Gemini response...", flush=True)
        start_time = time.time()
        response = get_gemini_response(prompt_text, marked_image)
        end_time = time.time()
        with raw_text_path.open("w", encoding="utf-8") as f:
            f.write(response.text)
        response_dict = parse_single_view_response(response.text)
        response_dict["image_rotated_180"] = False
        with response_json_path.open("w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=2)
    else:
        start_time = time.time()
        end_time = start_time

    dir_scores = response_dict["Direction scores"]
    dir_vis = draw_direction_arrows_on_image(marked_image, dir_scores, center=query_point)
    cv2.imwrite(str(dir_vis_path), dir_vis)

    return {
        "idx": "bev",
        "query_point": list(query_point),
        "display_query_point": list(query_point),
        "elapsed": float(end_time - start_time),
        "used_cached_response": bool(used_cached_response),
        "image_rotated_180": False,
        "input_vis_path": str(input_vis_path),
        "raw_text_path": str(raw_text_path),
        "response_json_path": str(response_json_path),
        "dir_vis_path": str(dir_vis_path),
    }


def main() -> None:
    args = _parse_args()
    load_dotenv(str(MV_VLM_DIR / ".env"))
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in mv_vlm/.env or export it.")
    os.environ["GEMINI_API_KEY"] = api_key

    scene_num_int = int(args.scene_num)
    scene_dir = DATASET_ROOT / f"{args.scene}_robot" / f"scene_{scene_num_int:03d}"
    views_dir = scene_dir / "views"
    rgb_dir = views_dir / "rgb"
    class_dir = views_dir / "class"
    pose_path = views_dir / "pose.json"
    subset_root = scene_dir / "active_perception" / "subset" if args.ap_only else scene_dir / "subset"
    subset_path = subset_root / f"best_subset_{int(args.num_views)}.json"

    bev_dir = scene_dir / "bev"
    bev_rgb_path = bev_dir / "rgb_bev.png"
    bev_target_path = bev_dir / "target_point.json"
    bev_pose_path = bev_dir / "pose.json"
    intrinsics_path = views_dir / "intrinsics.json"

    if not subset_path.is_file():
        raise FileNotFoundError(f"Subset json not found: {subset_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"pose.json not found: {pose_path}")
    if not bev_rgb_path.is_file():
        raise FileNotFoundError(f"BEV RGB image not found: {bev_rgb_path}")
    if not bev_target_path.is_file():
        raise FileNotFoundError(f"BEV target_point.json not found: {bev_target_path}")
    if not bev_pose_path.is_file():
        raise FileNotFoundError(f"BEV pose.json not found: {bev_pose_path}")
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"intrinsics.json not found: {intrinsics_path}")

    bev_target = _load_json(bev_target_path)
    grasp_pixel_uv = bev_target.get("grasp_pixel_uv")
    if grasp_pixel_uv is None:
        raise KeyError(f"grasp_pixel_uv not found in {bev_target_path}")
    grasp_world = np.array(bev_target["grasp_position_world"], dtype=np.float64)

    intr = _load_json(intrinsics_path)
    K = np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    bev_pose_data = _load_json(bev_pose_path)
    bev_c2w = np.array(bev_pose_data["poses"][0]["cam_matrix"], dtype=np.float64)
    bev_w2c = np.linalg.inv(bev_c2w)

    out_dir = scene_dir / "active_perception" / "vlm_ours" if args.ap_only else scene_dir / "vlm_ours"
    idx_output_dir = out_dir / "idx_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_output_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "vlm_result.json"
    three_d_path = out_dir / "final_3d_direction.json"

    subset = _load_json(subset_path)
    selected_indices = [int(i) for i in subset.get("selected_indices", [])]
    if len(selected_indices) == 0:
        raise RuntimeError(f"No selected_indices in subset json: {subset_path}")

    if len(selected_indices) != int(args.num_views):
        print(
            f"[WARN] Requested num_views={args.num_views}, but subset has {len(selected_indices)} indices. "
            f"Using subset size {len(selected_indices)}.",
            flush=True,
        )

    reached_indices = set(get_pose_reached_indices(str(pose_path)))
    pose_map = get_pose_index_to_transforms(str(pose_path))
    target_class_id = get_target_class_id(str(pose_path))

    for idx in selected_indices:
        if idx not in reached_indices:
            raise ValueError(f"Subset idx {idx} is not in reached_indices")
        if idx not in pose_map:
            raise ValueError(f"Subset idx {idx} missing in pose map")
        if not (rgb_dir / f"{idx:04d}.png").is_file():
            raise FileNotFoundError(f"RGB image missing for idx {idx}: {rgb_dir / f'{idx:04d}.png'}")
        if not (class_dir / f"{idx:04d}.png").is_file():
            raise FileNotFoundError(f"Class image missing for idx {idx}: {class_dir / f'{idx:04d}.png'}")

    rotate_flags = {idx: should_rotate_image_180(pose_map[idx]["c2w"]) for idx in selected_indices}
    worker_args = [
        (
            idx,
            target_class_id,
            str(out_dir),
            str(idx_output_dir),
            bool(args.skip_gemini),
            bool(rotate_flags[idx]),
            build_image_path(str(rgb_dir), idx),
            build_image_path(str(class_dir), idx),
            prompt_single,
        )
        for idx in selected_indices
    ]

    workers = int(args.workers)
    if workers <= 0:
        workers = min(len(worker_args), os.cpu_count() or 1)
    workers = max(1, min(workers, len(worker_args)))

    if args.add_bev:
        print(f"[INFO] Running Gemini on BEV + {len(worker_args)} subset views (workers={workers})", flush=True)
    else:
        print(f"[INFO] Running Gemini on {len(worker_args)} subset views only (workers={workers}); BEV excluded from aggregation", flush=True)
    start_parallel = time.time()
    if workers == 1:
        bev_result = _process_bev(bev_rgb_path, grasp_pixel_uv, out_dir, idx_output_dir, bool(args.skip_gemini), prompt_single)
        subset_results = [process_single_index(x) for x in worker_args]
    else:
        with mp.Pool(processes=workers) as pool:
            async_result = pool.map_async(process_single_index, worker_args)
            # BEV runs on main thread while subset views are processed in parallel.
            bev_result = _process_bev(bev_rgb_path, grasp_pixel_uv, out_dir, idx_output_dir, bool(args.skip_gemini), prompt_single)
            subset_results = async_result.get()
    parallel_execution_sec = time.time() - start_parallel

    # BEV is optional in aggregation; when enabled, keep it first in result list.
    subset_results = sorted(subset_results, key=lambda item: int(item["idx"]))
    results = ([bev_result] + subset_results) if args.add_bev else subset_results
    vis_images = []
    per_view = []
    views_info: list[dict] = []  # per-view (w2c, K, clock_scores) for 3D aggregation

    for item in results:
        with open(item["response_json_path"], "r", encoding="utf-8") as f:
            response = json.load(f)
        direction_scores = response.get("Direction scores", response.get("direction_scores", None))
        if direction_scores is None:
            raise ValueError(f"Direction scores missing in {item['response_json_path']}")
        direction_scores = [float(x) for x in direction_scores]
        if len(direction_scores) != 12:
            raise ValueError(f"Direction scores must be 12 values, got {len(direction_scores)} for idx={item['idx']}")

        img = cv2.imread(item["dir_vis_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read vis image: {item['dir_vis_path']}")
        vis_images.append(img)

        # Build camera info for 3D aggregation.
        # Clock scores must be in the *original* camera frame: un-rotate if image was rotated.
        rotated = bool(item["image_rotated_180"])
        clock_arr = np.asarray(direction_scores, dtype=np.float32)
        if rotated:
            clock_arr = rotate_direction_scores_180(clock_arr)
        idx_val = item["idx"]
        if idx_val == "bev":
            view_w2c = bev_w2c
        else:
            view_w2c = pose_map[int(idx_val)]["w2c"].astype(np.float64)
        views_info.append({"idx": idx_val, "w2c": view_w2c, "K": K, "clock_scores": clock_arr})

        per_view.append(
            {
                "index": idx_val,  # "bev" or int
                "query_point_uv": [int(item["query_point"][0]), int(item["query_point"][1])],
                "display_query_point_uv": [int(item["display_query_point"][0]), int(item["display_query_point"][1])],
                "image_rotated_180": rotated,
                "inference_sec": float(item["elapsed"]),
                "used_cached_response": bool(item["used_cached_response"]),
                "direction_scores": direction_scores,
                "parsed_response": response,
                "input_vis_path": str(item["input_vis_path"]),
                "raw_response_path": str(item["raw_text_path"]),
                "vis_path": str(item["dir_vis_path"]),
                "response_json_path": str(item["response_json_path"]),
            }
        )

    # --- Aggregate: per-view projection weighting (paper §VLM-guided 3D Reasoning) ---
    # For each candidate 3D direction v, project x and x+rho*v into each view,
    # interpolate that view's clock scores at the resulting 2D angle, and weight
    # by the projected displacement length relative to the max in-plane projection.
    candidate_dirs = fibonacci_upper_hemisphere(DISTILL_NUM_DIRECTIONS, up_axis=LATITUDE_UP_AXIS)
    # Keep only directions with latitude >= MIN_LATITUDE_DEG (z-component >= sin(lat))
    min_z = float(np.sin(np.deg2rad(MIN_LATITUDE_DEG)))
    candidate_dirs = candidate_dirs[candidate_dirs[:, 2] >= min_z]
    distill_scores = _compute_weighted_direction_scores(
        candidate_dirs, grasp_world, views_info,
        rho=DISTILL_RHO,
        weight_power=DISTILL_WEIGHT_POWER,
        weight_min_ratio=DISTILL_WEIGHT_MIN_RATIO,
    )

    debug_dir_unit: np.ndarray | None = None
    debug_rows: list[dict] = []
    if args.debug and args.dir:
        try:
            debug_dir_unit = _parse_debug_direction_arg(args.dir)
        except Exception as e:
            raise ValueError(f"Invalid --dir value '{args.dir}': {e}") from e

    if args.debug and args.dir:
        try:
            assert debug_dir_unit is not None
            debug_rows, debug_score = _compute_single_direction_debug_rows(
                debug_dir_unit,
                grasp_world,
                views_info,
                rho=DISTILL_RHO,
                weight_power=DISTILL_WEIGHT_POWER,
                weight_min_ratio=DISTILL_WEIGHT_MIN_RATIO,
            )
            print("[DEBUG] Manual direction per-view terms")
            print(f"[DEBUG] input_dir={args.dir} normalized={[round(float(v), 6) for v in debug_dir_unit.tolist()]}")
            print("[DEBUG] idx\tdx\tdy\tinterpolated_vlm_score\tweight")
            for row in debug_rows:
                idx_val = row["idx"]
                dx_val = row["dx"]
                dy_val = row["dy"]
                s_val = row["interpolated_vlm_score"]
                w_val = row["weight"]
                if dx_val is None or dy_val is None or s_val is None:
                    print(f"[DEBUG] {idx_val}\tN/A\tN/A\tN/A\t{w_val:.6f} ({row.get('reason', 'invalid')})")
                else:
                    print(f"[DEBUG] {idx_val}\t{dx_val:.6f}\t{dy_val:.6f}\t{s_val:.6f}\t{w_val:.6f}")
            print(f"[DEBUG] weighted_score={debug_score:.6f}")
        except Exception as e:
            raise ValueError(f"Invalid --dir value '{args.dir}': {e}") from e

    # --- TSDF geometry scoring ---
    depth_dir = views_dir / "depth"
    geometry_scores = None
    geometry_best_idx = None
    geometry_vis_path = None
    if args.skip_geometry or not depth_dir.is_dir():
        if not args.skip_geometry:
            print(f"[WARN] depth dir not found ({depth_dir}); skipping geometry scoring.", flush=True)
        final_scores = distill_scores
        final_best_idx = int(np.argmax(distill_scores))
    else:
        print("[INFO] Building TSDF volume and computing geometry scores...", flush=True)
        target_class_id_geo = get_target_class_id(str(pose_path))
        background_class_id_geo = get_background_class_id(str(pose_path))
        target_points_world, _ = collect_target_object_points(
            selected_indices,
            pose_map,
            target_class_id_geo,
            str(intrinsics_path),
            str(rgb_dir),
            str(depth_dir),
            str(class_dir),
            GEO_TARGET_POINT_MAX_POINTS,
            GEO_TARGET_POINT_STRIDE,
        )
        volume_min, volume_max = compute_volume_bounds(
            target_points_world, grasp_world.astype(np.float32), TSDF_VOLUME_MARGIN_M
        )
        tsdf_volume = build_tsdf_volume(
            selected_indices,
            pose_map,
            str(intrinsics_path),
            str(rgb_dir),
            str(depth_dir),
            str(class_dir),
            volume_min,
            volume_max,
            TSDF_VOXEL_SIZE_M,
            TSDF_TRUNCATION_M,
            target_class_id=target_class_id_geo,
            background_class_id=background_class_id_geo,
            exclude_background=True,
            mask_dilate_px=TSDF_TARGET_MASK_DILATE_PX,
            integration_stride=TSDF_INTEGRATION_STRIDE,
        )
        rng = np.random.default_rng(0)
        num_sample = min(100, target_points_world.shape[0])
        sampled_pts = target_points_world[rng.choice(target_points_world.shape[0], num_sample, replace=False)]
        geometry_scores, _, geometry_best_idx = compute_geometry_scores(
            candidate_dirs,
            tsdf_volume,
            sweep_start_m=GEO_SWEEP_START_M,
            sweep_end_m=GEO_SWEEP_END_M,
            sweep_num_steps=GEO_SWEEP_NUM_STEPS,
            coll_clearance_m=GEO_COLLISION_CLEARANCE_M,
            safe_clearance_m=GEO_SAFE_CLEARANCE_M,
            adaptive_thresholds=GEO_ADAPTIVE_THRESHOLDS,
            adaptive_low_q=GEO_ADAPTIVE_LOW_Q,
            adaptive_high_q=GEO_ADAPTIVE_HIGH_Q,
            target_points=sampled_pts,
        )
        # compute_distilled_direction_scores: alpha*distill + (1-alpha)*geometry, normalized
        final_scores, final_best_idx = compute_distilled_direction_scores(
            candidate_dirs,
            distill_scores,
            geometry_scores,
            alpha=GEO_LAMBDA,
        )
        print(f"[INFO] Geometry scoring done. best_idx: distill={int(np.argmax(distill_scores))} geo={geometry_best_idx} final={final_best_idx}", flush=True)

    best_dir_idx = final_best_idx
    best_direction = [float(x) for x in candidate_dirs[best_dir_idx].tolist()]

    # Clock-level best derived from the winning 3D direction (for reference / robot fallback)
    best_dir_xy = candidate_dirs[best_dir_idx][:2]
    selected_score = float(final_scores[best_dir_idx])

    # For visualization: simple mean of all un-rotated per-view clock scores.
    mean_unrot_scores = np.mean(np.stack([v["clock_scores"] for v in views_info], axis=0), axis=0)

    # --- Visualizations ---
    # Per-view grid
    cols = 2 if len(vis_images) <= 4 else 3
    grid_vis_path = out_dir / "direction_scores_grid.png"
    _concat_images_grid(vis_images, grid_vis_path, cols=cols, padding=12)

    debug_grid_path = None
    if args.debug and debug_dir_unit is not None and debug_rows:
        debug_row_by_idx = {row.get("idx"): row for row in debug_rows}
        debug_images = []
        for item in per_view:
            img = cv2.imread(item["vis_path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            idx_val = item["index"]
            row = debug_row_by_idx.get(idx_val)
            if row is None or row.get("dx") is None or row.get("dy") is None or row.get("interpolated_vlm_score") is None:
                cv2.putText(img, "dbg: projection invalid", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)
                debug_images.append(img)
                continue
            dx = float(row["dx"])
            dy = float(row["dy"])
            if bool(item["image_rotated_180"]):
                dx, dy = -dx, -dy
            # Debug overlay convention: mirror along y-axis for intended direction interpretation.
            dx = -dx
            center = (
                int(item["display_query_point_uv"][0]),
                int(item["display_query_point_uv"][1]),
            )
            img_dbg = _draw_debug_direction_arrow(
                img,
                center,
                dx,
                dy,
                float(row["interpolated_vlm_score"]),
                float(row["weight"]),
            )
            debug_images.append(img_dbg)
        if debug_images:
            debug_grid_path = out_dir / "direction_debug.png"
            _concat_images_grid(debug_images, debug_grid_path, cols=cols, padding=12)

    # Aggregated arrows on BEV (results[0] is always BEV)
    agg_center = (int(round(grasp_pixel_uv[0])), int(round(grasp_pixel_uv[1])))
    # Start from a clean BEV image (query point only), not from an already-annotated vis image.
    agg_vis_input = draw_yellow_circle(str(bev_rgb_path), agg_center)
    distill_clock_scores = np.asarray(mean_unrot_scores, dtype=np.float32)
    geometry_clock_scores = (
        _build_clock_bin_scores_from_candidates(candidate_dirs, geometry_scores)
        if geometry_scores is not None
        else None
    )
    total_clock_scores = _build_clock_bin_scores_from_candidates(candidate_dirs, final_scores)
    agg_vis = _draw_clock_scores_world_aligned(
        agg_vis_input,
        agg_center,
        distill_clock_scores,
        geometry_clock_scores,
        total_clock_scores,
    )
    agg_vis = _draw_best_arrows_on_bev(
        agg_vis,
        agg_center,
        candidate_dirs,
        distill_best_idx=int(np.argmax(distill_scores)),
        total_best_idx=int(best_dir_idx),
        geometry_best_idx=(int(geometry_best_idx) if geometry_scores is not None else None),
    )
    # Keep rendering logic for debugging consistency, but do not persist this image.
    agg_vis_path = None

    # VLM score visualizations
    distill_vis_path = out_dir / "score_vlm.png"
    distill_contour_path = out_dir / "contour_vlm.png"
    plot_candidate_scores_like_vis(
        candidate_dirs, distill_scores, int(np.argmax(distill_scores)),
        "Distill score (projection weighted)", str(distill_vis_path),
        up_axis=LATITUDE_UP_AXIS,
        best_edge_color="black",
    )
    plot_candidate_scores_contour_like_vis(
        candidate_dirs,
        distill_scores,
        int(np.argmax(distill_scores)),
        "Distill score (projection weighted)",
        str(distill_contour_path),
        level_step=0.2,
        up_axis=LATITUDE_UP_AXIS,
        marker_direction=debug_dir_unit,
        marker_color="blue",
    )

    # Geometry score visualizations
    geometry_vis_path = None
    geometry_contour_path = None
    if geometry_scores is not None:
        geometry_vis_path = out_dir / "score_geo.png"
        geometry_contour_path = out_dir / "contour_geo.png"
        plot_candidate_scores_like_vis(
            candidate_dirs, geometry_scores, geometry_best_idx,
            "Geometry score (TSDF)", str(geometry_vis_path),
            up_axis=LATITUDE_UP_AXIS,
            best_edge_color="black",
        )
        plot_candidate_scores_contour_like_vis(
            candidate_dirs,
            geometry_scores,
            geometry_best_idx,
            "Geometry score (TSDF)",
            str(geometry_contour_path),
            level_step=0.2,
            up_axis=LATITUDE_UP_AXIS,
        )

    # Final score visualizations
    final_vis_path = out_dir / "score_total.png"
    final_contour_path = out_dir / "contour_total.png"
    plot_candidate_scores_like_vis(
        candidate_dirs, final_scores, best_dir_idx,
        "Final score (distill + geometry)", str(final_vis_path),
        up_axis=LATITUDE_UP_AXIS,
        best_edge_color="black",
    )
    plot_candidate_scores_contour_like_vis(
        candidate_dirs,
        final_scores,
        best_dir_idx,
        "Final score (distill + geometry)",
        str(final_contour_path),
        level_step=0.2,
        up_axis=LATITUDE_UP_AXIS,
    )

    combined_best_vis_path = out_dir / "best_direction_comparison_vis.png"
    best_specs = [
        {"label": "vlm", "idx": int(np.argmax(distill_scores)), "color": "cyan"},
        {"label": "total", "idx": int(best_dir_idx), "color": "red"},
    ]
    if geometry_scores is not None:
        best_specs.insert(1, {"label": "geo", "idx": int(geometry_best_idx), "color": "lime"})
    plot_best_directions_comparison(
        candidate_dirs,
        final_scores,
        best_specs,
        "Best directions comparison (vlm / geo / total)",
        str(combined_best_vis_path),
        up_axis=LATITUDE_UP_AXIS,
    )

    # Dedicated BEV arrows for vlm / geometry / total selected directions.
    bev_total_arrow_vis = agg_vis_input.copy()
    bev_total_arrow_vis = _draw_best_arrows_on_bev(
        bev_total_arrow_vis,
        agg_center,
        candidate_dirs,
        distill_best_idx=int(np.argmax(distill_scores)),
        total_best_idx=int(best_dir_idx),
        geometry_best_idx=(int(geometry_best_idx) if geometry_scores is not None else None),
    )
    bev_total_arrow_vis_path = out_dir / "bev_total_direction_arrow_vis.png"
    cv2.imwrite(str(bev_total_arrow_vis_path), bev_total_arrow_vis)

    ply_paths = None
    ply_geo_paths = None
    ply_final_paths = None
    if args.ply:
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Depth directory not found for PLY export: {depth_dir}")
        local_points, local_colors = collect_points_near_center(
            selected_indices,
            pose_map,
            grasp_world.astype(np.float32),
            PLY_LOCAL_RADIUS_M,
            str(intrinsics_path),
            str(rgb_dir),
            str(depth_dir),
        )
        local_points, local_colors = random_downsample_points(
            local_points,
            local_colors,
            max_points=PLY_LOCAL_MAX_POINTS,
            seed=0,
        )

        ply_dir = out_dir / "ply"
        ply_paths = build_scene_visualization_plys(
            str(ply_dir),
            "distill_scene",
            local_points,
            local_colors,
            candidate_dirs,
            distill_scores,
            grasp_world.astype(np.float32),
            int(np.argmax(distill_scores)),
            sphere_radius=0.15,
            arrow_length=0.18,
        )
        if geometry_scores is not None:
            ply_geo_paths = build_scene_visualization_plys(
                str(ply_dir),
                "geometry_scene",
                local_points,
                local_colors,
                candidate_dirs,
                geometry_scores,
                grasp_world.astype(np.float32),
                geometry_best_idx,
                sphere_radius=0.15,
                arrow_length=0.18,
            )
        ply_final_paths = build_scene_visualization_plys(
            str(ply_dir),
            "final_scene",
            local_points,
            local_colors,
            candidate_dirs,
            final_scores,
            grasp_world.astype(np.float32),
            best_dir_idx,
            sphere_radius=0.15,
            arrow_length=0.18,
        )

    # --- Save results ---
    three_d = {
        "scene": args.scene,
        "scene_num": f"{scene_num_int:03d}",
        "source": "active_perception/vlm_ours" if args.ap_only else "vlm_ours",
        "ap_only": bool(args.ap_only),
        "method": "multi_view_projection_weighted_distill_with_geometry",
        "distill_rho": float(DISTILL_RHO),
        "geo_lambda": float(GEO_LAMBDA),
        "geometry_enabled": geometry_scores is not None,
        "best_dir_idx": int(best_dir_idx),
        "best_direction": best_direction,
        "selected_indices": [int(i) for i in selected_indices],
        "bev_included": bool(args.add_bev),
        "num_views_in_aggregate": int(len(views_info)),
        "subset_json_path": str(subset_path),
        "distill_vis_path": str(distill_vis_path),
        "distill_contour_path": str(distill_contour_path),
        "geometry_vis_path": str(geometry_vis_path) if geometry_vis_path else None,
        "geometry_contour_path": str(geometry_contour_path) if geometry_contour_path else None,
        "final_vis_path": str(final_vis_path),
        "final_contour_path": str(final_contour_path),
        "best_direction_comparison_vis_path": str(combined_best_vis_path),
        "bev_total_direction_arrow_vis_path": str(bev_total_arrow_vis_path),
        "direction_debug_vis_path": str(debug_grid_path) if debug_grid_path else None,
        "ply_visualization_paths": ply_paths,
        "ply_geometry_paths": ply_geo_paths,
        "ply_final_paths": ply_final_paths,
    }
    with three_d_path.open("w", encoding="utf-8") as f:
        json.dump(three_d, f, indent=2)

    result = {
        "scene": args.scene,
        "scene_num": f"{scene_num_int:03d}",
        "ap_only": bool(args.ap_only),
        "provider": "gemini",
        "model": gemini_version,
        "subset_json_path": str(subset_path),
        "num_views_requested": int(args.num_views),
        "num_views_used": int(len(selected_indices)) + (1 if args.add_bev else 0),
        "bev_included": bool(args.add_bev),
        "selected_indices": [int(i) for i in selected_indices],
        "parallel_execution_sec": float(parallel_execution_sec),
        "selected_score": selected_score,
        "best_direction": best_direction,
        "per_view_results": per_view,
        "aggregated_vis_path": str(agg_vis_path) if agg_vis_path else None,
        "distill_vis_path": str(distill_vis_path),
        "distill_contour_path": str(distill_contour_path),
        "geometry_vis_path": str(geometry_vis_path) if geometry_vis_path else None,
        "geometry_contour_path": str(geometry_contour_path) if geometry_contour_path else None,
        "final_vis_path": str(final_vis_path),
        "final_contour_path": str(final_contour_path),
        "best_direction_comparison_vis_path": str(combined_best_vis_path),
        "bev_total_direction_arrow_vis_path": str(bev_total_arrow_vis_path),
        "direction_debug_vis_path": str(debug_grid_path) if debug_grid_path else None,
        "grid_vis_path": str(grid_vis_path),
        "final_3d_direction_path": str(three_d_path),
        "ply_visualization_paths": ply_paths,
        "ply_geometry_paths": ply_geo_paths,
        "ply_final_paths": ply_final_paths,
    }

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== VLM aggregation done ===")
    print(f"scene: {args.scene} / scene_num: {scene_num_int:03d}")
    print(f"selected indices: {selected_indices}")
    print(f"selected score: {selected_score:.4f}")
    print(f"best 3D direction: {[round(x,4) for x in best_direction]}")
    print(f"saved: {result_path}")
    print(f"saved: {grid_vis_path}")
    print(f"saved: {distill_vis_path}")
    print(f"saved: {distill_contour_path}")
    if geometry_vis_path:
        print(f"saved: {geometry_vis_path}")
    if geometry_contour_path:
        print(f"saved: {geometry_contour_path}")
    print(f"saved: {final_vis_path}")
    print(f"saved: {final_contour_path}")
    print(f"saved: {combined_best_vis_path}")
    print(f"saved: {bev_total_arrow_vis_path}")
    if debug_grid_path is not None:
        print(f"saved: {debug_grid_path}")
    print(f"saved: {three_d_path}")
    if ply_paths is not None:
        print(f"saved: {ply_paths.get('rgb')}")
        print(f"saved: {ply_paths.get('gray')}")
    if ply_geo_paths is not None:
        print(f"saved: {ply_geo_paths.get('rgb')}")
    if ply_final_paths is not None:
        print(f"saved: {ply_final_paths.get('rgb')}")


if __name__ == "__main__":
    main()
