from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROMPT_SINGLE = """You are a professional roboticist, and your task is to perform manipulation planning based on a single image.

Key assumptions:
- The image shows a static scene.
- The yellow circle marks the target object.
- The floor is horizontal.

Direction definition:
- Directions are expressed using clock directions relative to image frame.
- 12 o'clock means upward toward the TOP EDGE of the image.
- 6 o'clock means downward toward the BOTTOM EDGE of the image.
- 3 o'clock is to the RIGHT, 9 o'clock is to the LEFT.
- Clock directions increase clockwise in 30-degree increments.
- Each clock direction corresponds to a candidate short-distance translation direction of the target object in that image.
- Because of depth ambiguity, each clock direction may correspond to multiple possible 3D directions in the world.

Instructions:
1) Identify the target object at the yellow circle and classify its material using: metal, glass, ceramic, plastic, paper, rubber.
2) Identify nearby objects in contact or near-contact with the target.
3) Analyze contacts and near-contacts.
4) For each clock direction d in {1..12}, assign a safety score in [0,1].
5) Choose one retrieval speed recommendation: fast or slow.

Output JSON:
{
  "Target object": "target(material=...)",
  "Surrounding objects": ["obj1(material=...)", "..."],
  "Physical relationships": "Brief scene relationship description.",
  "Speed": "fast/slow",
  "Reason": "Brief explanation.",
  "Direction scores": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
  "Best direction reason": "Why the highest-scored direction is safest.",
  "Worst direction reason": "Why the lowest-scored direction is most dangerous."
}

Rules:
- Use exactly 12 direction scores.
- Use two decimal places for scores.
- Return only JSON."""


DISTILL_NUM_DIRECTIONS = 1000
DISTILL_RHO_M = 0.10
DISTILL_WEIGHT_POWER = 1.5
DISTILL_WEIGHT_MIN_RATIO = 0.05
GEO_SWEEP_START_M = 0.03
GEO_SWEEP_END_M = 0.18
GEO_SWEEP_NUM_STEPS = 16
GEO_COLLISION_CLEARANCE_M = 0.015
GEO_SAFE_CLEARANCE_M = 0.10
GEO_TARGET_IGNORE_RADIUS_M = 0.04
GEO_LAMBDA = 0.5


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_dir:
        return Path(args.experiment_dir).expanduser().resolve()
    root = Path(args.output_root).expanduser().resolve()
    return root / str(args.experiment_name)


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
    out = []
    for item in raw.replace(",", " ").split():
        out.append(int(item))
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
        "raw": data,
    }


def _load_pose_map(views_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    pose_doc = _load_json(views_dir / "pose.json")
    pose_map = {}
    for pose in pose_doc.get("poses", []):
        idx = int(pose["index"])
        c2w = np.asarray(pose["cam_matrix"], dtype=np.float64)
        pose_map[idx] = {
            "pose": pose,
            "c2w": c2w,
            "w2c": np.linalg.inv(c2w),
        }
    if not pose_map:
        raise ValueError(f"No poses found in {views_dir / 'pose.json'}")
    return pose_map, pose_doc


def _read_depth_m(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path))
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def _sample_depth_m(depth_m: np.ndarray, u: int, v: int, radius: int = 2) -> float:
    h, w = depth_m.shape[:2]
    x0 = max(0, int(u) - radius)
    x1 = min(w, int(u) + radius + 1)
    y0 = max(0, int(v) - radius)
    y1 = min(h, int(v) + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError(f"No valid depth near target pixel {(u, v)}")
    return float(np.median(valid))


def _unproject_pixel_to_world(u: float, v: float, z: float, c2w: np.ndarray, intr: dict[str, Any]) -> np.ndarray:
    x = (float(u) - intr["cx"]) * float(z) / intr["fx"]
    y = (float(v) - intr["cy"]) * float(z) / intr["fy"]
    p_cam = np.asarray([x, y, float(z), 1.0], dtype=np.float64)
    return (c2w @ p_cam)[:3]


def _project_world_to_pixel(point_world: np.ndarray, c2w: np.ndarray, intr: dict[str, Any]) -> tuple[float, float, float] | None:
    p_world = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    p_cam = np.linalg.inv(c2w) @ p_world
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    u = intr["fx"] * float(p_cam[0]) / z + intr["cx"]
    v = intr["fy"] * float(p_cam[1]) / z + intr["cy"]
    return float(u), float(v), z


def _target_world_from_args(args: argparse.Namespace, exp_dir: Path, views_dir: Path, pose_map: dict[int, dict[str, Any]], intr: dict[str, Any]) -> np.ndarray:
    manual = _parse_vec3(args.grasp_world, "--grasp-world")
    if manual is not None:
        return manual

    if args.target_view_index is None or args.target_pixel is None:
        existing = exp_dir / "offline" / "target_point.json"
        if existing.is_file():
            data = _load_json(existing)
            return np.asarray(data["grasp_position_world"], dtype=np.float64)
        raise ValueError("Provide --grasp-world x y z, or --target-view-index IDX --target-pixel U V")

    target_idx = int(args.target_view_index)
    if target_idx not in pose_map:
        raise ValueError(f"--target-view-index {target_idx} is not in pose.json")
    if len(args.target_pixel) != 2:
        raise ValueError("--target-pixel expects U V")
    u, v = int(args.target_pixel[0]), int(args.target_pixel[1])
    depth = _read_depth_m(views_dir / "depth" / f"{target_idx:04d}.png")
    z = _sample_depth_m(depth, u, v)
    return _unproject_pixel_to_world(u, v, z, pose_map[target_idx]["c2w"], intr)


def _draw_target_circle(image: Image.Image, uv: tuple[float, float], radius: int = 18) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    u, v = float(uv[0]), float(uv[1])
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(0, 0, 0))
    inner = max(2, radius - 4)
    draw.ellipse((u - inner, v - inner, u + inner, v + inner), fill=(255, 230, 0))
    return out


def _prepare_vlm_inputs(
    exp_dir: Path,
    view_indices: list[int],
    grasp_world: np.ndarray,
    views_dir: Path,
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
) -> list[dict[str, Any]]:
    input_dir = exp_dir / "offline" / "vlm_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for idx in view_indices:
        rgb_path = views_dir / "rgb" / f"{idx:04d}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB view: {rgb_path}")
        projected = _project_world_to_pixel(grasp_world, pose_map[idx]["c2w"], intr)
        if projected is None:
            prepared.append({"index": idx, "usable": False, "reason": "target_behind_camera", "rgb_path": str(rgb_path)})
            continue
        u, v, z = projected
        usable = 0.0 <= u < intr["width"] and 0.0 <= v < intr["height"]
        out_path = input_dir / f"idx_{idx:04d}_input_with_query.png"
        if usable:
            marked = _draw_target_circle(Image.open(rgb_path), (u, v))
            marked.save(out_path)
        prepared.append(
            {
                "index": idx,
                "usable": bool(usable),
                "target_pixel_uv": [float(u), float(v)],
                "target_depth_m": float(z),
                "rgb_path": str(rgb_path),
                "input_image_path": str(out_path) if usable else None,
            }
        )
    usable_count = sum(1 for item in prepared if item.get("usable"))
    if usable_count == 0:
        raise RuntimeError("Target point is not visible in any selected view")
    return prepared


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left == -1 or right == -1 or right <= left:
        raise ValueError("Could not find JSON object in API response")
    return stripped[left : right + 1]


def _parse_vlm_response(text: str) -> dict[str, Any]:
    parsed = json.loads(_extract_json_text(text))
    scores = parsed.get("Direction scores", parsed.get("direction_scores"))
    if not isinstance(scores, list) or len(scores) != 12:
        raise ValueError("VLM response must contain 12 Direction scores")
    parsed["Direction scores"] = [float(np.clip(float(s), 0.0, 1.0)) for s in scores]
    return parsed


def _call_gemini(prompt_text: str, image_path: Path, model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("google-genai is not installed. Install it or run with cached responses.") from exc
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client()
    data = image_path.read_bytes()
    response = client.models.generate_content(
        model=model,
        contents=[
            prompt_text,
            types.Part.from_bytes(data=data, mime_type="image/png"),
        ],
    )
    return str(response.text)


def _run_vlm(
    exp_dir: Path,
    prepared: list[dict[str, Any]],
    call_api: bool,
    provider: str,
    model: str,
) -> list[dict[str, Any]]:
    result_dir = exp_dir / "offline" / "vlm_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in prepared:
        idx = int(item["index"])
        if not item.get("usable"):
            continue
        raw_path = result_dir / f"idx_{idx:04d}_response_raw.txt"
        json_path = result_dir / f"idx_{idx:04d}_response.json"
        if json_path.is_file():
            response_json = _load_json(json_path)
            if "Direction scores" not in response_json and "direction_scores" in response_json:
                response_json["Direction scores"] = response_json["direction_scores"]
            if not isinstance(response_json.get("Direction scores"), list):
                raise ValueError(f"Cached response has no Direction scores: {json_path}")
            used_cache = True
        else:
            if not call_api:
                raise RuntimeError(
                    f"No cached VLM response for view {idx}. Re-run with --call-api, or create {json_path} manually."
                )
            start = time.time()
            if provider != "gemini":
                raise ValueError(f"Unsupported provider: {provider}")
            raw_text = _call_gemini(PROMPT_SINGLE, Path(item["input_image_path"]), model)
            raw_path.write_text(raw_text, encoding="utf-8")
            response_json = _parse_vlm_response(raw_text)
            response_json["elapsed_s"] = time.time() - start
            response_json["provider"] = provider
            response_json["model"] = model
            _write_json(json_path, response_json)
            used_cache = False
        results.append(
            {
                "index": idx,
                "target_pixel_uv": item["target_pixel_uv"],
                "input_image_path": item["input_image_path"],
                "response_json_path": str(json_path),
                "used_cache": bool(used_cache),
                "direction_scores": [float(s) for s in response_json["Direction scores"]],
                "response": response_json,
            }
        )
    if not results:
        raise RuntimeError("No VLM results available")
    _write_json(result_dir / "vlm_results.json", {"views": results})
    return results


def _fibonacci_upper_hemisphere(n: int) -> np.ndarray:
    n = max(1, int(n))
    dirs = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    i = 0
    while len(dirs) < n:
        z = (i + 0.5) / max(n, 1)
        theta = i * golden
        r = math.sqrt(max(0.0, 1.0 - z * z))
        dirs.append([r * math.cos(theta), r * math.sin(theta), z])
        i += 1
    return np.asarray(dirs, dtype=np.float64)


def _interpolate_clock_score(clock_scores: np.ndarray, dx: float, dy: float) -> float:
    scores = np.asarray(clock_scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != 12:
        raise ValueError("clock_scores must have length 12")
    angle_clock_deg = float(np.degrees(np.arctan2(float(dx), float(-dy))) % 360.0)
    clock_pos = angle_clock_deg / 30.0
    pos = (clock_pos - 1.0) % 12.0
    lo = int(np.floor(pos)) % 12
    hi = (lo + 1) % 12
    t = float(pos - np.floor(pos))
    return float((1.0 - t) * scores[lo] + t * scores[hi])


def _distill_vlm_scores(
    candidate_dirs: np.ndarray,
    grasp_world: np.ndarray,
    vlm_results: list[dict[str, Any]],
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
) -> np.ndarray:
    grasp = np.asarray(grasp_world, dtype=np.float64)
    out = np.zeros((candidate_dirs.shape[0],), dtype=np.float64)
    for ci, direction in enumerate(candidate_dirs):
        displaced = grasp + DISTILL_RHO_M * direction
        total_w = 0.0
        total_ws = 0.0
        for result in vlm_results:
            idx = int(result["index"])
            c2w = pose_map[idx]["c2w"]
            p0 = _project_world_to_pixel(grasp, c2w, intr)
            p1 = _project_world_to_pixel(displaced, c2w, intr)
            if p0 is None or p1 is None:
                continue
            u0, v0, z0 = p0
            u1, v1, _ = p1
            dx = u1 - u0
            dy = v1 - v0
            d_i = float(math.hypot(dx, dy))
            f_avg = 0.5 * (intr["fx"] + intr["fy"])
            l_i = f_avg * DISTILL_RHO_M / max(abs(z0), 1e-8)
            ratio = d_i / max(l_i, 1e-8)
            if ratio < DISTILL_WEIGHT_MIN_RATIO:
                continue
            weight = ratio ** DISTILL_WEIGHT_POWER
            score = _interpolate_clock_score(np.asarray(result["direction_scores"], dtype=np.float64), dx, dy)
            total_w += weight
            total_ws += weight * score
        out[ci] = total_ws / max(total_w, 1e-8)
    return out.astype(np.float32)


def _read_occupied_points_from_ply(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.empty((0, 3), dtype=np.float32)
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    vertex_count = 0
    header_end = None
    prop_names = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
        elif stripped.startswith("property"):
            prop_names.append(stripped.split()[-1])
        elif stripped == "end_header":
            header_end = i + 1
            break
    if header_end is None or vertex_count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    try:
        state_idx = prop_names.index("state")
    except ValueError:
        state_idx = None
    points = []
    for line in lines[header_end : header_end + vertex_count]:
        parts = line.split()
        if len(parts) < 3:
            continue
        if state_idx is not None and int(float(parts[state_idx])) != 1:
            continue
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(points, dtype=np.float32)


def _geometry_scores_from_occupancy(
    candidate_dirs: np.ndarray,
    grasp_world: np.ndarray,
    occupied_points: np.ndarray,
    target_ignore_radius_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if occupied_points.shape[0] == 0:
        return np.ones((candidate_dirs.shape[0],), dtype=np.float32), {"occupied_points_used": 0}
    grasp = np.asarray(grasp_world, dtype=np.float32)
    dist_to_grasp = np.linalg.norm(occupied_points - grasp[None, :], axis=1)
    obstacle_points = occupied_points[dist_to_grasp > float(target_ignore_radius_m)]
    if obstacle_points.shape[0] == 0:
        return np.ones((candidate_dirs.shape[0],), dtype=np.float32), {
            "occupied_points_used": 0,
            "occupied_points_ignored_near_target": int(occupied_points.shape[0]),
        }

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(obstacle_points)
        use_tree = True
    except Exception:
        tree = None
        use_tree = False

    sweep = np.linspace(GEO_SWEEP_START_M, GEO_SWEEP_END_M, GEO_SWEEP_NUM_STEPS, dtype=np.float32)
    scores = np.zeros((candidate_dirs.shape[0],), dtype=np.float32)
    min_clearances = np.zeros_like(scores)
    for i, direction in enumerate(candidate_dirs.astype(np.float32)):
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        samples = grasp[None, :] + sweep[:, None] * direction[None, :]
        if use_tree:
            dists, _ = tree.query(samples)
            min_dist = float(np.min(dists))
        else:
            min_dist = float(np.min(np.linalg.norm(samples[:, None, :] - obstacle_points[None, :, :], axis=2)))
        min_clearances[i] = min_dist
        score = (min_dist - GEO_COLLISION_CLEARANCE_M) / max(GEO_SAFE_CLEARANCE_M - GEO_COLLISION_CLEARANCE_M, 1e-8)
        scores[i] = float(np.clip(score, 0.0, 1.0))
    return scores, {
        "occupied_points_total": int(occupied_points.shape[0]),
        "occupied_points_used": int(obstacle_points.shape[0]),
        "target_ignore_radius_m": float(target_ignore_radius_m),
        "min_clearance_m": min_clearances.tolist(),
    }


def _write_direction_ply(path: Path, center: np.ndarray, direction: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    points = [center, center + 0.20 * direction]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex 2\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element edge 1\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
        f.write("0 1\n")


def _fuse_direction(
    exp_dir: Path,
    grasp_world: np.ndarray,
    view_indices: list[int],
    vlm_results: list[dict[str, Any]],
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
    alpha: float,
    target_ignore_radius_m: float,
) -> dict[str, Any]:
    candidate_dirs = _fibonacci_upper_hemisphere(DISTILL_NUM_DIRECTIONS)
    vlm_scores = _distill_vlm_scores(candidate_dirs, grasp_world, vlm_results, pose_map, intr)
    occupied_points = _read_occupied_points_from_ply(exp_dir / "occupancy_final.ply")
    geometry_scores, geometry_meta = _geometry_scores_from_occupancy(
        candidate_dirs,
        grasp_world,
        occupied_points,
        target_ignore_radius_m=target_ignore_radius_m,
    )
    final_scores = float(alpha) * vlm_scores + (1.0 - float(alpha)) * geometry_scores
    best_idx = int(np.argmax(final_scores))
    vlm_best_idx = int(np.argmax(vlm_scores))
    geometry_best_idx = int(np.argmax(geometry_scores))

    out_dir = exp_dir / "offline"
    _write_direction_ply(out_dir / "best_direction.ply", grasp_world, candidate_dirs[best_idx], (255, 0, 0))
    result = {
        "format": "fetchbench_real_offline_direction_v1",
        "experiment_dir": str(exp_dir),
        "view_indices": [int(v) for v in view_indices],
        "grasp_position_world": [float(v) for v in grasp_world.tolist()],
        "alpha_vlm": float(alpha),
        "best_idx": best_idx,
        "best_direction": [float(v) for v in candidate_dirs[best_idx].tolist()],
        "best_score": float(final_scores[best_idx]),
        "vlm_best_idx": vlm_best_idx,
        "vlm_best_direction": [float(v) for v in candidate_dirs[vlm_best_idx].tolist()],
        "geometry_best_idx": geometry_best_idx,
        "geometry_best_direction": [float(v) for v in candidate_dirs[geometry_best_idx].tolist()],
        "scores": {
            "final": final_scores.astype(float).tolist(),
            "vlm": vlm_scores.astype(float).tolist(),
            "geometry": geometry_scores.astype(float).tolist(),
        },
        "geometry": geometry_meta,
        "best_direction_ply": str(out_dir / "best_direction.ply"),
    }
    _write_json(out_dir / "final_3d_direction.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline FetchBench real VLM+geometry direction pipeline")
    parser.add_argument("--output-root", default="/workspace/ros2_ws/ours_experiment")
    parser.add_argument("--experiment-name", default="ex2")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--view-indices", default="all", help="'all' or comma/space separated view indices")
    parser.add_argument("--grasp-world", nargs="+", default=None, help="Manual grasp point in world frame: x y z")
    parser.add_argument("--target-view-index", type=int, default=None, help="View index containing a clicked target pixel")
    parser.add_argument("--target-pixel", nargs=2, type=int, default=None, metavar=("U", "V"))
    parser.add_argument("--call-api", action="store_true", help="Actually call the configured VLM API for missing responses")
    parser.add_argument("--api-provider", choices=["gemini"], default="gemini")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--alpha-vlm", type=float, default=GEO_LAMBDA, help="final = alpha*VLM + (1-alpha)*geometry")
    parser.add_argument("--target-ignore-radius-m", type=float, default=GEO_TARGET_IGNORE_RADIUS_M)
    parser.add_argument("--prepare-only", action="store_true", help="Only write marked VLM input images and metadata")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    exp_dir = _resolve_experiment_dir(args)
    views_dir = exp_dir / "views"
    if not views_dir.is_dir():
        raise FileNotFoundError(f"Views directory not found: {views_dir}")
    intr = _load_intrinsics(views_dir)
    pose_map, _pose_doc = _load_pose_map(views_dir)
    available = sorted(pose_map.keys())
    view_indices = _parse_indices(args.view_indices, available)
    grasp_world = _target_world_from_args(args, exp_dir, views_dir, pose_map, intr)

    offline_dir = exp_dir / "offline"
    _write_json(
        offline_dir / "target_point.json",
        {
            "grasp_position_world": [float(v) for v in grasp_world.tolist()],
            "source": "grasp_world" if args.grasp_world is not None else "target_pixel_or_cached",
        },
    )
    prepared = _prepare_vlm_inputs(exp_dir, view_indices, grasp_world, views_dir, pose_map, intr)
    _write_json(offline_dir / "prepared_views.json", {"views": prepared})

    print(f"Prepared {sum(1 for item in prepared if item.get('usable'))} VLM input images in {offline_dir / 'vlm_inputs'}")
    if args.prepare_only:
        print("prepare-only requested; stopping before VLM/API and fusion")
        return

    vlm_results = _run_vlm(
        exp_dir=exp_dir,
        prepared=prepared,
        call_api=bool(args.call_api),
        provider=str(args.api_provider),
        model=str(args.model),
    )
    result = _fuse_direction(
        exp_dir=exp_dir,
        grasp_world=grasp_world,
        view_indices=view_indices,
        vlm_results=vlm_results,
        pose_map=pose_map,
        intr=intr,
        alpha=float(args.alpha_vlm),
        target_ignore_radius_m=float(args.target_ignore_radius_m),
    )
    print(f"Best direction: {result['best_direction']}")
    print(f"Saved: {exp_dir / 'offline' / 'final_3d_direction.json'}")


if __name__ == "__main__":
    main()
