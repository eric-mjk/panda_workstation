from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from .select_subset import DEFAULT_NUM_STARTS, DEFAULT_SIM_WEIGHT, select_best_subset_indices
except ImportError:
    from select_subset import DEFAULT_NUM_STARTS, DEFAULT_SIM_WEIGHT, select_best_subset_indices


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
MIN_LATITUDE_DEG = 10.0
DISTILL_WEIGHT_POWER = 1.5
DISTILL_WEIGHT_MIN_RATIO = 0.05
TSDF_VOXEL_SIZE_M = 0.01
TSDF_TRUNCATION_M = 0.04
TSDF_TARGET_MASK_DILATE_PX = 0
TSDF_EXCLUDE_BACKGROUND = False
GEO_TARGET_POINT_STRIDE = 4
GEO_TARGET_POINT_MAX_POINTS = 2500
GEO_SWEEP_START_M = 0.00
GEO_SWEEP_END_M = 0.15
GEO_SWEEP_NUM_STEPS = 16
GEO_COLLISION_CLEARANCE_M = 0.01
GEO_SAFE_CLEARANCE_M = 0.10
GEO_QUANTILE = 0.10
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


def _resolve_views_dir(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    if (exp_dir / "pose.json").is_file() and (exp_dir / "intrinsics.json").is_file():
        return exp_dir
    legacy = exp_dir / "views"
    if (legacy / "pose.json").is_file() and (legacy / "intrinsics.json").is_file():
        return legacy
    return exp_dir


def _resolve_mask_dir(exp_dir: Path, views_dir: Path, explicit: str = "") -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates.extend([exp_dir / "masks", views_dir / "class"])
    for path in candidates:
        if path.is_dir():
            return path
    return None


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


def _scaled_intrinsics(intr: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    width = int(width)
    height = int(height)
    base_w = max(float(intr["width"]), 1.0)
    base_h = max(float(intr["height"]), 1.0)
    sx = float(width) / base_w
    sy = float(height) / base_h
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
    if np.issubdtype(depth.dtype, np.integer):
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


def _unproject_pixel_to_world(
    u: float,
    v: float,
    z: float,
    c2w: np.ndarray,
    intr: dict[str, Any],
) -> np.ndarray:
    x = (float(u) - intr["cx"]) * float(z) / intr["fx"]
    y = (float(v) - intr["cy"]) * float(z) / intr["fy"]
    p_cam = np.asarray([x, y, float(z), 1.0], dtype=np.float64)
    return (c2w @ p_cam)[:3]


def _project_world_to_pixel(
    point_world: np.ndarray,
    c2w: np.ndarray,
    intr: dict[str, Any],
) -> tuple[float, float, float] | None:
    p_world = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    p_cam = np.linalg.inv(c2w) @ p_world
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    u = intr["fx"] * float(p_cam[0]) / z + intr["cx"]
    v = intr["fy"] * float(p_cam[1]) / z + intr["cy"]
    return float(u), float(v), z


def _target_world_from_args(
    args: argparse.Namespace,
    exp_dir: Path,
    views_dir: Path,
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
) -> np.ndarray:
    manual = _parse_vec3(args.grasp_world, "--grasp-world")
    if manual is not None:
        return manual

    if args.target_view_index is None or args.target_pixel is None:
        for existing in (exp_dir / "target_point.json", exp_dir / "offline" / "target_point.json"):
            if existing.is_file():
                data = _load_json(existing)
                return np.asarray(data["grasp_position_world"], dtype=np.float64)
        raise ValueError("Provide --grasp-world x y z, or --target-view-index IDX --target-pixel U V")

    target_idx = int(args.target_view_index)
    if target_idx not in pose_map:
        raise ValueError(f"--target-view-index {target_idx} is not in pose.json")
    if len(args.target_pixel) != 2:
        raise ValueError("--target-pixel expects U V")
    u_rgb, v_rgb = float(args.target_pixel[0]), float(args.target_pixel[1])
    depth = _read_depth_m(views_dir / "depth" / f"{target_idx:04d}.png")
    rgb_path = views_dir / "rgb" / f"{target_idx:04d}.png"
    rgb_w, rgb_h = _image_size(rgb_path) if rgb_path.is_file() else (int(intr["width"]), int(intr["height"]))
    depth_h, depth_w = depth.shape[:2]
    depth_intr = _scaled_intrinsics(intr, depth_w, depth_h)
    u_depth = u_rgb * float(depth_w) / max(float(rgb_w), 1.0)
    v_depth = v_rgb * float(depth_h) / max(float(rgb_h), 1.0)
    z = _sample_depth_m(depth, int(round(u_depth)), int(round(v_depth)))
    return _unproject_pixel_to_world(u_depth, v_depth, z, pose_map[target_idx]["c2w"], depth_intr)


def _draw_target_circle(image: Image.Image, uv: tuple[float, float], radius: int = 18) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    u, v = float(uv[0]), float(uv[1])
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(0, 0, 0))
    inner = max(2, radius - 4)
    draw.ellipse((u - inner, v - inner, u + inner, v + inner), fill=(255, 230, 0))
    return out


def _should_rotate_image_180(c2w: np.ndarray) -> bool:
    return float(np.asarray(c2w, dtype=np.float64)[2, 1]) < 0.0


def _rotate_point_180(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    u, v = point
    return float(width - 1 - u), float(height - 1 - v)


def _rotate_direction_scores_180(scores: list[float] | np.ndarray) -> list[float]:
    return [float(v) for v in np.roll(np.asarray(scores, dtype=np.float32), -6).tolist()]


def _prepare_vlm_inputs(
    exp_dir: Path,
    view_indices: list[int],
    grasp_world: np.ndarray,
    views_dir: Path,
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    input_dir = output_dir if output_dir is not None else exp_dir / "rgb_vlm_in"
    input_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for idx in view_indices:
        rgb_path = views_dir / "rgb" / f"{idx:04d}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB view: {rgb_path}")
        rgb_w, rgb_h = _image_size(rgb_path)
        rgb_intr = _scaled_intrinsics(intr, rgb_w, rgb_h)
        projected = _project_world_to_pixel(grasp_world, pose_map[idx]["c2w"], rgb_intr)
        if projected is None:
            prepared.append({"index": idx, "usable": False, "reason": "target_behind_camera", "rgb_path": str(rgb_path)})
            continue
        u, v, z = projected
        usable = 0.0 <= u < rgb_w and 0.0 <= v < rgb_h
        out_path = input_dir / f"idx_{idx:04d}_input_with_query.png"
        rotate_180 = _should_rotate_image_180(pose_map[idx]["c2w"])
        display_u, display_v = _rotate_point_180((u, v), int(rgb_w), int(rgb_h)) if rotate_180 else (u, v)
        if usable:
            marked = _draw_target_circle(Image.open(rgb_path), (u, v))
            if rotate_180:
                marked = marked.transpose(Image.Transpose.ROTATE_180)
            marked.save(out_path)
        prepared.append(
            {
                "index": idx,
                "usable": bool(usable),
                "target_pixel_uv": [float(u), float(v)],
                "display_target_pixel_uv": [float(display_u), float(display_v)],
                "target_depth_m": float(z),
                "image_rotated_180": bool(rotate_180),
                "rgb_size": [int(rgb_w), int(rgb_h)],
                "projection_intrinsics": {
                    "width": int(rgb_intr["width"]),
                    "height": int(rgb_intr["height"]),
                    "fx": float(rgb_intr["fx"]),
                    "fy": float(rgb_intr["fy"]),
                    "cx": float(rgb_intr["cx"]),
                    "cy": float(rgb_intr["cy"]),
                },
                "rgb_path": str(rgb_path),
                "input_image_path": str(out_path) if usable else None,
            }
        )
    usable_count = sum(1 for item in prepared if item.get("usable"))
    if usable_count == 0:
        raise RuntimeError("Target point is not visible in any selected view")
    return prepared


def _overlay_font(size_px: int):
    try:
        from PIL import ImageFont

        for font_path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ):
            if Path(font_path).is_file():
                return ImageFont.truetype(font_path, size=int(size_px))
    except Exception:
        pass
    return None


def _text_size(draw: ImageDraw.ImageDraw, text: str, font=None) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0]), int(box[3] - box[1])
    except Exception:
        return tuple(int(v) for v in draw.textsize(text, font=font))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    head_len = max(10.0, float(width) * 4.0)
    head_angle = math.radians(28.0)
    for sign in (-1.0, 1.0):
        hx = ex - head_len * math.cos(angle + sign * head_angle)
        hy = ey - head_len * math.sin(angle + sign * head_angle)
        draw.line((ex, ey, hx, hy), fill=color, width=width)


def _score_color(score: float) -> tuple[int, int, int]:
    score = float(np.clip(score, 0.0, 1.0))
    red = int(round(255.0 * (1.0 - score)))
    green = int(round(220.0 * score + 30.0))
    return red, green, 40


def _draw_vlm_score_overlay(
    input_image_path: Path,
    scores: list[float],
    output_path: Path,
    center_uv: tuple[float, float] | None = None,
) -> None:
    image = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    if center_uv is None:
        cx = w * 0.5
        cy = h * 0.5
    else:
        cx = float(np.clip(center_uv[0], 0.0, max(0, w - 1)))
        cy = float(np.clip(center_uv[1], 0.0, max(0, h - 1)))
    radius = min(w, h) * 0.24
    margin = max(24.0, min(w, h) * 0.035)
    radius = min(radius, max(margin, cx - margin), max(margin, w - 1 - cx), max(margin, cy - margin), max(margin, h - 1 - cy))
    radius = max(18.0, radius)
    line_width = max(2, int(round(min(w, h) / 180.0)))
    font = _overlay_font(max(18, int(round(min(w, h) / 24.0))))
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(12)
    for i, score in enumerate(scores_arr):
        angle = math.radians(i * 30.0)
        tip_x = cx + radius * math.sin(angle)
        tip_y = cy - radius * math.cos(angle)
        color = _score_color(float(score))
        _draw_arrow(draw, (cx, cy), (tip_x, tip_y), color, line_width)
        text = f"{float(score):.2f}"
        tw, th = _text_size(draw, text, font=font)
        text_radius = radius + max(18.0, min(w, h) * 0.028)
        x = cx + text_radius * math.sin(angle)
        y = cy - text_radius * math.cos(angle)
        x = float(np.clip(x, tw / 2 + 4, w - tw / 2 - 4))
        y = float(np.clip(y, th / 2 + 4, h - th / 2 - 4))
        pad = max(4, int(round(min(w, h) / 150.0)))
        draw.rectangle((x - tw / 2 - pad, y - th / 2 - pad, x + tw / 2 + pad, y + th / 2 + pad), fill=(0, 0, 0))
        draw.text((x - tw / 2, y - th / 2), text, fill=(255, 255, 255), font=font)
    dot_r = max(4, int(round(min(w, h) / 150.0)))
    draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=(255, 230, 0), outline=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    result_dir = output_dir if output_dir is not None else exp_dir / "rgb_vlm_out"
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    usable_items = [item for item in prepared if item.get("usable")]
    print(
        f"[VLM] Starting VLM stage: usable_views={len(usable_items)} provider={provider} model={model} output={result_dir}",
        flush=True,
    )
    for view_counter, item in enumerate(usable_items, start=1):
        idx = int(item["index"])
        raw_path = result_dir / f"idx_{idx:04d}_response_raw.txt"
        json_path = result_dir / f"idx_{idx:04d}_response.json"
        print(f"[VLM] ({view_counter}/{len(usable_items)}) view={idx:04d} input={item.get('input_image_path')}", flush=True)
        if json_path.is_file():
            print(f"[VLM] ({view_counter}/{len(usable_items)}) view={idx:04d} using cached response: {json_path}", flush=True)
            response_json = _load_json(json_path)
            if "Direction scores" not in response_json and "direction_scores" in response_json:
                response_json["Direction scores"] = response_json["direction_scores"]
            if not isinstance(response_json.get("Direction scores"), list):
                raise ValueError(f"Cached response has no Direction scores: {json_path}")
            cached_rotation = bool(response_json.get("image_rotated_180", False))
            if cached_rotation != bool(item.get("image_rotated_180", False)):
                raise ValueError(
                    f"Cached response rotation flag does not match current pose for view {idx}: {json_path}"
                )
            used_cache = True
        else:
            if not call_api:
                raise RuntimeError(
                    f"No cached VLM response for view {idx}. Re-run with --call-api, or create {json_path} manually."
                )
            start = time.time()
            if provider != "gemini":
                raise ValueError(f"Unsupported provider: {provider}")
            print(f"[VLM] ({view_counter}/{len(usable_items)}) view={idx:04d} calling {provider} API...", flush=True)
            raw_text = _call_gemini(PROMPT_SINGLE, Path(item["input_image_path"]), model)
            raw_path.write_text(raw_text, encoding="utf-8")
            response_json = _parse_vlm_response(raw_text)
            elapsed_s = time.time() - start
            response_json["elapsed_s"] = elapsed_s
            response_json["provider"] = provider
            response_json["model"] = model
            response_json["image_rotated_180"] = bool(item.get("image_rotated_180", False))
            _write_json(json_path, response_json)
            print(
                f"[VLM] ({view_counter}/{len(usable_items)}) view={idx:04d} API response saved in {elapsed_s:.1f}s: {json_path}",
                flush=True,
            )
            used_cache = False
        direction_scores = [float(s) for s in response_json["Direction scores"]]
        overlay_path = result_dir / f"idx_{idx:04d}_scores.png"
        if item.get("input_image_path"):
            center = item.get("display_target_pixel_uv") or item.get("target_pixel_uv")
            center_uv = (float(center[0]), float(center[1])) if isinstance(center, list) and len(center) == 2 else None
            _draw_vlm_score_overlay(Path(item["input_image_path"]), direction_scores, overlay_path, center_uv=center_uv)
            print(f"[VLM] ({view_counter}/{len(usable_items)}) view={idx:04d} score overlay saved: {overlay_path}", flush=True)
        if bool(item.get("image_rotated_180", False)):
            direction_scores = _rotate_direction_scores_180(direction_scores)
        results.append(
            {
                "index": idx,
                "target_pixel_uv": item["target_pixel_uv"],
                "display_target_pixel_uv": item["display_target_pixel_uv"],
                "image_rotated_180": bool(item.get("image_rotated_180", False)),
                "input_image_path": item["input_image_path"],
                "response_json_path": str(json_path),
                "score_overlay_path": str(overlay_path),
                "used_cache": bool(used_cache),
                "direction_scores": direction_scores,
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
    # Match simulation_source/src/vlm_aggregate.py. This mirrors the image x-axis
    # before clock interpolation so the real pipeline and paper pipeline agree.
    angle_clock_deg = float(np.degrees(np.arctan2(float(-dx), float(-dy))) % 360.0)
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
    views_dir: Path,
) -> np.ndarray:
    grasp = np.asarray(grasp_world, dtype=np.float64)
    out = np.zeros((candidate_dirs.shape[0],), dtype=np.float64)
    view_intr_cache: dict[int, dict[str, Any]] = {}
    for ci, direction in enumerate(candidate_dirs):
        displaced = grasp + DISTILL_RHO_M * direction
        total_w = 0.0
        total_ws = 0.0
        for result in vlm_results:
            idx = int(result["index"])
            c2w = pose_map[idx]["c2w"]
            if idx not in view_intr_cache:
                rgb_path = views_dir / "rgb" / f"{idx:04d}.png"
                if rgb_path.is_file():
                    rgb_w, rgb_h = _image_size(rgb_path)
                    view_intr_cache[idx] = _scaled_intrinsics(intr, rgb_w, rgb_h)
                else:
                    view_intr_cache[idx] = intr
            view_intr = view_intr_cache[idx]
            p0 = _project_world_to_pixel(grasp, c2w, view_intr)
            p1 = _project_world_to_pixel(displaced, c2w, view_intr)
            if p0 is None or p1 is None:
                continue
            u0, v0, z0 = p0
            u1, v1, _ = p1
            dx = u1 - u0
            dy = v1 - v0
            d_i = float(math.hypot(dx, dy))
            f_avg = 0.5 * (view_intr["fx"] + view_intr["fy"])
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


def _read_class_map(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


def _maybe_dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    radius_px = int(radius_px)
    if radius_px <= 0 or not np.any(mask):
        return mask
    try:
        from scipy.ndimage import binary_dilation
    except Exception:
        return mask
    structure = np.ones((radius_px * 2 + 1, radius_px * 2 + 1), dtype=bool)
    return binary_dilation(mask, structure=structure)


def _pixels_to_world(
    xs: np.ndarray,
    ys: np.ndarray,
    depth_m: np.ndarray,
    c2w: np.ndarray,
    intr: dict[str, Any],
) -> np.ndarray:
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - float(intr["cx"])) * z / float(intr["fx"])
    y = (ys.astype(np.float32) - float(intr["cy"])) * z / float(intr["fy"])
    p_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (np.asarray(c2w, dtype=np.float64) @ p_cam.T).T[:, :3].astype(np.float32)


def _collect_target_object_points(
    view_indices: list[int],
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
    views_dir: Path,
    class_dir: Path | None,
    target_class_id: int,
    max_points: int,
    stride: int,
) -> np.ndarray:
    if class_dir is None or not class_dir.is_dir():
        return np.empty((0, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    chunks = []
    stride = max(1, int(stride))
    per_view_max = max(1, int(max_points))
    for idx in view_indices:
        depth_path = views_dir / "depth" / f"{idx:04d}.png"
        class_path = class_dir / f"{idx:04d}.png"
        if not depth_path.is_file() or not class_path.is_file() or idx not in pose_map:
            continue
        depth = _read_depth_m(depth_path)
        depth_h, depth_w = depth.shape[:2]
        depth_intr = _scaled_intrinsics(intr, depth_w, depth_h)
        cls = _read_class_map(class_path)
        if cls.shape[:2] != depth.shape[:2]:
            continue
        ys, xs = np.where((depth > 0.0) & (cls == int(target_class_id)))
        if ys.size == 0:
            continue
        if stride > 1:
            keep = ((ys % stride) == 0) & ((xs % stride) == 0)
            ys = ys[keep]
            xs = xs[keep]
        if ys.size == 0:
            continue
        if ys.size > per_view_max:
            choice = rng.choice(ys.size, size=per_view_max, replace=False)
            ys = ys[choice]
            xs = xs[choice]
        chunks.append(_pixels_to_world(xs, ys, depth, pose_map[idx]["c2w"], depth_intr))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    points = np.concatenate(chunks, axis=0)
    if points.shape[0] > int(max_points):
        choice = rng.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[choice]
    return points.astype(np.float32)


def _open3d_extrinsic(w2c: np.ndarray) -> np.ndarray:
    return np.asarray(w2c, dtype=np.float64)


def _build_tsdf_volume(
    view_indices: list[int],
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
    views_dir: Path,
    class_dir: Path | None,
    target_class_id: int | None,
    background_class_id: int | None,
    exclude_background: bool = TSDF_EXCLUDE_BACKGROUND,
) -> dict[str, Any]:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("open3d is required for FetchBench TSDF geometry scoring. Re-run with --skip-geometry to bypass it.") from exc

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(TSDF_VOXEL_SIZE_M),
        sdf_trunc=float(TSDF_TRUNCATION_M),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    integrated = 0
    for idx in view_indices:
        if idx not in pose_map:
            continue
        depth_path = views_dir / "depth" / f"{idx:04d}.png"
        if not depth_path.is_file():
            continue
        depth = _read_depth_m(depth_path).astype(np.float32)
        depth_intr = _scaled_intrinsics(intr, int(depth.shape[1]), int(depth.shape[0]))
        if class_dir is not None and class_dir.is_dir():
            class_path = class_dir / f"{idx:04d}.png"
            if class_path.is_file() and target_class_id is not None:
                cls = _read_class_map(class_path)
                if cls.shape[:2] == depth.shape[:2]:
                    exclude_mask = cls == int(target_class_id)
                    if bool(exclude_background) and background_class_id is not None:
                        exclude_mask = exclude_mask | (cls == int(background_class_id))
                    exclude_mask = _maybe_dilate_mask(exclude_mask, TSDF_TARGET_MASK_DILATE_PX)
                    depth = depth.copy()
                    depth[exclude_mask] = 0.0

        depth_img = o3d.geometry.Image(depth)
        color_img = o3d.geometry.Image(np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(depth.shape[1]),
            int(depth.shape[0]),
            float(depth_intr["fx"]),
            float(depth_intr["fy"]),
            float(depth_intr["cx"]),
            float(depth_intr["cy"]),
        )
        volume.integrate(rgbd, intrinsic, _open3d_extrinsic(pose_map[idx]["w2c"]))
        integrated += 1
    if integrated == 0:
        raise RuntimeError("No depth images were integrated into the TSDF volume.")

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0:
        raise RuntimeError("Open3D TSDF produced an empty mesh.")
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    return {"scene": scene, "mesh": mesh, "integrated_views": integrated}


def _query_open3d_distance(tsdf_data: dict[str, Any], points: np.ndarray) -> np.ndarray:
    import open3d as o3d

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    tensor_points = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    return tsdf_data["scene"].compute_distance(tensor_points).numpy().astype(np.float32)


def _geometry_scores_from_tsdf(
    candidate_dirs: np.ndarray,
    grasp_world: np.ndarray,
    tsdf_data: dict[str, Any],
    target_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(0)
    if target_points.shape[0] == 0:
        raise RuntimeError("No target-class points were available for FetchBench TSDF geometry scoring.")
    sample_count = min(100, target_points.shape[0])
    choice = rng.choice(target_points.shape[0], size=sample_count, replace=False)
    scoring_points = target_points[choice].astype(np.float32)
    source = "target_class_points"

    deltas = np.linspace(GEO_SWEEP_START_M, GEO_SWEEP_END_M, GEO_SWEEP_NUM_STEPS, dtype=np.float32)
    quantiles = np.zeros((candidate_dirs.shape[0],), dtype=np.float32)
    for i, direction in enumerate(candidate_dirs.astype(np.float32)):
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        swept = scoring_points[:, None, :] + deltas[None, :, None] * direction[None, None, :]
        distances = _query_open3d_distance(tsdf_data, swept.reshape(-1, 3))
        quantiles[i] = float(np.quantile(distances, GEO_QUANTILE))

    scores = (quantiles - GEO_COLLISION_CLEARANCE_M) / max(GEO_SAFE_CLEARANCE_M - GEO_COLLISION_CLEARANCE_M, 1e-8)
    scores = np.clip(scores, 0.0, 1.0).astype(np.float32)
    meta = {
        "method": "open3d_tsdf_quantile_clearance",
        "integrated_views": int(tsdf_data["integrated_views"]),
        "scoring_point_source": source,
        "scoring_points": int(scoring_points.shape[0]),
        "target_points_available": int(target_points.shape[0]),
        "tsdf_voxel_size_m": float(TSDF_VOXEL_SIZE_M),
        "tsdf_truncation_m": float(TSDF_TRUNCATION_M),
        "tsdf_exclude_background": bool(TSDF_EXCLUDE_BACKGROUND),
        "sweep_start_m": float(GEO_SWEEP_START_M),
        "sweep_end_m": float(GEO_SWEEP_END_M),
        "sweep_num_steps": int(GEO_SWEEP_NUM_STEPS),
        "quantile": float(GEO_QUANTILE),
        "collision_clearance_m": float(GEO_COLLISION_CLEARANCE_M),
        "safe_clearance_m": float(GEO_SAFE_CLEARANCE_M),
        "clearance_quantiles_m": quantiles.astype(float).tolist(),
    }
    debug = {
        "scoring_points": scoring_points,
        "deltas": deltas,
    }
    return scores, meta, debug


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


def _write_point_cloud_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {color[0]} {color[1]} {color[2]}\n")


def _write_sweep_points_ply(
    path: Path,
    scoring_points: np.ndarray,
    direction: np.ndarray,
    deltas: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.asarray(scoring_points, dtype=np.float32).reshape(-1, 3)
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    deltas = np.asarray(deltas, dtype=np.float32).reshape(-1)
    swept = base[:, None, :] + deltas[None, :, None] * direction[None, None, :]
    pts = swept.reshape(-1, 3)
    step_ids = np.tile(np.arange(deltas.size, dtype=np.float32), base.shape[0])
    denom = max(float(deltas.size - 1), 1.0)
    t = step_ids / denom
    reds = np.round(255.0 * t).astype(np.uint8)
    greens = np.round(255.0 * (1.0 - np.abs(t - 0.5) * 2.0)).astype(np.uint8)
    blues = np.round(255.0 * (1.0 - t)).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, r, g, b in zip(pts, reds, greens, blues):
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(r)} {int(g)} {int(b)}\n")


def _write_tsdf_mesh_ply(tsdf_data: dict[str, Any], path: Path) -> None:
    try:
        import open3d as o3d
    except Exception:
        return
    mesh = tsdf_data.get("mesh")
    if mesh is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    return ((scores - float(np.min(scores))) / (float(np.max(scores) - np.min(scores)) + 1e-8)).astype(np.float32)


def _direction_record(
    name: str,
    alpha_vlm: float,
    candidate_dirs: np.ndarray,
    scores: np.ndarray,
    ply_path: Path,
) -> dict[str, Any]:
    best_idx = int(np.argmax(scores))
    return {
        "name": name,
        "alpha_vlm": float(alpha_vlm),
        "idx": best_idx,
        "direction": [float(v) for v in candidate_dirs[best_idx].tolist()],
        "score": float(scores[best_idx]),
        "scores": np.asarray(scores, dtype=np.float32).astype(float).tolist(),
        "direction_ply": str(ply_path),
    }


def _fuse_direction(
    exp_dir: Path,
    grasp_world: np.ndarray,
    vlm_view_indices: list[int],
    view_indices: list[int],
    vlm_results: list[dict[str, Any]],
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
    alpha: float,
    skip_geometry: bool,
    class_dir: Path | None,
    target_class_id: int | None,
    background_class_id: int | None,
    views_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    views_dir = views_dir if views_dir is not None else _resolve_views_dir(exp_dir)
    candidate_dirs = _fibonacci_upper_hemisphere(DISTILL_NUM_DIRECTIONS)
    min_z = float(np.sin(np.deg2rad(MIN_LATITUDE_DEG)))
    candidate_dirs = candidate_dirs[candidate_dirs[:, 2] >= min_z]
    vlm_scores = _distill_vlm_scores(candidate_dirs, grasp_world, vlm_results, pose_map, intr, views_dir)

    geometry_meta: dict[str, Any]
    geometry_debug: dict[str, Any] = {}
    geometry_debug_plys: dict[str, str] = {}
    if skip_geometry:
        geometry_scores = np.ones_like(vlm_scores, dtype=np.float32)
        geometry_meta = {"method": "skipped", "reason": "skip_geometry=true"}
        final_scores = _normalize_scores(vlm_scores)
    else:
        target_points = _collect_target_object_points(
            view_indices=view_indices,
            pose_map=pose_map,
            intr=intr,
            views_dir=views_dir,
            class_dir=class_dir,
            target_class_id=int(target_class_id) if target_class_id is not None else 1,
            max_points=GEO_TARGET_POINT_MAX_POINTS,
            stride=GEO_TARGET_POINT_STRIDE,
        )
        tsdf_data = _build_tsdf_volume(
            view_indices=view_indices,
            pose_map=pose_map,
            intr=intr,
            views_dir=views_dir,
            class_dir=class_dir,
            target_class_id=target_class_id,
            background_class_id=background_class_id,
            exclude_background=TSDF_EXCLUDE_BACKGROUND,
        )
        geometry_scores, geometry_meta, geometry_debug = _geometry_scores_from_tsdf(
            candidate_dirs,
            grasp_world,
            tsdf_data,
            target_points,
        )
        final_scores = float(alpha) * vlm_scores + (1.0 - float(alpha)) * geometry_scores
        final_scores = _normalize_scores(final_scores)

    out_dir = output_dir if output_dir is not None else exp_dir / "directions"
    vlm_ply = out_dir / "vlm_only_direction.ply"
    geometry_ply = out_dir / "geometry_only_direction.ply"
    aggregate_ply = out_dir / "aggregate_direction.ply"

    vlm_record = _direction_record("vlm_only", 1.0, candidate_dirs, vlm_scores, vlm_ply)
    geometry_record = _direction_record("geometry_only", 0.0, candidate_dirs, geometry_scores, geometry_ply)
    aggregate_record = _direction_record("aggregate", float(alpha), candidate_dirs, final_scores, aggregate_ply)

    aggregate_idx = int(aggregate_record["idx"])
    vlm_idx = int(vlm_record["idx"])
    geometry_idx = int(geometry_record["idx"])

    _write_direction_ply(vlm_ply, grasp_world, candidate_dirs[vlm_idx], (0, 110, 255))
    _write_direction_ply(geometry_ply, grasp_world, candidate_dirs[geometry_idx], (0, 180, 80))
    _write_direction_ply(aggregate_ply, grasp_world, candidate_dirs[aggregate_idx], (255, 0, 0))

    if not skip_geometry:
        debug_dir = out_dir / "debug_geometry"
        tsdf_ply = debug_dir / "tsdf_without_target.ply"
        target_points_ply = debug_dir / "target_surface_points.ply"
        scoring_points_ply = debug_dir / "scoring_points_100.ply"
        geometry_sweep_ply = debug_dir / "sweep_geometry_only_direction.ply"
        vlm_sweep_ply = debug_dir / "sweep_vlm_only_direction.ply"
        aggregate_sweep_ply = debug_dir / "sweep_aggregate_direction.ply"
        _write_tsdf_mesh_ply(tsdf_data, tsdf_ply)
        _write_point_cloud_ply(target_points_ply, target_points, (0, 220, 80))
        scoring_points = np.asarray(geometry_debug.get("scoring_points", np.empty((0, 3))), dtype=np.float32)
        deltas = np.asarray(geometry_debug.get("deltas", np.empty((0,))), dtype=np.float32)
        _write_point_cloud_ply(scoring_points_ply, scoring_points, (255, 0, 255))
        _write_sweep_points_ply(geometry_sweep_ply, scoring_points, candidate_dirs[geometry_idx], deltas)
        _write_sweep_points_ply(vlm_sweep_ply, scoring_points, candidate_dirs[vlm_idx], deltas)
        _write_sweep_points_ply(aggregate_sweep_ply, scoring_points, candidate_dirs[aggregate_idx], deltas)
        geometry_debug_plys = {
            "tsdf_without_target": str(tsdf_ply),
            "target_surface_points": str(target_points_ply),
            "scoring_points": str(scoring_points_ply),
            "sweep_geometry_only": str(geometry_sweep_ply),
            "sweep_vlm_only": str(vlm_sweep_ply),
            "sweep_aggregate": str(aggregate_sweep_ply),
        }
        geometry_meta["debug_plys"] = geometry_debug_plys

    _write_json(
        out_dir / "vlm_only_direction.json",
        {
            **vlm_record,
            "format": "fetchbench_real_offline_ablation_direction_v1",
            "experiment_dir": str(exp_dir),
            "view_indices": [int(v) for v in vlm_view_indices],
            "vlm_view_indices": [int(v) for v in vlm_view_indices],
            "grasp_position_world": [float(v) for v in grasp_world.tolist()],
            "score_source": "vlm",
        },
    )
    _write_json(
        out_dir / "geometry_only_direction.json",
        {
            **geometry_record,
            "format": "fetchbench_real_offline_ablation_direction_v1",
            "experiment_dir": str(exp_dir),
            "view_indices": [int(v) for v in view_indices],
            "vlm_view_indices": [int(v) for v in vlm_view_indices],
            "grasp_position_world": [float(v) for v in grasp_world.tolist()],
            "score_source": "geometry",
            "geometry": geometry_meta,
        },
    )
    _write_json(
        out_dir / "aggregate_direction.json",
        {
            **aggregate_record,
            "format": "fetchbench_real_offline_ablation_direction_v1",
            "experiment_dir": str(exp_dir),
            "view_indices": [int(v) for v in view_indices],
            "vlm_view_indices": [int(v) for v in vlm_view_indices],
            "grasp_position_world": [float(v) for v in grasp_world.tolist()],
            "score_source": "alpha_vlm_fusion",
            "component_scores": {
                "vlm": vlm_scores.astype(float).tolist(),
                "geometry": geometry_scores.astype(float).tolist(),
            },
            "geometry": geometry_meta,
        },
    )

    result = {
        "format": "fetchbench_real_offline_direction_v1",
        "experiment_dir": str(exp_dir),
        "view_indices": [int(v) for v in view_indices],
        "vlm_view_indices": [int(v) for v in vlm_view_indices],
        "grasp_position_world": [float(v) for v in grasp_world.tolist()],
        "alpha_vlm": float(alpha),
        "min_latitude_deg": float(MIN_LATITUDE_DEG),
        "directions": {
            "geometry_only": geometry_record,
            "vlm_only": vlm_record,
            "aggregate": aggregate_record,
        },
        "aggregate_idx": aggregate_idx,
        "aggregate_direction": [float(v) for v in candidate_dirs[aggregate_idx].tolist()],
        "aggregate_score": float(final_scores[aggregate_idx]),
        "vlm_only_idx": vlm_idx,
        "vlm_only_direction": [float(v) for v in candidate_dirs[vlm_idx].tolist()],
        "vlm_only_score": float(vlm_scores[vlm_idx]),
        "geometry_only_idx": geometry_idx,
        "geometry_only_direction": [float(v) for v in candidate_dirs[geometry_idx].tolist()],
        "geometry_only_score": float(geometry_scores[geometry_idx]),
        "scores": {
            "final": final_scores.astype(float).tolist(),
            "vlm": vlm_scores.astype(float).tolist(),
            "geometry": geometry_scores.astype(float).tolist(),
        },
        "geometry": geometry_meta,
        "geometry_debug_plys": geometry_debug_plys,
        "direction_plys": {
            "geometry_only": str(geometry_ply),
            "vlm_only": str(vlm_ply),
            "aggregate": str(aggregate_ply),
        },
    }
    _write_json(out_dir / "final_3d_direction.json", result)
    return result


def run_prep(args: argparse.Namespace) -> dict[str, Any]:
    exp_dir = _resolve_experiment_dir(args)
    views_dir = _resolve_views_dir(exp_dir)
    if not views_dir.is_dir():
        raise FileNotFoundError(f"AP output directory not found: {views_dir}")
    intr = _load_intrinsics(views_dir)
    pose_map, _pose_doc = _load_pose_map(views_dir)
    available = sorted(pose_map.keys())
    view_indices = _parse_indices(args.view_indices, available)
    grasp_world = _target_world_from_args(args, exp_dir, views_dir, pose_map, intr)

    if args.vlm_view_indices:
        vlm_view_indices = _parse_indices(args.vlm_view_indices, available)
    elif args.disable_subset_selection:
        vlm_view_indices = view_indices[: max(1, int(args.num_vlm_views))]
    else:
        vlm_view_indices, _ = select_best_subset_indices(
            exp_dir=exp_dir,
            view_indices=view_indices,
            grasp_world=grasp_world,
            num_views=int(args.num_vlm_views),
            sim_weight=float(args.subset_sim_weight),
            num_starts=int(args.subset_num_starts),
            output_dir=None,
            label="vlm",
        )

    target_doc = {
        "grasp_position_world": [float(v) for v in grasp_world.tolist()],
        "source": "grasp_world" if args.grasp_world is not None else "target_pixel_or_cached",
    }
    _write_json(exp_dir / "target_point.json", target_doc)

    prepared = _prepare_vlm_inputs(
        exp_dir,
        vlm_view_indices,
        grasp_world,
        views_dir,
        pose_map,
        intr,
        output_dir=exp_dir / "rgb_vlm_in",
    )

    masks_dir = exp_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_manifest = {
        "format": "fetchbench_real_mask_manifest_v1",
        "status": "pending_sam",
        "note": "Placeholder. SAM or another real segmentation source should write one class-mask PNG per AP view here.",
        "required_view_indices": [int(v) for v in view_indices],
        "target_class_id": int(getattr(args, "target_class_id", 1) or 1),
        "background_class_id": int(getattr(args, "background_class_id", 0) or 0),
        "expected_files": [str(masks_dir / f"{int(v):04d}.png") for v in view_indices],
    }
    _write_json(masks_dir / "masks_manifest.json", mask_manifest)

    subset_doc = {
        "format": "fetchbench_real_prep_v1",
        "experiment_dir": str(exp_dir),
        "views_dir": str(views_dir),
        "view_indices": [int(v) for v in view_indices],
        "vlm_view_indices": [int(v) for v in vlm_view_indices],
        "num_vlm_views_requested": int(args.num_vlm_views),
        "grasp_position_world": [float(v) for v in grasp_world.tolist()],
        "target_class_id": int(getattr(args, "target_class_id", 1) or 1),
        "background_class_id": int(getattr(args, "background_class_id", 0) or 0),
        "target_point_json": str(exp_dir / "target_point.json"),
        "rgb_vlm_in_dir": str(exp_dir / "rgb_vlm_in"),
        "masks_dir": str(masks_dir),
        "mask_manifest": str(masks_dir / "masks_manifest.json"),
        "prepared_vlm_inputs": prepared,
        "sam_placeholder": {
            "implemented": False,
            "expected_output": "Write class-mask PNGs into masks/ named 0000.png, 0001.png, ... using target_class_id for target pixels.",
        },
    }
    _write_json(exp_dir / "vlm_subset.json", subset_doc)
    return subset_doc


def _load_prep_doc(exp_dir: Path) -> dict[str, Any]:
    prep_path = exp_dir / "vlm_subset.json"
    if not prep_path.is_file():
        raise FileNotFoundError(f"Prep file missing: {prep_path}. Run fetchbench_prep first.")
    return _load_json(prep_path)


def run_vlm_stage(args: argparse.Namespace) -> dict[str, Any]:
    exp_dir = _resolve_experiment_dir(args)
    prep_doc = _load_prep_doc(exp_dir)
    prepared = prep_doc.get("prepared_vlm_inputs", [])
    if not isinstance(prepared, list) or not prepared:
        raise RuntimeError(f"No prepared VLM inputs in {exp_dir / 'vlm_subset.json'}")
    results = _run_vlm(
        exp_dir=exp_dir,
        prepared=prepared,
        call_api=bool(args.call_api),
        provider=str(args.api_provider),
        model=str(args.model),
        output_dir=exp_dir / "rgb_vlm_out",
    )
    out = {
        "format": "fetchbench_real_vlm_stage_v1",
        "experiment_dir": str(exp_dir),
        "vlm_view_indices": [int(item["index"]) for item in results],
        "rgb_vlm_in_dir": str(exp_dir / "rgb_vlm_in"),
        "rgb_vlm_out_dir": str(exp_dir / "rgb_vlm_out"),
        "views": results,
    }
    _write_json(exp_dir / "rgb_vlm_out" / "vlm_results.json", out)
    return out


def run_direction_stage(args: argparse.Namespace) -> dict[str, Any]:
    exp_dir = _resolve_experiment_dir(args)
    views_dir = _resolve_views_dir(exp_dir)
    intr = _load_intrinsics(views_dir)
    pose_map, pose_doc = _load_pose_map(views_dir)
    prep_doc = _load_prep_doc(exp_dir)
    vlm_results_path = exp_dir / "rgb_vlm_out" / "vlm_results.json"
    if not vlm_results_path.is_file():
        raise FileNotFoundError(f"VLM results missing: {vlm_results_path}. Run fetchbench_vlm first.")
    vlm_doc = _load_json(vlm_results_path)
    vlm_results = vlm_doc.get("views", [])
    if not isinstance(vlm_results, list) or not vlm_results:
        raise RuntimeError(f"No VLM results in {vlm_results_path}")

    grasp_world = np.asarray(prep_doc["grasp_position_world"], dtype=np.float64)
    vlm_view_indices = [int(v) for v in prep_doc["vlm_view_indices"]]
    view_indices_raw = prep_doc.get("view_indices", prep_doc.get("tsdf_view_indices", prep_doc.get("captured_view_pool_indices", [])))
    view_indices = [int(v) for v in view_indices_raw]
    if not view_indices:
        raise RuntimeError(f"No view_indices in {exp_dir / 'vlm_subset.json'}")

    class_dir = _resolve_mask_dir(exp_dir, views_dir, str(args.class_dir or ""))
    target_class_id = args.target_class_id
    background_class_id = args.background_class_id
    if target_class_id is None and isinstance(pose_doc, dict) and "target_class_id" in pose_doc:
        target_class_id = int(pose_doc["target_class_id"])
    if background_class_id is None and isinstance(pose_doc, dict) and "background_class_id" in pose_doc:
        background_class_id = int(pose_doc["background_class_id"])
    if target_class_id is None:
        target_class_id = int(prep_doc.get("target_class_id", 1))
    if background_class_id is None:
        background_class_id = int(prep_doc.get("background_class_id", 0))

    if not bool(args.skip_geometry) and class_dir is None:
        raise RuntimeError(
            "Direction geometry requires class masks in <experiment>/masks or --class-dir. "
            "Run the future SAM step or use --skip-geometry for VLM-only debugging."
        )
    if not bool(args.skip_geometry) and class_dir is not None:
        missing_masks = [int(idx) for idx in view_indices if not (class_dir / f"{int(idx):04d}.png").is_file()]
        if missing_masks:
            raise RuntimeError(
                f"Missing class-mask PNGs for AP views in {class_dir}: {missing_masks}. "
                "The current masks/ directory is only a placeholder until SAM is implemented."
            )

    return _fuse_direction(
        exp_dir=exp_dir,
        grasp_world=grasp_world,
        vlm_view_indices=vlm_view_indices,
        view_indices=view_indices,
        vlm_results=vlm_results,
        pose_map=pose_map,
        intr=intr,
        alpha=float(args.alpha_vlm),
        skip_geometry=bool(args.skip_geometry),
        class_dir=class_dir,
        target_class_id=target_class_id,
        background_class_id=background_class_id,
        views_dir=views_dir,
        output_dir=exp_dir / "directions",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline FetchBench real VLM+geometry direction pipeline")
    parser.add_argument("--output-root", default="/workspace/ros2_ws/ours_experiment")
    parser.add_argument("--experiment-name", default="ex2")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--view-indices", default="all", help="Captured-view pool: 'all' or comma/space separated view indices")
    parser.add_argument("--vlm-view-indices", default="", help="Explicit VLM views. Default: FetchBench subset from AP view_indices.")
    parser.add_argument("--num-vlm-views", type=int, default=4, help="Number of subset-selected views sent to VLM.")
    parser.add_argument("--disable-subset-selection", action="store_true", help="Use provided view lists directly; do not run subset selection.")
    parser.add_argument("--subset-sim-weight", type=float, default=DEFAULT_SIM_WEIGHT)
    parser.add_argument("--subset-num-starts", type=int, default=DEFAULT_NUM_STARTS)
    parser.add_argument("--grasp-world", nargs="+", default=None, help="Manual grasp point in world frame: x y z")
    parser.add_argument("--target-view-index", type=int, default=None, help="View index containing a clicked target pixel")
    parser.add_argument("--target-pixel", nargs=2, type=int, default=None, metavar=("U", "V"))
    parser.add_argument("--call-api", action="store_true", help="Actually call the configured VLM API for missing responses")
    parser.add_argument("--api-provider", choices=["gemini"], default="gemini")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--alpha-vlm", type=float, default=GEO_LAMBDA, help="final = alpha*VLM + (1-alpha)*geometry")
    parser.add_argument("--skip-geometry", action="store_true", help="Use VLM distillation only; do not build TSDF geometry.")
    parser.add_argument("--class-dir", default="", help="Optional class-mask directory. Defaults to <experiment>/masks, then legacy <views>/class.")
    parser.add_argument("--target-class-id", type=int, default=None, help="Optional target class id for sim-equivalent TSDF masking/scoring.")
    parser.add_argument("--background-class-id", type=int, default=None, help="Optional background class id for sim-equivalent TSDF masking.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write marked VLM input images and metadata")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prep = run_prep(args)
    print(f"Prepared {sum(1 for item in prep['prepared_vlm_inputs'] if item.get('usable'))} VLM input images in {_resolve_experiment_dir(args) / 'rgb_vlm_in'}")
    print(f"View indices: {prep['view_indices']}")
    print(f"VLM view indices: {prep['vlm_view_indices']}")
    if args.prepare_only:
        print("prepare-only requested; stopping before VLM/API and fusion")
        return

    run_vlm_stage(args)
    result = run_direction_stage(args)
    print(f"Geometry-only direction: {result['geometry_only_direction']}")
    print(f"VLM-only direction: {result['vlm_only_direction']}")
    print(f"Aggregate direction alpha_vlm={float(args.alpha_vlm):.3f}: {result['aggregate_direction']}")
    print(f"Saved: {_resolve_experiment_dir(args) / 'directions' / 'final_3d_direction.json'}")


if __name__ == "__main__":
    main()
