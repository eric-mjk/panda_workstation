from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from .pipeline import (
        _load_intrinsics,
        _load_json,
        _load_pose_map,
        _project_world_to_pixel,
        _resolve_experiment_dir,
        _resolve_views_dir,
        _scaled_intrinsics,
        _write_json,
    )
except ImportError:
    from pipeline import (
        _load_intrinsics,
        _load_json,
        _load_pose_map,
        _project_world_to_pixel,
        _resolve_experiment_dir,
        _resolve_views_dir,
        _scaled_intrinsics,
        _write_json,
    )


DEFAULT_PORT = 5050


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while waiting for SAM3 response")
        data += chunk
    return data


def _send_request(sock: socket.socket, metadata: dict[str, Any], payload: bytes) -> None:
    meta_bytes = json.dumps(metadata).encode("utf-8")
    sock.sendall(struct.pack(">I", len(meta_bytes)))
    sock.sendall(meta_bytes)
    sock.sendall(struct.pack(">I", len(payload)))
    sock.sendall(payload)


def _recv_json(sock: socket.socket) -> dict[str, Any]:
    result_len_bytes = _recv_exact(sock, 4)
    (result_len,) = struct.unpack(">I", result_len_bytes)
    return json.loads(_recv_exact(sock, result_len).decode("utf-8"))


def _request_sam3_mask(
    *,
    server_ip: str,
    port: int,
    timeout: float,
    connect_retries: int,
    retry_sleep_s: float,
    image_path: Path,
    prompt: str | None,
    point_xy: tuple[float, float] | None,
    box_xyxy: tuple[float, float, float, float] | None,
    return_instances: bool,
) -> dict[str, Any]:
    metadata = {
        "type": "sam3_text_mask",
        "image_name": image_path.name,
        "return_instances": bool(return_instances),
    }
    if prompt and (point_xy is not None or box_xyxy is not None):
        metadata["mode"] = "text_geometric"
    elif prompt:
        metadata["mode"] = "text"
    else:
        metadata["mode"] = "interactive"
    if prompt:
        metadata["prompt"] = str(prompt)
    if point_xy is not None:
        metadata["positive_points"] = [[float(point_xy[0]), float(point_xy[1])]]
    if box_xyxy is not None:
        metadata["box_xyxy"] = [float(v) for v in box_xyxy]
    payload = image_path.read_bytes()
    last_exc: Exception | None = None
    attempts = max(1, int(connect_retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            with socket.create_connection((server_ip, int(port)), timeout=float(timeout)) as sock:
                _send_request(sock, metadata, payload)
                return _recv_json(sock)
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"[MASK] SAM3 connection failed ({type(exc).__name__}: {exc}); "
                f"retrying {attempt}/{attempts - 1} after {float(retry_sleep_s):.1f}s",
                flush=True,
            )
            time.sleep(max(0.0, float(retry_sleep_s)))
    raise ConnectionError(
        f"Could not connect to SAM3 server at {server_ip}:{int(port)} after {attempts} attempt(s). "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _decode_mask_png(mask_png_b64: str, expected_size: tuple[int, int]) -> Image.Image:
    data = base64.b64decode(mask_png_b64.encode("ascii"))
    import io

    mask_img = Image.open(io.BytesIO(data)).convert("L")
    if mask_img.size != expected_size:
        mask_img = mask_img.resize(expected_size, Image.Resampling.NEAREST)
    return mask_img


def _write_class_mask(
    *,
    mask_img: Image.Image,
    out_path: Path,
    target_class_id: int,
    background_class_id: int,
    threshold: int,
) -> int:
    arr = np.asarray(mask_img, dtype=np.uint8)
    class_arr = np.full(arr.shape, int(background_class_id), dtype=np.uint8)
    class_arr[arr > int(threshold)] = np.uint8(int(target_class_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_arr, mode="L").save(out_path)
    return int(np.count_nonzero(class_arr == int(target_class_id)))


def _write_mask_overlay(
    *,
    rgb_path: Path,
    raw_mask_img: Image.Image,
    out_path: Path,
    threshold: int,
) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    mask = raw_mask_img
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    rgb_arr = np.asarray(rgb, dtype=np.uint8).copy()
    mask_arr = np.asarray(mask, dtype=np.uint8) > int(threshold)
    if np.any(mask_arr):
        fill_color = np.asarray([255, 0, 0], dtype=np.float32)
        rgb_arr[mask_arr] = (0.20 * rgb_arr[mask_arr].astype(np.float32) + 0.80 * fill_color).astype(np.uint8)
        interior = mask_arr.copy()
        interior[1:-1, 1:-1] = (
            mask_arr[1:-1, 1:-1]
            & mask_arr[:-2, 1:-1]
            & mask_arr[2:, 1:-1]
            & mask_arr[1:-1, :-2]
            & mask_arr[1:-1, 2:]
        )
        contour = mask_arr & ~interior
        thick_contour = contour.copy()
        thick_contour[1:, :] |= contour[:-1, :]
        thick_contour[:-1, :] |= contour[1:, :]
        thick_contour[:, 1:] |= contour[:, :-1]
        thick_contour[:, :-1] |= contour[:, 1:]
        rgb_arr[thick_contour] = np.asarray([255, 255, 0], dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_arr, mode="RGB").save(out_path)


def _write_raw_mask(mask_img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(out_path)


def _parse_indices(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or str(raw).strip().lower() in ("", "all"):
        return [int(v) for v in default]
    return sorted(dict.fromkeys(int(item) for item in str(raw).replace(",", " ").split()))


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _is_inside_image(point_xy: tuple[float, float], size: tuple[int, int]) -> bool:
    width, height = size
    x, y = point_xy
    return 0.0 <= float(x) < float(width) and 0.0 <= float(y) < float(height)


def _auto_project_point(
    *,
    idx: int,
    pose_map: dict[int, dict[str, Any]],
    intr: dict[str, Any],
    grasp_world: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    pose = pose_map.get(int(idx))
    if pose is None:
        return None
    rgb_intr = _scaled_intrinsics(intr, int(image_size[0]), int(image_size[1]))
    projected = _project_world_to_pixel(grasp_world, pose["c2w"], rgb_intr)
    if projected is None:
        return None
    u, v, _ = projected
    point_xy = (float(u), float(v))
    if not _is_inside_image(point_xy, image_size):
        return None
    return point_xy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FetchBench class masks using the SAM3 mask server")
    parser.add_argument("--output-root", default="/workspace/ros2_ws/ours_experiment")
    parser.add_argument("--experiment-name", default="ex2")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default="", help='Optional text prompt for the target object, e.g. "the mustard bottle"')
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Fixed image pixel point to send to SAM3 for every view. X is column, Y is row in the RGB image.",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Fixed image pixel box to send to SAM3 for every view.",
    )
    parser.add_argument(
        "--point-mode",
        choices=("auto", "fixed", "none"),
        default="auto",
        help="auto projects grasp_position_world into each RGB image; fixed uses --point; none sends no point.",
    )
    parser.add_argument("--view-indices", default="all", help="AP view indices to mask. Default: all view_indices from vlm_subset.json")
    parser.add_argument("--target-class-id", type=int, default=1)
    parser.add_argument("--background-class-id", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--connect-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-s", type=float, default=2.0)
    parser.add_argument("--return-instances", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    exp_dir = _resolve_experiment_dir(args)
    views_dir = _resolve_views_dir(exp_dir)
    prep_path = exp_dir / "vlm_subset.json"
    if not prep_path.is_file():
        raise FileNotFoundError(f"Prep file missing: {prep_path}. Run fetchbench_prep first.")
    prep_doc = _load_json(prep_path)
    default_indices = prep_doc.get(
        "view_indices",
        prep_doc.get("tsdf_view_indices", prep_doc.get("captured_view_pool_indices", [])),
    )
    if not isinstance(default_indices, list) or not default_indices:
        raise RuntimeError(
            f"No view_indices found in {prep_path}. "
            "Run fetchbench_prep again, or clean non-AP outputs with fetchbench_clean and rerun prep."
        )
    view_indices = _parse_indices(args.view_indices, [int(v) for v in default_indices])
    prompt = str(args.prompt).strip()
    fixed_point = tuple(float(v) for v in args.point) if args.point is not None else None
    fixed_box = tuple(float(v) for v in args.box) if args.box is not None else None
    if fixed_point is not None:
        point_mode = "fixed"
    else:
        point_mode = str(args.point_mode)
    if point_mode == "fixed" and fixed_point is None:
        raise ValueError("--point-mode fixed requires --point X Y")
    if not prompt and point_mode == "none" and fixed_box is None:
        raise ValueError("Mask stage needs at least one SAM3 cue: --prompt, --point/--point-mode auto, or --box")

    intr = _load_intrinsics(views_dir)
    pose_map, _ = _load_pose_map(views_dir)
    grasp_raw = prep_doc.get("grasp_position_world")
    if point_mode == "auto":
        if not isinstance(grasp_raw, list) or len(grasp_raw) != 3:
            raise RuntimeError(f"point-mode=auto requires grasp_position_world in {prep_path}")
        grasp_world = np.asarray(grasp_raw, dtype=np.float64)
    else:
        grasp_world = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    masks_dir = exp_dir / "masks"
    raw_dir = masks_dir / "raw_sam3"
    overlay_dir = masks_dir / "overlay"
    meta_dir = masks_dir / "sam3_results"
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(
        f"[MASK] Starting SAM3 mask stage: views={view_indices} prompt={prompt!r} "
        f"point_mode={point_mode} point={fixed_point} box={fixed_box} "
        f"server={args.server_ip}:{int(args.port)} output={masks_dir}",
        flush=True,
    )
    for counter, idx in enumerate(view_indices, start=1):
        rgb_path = views_dir / "rgb" / f"{int(idx):04d}.png"
        depth_path = views_dir / "depth" / f"{int(idx):04d}.png"
        out_mask_path = masks_dir / f"{int(idx):04d}.png"
        overlay_path = overlay_dir / f"{int(idx):04d}_overlay.png"
        raw_mask_path = raw_dir / f"{int(idx):04d}_sam3_union.png"
        meta_path = meta_dir / f"{int(idx):04d}_sam3_result.json"
        if out_mask_path.is_file() and not bool(args.overwrite):
            print(f"[MASK] ({counter}/{len(view_indices)}) view={int(idx):04d} exists, skipping: {out_mask_path}", flush=True)
            rows.append({"index": int(idx), "status": "skipped_existing", "mask_path": str(out_mask_path)})
            continue
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB image missing for view {int(idx):04d}: {rgb_path}")
        rgb_size = _image_size(rgb_path)
        depth_size = _image_size(depth_path) if depth_path.is_file() else rgb_size
        point_xy = fixed_point
        if point_mode == "auto":
            point_xy = _auto_project_point(
                idx=int(idx),
                pose_map=pose_map,
                intr=intr,
                grasp_world=grasp_world,
                image_size=rgb_size,
            )
            if point_xy is None and not prompt:
                raise RuntimeError(
                    f"Could not project grasp_position_world into view {int(idx):04d}; "
                    "point-only auto masking has no fallback cue."
                )
            if point_xy is None:
                print(
                    f"[MASK] ({counter}/{len(view_indices)}) view={int(idx):04d} "
                    "target point is outside the RGB image; falling back to prompt-only",
                    flush=True,
                )

        start = time.time()
        print(
            f"[MASK] ({counter}/{len(view_indices)}) view={int(idx):04d} requesting SAM3 "
            f"prompt={prompt!r} point={point_xy} box={fixed_box}",
            flush=True,
        )
        result = _request_sam3_mask(
            server_ip=str(args.server_ip),
            port=int(args.port),
            timeout=float(args.timeout),
            connect_retries=int(args.connect_retries),
            retry_sleep_s=float(args.retry_sleep_s),
            image_path=rgb_path,
            prompt=prompt or None,
            point_xy=point_xy,
            box_xyxy=fixed_box,
            return_instances=bool(args.return_instances),
        )
        printable = {k: v for k, v in result.items() if not k.endswith("_b64")}
        _write_json(meta_path, printable)
        if result.get("status") != "ok":
            raise RuntimeError(f"SAM3 failed for view {int(idx):04d}: {printable}")
        mask_b64 = result.get("mask_png_b64")
        if isinstance(mask_b64, str) and mask_b64:
            raw_mask_img = _decode_mask_png(mask_b64, rgb_size)
        else:
            raw_mask_img = Image.new("L", rgb_size, 0)
            print(
                f"[MASK] ({counter}/{len(view_indices)}) view={int(idx):04d} SAM3 returned no masks; "
                "writing all-background mask",
                flush=True,
            )
        _write_raw_mask(raw_mask_img, raw_mask_path)
        class_mask_img = raw_mask_img
        if class_mask_img.size != depth_size:
            class_mask_img = class_mask_img.resize(depth_size, Image.Resampling.NEAREST)
        target_pixels = _write_class_mask(
            mask_img=class_mask_img,
            out_path=out_mask_path,
            target_class_id=int(args.target_class_id),
            background_class_id=int(args.background_class_id),
            threshold=int(args.threshold),
        )
        _write_mask_overlay(
            rgb_path=rgb_path,
            raw_mask_img=raw_mask_img,
            out_path=overlay_path,
            threshold=int(args.threshold),
        )
        elapsed = time.time() - start
        row = {
            "index": int(idx),
            "status": "ok",
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "mask_path": str(out_mask_path),
            "overlay_path": str(overlay_path),
            "raw_mask_path": str(raw_mask_path),
            "sam3_result_json": str(meta_path),
            "prompt": prompt or None,
            "point_xy": [float(point_xy[0]), float(point_xy[1])] if point_xy is not None else None,
            "box_xyxy": [float(v) for v in fixed_box] if fixed_box is not None else None,
            "rgb_size": [int(rgb_size[0]), int(rgb_size[1])],
            "depth_size": [int(depth_size[0]), int(depth_size[1])],
            "target_pixels": int(target_pixels),
            "elapsed_s": float(elapsed),
        }
        rows.append(row)
        print(
            f"[MASK] ({counter}/{len(view_indices)}) view={int(idx):04d} saved {out_mask_path} "
            f"target_pixels={target_pixels} elapsed={elapsed:.1f}s",
            flush=True,
        )

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    skipped_count = sum(1 for row in rows if row.get("status") == "skipped_existing")
    manifest = {
        "format": "fetchbench_real_mask_manifest_v1",
        "status": "complete" if ok_count + skipped_count == len(view_indices) else "partial",
        "source": "sam3_text_mask",
        "prompt": prompt or None,
        "point_mode": point_mode,
        "fixed_point": [float(v) for v in fixed_point] if fixed_point is not None else None,
        "fixed_box": [float(v) for v in fixed_box] if fixed_box is not None else None,
        "server_ip": str(args.server_ip),
        "port": int(args.port),
        "view_indices": [int(v) for v in view_indices],
        "target_class_id": int(args.target_class_id),
        "background_class_id": int(args.background_class_id),
        "threshold": int(args.threshold),
        "masks_dir": str(masks_dir),
        "raw_masks_dir": str(raw_dir),
        "overlay_dir": str(overlay_dir),
        "results_dir": str(meta_dir),
        "views": rows,
    }
    _write_json(masks_dir / "masks_manifest.json", manifest)
    print(f"[MASK] Done. ok={ok_count} skipped={skipped_count} manifest={masks_dir / 'masks_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
