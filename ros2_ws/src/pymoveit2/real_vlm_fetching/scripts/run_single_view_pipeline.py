#!/usr/bin/env python3
"""
Thin orchestrator: runs the full single-view VLM fetching preparation pipeline.
Does NOT command the robot or call any external API.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PY = sys.executable


def _run(cmd: list[str]) -> str:
    """Run a command with check=True and return the joined string for the manifest."""
    cmd_str = " ".join(cmd)
    print(f"\n>>> {cmd_str}")
    subprocess.run(cmd, check=True)
    return cmd_str


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-view VLM fetching preparation pipeline (no robot commands)."
    )
    p.add_argument("--out-dir", required=True, metavar="PATH",
                   help="Output directory for all pipeline artefacts.")
    p.add_argument("--image", default=None, metavar="PATH",
                   help="Input RGB image. If omitted, capture_once.py is called.")
    p.add_argument("--target", default=None, metavar="TEXT",
                   help="Target object name (required if --target-json is omitted).")
    p.add_argument("--target-json", default=None, metavar="PATH",
                   help="Path to pre-existing target JSON. If omitted, written from "
                        "--target / --target-u / --target-v.")
    p.add_argument("--target-u", type=int, default=None, metavar="INT",
                   help="Pixel column of target centre (required when writing target JSON).")
    p.add_argument("--target-v", type=int, default=None, metavar="INT",
                   help="Pixel row of target centre (required when writing target JSON).")
    p.add_argument("--mock-clock", type=int, required=True, metavar="INT",
                   help="Mock VLM clock value (1–12).")
    p.add_argument("--horizontal-distance-m", type=float, default=0.10, metavar="M")
    p.add_argument("--vertical-clearance-m", type=float, default=0.05, metavar="M")
    p.add_argument("--capture-index", type=int, default=0, metavar="INT")
    p.add_argument("--warmup-frames", type=int, default=10, metavar="INT")
    p.add_argument("--camera-serial", default=None, metavar="TEXT")
    p.add_argument("--radius-px", type=int, default=80, metavar="INT")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands_run: list[str] = []

    # ── Step 1: capture image if not provided ─────────────────────────────────
    if args.image is not None:
        image_path = Path(args.image)
    else:
        cmd = [
            _PY, str(_SCRIPTS_DIR / "capture_once.py"),
            "--out-dir", str(out_dir),
            "--index", str(args.capture_index),
            "--warmup-frames", str(args.warmup_frames),
        ]
        if args.camera_serial is not None:
            cmd += ["--camera-serial", args.camera_serial]
        commands_run.append(_run(cmd))
        image_path = out_dir / "rgb" / f"{args.capture_index:05d}.png"

    # ── Step 2: resolve / write target JSON ───────────────────────────────────
    if args.target_json is not None:
        target_json_path = Path(args.target_json)
        with open(target_json_path) as f:
            _tj = json.load(f)
        target_name = _tj.get("target") or _tj.get("name")
    else:
        if args.target is None:
            print("ERROR: --target is required when --target-json is not provided.", file=sys.stderr)
            sys.exit(1)
        if args.target_u is None or args.target_v is None:
            print("ERROR: --target-u and --target-v are required when --target-json is not provided.",
                  file=sys.stderr)
            sys.exit(1)
        target_json_path = out_dir / "target.json"
        target_json_path.write_text(json.dumps({
            "target": args.target,
            "u": args.target_u,
            "v": args.target_v,
        }, indent=2))
        print(f"Wrote target JSON: {target_json_path}")
        target_name = args.target

    # ── Step 3: make VLM overlay ───────────────────────────────────────────────
    overlay_path = out_dir / "vlm_input_overlay.png"
    cmd = [
        _PY, str(_SCRIPTS_DIR / "make_vlm_overlay.py"),
        "--image", str(image_path),
        "--target-json", str(target_json_path),
        "--out-dir", str(out_dir),
        "--radius-px", str(args.radius_px),
    ]
    commands_run.append(_run(cmd))

    # ── Step 4: offline VLM direction ─────────────────────────────────────────
    vlm_result_path = out_dir / "vlm_result.json"
    cmd = [
        _PY, str(_SCRIPTS_DIR / "run_vlm_direction_offline.py"),
        "--overlay", str(overlay_path),
        "--target", target_name,
        "--out", str(vlm_result_path),
        "--mock-clock", str(args.mock_clock),
    ]
    commands_run.append(_run(cmd))

    # ── Step 5: compute fetch displacement ────────────────────────────────────
    computed_displacement_path = out_dir / "computed_displacement.json"
    cmd = [
        _PY, str(_SCRIPTS_DIR / "run_fetch_trial_from_json.py"),
        "--vlm-result", str(vlm_result_path),
        "--horizontal-distance-m", str(args.horizontal_distance_m),
        "--vertical-clearance-m", str(args.vertical_clearance_m),
    ]
    commands_run.append(_run(cmd))

    # ── Step 6: write manifest ─────────────────────────────────────────────────
    manifest_path = out_dir / "pipeline_manifest.json"
    manifest = {
        "image": str(image_path),
        "target_json": str(target_json_path),
        "overlay": str(overlay_path),
        "vlm_result": str(vlm_result_path),
        "computed_displacement": str(computed_displacement_path),
        "target": target_name,
        "mock_clock": args.mock_clock,
        "horizontal_distance_m": args.horizontal_distance_m,
        "vertical_clearance_m": args.vertical_clearance_m,
        "commands_run": commands_run,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nPipeline complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
