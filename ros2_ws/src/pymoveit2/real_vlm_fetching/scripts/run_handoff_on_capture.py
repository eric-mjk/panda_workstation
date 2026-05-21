#!/usr/bin/env python3
"""
Thin adapter: RealSense capture → handoff VLM batch baseline.
Does NOT command the robot or call any external API.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parents[1]          # scripts → real_vlm_fetching → pymoveit2
_HANDOFF_CWD = _REPO_ROOT / "vlm_single_view_fetching"
_PY = sys.executable

_DEFAULT_TASK_TEXT = (
    "fetch the target object at the clicked grasp/extraction marker "
    "while disturbing surrounding objects as little as possible"
)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n>>> {cmd_str}")
    subprocess.run(cmd, check=True, cwd=cwd)
    return cmd_str


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture + handoff VLM baseline adapter (no robot commands)."
    )
    p.add_argument("--out-dir", required=True, metavar="PATH",
                   help="Top-level output directory for this real trial.")
    p.add_argument("--target-tag", required=True, metavar="TEXT",
                   help="Short tag for the target (e.g. mustard).")
    p.add_argument("--target-u", type=int, default=None, metavar="INT")
    p.add_argument("--target-v", type=int, default=None, metavar="INT")
    p.add_argument("--click-target", action="store_true",
                   help="Open image picker to select the target pixel interactively.")
    p.add_argument("--target-pixel-json", default=None, metavar="PATH",
                   help="Load u/v from a pre-saved target pixel JSON; skips the picker.")
    p.add_argument("--object-class", required=True, metavar="TEXT",
                   help="Human-readable object class (e.g. 'yellow mustard bottle').")
    p.add_argument("--task-text", default=_DEFAULT_TASK_TEXT, metavar="TEXT")
    p.add_argument("--image", default=None, metavar="PATH",
                   help="Use this image directly; skip capture if provided.")
    p.add_argument("--capture-index", type=int, default=0, metavar="INT")
    p.add_argument("--warmup-frames", type=int, default=10, metavar="INT")
    p.add_argument("--camera-serial", default=None, metavar="TEXT")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Pass --dry-run to handoff runner (always on for now).")
    p.add_argument("--handoff-out-subdir", default="handoff_output", metavar="TEXT")
    return p.parse_args()


def _read_uv(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    u = data.get("u") if data.get("u") is not None else data.get("x")
    v = data.get("v") if data.get("v") is not None else data.get("y")
    if u is None or v is None:
        print(f"ERROR: {path} has no u/v or x/y fields.", file=sys.stderr)
        sys.exit(1)
    return int(u), int(v)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands_run: list[str] = []

    # ── Step 1/2: capture or use provided image ───────────────────────────────
    if args.image is not None:
        image_path = Path(args.image).resolve()
    else:
        capture_out = out_dir / "capture"
        cmd = [
            _PY, str(_SCRIPTS_DIR / "capture_once.py"),
            "--out-dir", str(capture_out),
            "--index", str(args.capture_index),
            "--warmup-frames", str(args.warmup_frames),
        ]
        if args.camera_serial is not None:
            cmd += ["--camera-serial", args.camera_serial]
        commands_run.append(_run(cmd))
        image_path = (capture_out / "rgb" / f"{args.capture_index:05d}.png").resolve()

    # ── Step 2b: resolve target u/v ──────────────────────────────────────────
    target_pixel_json: Path | None = None
    if args.click_target:
        target_pixel_json = out_dir / "target_pixel.json"
        cmd = [
            _PY,
            str(_REPO_ROOT / "vlm_single_view_fetching" / "fetching_baseline" / "pick_target_pixel.py"),
            "--image", str(image_path),
            "--out", str(target_pixel_json),
        ]
        commands_run.append(_run(cmd))
        target_u, target_v = _read_uv(target_pixel_json)
        print(f"Selected pixel   : u={target_u}, v={target_v}")
    elif args.target_pixel_json is not None:
        target_pixel_json = Path(args.target_pixel_json)
        target_u, target_v = _read_uv(target_pixel_json)
        print(f"Selected pixel   : u={target_u}, v={target_v}")
    else:
        if args.target_u is None or args.target_v is None:
            print(
                "ERROR: provide --target-u/--target-v, --click-target, or --target-pixel-json.",
                file=sys.stderr,
            )
            sys.exit(1)
        target_u, target_v = args.target_u, args.target_v

    # ── Step 3: write handoff target-set JSON ─────────────────────────────────
    target_set_json_path = out_dir / "handoff_target_set.json"
    target_set = {
        "image_path": str(image_path),
        "target_provider": "manual_multi",
        "targets": [
            {
                "id": 1,
                "tag": args.target_tag,
                "u": target_u,
                "v": target_v,
                "object_class": args.object_class,
                "task_text": args.task_text,
                "target_provider": "manual_multi",
                "target_note": "real capture adapter",
            }
        ],
    }
    target_set_json_path.write_text(json.dumps(target_set, indent=2))
    print(f"Wrote target-set JSON: {target_set_json_path}")

    # ── Step 4: call handoff batch runner (always dry-run for now) ────────────
    handoff_out_dir = out_dir / args.handoff_out_subdir
    cmd = [
        _PY, "fetching_baseline/run_batch_vlm_fetching_direction.py",
        "--target-set-json", str(target_set_json_path),
        "--out-root", str(handoff_out_dir),
        "--dry-run",
    ]
    commands_run.append(_run(cmd, cwd=_HANDOFF_CWD))

    # ── Step 5: write adapter manifest ────────────────────────────────────────
    manifest_path = out_dir / "adapter_manifest.json"
    manifest = {
        "image_path": str(image_path),
        "target_set_json": str(target_set_json_path),
        "handoff_output_dir": str(handoff_out_dir),
        "target_tag": args.target_tag,
        "target_u": target_u,
        "target_v": target_v,
        "target_pixel_json": str(target_pixel_json) if target_pixel_json else None,
        "object_class": args.object_class,
        "task_text": args.task_text,
        "commands_run": commands_run,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # ── Step 6: print useful paths ────────────────────────────────────────────
    summary_csv = handoff_out_dir / "summary.csv"
    print(f"\nImage path       : {image_path}")
    print(f"Target-set JSON  : {target_set_json_path}")
    print(f"Handoff output   : {handoff_out_dir}")
    print(f"Summary CSV      : {summary_csv}")
    pngs = sorted(handoff_out_dir.rglob("*.png"))
    if pngs:
        print("PNG outputs:")
        for png in pngs:
            print(f"  {png}")
    print(f"\nAdapter manifest : {manifest_path}")


if __name__ == "__main__":
    main()
