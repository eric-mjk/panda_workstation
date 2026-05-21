#!/usr/bin/env python3
"""
Capture one RGB + depth + depth_colormap frame from the RealSense camera.
Does NOT command the robot.
"""
import argparse
import sys
from pathlib import Path

# examples/ contains utils_camera and robot_arm_config
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples"))

from robot_arm_config import CAMERA_SERIAL
from utils_camera import set_camera, get_frames, capture_img


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture one frame from the RealSense camera (no robot commands)."
    )
    p.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "real_vlm_fetching" / "runs" / "capture_once"),
        metavar="PATH",
        help="Output directory (default: real_vlm_fetching/runs/capture_once).",
    )
    p.add_argument(
        "--index", type=int, default=0, metavar="INT",
        help="Frame index used in output filenames (default: 0).",
    )
    p.add_argument(
        "--camera-serial", default=None, metavar="SERIAL",
        help="RealSense serial number override (default: value from robot_arm_config).",
    )
    p.add_argument(
        "--warmup-frames", type=int, default=10, metavar="INT",
        help="Number of frames to discard before capturing (default: 10).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    serial = args.camera_serial if args.camera_serial is not None else CAMERA_SERIAL

    print(f"Camera serial  : {serial}")
    print(f"Output dir     : {args.out_dir}")
    print(f"Warmup frames  : {args.warmup_frames}")

    pipeline, align = set_camera(serial)
    try:
        for _ in range(args.warmup_frames):
            get_frames(pipeline, align)

        ok = capture_img(pipeline, align, args.out_dir, args.index)
        if not ok:
            print("ERROR: capture_img returned no frames.", file=sys.stderr)
            sys.exit(1)
    finally:
        pipeline.stop()

    fmt = f"{args.index:05d}.png"
    for sub in ("rgb", "depth", "depth_colormap"):
        print(f"  {Path(args.out_dir) / sub / fmt}")


if __name__ == "__main__":
    main()
