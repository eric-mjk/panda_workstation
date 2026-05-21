#!/usr/bin/env python3
"""
Generate a clock-direction overlay image and target info JSON from a real RGB image.
Does NOT command the robot or call any external API.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_vlm_fetching.overlay import draw_clock_overlay
from real_vlm_fetching.pose_io import load_json, save_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Create VLM input overlay image and target info JSON."
    )
    p.add_argument("--image", required=True, metavar="PATH",
                   help="Input RGB image path.")
    p.add_argument("--target-json", required=True, metavar="PATH",
                   help='JSON with "target", "u", "v" fields.')
    p.add_argument("--out-dir", required=True, metavar="PATH",
                   help="Output directory.")
    p.add_argument("--radius-px", type=int, default=80, metavar="PX",
                   help="Clock arrow radius in pixels (default: 80).")
    return p.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    target_path = Path(args.target_json)
    if not target_path.exists():
        print(f"ERROR: target JSON not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    target = load_json(target_path)
    name = target.get("target") or target.get("name")
    u = int(target["u"])
    v = int(target["v"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_out = str(out_dir / "vlm_input_overlay.png")
    draw_clock_overlay(
        image_path=str(image_path),
        target_u=u,
        target_v=v,
        target_name=name,
        output_path=overlay_out,
        radius_px=args.radius_px,
    )

    info = {
        "source_image": str(image_path.resolve()),
        "target": name,
        "u": u,
        "v": v,
        "radius_px": args.radius_px,
        "overlay_image": str(Path(overlay_out).resolve()),
    }
    info_out = str(out_dir / "target_info.json")
    save_json(info_out, info)

    print(f"Saved overlay  : {overlay_out}")
    print(f"Saved info     : {info_out}")


if __name__ == "__main__":
    main()
