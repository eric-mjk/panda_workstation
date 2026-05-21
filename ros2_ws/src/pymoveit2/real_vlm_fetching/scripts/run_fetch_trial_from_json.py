#!/usr/bin/env python3
"""
Dry-run script: loads a VLM result JSON and prints the computed fetch displacement.
Does NOT command the robot.
"""
import argparse
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_vlm_fetching.pose_io import load_json, save_json
from real_vlm_fetching.frame_conventions import (
    clock_to_world_direction,
    compute_fetch_displacement,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute fetch displacement from a VLM result JSON (no robot commands)."
    )
    p.add_argument("--vlm-result", required=True, metavar="PATH",
                   help="Path to VLM result JSON file.")
    p.add_argument("--horizontal-distance-m", type=float, default=0.10,
                   metavar="M", help="Horizontal reach distance in metres (default: 0.10).")
    p.add_argument("--vertical-clearance-m", type=float, default=0.05,
                   metavar="M", help="Vertical lift clearance in metres (default: 0.05).")
    return p.parse_args()


def main():
    args = parse_args()

    vlm_path = Path(args.vlm_result)
    if not vlm_path.exists():
        print(f"ERROR: file not found: {vlm_path}", file=sys.stderr)
        sys.exit(1)

    data = load_json(vlm_path)

    clock = data.get("best_clock") or data.get("clock")
    if clock is None:
        print("ERROR: JSON must contain 'best_clock' or 'clock' field.", file=sys.stderr)
        sys.exit(1)
    clock = int(clock)

    h_dir = clock_to_world_direction(clock)
    displacement = compute_fetch_displacement(
        clock,
        args.horizontal_distance_m,
        args.vertical_clearance_m,
    )

    print(f"Selected clock   : {clock}")
    print(f"Horizontal dir   : {h_dir}")
    print(f"Fetch displacement: {displacement}")

    output_path = vlm_path.parent / "computed_displacement.json"
    save_json(output_path, {
        "vlm_result": str(vlm_path),
        "clock": clock,
        "horizontal_distance_m": args.horizontal_distance_m,
        "vertical_clearance_m": args.vertical_clearance_m,
        "horizontal_direction": h_dir,
        "fetch_displacement": displacement,
    })
    print(f"Saved             : {output_path}")


if __name__ == "__main__":
    main()
