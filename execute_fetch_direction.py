#!/usr/bin/env python3
"""
Execute a manually specified fetching direction on the real Panda.

This is a thin wrapper around execute_fetch_motion.py.
It converts a clock direction into a horizontal + vertical displacement JSON.

Clock convention:
  12 = +Y
   3 = +X
   6 = -Y
   9 = -X

Default motion:
  horizontal = 0.10 m
  vertical   = 0.05 m

Example:
  python3 real_vlm_fetching/scripts/execute_fetch_direction.py --clock 3 --scale 0.2 --execute
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def clock_to_unit(clock: int):
    if not (1 <= clock <= 12):
        raise ValueError(f"clock must be in [1, 12], got {clock}")

    # Image/BEV convention:
    # 12 o'clock = +Y, 3 o'clock = +X, 6 o'clock = -Y, 9 o'clock = -X.
    angle_deg = 90.0 - (clock % 12) * 30.0
    angle_rad = math.radians(angle_deg)

    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Clean tiny numerical noise.
    if abs(dx) < 1e-12:
        dx = 0.0
    if abs(dy) < 1e-12:
        dy = 0.0

    return [float(dx), float(dy), 0.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", type=int, required=True, help="Clock direction: 1..12")
    parser.add_argument("--horizontal-distance-m", type=float, default=0.10)
    parser.add_argument("--vertical-clearance-m", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-step-m", type=float, default=0.13)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--tmp-json",
        type=str,
        default="/tmp/manual_fetch_direction_motion.json",
        help="Temporary motion JSON path",
    )
    args = parser.parse_args()

    unit = clock_to_unit(args.clock)
    disp = [
        unit[0] * args.horizontal_distance_m,
        unit[1] * args.horizontal_distance_m,
        args.vertical_clearance_m,
    ]

    payload = {
        "source": "manual_clock_direction",
        "best_clock": args.clock,
        "mapping": "manual_clock_to_world",
        "horizontal_direction_world_unit": unit,
        "horizontal_distance_m": args.horizontal_distance_m,
        "vertical_clearance_m": args.vertical_clearance_m,
        "fetch_displacement_base_m": disp,
        "fetch_displacement_world_m": disp,
        "fetch_displacement_m": disp,
    }

    tmp_json = Path(args.tmp_json)
    tmp_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Manual clock        : {args.clock}")
    print(f"Horizontal unit     : [{unit[0]:+.4f}, {unit[1]:+.4f}, {unit[2]:+.4f}]")
    print(f"Raw displacement    : [{disp[0]:+.4f}, {disp[1]:+.4f}, {disp[2]:+.4f}] m")
    print(f"Scale               : {args.scale}")
    print(f"Scaled displacement : [{disp[0]*args.scale:+.4f}, {disp[1]*args.scale:+.4f}, {disp[2]*args.scale:+.4f}] m")
    print(f"Temporary JSON      : {tmp_json}")

    script = Path(__file__).resolve().parent / "execute_fetch_motion.py"

    cmd = [
        sys.executable,
        str(script),
        "--motion-json",
        str(tmp_json),
        "--scale",
        str(args.scale),
        "--max-step-m",
        str(args.max_step_m),
    ]

    if args.execute:
        cmd.append("--execute")

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
