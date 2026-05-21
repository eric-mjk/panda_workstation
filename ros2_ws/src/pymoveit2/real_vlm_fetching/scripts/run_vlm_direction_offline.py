#!/usr/bin/env python3
"""
Offline VLM direction script. Does NOT call any external API.
Use --mock-clock to inject a fake VLM result for pipeline testing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_vlm_fetching.pose_io import save_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Write a VLM result JSON (mock mode only — no API calls)."
    )
    p.add_argument("--overlay", required=True, metavar="PATH",
                   help="Path to the VLM input overlay image.")
    p.add_argument("--target", required=True, metavar="TEXT",
                   help="Target object name.")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="Output path for vlm_result.json.")
    p.add_argument("--mock-clock", type=int, metavar="INT",
                   help="If set, write a mock VLM result with this clock (1–12). No API is called.")
    return p.parse_args()


def main():
    args = parse_args()

    overlay_path = Path(args.overlay)
    if not overlay_path.exists():
        print(f"ERROR: overlay image not found: {overlay_path}", file=sys.stderr)
        sys.exit(1)

    if args.mock_clock is not None:
        if args.mock_clock not in range(1, 13):
            print(f"ERROR: --mock-clock must be 1–12, got {args.mock_clock}", file=sys.stderr)
            sys.exit(1)

        result = {
            "target": args.target,
            "best_clock": args.mock_clock,
            "best_score": 1.0,
            "model": "mock",
            "short_reason": "Mock result for pipeline testing.",
        }
        out_path = Path(args.out)
        save_json(out_path, result)
        print(f"Mock VLM result saved: {out_path}")
        print(f"  target     : {args.target}")
        print(f"  best_clock : {args.mock_clock}")
    else:
        print("No --mock-clock provided and real VLM API is not yet wired up.")
        print("Pass --mock-clock <1-12> to write a mock result for pipeline testing.")
        sys.exit(0)


if __name__ == "__main__":
    main()
