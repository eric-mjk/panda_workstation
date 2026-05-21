#!/usr/bin/env python3
"""
Read best_clock from handoff output and compute EE fetch displacement.
No robot commands. No VLM API calls. Dry-run only.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_vlm_fetching.frame_conventions import clock_to_world_direction
from real_vlm_fetching.camera_clock_mapping import (
    quaternion_xyzw_to_rotation_matrix,
    clock_to_base_direction as _clock_to_base_direction,
)

# Candidate JSON/CSV keys, checked in order.
_CLOCK_KEYS = (
    "best_clock",
    "selected_clock",
    "clock",
    "best_clock_direction",
    "selected_clock_direction",
)


def _parse_clock_value(raw):
    """Return int clock (1-12) from int, numeric string, or 'N o'clock' string. None if unparseable."""
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        pass
    if isinstance(raw, str):
        m = re.search(r'\b(\d{1,2})\b', raw)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 12:
                return v
    return None


def _clock_from_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    for key in _CLOCK_KEYS:
        v = _parse_clock_value(data.get(key))
        if v is not None:
            return v
    return None


def _clock_from_csv(path: Path, tag=None):
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if tag:
        tagged = [r for r in rows if r.get("tag", "").strip() == tag]
        if tagged:
            rows = tagged
    for row in rows:
        for key in _CLOCK_KEYS:
            v = _parse_clock_value(row.get(key, ""))
            if v is not None:
                return v
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute fetch displacement from handoff best_clock (no robot commands)."
    )
    p.add_argument("--handoff-output", default=None, metavar="PATH")
    p.add_argument("--vlm-result", default=None, metavar="PATH")
    p.add_argument("--summary-csv", default=None, metavar="PATH")
    p.add_argument("--clock", type=int, default=None, metavar="INT",
                   help="Override: use this clock directly (1-12).")
    p.add_argument("--target-tag", default=None, metavar="TAG")
    p.add_argument("--horizontal-distance-m", type=float, default=0.10, metavar="FLOAT")
    p.add_argument("--vertical-clearance-m", type=float, default=0.05, metavar="FLOAT")
    p.add_argument("--out", default=None, metavar="PATH")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--cam-pos", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Camera position in base frame (metres).")
    p.add_argument("--cam-quat", nargs=4, type=float, default=None,
                   metavar=("QX", "QY", "QZ", "QW"),
                   help="Camera orientation as quaternion xyzw in base frame.")
    p.add_argument("--camera-extrinsics", default=None, metavar="PATH",
                   help="JSON file with translation and quaternion_xyzw fields.")
    return p.parse_args()


def main():
    args = parse_args()

    source_handoff = None
    source_vlm = None
    source_csv = None
    best_clock = None

    # Priority 1: explicit --clock
    if args.clock is not None:
        best_clock = args.clock

    # Priority 2: --vlm-result JSON
    elif args.vlm_result is not None:
        source_vlm = str(Path(args.vlm_result).resolve())
        best_clock = _clock_from_json(Path(args.vlm_result))

    # Priority 3: --handoff-output directory
    elif args.handoff_output is not None:
        ho = Path(args.handoff_output)
        source_handoff = str(ho.resolve())
        result_jsons = sorted(ho.rglob("result.json"))
        if args.target_tag:
            tagged = [rj for rj in result_jsons if rj.parent.name == args.target_tag]
            if tagged:
                result_jsons = tagged
        for rj in result_jsons:
            best_clock = _clock_from_json(rj)
            if best_clock is not None:
                source_vlm = str(rj.resolve())
                break
        if best_clock is None:
            csv_path = ho / "summary.csv"
            if csv_path.exists():
                source_csv = str(csv_path.resolve())
                best_clock = _clock_from_csv(csv_path, args.target_tag)

    # Priority 4: --summary-csv
    elif args.summary_csv is not None:
        source_csv = str(Path(args.summary_csv).resolve())
        best_clock = _clock_from_csv(Path(args.summary_csv), args.target_tag)

    if best_clock is None:
        print(
            "ERROR: could not determine best_clock.\n"
            "Provide --clock, --vlm-result, --handoff-output, or --summary-csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    if best_clock not in range(1, 13):
        print(f"ERROR: best_clock={best_clock} is not in range 1-12.", file=sys.stderr)
        sys.exit(1)

    # Resolve camera extrinsics (file > CLI > fallback)
    cam_translation = None
    cam_quaternion = None  # [qx, qy, qz, qw]

    if args.camera_extrinsics is not None:
        ext = json.loads(Path(args.camera_extrinsics).read_text())
        cam_translation = ext["translation"]
        cam_quaternion = ext["quaternion_xyzw"]
    elif args.cam_pos is not None and args.cam_quat is not None:
        cam_translation = args.cam_pos
        cam_quaternion = args.cam_quat

    if cam_quaternion is not None:
        R = quaternion_xyzw_to_rotation_matrix(*cam_quaternion)
        mapping = _clock_to_base_direction(best_clock, R)
        mapping_source = "camera_transform"
        h_dir = mapping["horizontal_direction_base"]
        d_cam = mapping["d_cam"]
        d_base_raw = mapping["d_base_raw"]
    else:
        h_dir = clock_to_world_direction(best_clock)
        mapping_source = "fallback_hardcoded"
        d_cam = None
        d_base_raw = None

    displacement = [
        args.horizontal_distance_m * h_dir[0],
        args.horizontal_distance_m * h_dir[1],
        args.vertical_clearance_m,
    ]

    output = {
        "target_tag": args.target_tag,
        "best_clock": best_clock,
        "mapping_source": mapping_source,
        "d_cam": d_cam,
        "d_base_raw": d_base_raw,
        "horizontal_direction_base": h_dir,
        "fetch_displacement_base_m": displacement,
        "horizontal_distance_m": args.horizontal_distance_m,
        "vertical_clearance_m": args.vertical_clearance_m,
        "camera_transform": {
            "translation": cam_translation,
            "quaternion_xyzw": cam_quaternion,
        } if cam_quaternion is not None else None,
        "frame_convention": {
            "assumption": "12=image top, 3=image right, mapped to current base-frame convention",
        },
        "warning": "Verify image-clock to robot-base consistency before real execution.",
        "source": {
            "handoff_output": source_handoff,
            "vlm_result": source_vlm,
            "summary_csv": source_csv,
        },
        "dry_run": True,
    }

    if args.out is not None:
        out_path = Path(args.out)
    elif args.handoff_output is not None:
        out_path = Path(args.handoff_output) / "computed_fetch_motion.json"
    else:
        out_path = Path("/tmp/computed_fetch_motion.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"best_clock             : {best_clock}")
    if args.target_tag:
        print(f"target_tag             : {args.target_tag}")
    print(f"mapping_source         : {mapping_source}")
    if d_cam is not None:
        print(f"d_cam                  : {d_cam}")
    if d_base_raw is not None:
        print(f"d_base_raw             : {d_base_raw}")
    print(f"horizontal_direction   : {h_dir}")
    print(f"fetch_displacement_m   : {displacement}")
    print(f"Output JSON            : {out_path.resolve()}")
    print(
        "\nWARNING: Verify image-clock to robot-base consistency before real execution."
    )


if __name__ == "__main__":
    main()
