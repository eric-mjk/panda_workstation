from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .pipeline import _resolve_experiment_dir
except ImportError:
    from pipeline import _resolve_experiment_dir


AP_OUTPUT_NAMES = {
    "rgb",
    "depth",
    "depth_preview",
    "intrinsics.json",
    "pose.json",
    "occupancy_grid.ply",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete FetchBench ROS offline outputs while preserving AP captures."
    )
    parser.add_argument("--output-root", default="/workspace/ros2_ws/ours_experiment")
    parser.add_argument("--experiment-name", default="ex2")
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag, only prints what would be removed.",
    )
    return parser


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    args = _build_parser().parse_args()
    exp_dir = _resolve_experiment_dir(args)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    removable = []
    preserved = []
    for child in sorted(exp_dir.iterdir(), key=lambda p: p.name):
        if child.name in AP_OUTPUT_NAMES:
            preserved.append(child)
        else:
            removable.append(child)

    mode = "DELETE" if bool(args.yes) else "DRY RUN"
    print(f"[CLEAN] {mode}: {exp_dir}", flush=True)
    print("[CLEAN] Preserving AP outputs:", flush=True)
    for path in preserved:
        print(f"  keep   {path}", flush=True)

    if removable:
        print("[CLEAN] Removing offline/non-AP outputs:", flush=True)
    else:
        print("[CLEAN] Nothing to remove.", flush=True)

    for path in removable:
        print(f"  remove {path}", flush=True)
        if bool(args.yes):
            _remove_path(path)

    if not bool(args.yes) and removable:
        print("[CLEAN] Dry run only. Re-run with --yes to delete these files.", flush=True)
    elif bool(args.yes):
        print(f"[CLEAN] Done. Removed {len(removable)} path(s).", flush=True)


if __name__ == "__main__":
    main()
