#!/usr/bin/env python3
"""
Reads a config file specifying a sequence of candidate ranks, then moves the
Panda arm through each view pose by calling:

  ros2 run pymoveit2 panda_joint_goal.py --ros-args \
      -p joint_positions:="[j1, j2, j3, j4, j5, j6, j7]"

Press 'g' to advance to the next pose, Ctrl-C to abort.

Usage:
  python3 run_view_candidates.py [config.json]

Default config path: scripts/view_candidates_config.json
"""

import json
import os
import subprocess
import sys
import termios
import tty


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "view_candidates_config.json")


def read_key():
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG

    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)

    print(f"Loading config: {config_path}")
    config = load_json(config_path)

    candidates_path = config["candidates_json"]
    if not os.path.isabs(candidates_path):
        candidates_path = os.path.join(SCRIPT_DIR, candidates_path)

    print(f"Loading candidates: {candidates_path}")
    data = load_json(candidates_path)

    # Build rank -> candidate lookup
    by_rank = {c["rank"]: c for c in data["candidates"]}

    sequence = config["sequence"]
    total = len(sequence)

    print(f"\nSequence has {total} poses: {sequence}")
    print("Press 'g' to move to each pose, Ctrl-C to abort.\n")

    for step, rank in enumerate(sequence, start=1):
        if rank not in by_rank:
            print(f"[{step}/{total}] Rank {rank} not found in candidates — skipping.")
            continue

        c = by_rank[rank]
        joints = c["joint_angles"]
        joints_str = "[" + ", ".join(f"{j:.10f}" for j in joints) + "]"

        print(f"[{step}/{total}] Rank {rank}")
        print(f"  visible_voxels   : {c['visible_voxel_count']}")
        print(f"  manipulability   : {c['manipulability_score']:.4f}")
        print(f"  plannable_home   : {c['plannable_from_home']}")
        print(f"  joint_angles     : {[round(j, 4) for j in joints]}")
        print("Press 'g' to execute, Ctrl-C to quit ... ", end="", flush=True)

        while True:
            key = read_key()
            if key == "g":
                print("go")
                break
            if key == "\x03":  # Ctrl-C
                print("\nAborted.")
                sys.exit(0)

        cmd = [
            "ros2", "run", "pymoveit2", "panda_joint_goal.py",
            "--ros-args",
            "-p", f"joint_positions:={joints_str}",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: command exited with code {result.returncode}")
        print()

    print("All poses complete.")


if __name__ == "__main__":
    main()
