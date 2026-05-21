#!/usr/bin/env python3
"""
Translate the current EE by a fetch displacement from computed_fetch_motion.json.
No gripper. No grasp pose. Default dry-run — requires --execute to move.
Requires ROS 2 and MoveIt 2.
"""
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"

for _p in (str(_REPO_ROOT), str(_EXAMPLES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import math
import time
from threading import Thread

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import panda as robot

from robot_arm_config import PANDA_HAND_LINK
from utils_franka_moveit2 import (
    pose_to_transform,
    ensure_table_collision_box,
    ensure_wall_collision_box,
)
from utils_math import quaternion_from_rotation_matrix


def _fmt(vals, decimals=4):
    return "[" + ", ".join(f"{v:+.{decimals}f}" for v in vals) + "]"


def parse_args():
    p = argparse.ArgumentParser(
        description="Translate EE by fetch displacement (no gripper, no grasp pose).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--motion-json", required=True, metavar="PATH",
                   help="Path to computed_fetch_motion.json.")
    p.add_argument("--scale", type=float, default=1.0, metavar="FLOAT",
                   help="Scale factor applied to displacement.")
    p.add_argument("--max-step-m", type=float, default=0.03, metavar="FLOAT",
                   help="Refuse execution if scaled displacement norm exceeds this.")
    p.add_argument("--execute", action="store_true",
                   help="Actually move the robot. Without this flag only the plan is printed.")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Dry-run (default True). Overridden when --execute is passed.")
    p.add_argument("--allow-large-motion", action="store_true",
                   help="Override the --max-step-m safety limit.")
    p.add_argument("--speed-scale", type=float, default=0.2, metavar="FLOAT",
                   help="MoveIt2 max_velocity and max_acceleration.")
    return p.parse_args()


def _wait_joint_state(moveit2, timeout_sec=10.0):
    t0 = time.time()
    while moveit2.joint_state is None:
        if time.time() - t0 > timeout_sec:
            raise RuntimeError("Timed out waiting for /joint_states.")
        time.sleep(0.1)


def main():
    args = parse_args()

    # ── 1. Load motion JSON ───────────────────────────────────────────────────
    motion_path = Path(args.motion_json)
    if not motion_path.exists():
        print(f"ERROR: {motion_path} not found.", file=sys.stderr)
        sys.exit(1)

    motion = json.loads(motion_path.read_text())
    raw_disp = motion.get("fetch_displacement_base_m")
    if raw_disp is None:
        print("ERROR: 'fetch_displacement_base_m' missing in motion JSON.", file=sys.stderr)
        sys.exit(1)

    displacement = [v * args.scale for v in raw_disp]
    disp_norm = math.sqrt(sum(v * v for v in displacement))

    print(f"\nMotion JSON         : {motion_path}")
    print(f"best_clock          : {motion.get('best_clock', '?')}  "
          f"(mapping: {motion.get('mapping_source', 'unknown')})")
    print(f"Raw displacement    : {_fmt(raw_disp)} m")
    print(f"Scale               : {args.scale}")
    print(f"Displacement        : {_fmt(displacement)} m")
    print(f"Displacement norm   : {disp_norm:.4f} m  (limit: {args.max_step_m} m)")

    # ── 2. Safety check ───────────────────────────────────────────────────────
    exceeds = disp_norm > args.max_step_m and not args.allow_large_motion
    if exceeds:
        print(
            f"\nSAFETY: norm {disp_norm:.4f} m > --max-step-m {args.max_step_m} m.\n"
            f"Pass --allow-large-motion to override, or reduce --scale.",
            file=sys.stderr,
        )
        if args.execute:
            print("Execution blocked by safety check.", file=sys.stderr)
            sys.exit(1)
        print("(continuing dry-run for inspection)")

    # ── 3. ROS / MoveIt2 ─────────────────────────────────────────────────────
    rclpy.init()
    node = Node("execute_fetch_motion")
    cb_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name="panda_link8",
        group_name="panda_arm",
        callback_group=cb_group,
    )
    moveit2.planner_id = "RRTConnectkConfigDefault"
    moveit2.max_velocity = args.speed_scale
    moveit2.max_acceleration = args.speed_scale

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    node.create_rate(10.0).sleep()

    try:
        ensure_table_collision_box(moveit2)
        ensure_wall_collision_box(moveit2)
        _wait_joint_state(moveit2)

        # ── 4. Current EE pose via FK ─────────────────────────────────────────
        poses = moveit2.compute_fk(fk_link_names=["panda_link8"])
        if poses is None:
            print("ERROR: compute_fk returned None.", file=sys.stderr)
            sys.exit(1)

        base_to_ee = pose_to_transform(poses[0])
        cur_pos = base_to_ee[:3, 3].tolist()
        cur_quat = [float(v) for v in quaternion_from_rotation_matrix(base_to_ee[:3, :3])]

        tgt_pos = [cur_pos[i] + displacement[i] for i in range(3)]
        tgt_quat = cur_quat  # orientation unchanged

        print(f"\nCurrent EE pos      : {_fmt(cur_pos, 3)}")
        print(f"Current EE quat     : {_fmt(cur_quat, 3)}")
        print(f"Target EE pos       : {_fmt(tgt_pos, 3)}")
        print(f"Target EE quat      : {_fmt(tgt_quat, 3)}  (unchanged)")
        print(f"Execute requested   : {args.execute}")

        if not args.execute:
            print("\nDry-run — robot not moved. Pass --execute to move.")
            print("Motion executed     : False")
            return

        # ── 5. Execute ────────────────────────────────────────────────────────
        print("\nExecuting cartesian motion...")
        moveit2.move_to_pose(
            position=tuple(tgt_pos),
            quat_xyzw=tuple(tgt_quat),
            cartesian=True,
            cartesian_max_step=0.0025,
            cartesian_fraction_threshold=0.0,
        )

        if moveit2.query_state() == MoveIt2State.IDLE:
            print("WARNING: MoveIt2 IDLE immediately — planning may have failed.")
            print("Motion executed     : False")
        else:
            success = moveit2.wait_until_executed()
            print(f"Motion executed     : {success}")

    finally:
        executor.shutdown()
        executor_thread.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
