from __future__ import annotations

import atexit
import json
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image as PILImage
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
import tf2_ros

from .core import (
    CandidateScoreManager,
    UnknownVoxelScorer,
    VoxelAccumulator,
    default_candidates_path,
    intrinsics_matrix_from_camera_info,
    load_candidates,
    transform_msg_to_matrix,
)


def _scale_intrinsics_matrix(intrinsics: np.ndarray, width: int, height: int, base_width: int, base_height: int) -> np.ndarray:
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    sx = float(width) / max(float(base_width), 1.0)
    sy = float(height) / max(float(base_height), 1.0)
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


@dataclass
class CaptureBundle:
    rgb_msg: Image
    depth_msg: Image
    camera_info: CameraInfo
    sync_index: int


class ActivePerceptionCoordinator(Node):
    """Real-robot coordinator for the copied FetchBench AP algorithm."""

    def __init__(self) -> None:
        super().__init__("fetchbench_active_perception")
        self._declare_parameters()
        self._camera_config = self._load_camera_config()
        camera_source_params = self._get_camera_source_parameters(self._camera_config)
        camera_info_from_config = self._camera_info_from_config(self._camera_config)

        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._rgb_topic = str(camera_source_params.get("rgb_topic", self.get_parameter("rgb_topic").value))
        self._depth_topic = str(camera_source_params.get("depth_topic", self.get_parameter("depth_topic").value))
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._world_frame = str(self._get_nested_config_parameter(
            self._camera_config,
            ("tsdf_debug", "ros__parameters"),
            "world_frame",
            self.get_parameter("world_frame").value,
        ))
        self._camera_frame_param = str(camera_source_params.get("frame_id", self.get_parameter("camera_frame").value))
        if camera_info_from_config is not None and camera_info_from_config.header.frame_id:
            self._camera_frame_param = camera_info_from_config.header.frame_id
        self._depth_scale = float(camera_source_params.get("depth_scale", self.get_parameter("depth_scale").value))
        self._max_depth_m = float(camera_source_params.get("max_depth", self.get_parameter("max_depth_m").value))
        self._sync_slop = float(camera_source_params.get("sync_slop", self.get_parameter("sync_slop").value))
        self._capture_timeout_s = float(self.get_parameter("capture_timeout_s").value)
        self._tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)

        self._latest_info: CameraInfo | None = camera_info_from_config
        self._latest_bundle: CaptureBundle | None = None
        self._synced_count = 0
        self._rgbd_without_info_count = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._rgb_sub = Subscriber(self, Image, self._rgb_topic, qos_profile=sensor_qos)
        self._depth_sub = Subscriber(self, Image, self._depth_topic, qos_profile=sensor_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=10,
            slop=self._sync_slop,
        )
        self._sync.registerCallback(self._rgbd_callback)
        if camera_info_from_config is None:
            self.create_subscription(CameraInfo, self._camera_info_topic, self._camera_info_callback, 10)

        candidates_path = str(self.get_parameter("candidates_json").value)
        if not candidates_path:
            candidates_path = str(default_candidates_path())
        self._candidates_path = Path(candidates_path).expanduser().resolve()
        self._candidates, self._candidates_doc = load_candidates(
            self._candidates_path,
            int(self.get_parameter("top_k").value),
        )

        self._accumulator: VoxelAccumulator | None = None
        self._scorer: UnknownVoxelScorer | None = None
        self._candidate_ranker: CandidateScoreManager | None = None
        self._intrinsics: np.ndarray | None = None

        self._used_indices: set[int] = set()
        self._view_records: list[dict] = []
        self._pose_records: list[dict] = []
        self._reached_view_indices: set[int] = set()
        self._planned_moves: list[dict] = []
        self._stop_reason = "not_started"
        self._pending_candidate_idx: int | None = None
        self._pending_candidate_meta: dict | None = None
        self._pending_candidate_step: int | None = None
        self._current_view_candidate_idx: int | None = None
        self._current_view_expected_gain: dict | None = None
        self._interactive_step_idx = 1
        self._interactive_quit_requested = False
        self._terminal_settings = None

        self._output_root = Path(str(self.get_parameter("output_dir").value)).expanduser().resolve()
        self._experiment_name = str(self.get_parameter("experiment_name").value).strip()
        self._output_dir = self._output_root / self._experiment_name if self._experiment_name else self._output_root
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._views_dir = self._output_dir
        self._rgb_dir = self._output_dir / "rgb"
        self._depth_dir = self._output_dir / "depth"
        self._depth_preview_dir = self._output_dir / "depth_preview"
        for directory in (self._rgb_dir, self._depth_dir, self._depth_preview_dir):
            directory.mkdir(parents=True, exist_ok=True)

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        marker_prefix = str(self.get_parameter("voxel_marker_topic_prefix").value).rstrip("/")
        self._occupied_marker_pub = self.create_publisher(Marker, f"{marker_prefix}/occupied_voxels", marker_qos)
        self._unknown_marker_pub = self.create_publisher(Marker, f"{marker_prefix}/unknown_voxels", marker_qos)
        self._free_marker_pub = self.create_publisher(Marker, f"{marker_prefix}/free_voxels", marker_qos)
        self._next_best_view_pub = self.create_publisher(Marker, f"{marker_prefix}/next_best_view", marker_qos)

        self.get_logger().info("FetchBench real active-perception coordinator initialized")
        self.get_logger().info(f"RGB topic: {self._rgb_topic}")
        self.get_logger().info(f"Depth topic: {self._depth_topic}")
        if camera_info_from_config is None:
            self.get_logger().info(f"CameraInfo topic: {self._camera_info_topic}")
        else:
            self.get_logger().info("CameraInfo: loaded from camera_config_file")
        self.get_logger().info(f"TF lookup: {self._world_frame} -> {self._camera_frame_param or '<camera_info frame>'}")
        self.get_logger().info(f"Depth scale: {self._depth_scale}, max depth: {self._max_depth_m}, sync slop: {self._sync_slop}")
        self.get_logger().info(f"Candidates: {self._candidates_path} ({len(self._candidates)} loaded)")
        self.get_logger().info(f"Output dir: {self._output_dir}")
        self.get_logger().info(f"Voxel marker topics: {marker_prefix}/occupied_voxels, {marker_prefix}/unknown_voxels")
        self.get_logger().info(f"Next-best-view marker topic: {marker_prefix}/next_best_view")

    def _declare_parameters(self) -> None:
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("world_frame", "panda_link0")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("candidates_json", str(default_candidates_path()))
        self.declare_parameter("camera_config_file", "")
        self.declare_parameter("output_dir", "/workspace/ros2_ws/ours_experiment")
        self.declare_parameter("experiment_name", "ex2")

        self.declare_parameter("depth_scale", 4000.0)
        self.declare_parameter("sync_slop", 0.10)
        self.declare_parameter("capture_timeout_s", 10.0)
        self.declare_parameter("tf_timeout_s", 2.0)
        self.declare_parameter("settle_time_s", 0.8)
        self.declare_parameter("capture_buffer_s", 2.0)

        self.declare_parameter("top_k", 100)
        self.declare_parameter("max_steps", 15)
        self.declare_parameter("min_gain", 1e-6)
        self.declare_parameter("ig_with_raycast", False)
        self.declare_parameter("max_scoring_voxels", 1200)
        self.declare_parameter("unresolved_target_penalty", 0.3)

        self.declare_parameter("x_range", [0.20, 0.75])
        self.declare_parameter("y_range", [-0.35, 0.35])
        self.declare_parameter("z_range", [-0.05, 0.35])
        self.declare_parameter("voxel_size_m", 0.02)
        self.declare_parameter("z_plane_m", 0.0)
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 1.50)
        self.declare_parameter("pixel_stride", 8)
        self.declare_parameter("min_component_voxels", 5)

        self.declare_parameter("dry_run_motion", True)
        self.declare_parameter("keyboard_control", False)
        self.declare_parameter("move_timeout_s", 120.0)
        self.declare_parameter("pymoveit2_package", "pymoveit2")
        self.declare_parameter("joint_goal_executable", "panda_joint_goal.py")
        self.declare_parameter("write_final_ply", True)
        self.declare_parameter("include_unknown_in_ply", True)

        self.declare_parameter("publish_voxel_markers", True)
        self.declare_parameter("publish_occupied_voxels", True)
        self.declare_parameter("publish_unknown_voxels", True)
        self.declare_parameter("publish_free_voxels", True)
        self.declare_parameter("max_marker_voxels", 20000)
        self.declare_parameter("voxel_marker_stride", 1)
        self.declare_parameter("voxel_marker_topic_prefix", "/fetchbench_active_perception")
        self.declare_parameter("next_best_view_marker_size_m", 0.06)

    def _load_camera_config(self) -> dict:
        config_file = str(self.get_parameter("camera_config_file").value)
        if not config_file:
            return {}
        config_path = Path(config_file).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"camera_config_file is not a YAML mapping: {config_path}")
        self.get_logger().info(f"Camera config: {config_path}")
        return config

    @staticmethod
    def _get_camera_source_parameters(camera_config: dict) -> dict:
        for node_name in ("sim_rgbd_to_pointcloud", "rgbd_to_pointcloud"):
            params = camera_config.get(node_name, {}).get("ros__parameters")
            if isinstance(params, dict):
                return params
        return {}

    @staticmethod
    def _get_nested_config_parameter(config: dict, path: tuple[str, ...], name: str, fallback):
        node = config
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            return node.get(name, fallback)
        return fallback

    @staticmethod
    def _camera_info_from_config(camera_config: dict) -> CameraInfo | None:
        camera_info = camera_config.get("/camera/color/camera_info")
        if not isinstance(camera_info, dict):
            return None

        k = camera_info.get("k")
        if not isinstance(k, list) or len(k) != 9:
            raise ValueError("camera_config_file /camera/color/camera_info.k must contain 9 values")

        msg = CameraInfo()
        msg.header.frame_id = str(camera_info.get("header", {}).get("frame_id", ""))
        msg.width = int(camera_info["width"])
        msg.height = int(camera_info["height"])
        msg.k = [float(v) for v in k]
        return msg

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    def _rgbd_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._latest_info is None:
            self._rgbd_without_info_count += 1
            return
        self._synced_count += 1
        self._latest_bundle = CaptureBundle(
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            camera_info=self._latest_info,
            sync_index=self._synced_count,
        )
        if self._synced_count == 1:
            self.get_logger().info(
                f"Received first synchronized RGB-D pair: RGB={rgb_msg.width}x{rgb_msg.height}, "
                f"depth={depth_msg.width}x{depth_msg.height}"
            )

    def run(self) -> None:
        try:
            if bool(self.get_parameter("keyboard_control").value):
                self._run_keyboard_controlled()
            else:
                self._run_automatic()
        except KeyboardInterrupt:
            self._stop_reason = "keyboard_interrupt"
            raise
        finally:
            self._write_outputs()

    def _run_automatic(self) -> None:
        target_captures = self._target_capture_count()
        self._capture_with_buffer(tag="initial", candidate_idx=None, expected_gain=None, move_success=True)
        if len(self._view_records) >= target_captures:
            self._stop_reason = "max_steps_reached"
            return
        if bool(self.get_parameter("dry_run_motion").value):
            self.get_logger().warn("dry_run_motion=true: selecting one next view and stopping before robot motion")
            self._select_and_optionally_move(step_idx=1, dry_run_stop=True)
        else:
            remaining_captures = max(0, target_captures - len(self._view_records))
            for step_idx in range(1, remaining_captures + 1):
                keep_going = self._select_and_optionally_move(step_idx=step_idx, dry_run_stop=False)
                if not keep_going:
                    break
                if len(self._view_records) >= target_captures:
                    break
            if self._stop_reason == "not_started":
                self._stop_reason = "max_steps_reached"

    def _target_capture_count(self) -> int:
        return 1 + max(0, int(self.get_parameter("max_steps").value))

    def _stop_if_capture_limit_reached(self) -> bool:
        if len(self._view_records) < self._target_capture_count():
            return False
        if self._stop_reason == "not_started":
            self._stop_reason = "max_steps_reached"
        self._interactive_quit_requested = True
        self._pending_candidate_idx = None
        self._pending_candidate_meta = None
        self._pending_candidate_step = None
        self.get_logger().info(
            f"[stop] reached max_steps capture limit: {len(self._view_records)}/{self._target_capture_count()}"
        )
        return True

    def _run_keyboard_controlled(self) -> None:
        self._setup_keyboard()
        self._log_keyboard_help()
        try:
            while rclpy.ok() and not self._interactive_quit_requested:
                rclpy.spin_once(self, timeout_sec=0.05)
                self._poll_keyboard()
        finally:
            self._restore_keyboard()
        if self._stop_reason == "not_started":
            self._stop_reason = "keyboard_quit"

    def _wait_for_new_capture(self) -> CaptureBundle:
        start_count = self._synced_count
        deadline = time.monotonic() + self._capture_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_bundle is not None and self._latest_bundle.sync_index > start_count:
                return self._latest_bundle
        raise TimeoutError(
            "Timed out waiting for a new synchronized RGB-D capture and CameraInfo. "
            f"Subscribed RGB={self._rgb_topic}, depth={self._depth_topic}, "
            f"camera_info={self._camera_info_topic}; "
            f"camera_info_received={self._latest_info is not None}, "
            f"rgbd_pairs_with_info={self._synced_count}, "
            f"rgbd_pairs_without_info={self._rgbd_without_info_count}."
        )

    def _capture_with_buffer(
        self,
        tag: str,
        candidate_idx: int | None,
        expected_gain: dict | None,
        move_success: bool,
    ) -> None:
        buffer_s = max(0.0, float(self.get_parameter("capture_buffer_s").value))
        if buffer_s > 0.0:
            self.get_logger().info(f"Waiting {buffer_s:.1f}s before RGB-D capture")
            self._spin_wait(buffer_s)
        capture = self._wait_for_new_capture()
        self._capture_and_update(
            capture,
            tag=tag,
            candidate_idx=candidate_idx,
            expected_gain=expected_gain,
            move_success=move_success,
        )
        if buffer_s > 0.0:
            self.get_logger().info(f"Waiting {buffer_s:.1f}s after RGB-D capture")
            self._spin_wait(buffer_s)

    def _spin_wait(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _initialize_ap_state(self, camera_info: CameraInfo) -> None:
        if self._accumulator is not None:
            return
        self._intrinsics = intrinsics_matrix_from_camera_info(camera_info)
        self._accumulator = VoxelAccumulator(
            x_range=tuple(float(v) for v in self.get_parameter("x_range").value),
            y_range=tuple(float(v) for v in self.get_parameter("y_range").value),
            z_range=tuple(float(v) for v in self.get_parameter("z_range").value),
            voxel_size=float(self.get_parameter("voxel_size_m").value),
            z_plane=float(self.get_parameter("z_plane_m").value),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=self._max_depth_m,
            pixel_stride=int(self.get_parameter("pixel_stride").value),
            min_component_voxels=int(self.get_parameter("min_component_voxels").value),
        )
        self._scorer = UnknownVoxelScorer(
            min_threshold=float(self.get_parameter("voxel_size_m").value),
            max_threshold=float(self.get_parameter("voxel_size_m").value) * 10.0,
            unresolved_target_penalty=float(self.get_parameter("unresolved_target_penalty").value),
        )
        self._candidate_ranker = CandidateScoreManager(
            intrinsics=self._intrinsics,
            width=int(camera_info.width),
            height=int(camera_info.height),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=self._max_depth_m,
            ig_with_raycast=bool(self.get_parameter("ig_with_raycast").value),
            max_scoring_voxels=int(self.get_parameter("max_scoring_voxels").value),
        )
        self.get_logger().info(
            "Initialized AP occupancy grid: "
            f"shape={self._accumulator.grid.shape}, voxel={self._accumulator.voxel_size:.3f}m"
        )

    def _depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_raw)
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / float(self._depth_scale)
        return depth.astype(np.float32)

    def _lookup_camera_pose(self, camera_frame: str) -> np.ndarray:
        tf_msg = self._tf_buffer.lookup_transform(
            self._world_frame,
            camera_frame,
            Time(),
            timeout=Duration(seconds=float(self._tf_timeout_s)),
        )
        return transform_msg_to_matrix(tf_msg.transform)

    def _capture_and_update(
        self,
        capture: CaptureBundle,
        tag: str,
        candidate_idx: int | None,
        expected_gain: dict | None,
        move_success: bool,
    ) -> None:
        self._initialize_ap_state(capture.camera_info)
        assert self._accumulator is not None
        assert self._scorer is not None
        assert self._intrinsics is not None

        try:
            depth_raw = self._bridge.imgmsg_to_cv2(capture.depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            raise RuntimeError(f"cv_bridge depth conversion failed: {exc}") from exc
        try:
            rgb_np = self._bridge.imgmsg_to_cv2(capture.rgb_msg, desired_encoding="rgb8")
        except Exception as exc:
            raise RuntimeError(f"cv_bridge RGB conversion failed: {exc}") from exc
        depth_m = self._depth_to_meters(depth_raw)
        camera_frame = self._camera_frame_param or capture.camera_info.header.frame_id
        cam_to_world = self._lookup_camera_pose(camera_frame)
        depth_intrinsics = _scale_intrinsics_matrix(
            self._intrinsics,
            width=int(depth_m.shape[1]),
            height=int(depth_m.shape[0]),
            base_width=int(capture.camera_info.width),
            base_height=int(capture.camera_info.height),
        )

        counts = self._accumulator.update(depth_m, cam_to_world, depth_intrinsics)
        unknown_points, unknown_scores = self._scorer.compute(self._accumulator)
        view_idx = len(self._view_records)
        saved_view = self._save_observation(
            view_idx=view_idx,
            tag=tag,
            rgb_np=rgb_np,
            depth_m=depth_m,
            camera_info=capture.camera_info,
            camera_frame=camera_frame,
            cam_to_world=cam_to_world,
            candidate_idx=candidate_idx,
            move_success=move_success,
            sync_index=capture.sync_index,
        )
        candidate = self._candidates[candidate_idx] if candidate_idx is not None else None
        record = {
            "view_index": int(view_idx),
            "tag": tag,
            "candidate_index": int(candidate_idx) if candidate_idx is not None else None,
            "candidate_rank": int(candidate.get("rank", candidate_idx + 1)) if candidate is not None else None,
            "move_success": bool(move_success),
            "camera_frame": camera_frame,
            "camera_matrix": cam_to_world.tolist(),
            "expected_gain": expected_gain,
            "grid_counts": counts,
            "occupancy_update_stats": dict(self._accumulator.last_update_stats),
            "scored_unknown_voxels": int(unknown_points.shape[0]),
            "score_mean": float(np.mean(unknown_scores)) if unknown_scores.size else 0.0,
            "rgb_shape_hw": [int(capture.rgb_msg.height), int(capture.rgb_msg.width)],
            "depth_shape_hw": [int(depth_m.shape[0]), int(depth_m.shape[1])],
            "sync_index": int(capture.sync_index),
            "saved_view": saved_view,
        }
        self._view_records.append(record)
        self._publish_voxel_markers()
        self.get_logger().info(
            f"[{tag}] occ={counts['occupied']} free={counts['free']} unknown={counts['unknown']} "
            f"scored_unknown={unknown_points.shape[0]} depth_valid={self._accumulator.last_update_stats.get('valid_depth_samples', 0)}"
        )

    def _select_and_optionally_move(self, step_idx: int, dry_run_stop: bool) -> bool:
        if not self._select_next_candidate(step_idx):
            return False

        if bool(self.get_parameter("dry_run_motion").value):
            if dry_run_stop:
                self._stop_reason = "dry_run_complete"
            return not dry_run_stop

        move_success = self._move_pending_candidate(process_after_move=True)
        if not move_success:
            assert self._pending_candidate_idx is not None
            self.get_logger().warn(
                f"[step {step_idx:02d}] move failed for candidate {self._pending_candidate_idx}; continuing"
            )
            self._pending_candidate_idx = None
            self._pending_candidate_meta = None
            self._pending_candidate_step = None
            return True

        self._interactive_step_idx = max(self._interactive_step_idx, step_idx + 1)
        self._pending_candidate_idx = None
        self._pending_candidate_meta = None
        self._pending_candidate_step = None
        self._stop_reason = "max_steps_reached"
        return True

    def _select_next_candidate(self, step_idx: int) -> bool:
        assert self._accumulator is not None
        assert self._scorer is not None
        assert self._candidate_ranker is not None

        if self._pending_candidate_idx is not None:
            self.get_logger().warn(
                f"Candidate {self._pending_candidate_idx} is already selected. "
                "Press m to move it, or q to quit."
            )
            return True

        unknown_points, unknown_scores = self._scorer.compute(self._accumulator)
        if unknown_points.shape[0] == 0:
            self._stop_reason = f"no_scored_unknown_voxels_step_{step_idx}"
            self.get_logger().info("[stop] no unknown voxels with positive score remain")
            return False

        best_idx, best_meta = self._candidate_ranker.select_best(
            self._candidates,
            unknown_points,
            unknown_scores,
            self._used_indices,
            self._accumulator,
            require_joint_angles=True,
            joint_dof_count=7,
        )
        if best_idx is None or best_meta is None:
            self._stop_reason = f"no_unused_candidate_step_{step_idx}"
            self.get_logger().info("[stop] no unused candidate remains")
            return False

        gain = float(best_meta["weighted_sum"])
        if gain <= float(self.get_parameter("min_gain").value):
            self._stop_reason = f"gain_below_threshold_step_{step_idx}"
            self.get_logger().info(f"[stop] best gain {gain:.6f} <= min_gain")
            return False

        visible, cam_pos, cam_quat = self._candidate_ranker.candidate_visible_mask(
            self._candidates[best_idx],
            unknown_points.astype(np.float64),
            self._accumulator,
        )
        center_weights = (
            self._candidate_ranker.image_center_weights(unknown_points[visible], cam_pos, cam_quat)
            if cam_pos is not None and cam_quat is not None and np.any(visible)
            else np.zeros((0,), dtype=np.float32)
        )
        self._scorer.record_targeted_voxels(self._accumulator, unknown_points[visible], center_weights)
        self._used_indices.add(best_idx)

        candidate = self._candidates[best_idx]
        planned = {
            "step": int(step_idx),
            "candidate_index": int(best_idx),
            "candidate_rank": int(candidate.get("rank", best_idx + 1)),
            "expected_gain": best_meta,
            "joint_angles": [float(v) for v in candidate.get("joint_angles", [])],
            "dry_run": bool(self.get_parameter("dry_run_motion").value),
        }
        self._planned_moves.append(planned)
        self._pending_candidate_idx = best_idx
        self._pending_candidate_meta = best_meta
        self._pending_candidate_step = int(step_idx)
        self._publish_next_best_view_marker(candidate)
        self.get_logger().info(
            f"[select {step_idx:02d}] idx={best_idx} rank={planned['candidate_rank']} "
            f"gain={gain:.6f} visible_unknown={best_meta['visible_unknown_voxels']}"
        )
        return True

    def _move_pending_candidate(self, process_after_move: bool) -> bool:
        if self._pending_candidate_idx is None:
            self.get_logger().warn("No selected candidate. Press n first.")
            return False
        if bool(self.get_parameter("dry_run_motion").value):
            self.get_logger().warn("dry_run_motion=true: refusing to move. Set dry_run_motion:=false to enable motion.")
            return False

        candidate_idx = self._pending_candidate_idx
        step_idx = self._pending_candidate_step or self._interactive_step_idx
        expected_gain = self._pending_candidate_meta

        move_success = self._move_to_candidate(candidate_idx)
        if not move_success:
            return False

        time.sleep(float(self.get_parameter("settle_time_s").value))
        if not process_after_move:
            self._current_view_candidate_idx = candidate_idx
            self._current_view_expected_gain = expected_gain
            self.get_logger().info(
                f"Moved to candidate {candidate_idx} and settled. Press p to process the new RGB-D frame."
            )
            return True

        self._capture_with_buffer(
            tag=f"step_{step_idx:03d}_cand_{candidate_idx:03d}",
            candidate_idx=candidate_idx,
            expected_gain=expected_gain,
            move_success=move_success,
        )
        return True

    def _keyboard_process_capture(self) -> None:
        if self._stop_if_capture_limit_reached():
            return
        try:
            tag = "initial" if self._accumulator is None else f"manual_{len(self._view_records):03d}"
            self._capture_with_buffer(
                tag=tag,
                candidate_idx=self._current_view_candidate_idx,
                expected_gain=self._current_view_expected_gain,
                move_success=True,
            )
            self._stop_if_capture_limit_reached()
        except Exception as exc:
            self.get_logger().error(f"Failed to process RGB-D frame: {exc}")

    def _keyboard_select_next(self) -> None:
        if self._stop_if_capture_limit_reached():
            return
        if self._accumulator is None:
            self.get_logger().warn("No occupancy grid yet. Press p first.")
            return
        try:
            self._select_next_candidate(self._interactive_step_idx)
        except Exception as exc:
            self.get_logger().error(f"Failed to select next candidate: {exc}")

    def _keyboard_move_selected(self) -> None:
        if self._stop_if_capture_limit_reached():
            return
        if self._pending_candidate_idx is None:
            self.get_logger().warn("No selected candidate. Press n first.")
            return
        try:
            if self._move_pending_candidate(process_after_move=False):
                self._pending_candidate_idx = None
                self._pending_candidate_meta = None
                self._pending_candidate_step = None
                self._interactive_step_idx += 1
        except Exception as exc:
            self.get_logger().error(f"Failed to move selected candidate: {exc}")

    def _save_observation(
        self,
        view_idx: int,
        tag: str,
        rgb_np: np.ndarray,
        depth_m: np.ndarray,
        camera_info: CameraInfo,
        camera_frame: str,
        cam_to_world: np.ndarray,
        candidate_idx: int | None,
        move_success: bool,
        sync_index: int,
    ) -> dict:
        file_stem = f"{int(view_idx):04d}"
        rgb_path = self._rgb_dir / f"{file_stem}.png"
        depth_path = self._depth_dir / f"{file_stem}.png"
        depth_preview_path = self._depth_preview_dir / f"{file_stem}.png"

        rgb_arr = np.asarray(rgb_np)
        if rgb_arr.ndim == 2:
            rgb_arr = np.repeat(rgb_arr[:, :, None], 3, axis=2)
        if rgb_arr.shape[2] > 3:
            rgb_arr = rgb_arr[:, :, :3]
        rgb_arr = np.ascontiguousarray(np.clip(rgb_arr, 0, 255).astype(np.uint8))
        PILImage.fromarray(rgb_arr, mode="RGB").save(rgb_path)

        depth_mm = np.asarray(depth_m, dtype=np.float32) * 1000.0
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm_u16 = np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)
        PILImage.fromarray(depth_mm_u16, mode="I;16").save(depth_path)

        max_depth = max(float(self._max_depth_m), 1e-6)
        preview = np.clip(depth_m / max_depth, 0.0, 1.0)
        preview = np.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        preview_u8 = (preview * 255.0).astype(np.uint8)
        PILImage.fromarray(preview_u8, mode="L").save(depth_preview_path)

        intrinsics_doc = self._write_intrinsics_json(
            camera_info,
            rgb_shape_hw=rgb_arr.shape[:2],
            depth_shape_hw=depth_m.shape[:2],
        )
        pose_record = {
            "index": int(view_idx),
            "view_index": int(view_idx),
            "candidate_index": int(candidate_idx) if candidate_idx is not None else None,
            "tag": tag,
            "move_success": bool(move_success),
            "sync_index": int(sync_index),
            "world_frame": self._world_frame,
            "camera_frame": camera_frame,
            "cam_position": [float(v) for v in cam_to_world[:3, 3].tolist()],
            "cam_matrix": cam_to_world.tolist(),
            "intrinsics": intrinsics_doc,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "depth_preview_path": str(depth_preview_path),
        }
        self._pose_records.append(pose_record)
        self._reached_view_indices.add(int(view_idx))
        self._write_pose_json()

        saved = {
            "view_index": int(view_idx),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "depth_preview_path": str(depth_preview_path),
            "pose_json": str(self._views_dir / "pose.json"),
            "intrinsics_json": str(self._views_dir / "intrinsics.json"),
        }
        self.get_logger().info(f"Saved AP observation {file_stem}: {rgb_path}")
        return saved

    def _write_intrinsics_json(
        self,
        camera_info: CameraInfo,
        rgb_shape_hw: tuple[int, int] | None = None,
        depth_shape_hw: tuple[int, int] | None = None,
    ) -> dict:
        k = camera_info.k
        intrinsics_doc = {
            "width": int(camera_info.width),
            "height": int(camera_info.height),
            "fx": float(k[0]),
            "fy": float(k[4]),
            "cx": float(k[2]),
            "cy": float(k[5]),
            "k": [float(v) for v in k],
            "camera_frame": camera_info.header.frame_id,
        }
        if rgb_shape_hw is not None:
            intrinsics_doc["saved_rgb_width"] = int(rgb_shape_hw[1])
            intrinsics_doc["saved_rgb_height"] = int(rgb_shape_hw[0])
        if depth_shape_hw is not None:
            intrinsics_doc["saved_depth_width"] = int(depth_shape_hw[1])
            intrinsics_doc["saved_depth_height"] = int(depth_shape_hw[0])
        path = self._views_dir / "intrinsics.json"
        path.write_text(json.dumps(intrinsics_doc, indent=2), encoding="utf-8")
        return intrinsics_doc

    def _write_pose_json(self) -> None:
        pose_doc = {
            "format": "fetchbench_real_views_v1",
            "world_frame": self._world_frame,
            "reached_indices": sorted(int(v) for v in self._reached_view_indices),
            "poses": self._pose_records,
        }
        path = self._views_dir / "pose.json"
        path.write_text(json.dumps(pose_doc, indent=2), encoding="utf-8")

    def _log_keyboard_help(self) -> None:
        self.get_logger().info("Keyboard control enabled")
        self.get_logger().info("Press p to process latest RGB-D frame and publish voxel markers")
        self.get_logger().info("Press n to select the next active-perception view")
        self.get_logger().info("Press m to move to the selected view and settle")
        self.get_logger().info("Press v to republish voxel markers")
        self.get_logger().info("Press w to write PLY output")
        self.get_logger().info("Press q to quit")

    def _poll_keyboard(self) -> None:
        if not sys.stdin or not select.select([sys.stdin], [], [], 0.0)[0]:
            return

        key = sys.stdin.read(1)
        if key == "\n":
            return
        if key == "p":
            self._keyboard_process_capture()
        elif key == "n":
            self._keyboard_select_next()
        elif key == "m":
            self._keyboard_move_selected()
        elif key == "v":
            self._publish_voxel_markers(force=True)
        elif key == "w":
            self._write_outputs()
        elif key == "q":
            self.get_logger().info("Quit requested")
            self._interactive_quit_requested = True
        else:
            self.get_logger().warn(f"Unknown key '{key}'. Press q to quit.")

    def _setup_keyboard(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().warn("stdin is not a TTY; keyboard commands may require pressing Enter")
            return
        self._terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        atexit.register(self._restore_keyboard)

    def _restore_keyboard(self) -> None:
        if self._terminal_settings is None:
            return
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal_settings)
        self._terminal_settings = None

    def _publish_voxel_markers(self, force: bool = False) -> None:
        if self._accumulator is None:
            self.get_logger().warn("No occupancy grid to publish yet. Press p first.")
            return
        if not force and not bool(self.get_parameter("publish_voxel_markers").value):
            return

        self._publish_grid_state_marker(
            state_value=1,
            enabled=bool(self.get_parameter("publish_occupied_voxels").value),
            publisher=self._occupied_marker_pub,
            ns="ap_occupied_voxels",
            marker_id=1,
            color=ColorRGBA(r=1.0, g=0.45, b=0.05, a=0.9),
        )
        self._publish_grid_state_marker(
            state_value=0,
            enabled=bool(self.get_parameter("publish_unknown_voxels").value),
            publisher=self._unknown_marker_pub,
            ns="ap_unknown_voxels",
            marker_id=2,
            color=ColorRGBA(r=0.55, g=0.55, b=0.55, a=0.06),
        )
        self._publish_grid_state_marker(
            state_value=-1,
            enabled=bool(self.get_parameter("publish_free_voxels").value),
            publisher=self._free_marker_pub,
            ns="ap_free_voxels",
            marker_id=3,
            color=ColorRGBA(r=0.15, g=0.50, b=1.0, a=0.04),
        )

    def _publish_next_best_view_marker(self, candidate: dict) -> None:
        pos = candidate.get("cam_position")
        if not isinstance(pos, list) or len(pos) != 3:
            self._next_best_view_pub.publish(self._make_delete_marker("ap_next_best_view", 10))
            self.get_logger().warn("Selected candidate has no valid cam_position; cannot publish next-best-view marker")
            return

        marker = Marker()
        marker.header.frame_id = self._world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ap_next_best_view"
        marker.id = 10
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        marker.pose.orientation.w = 1.0
        size = float(self.get_parameter("next_best_view_marker_size_m").value)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self._next_best_view_pub.publish(marker)
        self.get_logger().info(
            f"Published next-best-view marker at x={float(pos[0]):.3f}, "
            f"y={float(pos[1]):.3f}, z={float(pos[2]):.3f}"
        )

    def _publish_grid_state_marker(
        self,
        state_value: int,
        enabled: bool,
        publisher,
        ns: str,
        marker_id: int,
        color: ColorRGBA,
    ) -> None:
        assert self._accumulator is not None
        if not enabled:
            publisher.publish(self._make_delete_marker(ns, marker_id))
            return

        stride = max(1, int(self.get_parameter("voxel_marker_stride").value))
        max_voxels = max(1, int(self.get_parameter("max_marker_voxels").value))
        points = self._accumulator.iter_world_points(state_value, stride=stride)
        if points.shape[0] > max_voxels:
            skip = int(np.ceil(points.shape[0] / max_voxels))
            points = points[::skip]

        if points.shape[0] == 0:
            publisher.publish(self._make_delete_marker(ns, marker_id))
            return

        marker = Marker()
        marker.header.frame_id = self._world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(self._accumulator.voxel_size) * 0.92
        marker.scale.y = float(self._accumulator.voxel_size) * 0.92
        marker.scale.z = float(self._accumulator.voxel_size) * 0.92
        marker.color = color
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        marker.colors = [color for _ in range(points.shape[0])]
        publisher.publish(marker)
        self.get_logger().info(f"Published {points.shape[0]} {ns} markers")

    def _make_delete_marker(self, ns: str, marker_id: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = int(marker_id)
        marker.action = Marker.DELETE
        return marker

    def _move_to_candidate(self, candidate_idx: int) -> bool:
        candidate = self._candidates[candidate_idx]
        joints = candidate.get("joint_angles", [])
        if len(joints) != 7:
            self.get_logger().error(f"Candidate {candidate_idx} has invalid joint_angles: {joints}")
            return False
        joints_str = "[" + ", ".join(f"{float(j):.10f}" for j in joints) + "]"
        cmd = [
            "ros2",
            "run",
            str(self.get_parameter("pymoveit2_package").value),
            str(self.get_parameter("joint_goal_executable").value),
            "--ros-args",
            "-p",
            f"joint_positions:={joints_str}",
        ]
        self.get_logger().info(f"Executing candidate {candidate_idx}: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=False,
                timeout=float(self.get_parameter("move_timeout_s").value),
            )
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"Move command timed out for candidate {candidate_idx}")
            return False
        return result.returncode == 0

    def _write_outputs(self) -> None:
        if bool(self.get_parameter("write_final_ply").value) and self._accumulator is not None:
            ply_path = self._output_dir / "occupancy_grid.ply"
            self._accumulator.write_ply(ply_path, include_unknown=bool(self.get_parameter("include_unknown_in_ply").value))
            self.get_logger().info(f"Saved final occupancy PLY: {ply_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActivePerceptionCoordinator()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
