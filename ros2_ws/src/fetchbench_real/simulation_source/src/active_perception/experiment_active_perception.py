#!/isaac-sim/kit/python/bin/python3
"""Isaac Sim active perception experiment.

This is an Isaac-only port of the active-perception loop in
``ours/active_perception/examples/sim_robot_arm_active_perception.py``.
It keeps the 3D occupancy representation, candidate scoring, and iterative
view selection, but removes pymoveit2, ROS, RViz, and planning-scene collision
box updates.

The robot can still appear to explore: candidate joint configurations are
applied directly to the Isaac articulation. For debugging the 3D representation
without moving the robot, use ``--camera-only`` to move a standalone camera prim
through the candidate camera poses.
"""

import argparse
import functools
import http.server
import json
import math
import re
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from isaacsim import SimulationApp

try:
	from numba import njit
except Exception:
	njit = None


SCENE_INFO_PATH = Path(__file__).with_name("scene_info.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
JSON_FILENAME = "isaac_objects_for_moveit.json"
OBJECT_ROOT = "/objects"

DEFAULT_ROBOT_CAMERA_PRIM_PATH = "/Franka/panda_hand/FrankaCamera"
DEFAULT_STANDALONE_CAMERA_PRIM_PATH = "/World/ActivePerceptionCamera"
DEFAULT_CANDIDATES_JSON = WORKSPACE_ROOT / "ours" / "active_perception" / "view_candidates" / "view_candidates.json"

DEFAULT_ROI_X_RANGE = (0.20, 0.75)
DEFAULT_ROI_Y_RANGE = (-0.35, 0.35)
DEFAULT_ROI_Z_RANGE = (-0.05, 0.35)
DEFAULT_VOXEL_SIZE_M = 0.02
DEFAULT_Z_PLANE_M = 0.0
DEFAULT_MIN_DEPTH_M = 0.10
DEFAULT_MAX_DEPTH_M = 1.50
DEFAULT_UNKNOWN_SCORE_MAX_DISTANCE_M = DEFAULT_VOXEL_SIZE_M * 10
DEFAULT_UNRESOLVED_TARGET_PENALTY = 0.3
DEFAULT_MAX_SCORING_VOXELS = 1200

DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
SETTLE_FRAMES = 8
JOINT_REACH_MAX_STEPS = 180
JOINT_REACH_TOL_RAD = 2e-3
CAMERA_ONLY_ABOVE_OFFSET_M = 0.25
FALLBACK_TOPDOWN_HEIGHT_M = 0.45
HOME_JOINTS = [0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816]
ABOVE_JOINTS = [0.0, -0.205, 0.0, -0.951, 0.0, 0.749, 0.783]


# ---------------------------------------------------------------------------
# Argument/bootstrap helpers
# ---------------------------------------------------------------------------


def _resolve_repo_path(path_str: str) -> Path:
	path = Path(path_str)
	if path.is_absolute():
		return path.resolve()
	return (REPO_ROOT / path).resolve()


def _load_json_loose(path: Path) -> dict[str, Any]:
	text = path.read_text(encoding="utf-8")
	text = re.sub(r",\s*([}\]])", r"\1", text)
	return json.loads(text)


def _load_scene_info(scene_key: str) -> dict[str, Any]:
	with SCENE_INFO_PATH.open(encoding="utf-8") as f:
		scene_info_all = json.load(f)
	if scene_key not in scene_info_all:
		available = ", ".join(sorted(scene_info_all.get("key", scene_info_all.get("keys", []))))
		raise KeyError(f"Unknown scene '{scene_key}'. Available scenes: {available}")
	return scene_info_all[scene_key]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run an Isaac-only active perception loop: direct robot/camera view changes, "
			"depth-to-voxel occupancy updates, and unknown-voxel view selection."
		)
	)
	parser.add_argument("--scene", type=str, default=None, help="Scene key from scene_info.json, e.g. 01.")
	parser.add_argument("--scene-num", type=str, default=None, help="Scene index in dataset root, e.g. 001.")
	parser.add_argument("--scene-json", type=str, default=None, help="Explicit path to isaac_objects_for_moveit.json.")
	parser.add_argument("--base-usd", type=str, default=None, help="Explicit base USD path.")
	parser.add_argument("--robot-prim-path", type=str, default="/Franka")
	parser.add_argument("--camera-prim-path", type=str, default=DEFAULT_ROBOT_CAMERA_PRIM_PATH)
	parser.add_argument("--candidates-json", type=str, default=str(DEFAULT_CANDIDATES_JSON))
	parser.add_argument("--top-k", type=int, default=100, help="Use at most first K view candidates.")
	parser.add_argument("--max-steps", type=int, default=15, help="Number of active view-selection iterations after the initial view.")
	parser.add_argument("--min-gain", type=float, default=1e-6, help="Stop if the best expected gain is at or below this value.")
	parser.add_argument("--initial-candidate", type=int, default=0, help="Fallback candidate index used only if HOME/ABOVE robot initialization is unavailable.")
	parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to <scene_dir>/active_perception.")
	parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
	parser.add_argument("--x-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
	parser.add_argument("--y-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
	parser.add_argument("--z-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
	parser.add_argument("--voxel-size", type=float, default=None)
	parser.add_argument("--z-plane", type=float, default=DEFAULT_Z_PLANE_M)
	parser.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH_M)
	parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH_M)
	parser.add_argument("--pixel-stride", type=int, default=8)
	parser.add_argument("--min-component-voxels", type=int, default=5)
	parser.add_argument(
		"--max-scoring-voxels",
		type=int,
		default=DEFAULT_MAX_SCORING_VOXELS,
		help="Maximum scored unknown voxels used when ranking NBV candidates.",
	)
	parser.add_argument(
		"--unresolved-target-penalty",
		type=float,
		default=DEFAULT_UNRESOLVED_TARGET_PENALTY,
		help="Multiplier applied to unknown voxels that were targeted by the previous NBV but remain unknown.",
	)
	parser.add_argument(
		"--ig-with-raycast",
		action="store_true",
		help="When scoring next-best-view candidates, count unknown voxels only if no occupied voxel blocks the camera ray.",
	)
	parser.add_argument("--live-viz", action="store_true", help="Serve a lightweight live browser dashboard with occupancy, score heatmaps, and camera previews.")
	parser.add_argument("--live-viz-port", type=int, default=8765, help="Port for --live-viz. Use 0 to ask the OS for a free port.")
	parser.add_argument("--live-viz-open", action="store_true", help="Open the live visualization URL in the default browser.")
	parser.add_argument("--live-viz-refresh-ms", type=int, default=2000, help="Browser polling interval for --live-viz.")
	parser.add_argument(
		"--min-output",
		action="store_true",
		help="Save only summary.json plus combined RGB/depth-preview contact sheets; skip depth, PLY, and bbox files.",
	)
	parser.add_argument("--camera-only", action="store_true", help="Debug mode: move a standalone camera through candidate camera poses instead of moving Franka joints.")
	parser.add_argument("--headless", action="store_true")
	return parser.parse_args()


ARGS = _parse_args()
scene_info = _load_scene_info(ARGS.scene) if ARGS.scene else None

if ARGS.scene_json:
	scene_json_path = _resolve_repo_path(ARGS.scene_json)
elif scene_info is not None and ARGS.scene_num is not None:
	try:
		scene_num = int(ARGS.scene_num)
	except ValueError as exc:
		raise ValueError(f"Invalid --scene-num value: {ARGS.scene_num}") from exc
	dataset_root = _resolve_repo_path(str(scene_info.get("dataset_root")))
	scene_json_path = dataset_root / f"scene_{scene_num:03d}" / JSON_FILENAME
else:
	scene_json_path = None

if scene_info is not None and ARGS.base_usd is None:
	ARGS.base_usd = str(_resolve_repo_path(str(scene_info.get("scene_usd"))))

if ARGS.camera_only and ARGS.camera_prim_path == DEFAULT_ROBOT_CAMERA_PRIM_PATH:
	ARGS.camera_prim_path = DEFAULT_STANDALONE_CAMERA_PRIM_PATH

simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.usd
import PIL.Image
import PIL.ImageDraw
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Usd, UsdGeom


# ---------------------------------------------------------------------------
# Math / USD / capture helpers
# ---------------------------------------------------------------------------


def _step(world: World, n: int) -> None:
	for _ in range(max(0, int(n))):
		world.step(render=True)


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray | list[float]) -> np.ndarray:
	x, y, z, w = [float(v) for v in q_xyzw]
	norm = math.sqrt(x * x + y * y + z * z + w * w)
	if norm <= 1e-12:
		return np.eye(3, dtype=np.float64)
	x, y, z, w = x / norm, y / norm, z / norm, w / norm
	xx, yy, zz = x * x, y * y, z * z
	xy, xz, yz = x * y, x * z, y * z
	wx, wy, wz = w * x, w * y, w * z
	return np.array(
		[
			[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
			[2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
			[2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
		],
		dtype=np.float64,
	)


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
	m00, m01, m02 = rot[0]
	m10, m11, m12 = rot[1]
	m20, m21, m22 = rot[2]
	tr = m00 + m11 + m22
	if tr > 0.0:
		s = math.sqrt(tr + 1.0) * 2.0
		w = 0.25 * s
		x = (m21 - m12) / s
		y = (m02 - m20) / s
		z = (m10 - m01) / s
	elif m00 > m11 and m00 > m22:
		s = math.sqrt(max(1.0 + m00 - m11 - m22, 1e-12)) * 2.0
		w = (m21 - m12) / s
		x = 0.25 * s
		y = (m01 + m10) / s
		z = (m02 + m20) / s
	elif m11 > m22:
		s = math.sqrt(max(1.0 + m11 - m00 - m22, 1e-12)) * 2.0
		w = (m02 - m20) / s
		x = (m01 + m10) / s
		y = 0.25 * s
		z = (m12 + m21) / s
	else:
		s = math.sqrt(max(1.0 + m22 - m00 - m11, 1e-12)) * 2.0
		w = (m10 - m01) / s
		x = (m02 + m20) / s
		y = (m12 + m21) / s
		z = 0.25 * s
	q = np.array([x, y, z, w], dtype=np.float64)
	q /= max(float(np.linalg.norm(q)), 1e-12)
	return q


def _pose_to_matrix(position_xyz: np.ndarray | list[float], quat_xyzw: np.ndarray | list[float]) -> np.ndarray:
	mat = np.eye(4, dtype=np.float64)
	mat[:3, :3] = _quat_xyzw_to_rotmat(quat_xyzw)
	mat[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
	return mat


def _matrix_to_pose(mat: np.ndarray) -> tuple[list[float], list[float]]:
	pos = [float(v) for v in mat[:3, 3]]
	quat = [float(v) for v in _rotmat_to_quat_xyzw(mat[:3, :3])]
	return pos, quat


def _xyzw_to_wxyz(quat_xyzw: np.ndarray | list[float]) -> np.ndarray:
	q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
	return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _wxyz_to_xyzw(quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
	q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
	return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _usd_to_cv_camera_quat_xyzw(usd_quat_xyzw: np.ndarray | list[float]) -> np.ndarray:
	usd_rot = _quat_xyzw_to_rotmat(usd_quat_xyzw)
	usd_from_cv = np.diag([1.0, -1.0, -1.0])
	return _rotmat_to_quat_xyzw(usd_rot @ usd_from_cv)


def _get_prim_world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
	prim = stage.GetPrimAtPath(prim_path)
	if not prim.IsValid():
		return None, None
	m = omni.usd.get_world_transform_matrix(prim)
	t = m.ExtractTranslation()
	q = m.ExtractRotation().GetQuat()
	im = q.GetImaginary()
	pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
	quat_xyzw = np.array([float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())], dtype=np.float64)
	return pos, quat_xyzw


def _get_camera_cv_world_pose(stage: Usd.Stage, camera_prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
	pos, usd_quat = _get_prim_world_pose(stage, camera_prim_path)
	if pos is None or usd_quat is None:
		return None, None
	return pos, _usd_to_cv_camera_quat_xyzw(usd_quat)


def _get_camera_cv_sensor_world_pose(camera: Camera) -> tuple[np.ndarray | None, np.ndarray | None]:
	"""Read an Isaac Camera pose in the same +Z-forward frame as candidates."""
	try:
		pos, quat_wxyz = camera.get_world_pose(camera_axes="ros")
	except Exception:
		return None, None
	return np.asarray(pos, dtype=np.float64), _wxyz_to_xyzw(quat_wxyz)


def _set_camera_sensor_world_pose(
	camera: Camera,
	position_xyz: np.ndarray,
	quat_xyzw: np.ndarray,
	camera_axes: str = "ros",
) -> bool:
	"""Set an Isaac Camera pose from either +Z-forward ROS/CV or raw USD axes."""
	try:
		camera.set_world_pose(
			position=np.asarray(position_xyz, dtype=np.float64),
			orientation=_xyzw_to_wxyz(quat_xyzw),
			camera_axes=camera_axes,
		)
		return True
	except Exception:
		return False


def _clear_children(stage: Usd.Stage, root_path: str) -> None:
	root = stage.GetPrimAtPath(root_path)
	if not root.IsValid():
		return
	for child in list(root.GetChildren()):
		stage.RemovePrim(child.GetPath())


def _ensure_xform(stage: Usd.Stage, path: str) -> None:
	prim = stage.GetPrimAtPath(path)
	if not prim.IsValid():
		UsdGeom.Xform.Define(stage, path)


def _reset_xform_to_origin(stage: Usd.Stage, path: str) -> None:
	prim = stage.GetPrimAtPath(path)
	if not prim.IsValid():
		return
	xform = UsdGeom.Xformable(prim)
	xform.ClearXformOpOrder()
	xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
	xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
	xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _ensure_camera_prim_like_source(stage: Usd.Stage, src_camera_path: str, dst_camera_path: str) -> None:
	src = stage.GetPrimAtPath(src_camera_path)
	if not src.IsValid() or src.GetTypeName() != "Camera":
		return
	dst = stage.GetPrimAtPath(dst_camera_path)
	if not dst.IsValid():
		UsdGeom.Camera.Define(stage, dst_camera_path)
		dst = stage.GetPrimAtPath(dst_camera_path)
	for attr_name in [
		"projection",
		"cameraProjectionType",
		"focalLength",
		"horizontalAperture",
		"verticalAperture",
		"horizontalApertureOffset",
		"verticalApertureOffset",
		"clippingRange",
		"clippingPlanes",
		"fStop",
		"focusDistance",
	]:
		src_attr = src.GetAttribute(attr_name)
		if not src_attr or not src_attr.IsValid():
			continue
		value = src_attr.Get()
		if value is None:
			continue
		dst_attr = dst.GetAttribute(attr_name)
		if not dst_attr or not dst_attr.IsValid():
			dst_attr = dst.CreateAttribute(attr_name, src_attr.GetTypeName())
		dst_attr.Set(value)


def _reload_objects_from_usda(stage: Usd.Stage, objects_usda: Path, object_root: str, all_objects_data: list[dict]) -> None:
	_clear_children(stage, object_root)
	_ensure_xform(stage, object_root)
	_reset_xform_to_origin(stage, object_root)
	obj_stage = Usd.Stage.Open(str(objects_usda))
	if obj_stage is None:
		raise RuntimeError(f"Failed to open objects USDA: {objects_usda}")
	src_layer = obj_stage.GetRootLayer()
	dst_layer = stage.GetRootLayer()
	for obj_data in all_objects_data:
		src_path = obj_data.get("prim_path")
		if src_path:
			Sdf.CopySpec(src_layer, src_path, dst_layer, src_path)
	del obj_stage
	for _ in range(30):
		simulation_app.update()


def _camera_intrinsics(camera: Camera, width: int, height: int) -> np.ndarray:
	try:
		k = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
		if k.shape == (3, 3) and np.all(np.isfinite(k)):
			return k
	except Exception:
		pass
	focal_length = float(camera.get_focal_length())
	horizontal_aperture = float(camera.get_horizontal_aperture())
	vertical_aperture = float(camera.get_vertical_aperture())
	fx = width * focal_length / horizontal_aperture if abs(horizontal_aperture) > 1e-12 else width
	fy = height * focal_length / vertical_aperture if abs(vertical_aperture) > 1e-12 else height
	return np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def _is_empty_frame(frame: Any) -> bool:
	if frame is None:
		return True
	arr = np.asarray(frame)
	return arr.size == 0


def _depth_to_2d(depth: np.ndarray | None, expected_hw: tuple[int, int]) -> np.ndarray | None:
	if depth is None:
		return None
	arr = np.asarray(depth)
	if arr.ndim == 3 and arr.shape[2] == 1:
		arr = arr[:, :, 0]
	elif arr.ndim == 1 and arr.size == expected_hw[0] * expected_hw[1]:
		arr = arr.reshape(expected_hw)
	if arr.ndim != 2:
		return None
	arr = np.asarray(arr, dtype=np.float32)
	return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _capture_rgbd_with_retry(camera: Camera, world: World, expected_hw: tuple[int, int], max_retry_frames: int = 24) -> tuple[np.ndarray | None, np.ndarray | None]:
	for _ in range(max_retry_frames + 1):
		rgba = camera.get_rgba()
		depth = _depth_to_2d(camera.get_depth(), expected_hw)
		if not _is_empty_frame(rgba) and not _is_empty_frame(depth):
			return np.asarray(rgba), depth
		world.step(render=True)
	return None, None


def _save_rgb(rgba: np.ndarray | None, path: Path) -> None:
	if rgba is None:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	arr = np.asarray(rgba)
	if arr.ndim == 3 and arr.shape[2] == 4:
		arr = arr[:, :, :3]
	if arr.ndim != 3 or arr.shape[2] != 3:
		return
	PIL.Image.fromarray(arr.astype(np.uint8)).save(str(path))


def _save_depth_mm(depth_m: np.ndarray | None, path: Path) -> None:
	if depth_m is None:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	arr_mm = np.clip(np.asarray(depth_m, dtype=np.float32) * 1000.0, 0.0, 65535.0).astype(np.uint16)
	PIL.Image.fromarray(arr_mm).save(str(path))


def _save_depth_preview(depth_m: np.ndarray | None, path: Path, max_depth_m: float) -> None:
	if depth_m is None:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	arr = np.asarray(depth_m, dtype=np.float32)
	valid = arr > 0.0
	gray = np.zeros(arr.shape, dtype=np.uint8)
	gray[valid] = np.clip(255.0 * (1.0 - arr[valid] / max(max_depth_m, 1e-6)), 0.0, 255.0).astype(np.uint8)
	PIL.Image.fromarray(gray).save(str(path))


# ---------------------------------------------------------------------------
# View candidates
# ---------------------------------------------------------------------------


def _candidate_pose(candidate: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
	pos = candidate.get("cam_position")
	quat = candidate.get("cam_quaternion_xyzw")
	if not isinstance(pos, list) or len(pos) != 3:
		return None, None
	if not isinstance(quat, list) or len(quat) != 4:
		return None, None
	return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)


def _candidate_look_at_target(candidate: dict[str, Any]) -> np.ndarray | None:
	target = candidate.get("look_at_target")
	if not isinstance(target, list) or len(target) != 3:
		return None
	return np.asarray(target, dtype=np.float64)


def _camera_forward_xyzw(cam_quat_xyzw: np.ndarray | list[float]) -> np.ndarray:
	forward = _quat_xyzw_to_rotmat(cam_quat_xyzw)[:, 2]
	return forward / max(float(np.linalg.norm(forward)), 1e-12)


def _camera_target_alignment(
	cam_position: np.ndarray,
	cam_quat_xyzw: np.ndarray,
	target_position: np.ndarray,
) -> float:
	to_target = np.asarray(target_position, dtype=np.float64) - np.asarray(cam_position, dtype=np.float64)
	to_target /= max(float(np.linalg.norm(to_target)), 1e-12)
	return float(np.clip(np.dot(_camera_forward_xyzw(cam_quat_xyzw), to_target), -1.0, 1.0))


def _camera_only_seed_pose(
	scene_dir: Path,
	stage: Usd.Stage,
	scene_data: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
	bev_pose_path = scene_dir / "bev" / "pose.json"
	if bev_pose_path.is_file():
		try:
			doc = _load_json_loose(bev_pose_path)
			poses = doc.get("poses", [])
			if isinstance(poses, list) and poses:
				pose = poses[0]
				pos = pose.get("cam_position")
				raw_usd_quat = pose.get("cam_quaternion_xyzw")
				if isinstance(pos, list) and len(pos) == 3 and isinstance(raw_usd_quat, list) and len(raw_usd_quat) == 4:
					return (
						np.asarray(pos, dtype=np.float64),
						np.asarray(raw_usd_quat, dtype=np.float64),
						"usd",
						str(bev_pose_path),
					)
		except Exception:
			pass

	target_prim_path = scene_data.get("target_object_prim_path")
	if isinstance(target_prim_path, str):
		target_pos, _ = _get_prim_world_pose(stage, target_prim_path)
		if target_pos is not None:
			cam_pos = target_pos.astype(np.float64, copy=True)
			cam_pos[2] += FALLBACK_TOPDOWN_HEIGHT_M
			# Raw USD yaw -90deg with -Z forward gives a top-down image aligned like the BEV captures.
			raw_usd_quat = np.array([0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float64)
			return cam_pos, raw_usd_quat, "usd", f"{target_prim_path}+topdown"
	return None, None, "ros", "unavailable"


def _load_candidates(path: Path, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	doc = _load_json_loose(path)
	raw_candidates = doc.get("candidates", [])
	if not isinstance(raw_candidates, list) or not raw_candidates:
		raise ValueError(f"No candidates found in {path}")
	candidates: list[dict[str, Any]] = []
	for cand in raw_candidates:
		pos, quat = _candidate_pose(cand)
		if pos is None or quat is None:
			continue
		cand = dict(cand)
		cand["cam_position"] = [float(v) for v in pos]
		cand["cam_quaternion_xyzw"] = [float(v) for v in quat]
		candidates.append(cand)
		if len(candidates) >= max(1, int(top_k)):
			break
	if not candidates:
		raise ValueError(f"No valid candidates after schema check in {path}")
	return candidates, doc


def _range_from_args_or_doc(arg_value: list[float] | None, doc: dict[str, Any], key: str, fallback: tuple[float, float]) -> tuple[float, float]:
	if arg_value is not None:
		return float(arg_value[0]), float(arg_value[1])
	roi = doc.get("roi", {}) if isinstance(doc.get("roi"), dict) else {}
	value = roi.get(key)
	if isinstance(value, list) and len(value) == 2:
		return float(value[0]), float(value[1])
	return fallback


def _voxel_size_from_args_or_doc(arg_value: float | None, doc: dict[str, Any]) -> float:
	if arg_value is not None:
		return float(arg_value)
	roi = doc.get("roi", {}) if isinstance(doc.get("roi"), dict) else {}
	value = roi.get("grid_resolution")
	if value is not None:
		return float(value)
	return DEFAULT_VOXEL_SIZE_M


# ---------------------------------------------------------------------------
# Voxel representation and scoring
# ---------------------------------------------------------------------------


def _raycast_unblocked_mask_python(
	grid: np.ndarray,
	x_min: float,
	y_min: float,
	z_min: float,
	voxel_size: float,
	camera: np.ndarray,
	points: np.ndarray,
	step_len: float,
	end_margin: float,
) -> np.ndarray:
	mask = np.ones((points.shape[0],), dtype=bool)
	nz, ny, nx = grid.shape
	vecs = points - camera.reshape(1, 3)
	dists = np.linalg.norm(vecs, axis=1)
	valid_dist = dists > 1e-9
	if not np.any(valid_dist):
		return mask
	directions = np.zeros_like(vecs)
	directions[valid_dist] = vecs[valid_dist] / dists[valid_dist, None]
	max_step_per_point = np.floor(np.maximum(dists - end_margin, 0.0) / step_len).astype(np.int32)
	max_steps = int(np.max(max_step_per_point)) if max_step_per_point.size else 0
	for step in range(1, max_steps + 1):
		active = mask & valid_dist & (step <= max_step_per_point)
		if not np.any(active):
			break
		active_indices = np.flatnonzero(active)
		samples = camera.reshape(1, 3) + directions[active] * float(step * step_len)
		ix = np.floor((samples[:, 0] - x_min) / voxel_size).astype(np.int32)
		iy = np.floor((samples[:, 1] - y_min) / voxel_size).astype(np.int32)
		iz = np.floor((samples[:, 2] - z_min) / voxel_size).astype(np.int32)
		inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
		if not np.any(inside):
			continue
		hit_local = np.zeros((active_indices.shape[0],), dtype=bool)
		hit_local[inside] = grid[iz[inside], iy[inside], ix[inside]] == 1
		if np.any(hit_local):
			mask[active_indices[hit_local]] = False
	return mask


def _raycast_unblocked_mask_numba_impl(
	grid,
	x_min,
	y_min,
	z_min,
	voxel_size,
	camera,
	points,
	step_len,
	end_margin,
):
	n = points.shape[0]
	mask = np.ones(n, dtype=np.bool_)
	nz = grid.shape[0]
	ny = grid.shape[1]
	nx = grid.shape[2]
	for point_idx in range(n):
		vx = points[point_idx, 0] - camera[0]
		vy = points[point_idx, 1] - camera[1]
		vz = points[point_idx, 2] - camera[2]
		dist = math.sqrt(vx * vx + vy * vy + vz * vz)
		if dist <= 1e-9:
			continue
		inv_dist = 1.0 / dist
		dx = vx * inv_dist
		dy = vy * inv_dist
		dz = vz * inv_dist
		max_steps = int(math.floor(max(dist - end_margin, 0.0) / step_len))
		for step in range(1, max_steps + 1):
			t = step * step_len
			x = camera[0] + dx * t
			y = camera[1] + dy * t
			z = camera[2] + dz * t
			ix = int(math.floor((x - x_min) / voxel_size))
			iy = int(math.floor((y - y_min) / voxel_size))
			iz = int(math.floor((z - z_min) / voxel_size))
			if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and grid[iz, iy, ix] == 1:
				mask[point_idx] = False
				break
	return mask


_raycast_unblocked_mask_numba = (
	njit(cache=True)(_raycast_unblocked_mask_numba_impl) if njit is not None else None
)


def _warmup_raycast_kernel() -> bool:
	if _raycast_unblocked_mask_numba is None:
		return False
	dummy_grid = np.zeros((1, 1, 1), dtype=np.int8)
	dummy_camera = np.zeros((3,), dtype=np.float64)
	dummy_points = np.zeros((1, 3), dtype=np.float64)
	_raycast_unblocked_mask_numba(dummy_grid, 0.0, 0.0, 0.0, 1.0, dummy_camera, dummy_points, 0.5, 0.5)
	return True


class VoxelAccumulator:
	"""Depth-integrated occupancy grid.

	grid state: -1 free, 0 unknown, 1 occupied.
	"""

	def __init__(
		self,
		x_range: tuple[float, float],
		y_range: tuple[float, float],
		z_range: tuple[float, float],
		voxel_size: float,
		z_plane: float,
		min_depth_m: float,
		max_depth_m: float,
		pixel_stride: int,
		min_component_voxels: int,
	):
		self.x_range = tuple(float(v) for v in x_range)
		self.y_range = tuple(float(v) for v in y_range)
		self.z_range = tuple(float(v) for v in z_range)
		self.voxel_size = float(voxel_size)
		self.min_depth_m = float(min_depth_m)
		self.max_depth_m = float(max_depth_m)
		self.pixel_stride = max(1, int(pixel_stride))
		self.min_component_voxels = max(1, int(min_component_voxels))

		self.nx = int(np.ceil((self.x_range[1] - self.x_range[0]) / self.voxel_size))
		self.ny = int(np.ceil((self.y_range[1] - self.y_range[0]) / self.voxel_size))
		self.nz = int(np.ceil((self.z_range[1] - self.z_range[0]) / self.voxel_size))
		self.grid = np.zeros((self.nz, self.ny, self.nx), dtype=np.int8)
		self.log_odds = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
		self.checked_region = np.zeros((self.nz, self.ny, self.nx), dtype=bool)

		self.l_hit = 2.0
		self.l_free = 0.40
		self.l_occ_thresh = 1.0
		self.l_clamp = 10.0
		self.update_count = 0
		self.last_camera_position: np.ndarray | None = None
		self.last_update_stats: dict[str, int] = {}
		self.ground_layer_index: int | None = None
		if self.z_range[0] <= z_plane < self.z_range[1]:
			idx = int(np.floor((float(z_plane) - self.z_range[0]) / self.voxel_size))
			if 0 <= idx < self.nz:
				self.ground_layer_index = idx
		self._apply_ground_plane()

	def _apply_ground_plane(self) -> None:
		if self.ground_layer_index is not None:
			self.grid[: self.ground_layer_index + 1, :, :] = 1

	def world_to_voxel(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
		ix = np.floor((pts[:, 0] - self.x_range[0]) / self.voxel_size).astype(np.int32)
		iy = np.floor((pts[:, 1] - self.y_range[0]) / self.voxel_size).astype(np.int32)
		iz = np.floor((pts[:, 2] - self.z_range[0]) / self.voxel_size).astype(np.int32)
		valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (iz >= 0) & (iz < self.nz)
		return np.stack((ix, iy, iz), axis=1), valid

	def voxel_center(self, ix: int, iy: int, iz: int) -> np.ndarray:
		return np.array(
			[
				self.x_range[0] + (float(ix) + 0.5) * self.voxel_size,
				self.y_range[0] + (float(iy) + 0.5) * self.voxel_size,
				self.z_range[0] + (float(iz) + 0.5) * self.voxel_size,
			],
			dtype=np.float32,
		)

	def iter_world_points(self, state_value: int, stride: int = 1) -> np.ndarray:
		indices = np.argwhere(self.grid == int(state_value))
		if self.ground_layer_index is not None:
			indices = indices[indices[:, 0] > self.ground_layer_index]
		if indices.size == 0:
			return np.empty((0, 3), dtype=np.float32)
		stride = max(1, int(stride))
		if stride > 1:
			indices = indices[(indices[:, 0] % stride == 0) & (indices[:, 1] % stride == 0) & (indices[:, 2] % stride == 0)]
		if indices.size == 0:
			return np.empty((0, 3), dtype=np.float32)
		pts = np.empty((indices.shape[0], 3), dtype=np.float32)
		pts[:, 0] = self.x_range[0] + (indices[:, 2] + 0.5) * self.voxel_size
		pts[:, 1] = self.y_range[0] + (indices[:, 1] + 0.5) * self.voxel_size
		pts[:, 2] = self.z_range[0] + (indices[:, 0] + 0.5) * self.voxel_size
		return pts

	def _mark_free_ray(self, camera_xyz: np.ndarray, endpoint_xyz: np.ndarray) -> None:
		vec = endpoint_xyz - camera_xyz
		dist = float(np.linalg.norm(vec))
		if dist < 1e-6:
			return
		steps = max(1, int(np.ceil(dist / (self.voxel_size * 0.75))))
		for step in range(steps):
			point = camera_xyz + (step / steps) * vec
			idxs, valid = self.world_to_voxel(point.reshape(1, 3))
			if not valid[0]:
				continue
			ix, iy, iz = [int(v) for v in idxs[0]]
			self.checked_region[iz, iy, ix] = True
			self.log_odds[iz, iy, ix] = max(-self.l_clamp, float(self.log_odds[iz, iy, ix]) - self.l_free)

	def raycast_unblocked_mask(self, camera_xyz: np.ndarray, target_points_xyz: np.ndarray) -> np.ndarray:
		"""Return True for targets whose ray is not blocked by occupied voxels."""
		points = np.ascontiguousarray(np.asarray(target_points_xyz, dtype=np.float64).reshape(-1, 3))
		if points.shape[0] == 0:
			return np.zeros((0,), dtype=bool)
		camera = np.ascontiguousarray(np.asarray(camera_xyz, dtype=np.float64).reshape(3))
		step_len = max(self.voxel_size * 0.5, 1e-4)
		end_margin = self.voxel_size * 0.5
		if _raycast_unblocked_mask_numba is not None:
			return np.asarray(
				_raycast_unblocked_mask_numba(
					self.grid,
					float(self.x_range[0]),
					float(self.y_range[0]),
					float(self.z_range[0]),
					float(self.voxel_size),
					camera,
					points,
					float(step_len),
					float(end_margin),
				),
				dtype=bool,
			)
		return _raycast_unblocked_mask_python(
			self.grid,
			float(self.x_range[0]),
			float(self.y_range[0]),
			float(self.z_range[0]),
			float(self.voxel_size),
			camera,
			points,
			float(step_len),
			float(end_margin),
		)

	def _clear_camera_sphere(self, camera_xyz: np.ndarray, radius_m: float = 0.05) -> None:
		idxs, valid = self.world_to_voxel(camera_xyz.reshape(1, 3))
		if not valid[0]:
			return
		cix, ciy, ciz = [int(v) for v in idxs[0]]
		r_vox = int(np.ceil(radius_m / self.voxel_size))
		for iz in range(max(0, ciz - r_vox), min(self.nz, ciz + r_vox + 1)):
			for iy in range(max(0, ciy - r_vox), min(self.ny, ciy + r_vox + 1)):
				for ix in range(max(0, cix - r_vox), min(self.nx, cix + r_vox + 1)):
					if (ix - cix) ** 2 + (iy - ciy) ** 2 + (iz - ciz) ** 2 <= r_vox ** 2:
						self.log_odds[iz, iy, ix] = min(float(self.log_odds[iz, iy, ix]), -self.l_free)
						self.grid[iz, iy, ix] = -1

	def _prune_occupied_components(self, occupied: np.ndarray) -> np.ndarray:
		try:
			from scipy.ndimage import label
		except Exception:
			return occupied
		struct = np.ones((3, 3, 3), dtype=bool)
		labeled, num_features = label(occupied, structure=struct)
		if num_features <= 0:
			return np.zeros_like(occupied, dtype=bool)
		counts = np.bincount(labeled.ravel())
		keep_labels = np.where(counts >= self.min_component_voxels)[0]
		keep_labels = keep_labels[keep_labels > 0]
		if keep_labels.size == 0:
			return np.zeros_like(occupied, dtype=bool)
		return np.isin(labeled, keep_labels)

	def _apply_log_odds_to_grid(self) -> None:
		occupied = self.log_odds >= self.l_occ_thresh
		occupied = self._prune_occupied_components(occupied)
		free = self.log_odds < 0.0
		self.grid[:] = 0
		self.grid[free] = -1
		self.grid[occupied] = 1
		self._apply_ground_plane()

	def update(self, depth_m: np.ndarray, cam_to_world: np.ndarray, intrinsics: np.ndarray) -> dict[str, int]:
		self.update_count += 1
		depth = np.asarray(depth_m, dtype=np.float32)
		if depth.ndim != 2 or depth.size == 0:
			self.last_update_stats = {"valid_depth_samples": 0, "inside_roi_samples": 0, "unique_endpoint_voxels": 0}
			return self.counts()

		height, width = depth.shape
		u_coords = np.arange(0, width, self.pixel_stride, dtype=np.int32)
		v_coords = np.arange(0, height, self.pixel_stride, dtype=np.int32)
		uu, vv = np.meshgrid(u_coords, v_coords)
		z = depth[vv, uu]
		valid = np.isfinite(z) & (z >= self.min_depth_m) & (z <= self.max_depth_m)
		valid_depth_samples = int(np.count_nonzero(valid))
		if not np.any(valid):
			self.last_camera_position = cam_to_world[:3, 3].astype(np.float32)
			self.last_update_stats = {"valid_depth_samples": 0, "inside_roi_samples": 0, "unique_endpoint_voxels": 0}
			return self.counts()

		fx, fy, cx, cy = float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
		x = (uu.astype(np.float32) - cx) * z / max(fx, 1e-6)
		y = (vv.astype(np.float32) - cy) * z / max(fy, 1e-6)
		points_cam = np.stack((x[valid], y[valid], z[valid], np.ones(np.count_nonzero(valid), dtype=np.float32)), axis=0)
		points_world = (cam_to_world @ points_cam).T[:, :3]

		idxs, inside = self.world_to_voxel(points_world)
		if not np.any(inside):
			self.last_camera_position = cam_to_world[:3, 3].astype(np.float32)
			self.last_update_stats = {
				"valid_depth_samples": valid_depth_samples,
				"inside_roi_samples": 0,
				"unique_endpoint_voxels": 0,
			}
			return self.counts()

		camera_xyz = cam_to_world[:3, 3].astype(np.float32)
		self.last_camera_position = camera_xyz
		endpoint_indices = np.unique(idxs[inside], axis=0)
		self.last_update_stats = {
			"valid_depth_samples": valid_depth_samples,
			"inside_roi_samples": int(np.count_nonzero(inside)),
			"unique_endpoint_voxels": int(endpoint_indices.shape[0]),
		}
		camera_idx_arr, camera_inside = self.world_to_voxel(camera_xyz.reshape(1, 3))
		camera_idx = tuple(int(v) for v in camera_idx_arr[0]) if camera_inside[0] else None

		endpoint_voxels: list[tuple[int, int, int]] = []
		for ix, iy, iz in endpoint_indices:
			ix, iy, iz = int(ix), int(iy), int(iz)
			if camera_idx is not None and (ix, iy, iz) == camera_idx:
				continue
			endpoint = self.voxel_center(ix, iy, iz)
			self._mark_free_ray(camera_xyz, endpoint)
			endpoint_voxels.append((ix, iy, iz))

		# Apply hits after all free-space carving so later rays do not erase endpoints.
		for ix, iy, iz in endpoint_voxels:
			self.checked_region[iz, iy, ix] = True
			self.log_odds[iz, iy, ix] = min(self.l_clamp, float(self.log_odds[iz, iy, ix]) + self.l_hit)

		self._clear_camera_sphere(camera_xyz)
		self._apply_log_odds_to_grid()
		self._clear_camera_sphere(camera_xyz)
		return self.counts()

	def counts(self) -> dict[str, int]:
		return {
			"free": int(np.count_nonzero(self.grid == -1)),
			"unknown": int(np.count_nonzero(self.grid == 0)),
			"occupied": int(np.count_nonzero(self.grid == 1)),
			"checked": int(np.count_nonzero(self.checked_region)),
			"total": int(self.grid.size),
		}

	def write_ply(self, output_path: Path, include_unknown: bool = False) -> None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		chunks = []
		colors = []
		states = []
		for state, color in [(-1, (70, 150, 255)), (1, (255, 160, 40))]:
			pts = self.iter_world_points(state, stride=1)
			if pts.shape[0] > 0:
				chunks.append(pts)
				colors.append(np.tile(np.asarray(color, dtype=np.uint8), (pts.shape[0], 1)))
				states.append(np.full((pts.shape[0],), state, dtype=np.int32))
		if include_unknown:
			pts = self.iter_world_points(0, stride=1)
			if pts.shape[0] > 0:
				chunks.append(pts)
				colors.append(np.tile(np.asarray((90, 90, 90), dtype=np.uint8), (pts.shape[0], 1)))
				states.append(np.zeros((pts.shape[0],), dtype=np.int32))
		if chunks:
			points = np.concatenate(chunks, axis=0)
			rgb = np.concatenate(colors, axis=0)
			state_values = np.concatenate(states, axis=0)
		else:
			points = np.empty((0, 3), dtype=np.float32)
			rgb = np.empty((0, 3), dtype=np.uint8)
			state_values = np.empty((0,), dtype=np.int32)
		with output_path.open("w", encoding="utf-8") as f:
			f.write("ply\nformat ascii 1.0\n")
			f.write(f"element vertex {points.shape[0]}\n")
			f.write("property float x\nproperty float y\nproperty float z\n")
			f.write("property int state\n")
			f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
			f.write("end_header\n")
			for p, c, s in zip(points, rgb, state_values):
				f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(s)} {int(c[0])} {int(c[1])} {int(c[2])}\n")


class UnknownVoxelScorer:
	def __init__(
		self,
		min_threshold: float = DEFAULT_VOXEL_SIZE_M,
		max_threshold: float = DEFAULT_UNKNOWN_SCORE_MAX_DISTANCE_M,
		unresolved_target_penalty: float = DEFAULT_UNRESOLVED_TARGET_PENALTY,
		interior_dilation_radius: int = 1,
	):
		self.min_threshold = float(min_threshold)
		self.max_threshold = float(max_threshold)
		self.unresolved_target_penalty = float(unresolved_target_penalty)
		self.interior_dilation_radius = max(0, int(interior_dilation_radius))
		self.penalty_multiplier: np.ndarray | None = None
		self.pending_targeted_mask: np.ndarray | None = None
		self.pending_targeted_weights: np.ndarray | None = None
		self.latest_points = np.empty((0, 3), dtype=np.float32)
		self.latest_scores = np.empty((0,), dtype=np.float32)

	def record_targeted_voxels(
		self,
		accumulator: VoxelAccumulator,
		visible_unknown_points: np.ndarray,
		target_weights: np.ndarray | None = None,
	) -> None:
		if self.penalty_multiplier is None or self.penalty_multiplier.shape != accumulator.grid.shape:
			self.penalty_multiplier = np.ones(accumulator.grid.shape, dtype=np.float32)
		if visible_unknown_points.shape[0] == 0:
			self.pending_targeted_mask = None
			self.pending_targeted_weights = None
			return
		idxs, valid = accumulator.world_to_voxel(visible_unknown_points)
		mask = np.zeros(accumulator.grid.shape, dtype=bool)
		weights_grid = np.zeros(accumulator.grid.shape, dtype=np.float32)
		idxs = idxs[valid]
		if target_weights is None:
			weights = np.ones((visible_unknown_points.shape[0],), dtype=np.float32)
		else:
			weights = np.asarray(target_weights, dtype=np.float32).reshape(-1)
			if weights.shape[0] != visible_unknown_points.shape[0]:
				weights = np.ones((visible_unknown_points.shape[0],), dtype=np.float32)
		weights = np.clip(weights[valid], 0.0, 1.0)
		for (ix, iy, iz), weight in zip(idxs, weights):
			ix, iy, iz = int(ix), int(iy), int(iz)
			mask[iz, iy, ix] = True
			weights_grid[iz, iy, ix] = max(float(weights_grid[iz, iy, ix]), float(weight))
		self.pending_targeted_mask = mask
		self.pending_targeted_weights = weights_grid

	def _apply_pending_penalty(self, accumulator: VoxelAccumulator) -> None:
		if self.pending_targeted_mask is None or self.penalty_multiplier is None:
			return
		if self.pending_targeted_weights is None:
			target_weights = self.pending_targeted_mask.astype(np.float32)
		else:
			target_weights = np.clip(self.pending_targeted_weights, 0.0, 1.0)
		still_unknown = (accumulator.grid == 0) & self.pending_targeted_mask
		if np.any(still_unknown):
			penalty = 1.0 - target_weights[still_unknown] * (1.0 - self.unresolved_target_penalty)
			self.penalty_multiplier[still_unknown] *= penalty
		resolved = self.pending_targeted_mask & (accumulator.grid != 0)
		if np.any(resolved):
			self.penalty_multiplier[resolved] = 1.0
		self.pending_targeted_mask = None
		self.pending_targeted_weights = None

	def _query_nearest_distance(self, occupied_points: np.ndarray, unknown_points: np.ndarray) -> np.ndarray:
		try:
			from scipy.spatial import cKDTree
			tree = cKDTree(occupied_points)
			dists, _ = tree.query(unknown_points, workers=-1)
			return np.asarray(dists, dtype=np.float32)
		except Exception:
			# Small fallback for environments without scipy.
			out = np.empty((unknown_points.shape[0],), dtype=np.float32)
			batch = 2048
			for start in range(0, unknown_points.shape[0], batch):
				pts = unknown_points[start : start + batch]
				d2 = np.sum((pts[:, None, :] - occupied_points[None, :, :]) ** 2, axis=2)
				out[start : start + batch] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
			return out

	def compute(self, accumulator: VoxelAccumulator) -> tuple[np.ndarray, np.ndarray]:
		self._apply_pending_penalty(accumulator)
		occupied_points = accumulator.iter_world_points(1, stride=1)
		unknown_points = accumulator.iter_world_points(0, stride=1)
		if occupied_points.shape[0] == 0 or unknown_points.shape[0] == 0:
			self.latest_points = np.empty((0, 3), dtype=np.float32)
			self.latest_scores = np.empty((0,), dtype=np.float32)
			return self.latest_points, self.latest_scores
		dists = self._query_nearest_distance(occupied_points, unknown_points)
		span = max(self.max_threshold - self.min_threshold, 1e-9)
		scores = np.clip(1.0 - (dists - self.min_threshold) / span, 0.0, 1.0).astype(np.float32)
		if self.penalty_multiplier is not None and self.penalty_multiplier.shape == accumulator.grid.shape:
			idxs, valid = accumulator.world_to_voxel(unknown_points)
			idxs = idxs[valid]
			if idxs.shape[0] == unknown_points.shape[0]:
				scores *= self.penalty_multiplier[idxs[:, 2], idxs[:, 1], idxs[:, 0]]
		nonzero = scores > 0.0
		self.latest_points = unknown_points[nonzero].astype(np.float32, copy=True)
		self.latest_scores = scores[nonzero].astype(np.float32, copy=True)
		return self.latest_points, self.latest_scores


class CandidateScoreManager:
	def __init__(
		self,
		intrinsics: np.ndarray,
		width: int,
		height: int,
		min_depth_m: float,
		max_depth_m: float,
		ig_with_raycast: bool = False,
		max_scoring_voxels: int = DEFAULT_MAX_SCORING_VOXELS,
	):
		self.fx = float(intrinsics[0, 0])
		self.fy = float(intrinsics[1, 1])
		self.cx = float(intrinsics[0, 2])
		self.cy = float(intrinsics[1, 2])
		self.width = int(width)
		self.height = int(height)
		self.min_depth_m = float(min_depth_m)
		self.max_depth_m = float(max_depth_m)
		self.ig_with_raycast = bool(ig_with_raycast)
		self.max_scoring_voxels = max(1, int(max_scoring_voxels))

	def project_points(self, points_world: np.ndarray, cam_position: np.ndarray, cam_quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		if points_world.shape[0] == 0:
			empty = np.zeros((0,), dtype=np.float64)
			return empty, empty, empty
		r_world_cam = _quat_xyzw_to_rotmat(cam_quat_xyzw)
		points_cam = (r_world_cam.T @ (points_world - cam_position.reshape(1, 3)).T).T
		z = points_cam[:, 2]
		u = self.fx * (points_cam[:, 0] / np.maximum(z, 1e-6)) + self.cx
		v = self.fy * (points_cam[:, 1] / np.maximum(z, 1e-6)) + self.cy
		return u, v, z

	def visible_mask(self, points_world: np.ndarray, cam_position: np.ndarray, cam_quat_xyzw: np.ndarray) -> np.ndarray:
		if points_world.shape[0] == 0:
			return np.zeros((0,), dtype=bool)
		u, v, z = self.project_points(points_world, cam_position, cam_quat_xyzw)
		valid_depth = (z >= self.min_depth_m) & (z <= self.max_depth_m)
		inside = (u >= 0.0) & (u < self.width) & (v >= 0.0) & (v < self.height)
		return valid_depth & inside

	def image_center_weights(self, points_world: np.ndarray, cam_position: np.ndarray, cam_quat_xyzw: np.ndarray) -> np.ndarray:
		if points_world.shape[0] == 0:
			return np.zeros((0,), dtype=np.float32)
		u, v, _ = self.project_points(points_world.astype(np.float64), cam_position, cam_quat_xyzw)
		center_u = 0.5 * float(max(self.width - 1, 1))
		center_v = 0.5 * float(max(self.height - 1, 1))
		max_radius = math.hypot(max(center_u, self.width - 1 - center_u), max(center_v, self.height - 1 - center_v))
		radius = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
		return np.clip(1.0 - radius / max(max_radius, 1e-6), 0.0, 1.0).astype(np.float32)

	def candidate_visible_mask(
		self,
		candidate: dict[str, Any],
		points_world: np.ndarray,
		accumulator: VoxelAccumulator | None = None,
	) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
		cam_pos, cam_quat = _candidate_pose(candidate)
		if cam_pos is None or cam_quat is None or points_world.shape[0] == 0:
			return np.zeros((points_world.shape[0],), dtype=bool), cam_pos, cam_quat
		visible = self.visible_mask(points_world.astype(np.float64), cam_pos, cam_quat)
		if self.ig_with_raycast and accumulator is not None and np.any(visible):
			raycast_visible = accumulator.raycast_unblocked_mask(cam_pos, points_world[visible])
			out = visible.copy()
			out[visible] = raycast_visible
			visible = out
		return visible, cam_pos, cam_quat

	def score_candidate(
		self,
		candidate: dict[str, Any],
		unknown_points: np.ndarray,
		unknown_scores: np.ndarray,
		accumulator: VoxelAccumulator | None = None,
	) -> tuple[float, int, float]:
		if unknown_points.shape[0] == 0:
			return 0.0, 0, 0.0
		visible, cam_pos, _ = self.candidate_visible_mask(candidate, unknown_points.astype(np.float64), accumulator)
		if cam_pos is None:
			return 0.0, 0, 0.0
		if not np.any(visible):
			return 0.0, 0, 0.0
		pts = unknown_points[visible]
		scores = unknown_scores[visible]
		dists = np.linalg.norm(pts - cam_pos.reshape(1, 3), axis=1)
		weights = 1.0 / (dists + 1e-3)
		return float(np.sum(scores * weights)), int(pts.shape[0]), float(np.mean(dists))

	def select_best(
		self,
		candidates: list[dict[str, Any]],
		unknown_points: np.ndarray,
		unknown_scores: np.ndarray,
		used_indices: set[int],
		accumulator: VoxelAccumulator | None = None,
		require_joint_angles: bool = False,
		joint_dof_count: int | None = None,
	) -> tuple[int | None, dict[str, Any] | None]:
		if unknown_points.shape[0] > self.max_scoring_voxels:
			keep = np.argpartition(unknown_scores, -self.max_scoring_voxels)[-self.max_scoring_voxels :]
			unknown_points = unknown_points[keep]
			unknown_scores = unknown_scores[keep]
		best_idx = None
		best_meta = None
		best_score = -1.0
		for idx, cand in enumerate(candidates):
			if idx in used_indices:
				continue
			if require_joint_angles and not _candidate_has_joint_angles(cand, joint_dof_count):
				continue
			score, visible_count, mean_dist = self.score_candidate(cand, unknown_points, unknown_scores, accumulator)
			if score <= best_score:
				continue
			best_score = score
			best_idx = idx
			best_meta = {
				"weighted_sum": float(score),
				"visible_unknown_voxels": int(visible_count),
				"mean_distance": float(mean_dist),
			}
		return best_idx, best_meta


def _estimate_occluded_components(accumulator: VoxelAccumulator, min_component_voxels: int, max_components: int = 8) -> dict[str, Any]:
	unknown = accumulator.grid == 0
	occupied = accumulator.grid == 1
	if accumulator.ground_layer_index is not None:
		unknown[: accumulator.ground_layer_index + 1, :, :] = False
	try:
		from scipy.ndimage import binary_dilation, label
		near_surface = unknown & binary_dilation(occupied, structure=np.ones((3, 3, 3), dtype=bool), iterations=2)
		labeled, num_features = label(near_surface, structure=np.ones((3, 3, 3), dtype=bool))
		components = []
		for comp_id in range(1, num_features + 1):
			idx = np.argwhere(labeled == comp_id)
			if idx.shape[0] < min_component_voxels:
				continue
			components.append(_component_from_indices(accumulator, idx))
	except Exception:
		idx = np.argwhere(unknown)
		components = [_component_from_indices(accumulator, idx)] if idx.shape[0] >= min_component_voxels else []
	components.sort(key=lambda item: item["num_unseen_voxels"], reverse=True)
	return {"components": components[:max(1, int(max_components))]}


def _component_from_indices(accumulator: VoxelAccumulator, indices_zyx: np.ndarray) -> dict[str, Any]:
	mins = np.min(indices_zyx, axis=0)
	maxs = np.max(indices_zyx, axis=0)
	z0, y0, x0 = [int(v) for v in mins]
	z1, y1, x1 = [int(v) for v in maxs]
	bbox_min = [
		accumulator.x_range[0] + x0 * accumulator.voxel_size,
		accumulator.y_range[0] + y0 * accumulator.voxel_size,
		accumulator.z_range[0] + z0 * accumulator.voxel_size,
	]
	bbox_max = [
		accumulator.x_range[0] + (x1 + 1) * accumulator.voxel_size,
		accumulator.y_range[0] + (y1 + 1) * accumulator.voxel_size,
		accumulator.z_range[0] + (z1 + 1) * accumulator.voxel_size,
	]
	size = [float(bbox_max[i] - bbox_min[i]) for i in range(3)]
	center = [float(0.5 * (bbox_min[i] + bbox_max[i])) for i in range(3)]
	return {
		"num_unseen_voxels": int(indices_zyx.shape[0]),
		"bbox_min": [float(v) for v in bbox_min],
		"bbox_max": [float(v) for v in bbox_max],
		"size_xyz": size,
		"center_xyz": center,
		"height": float(size[2]),
	}


# ---------------------------------------------------------------------------
# Robot/camera movement and active loop
# ---------------------------------------------------------------------------


def _candidate_has_joint_angles(candidate: dict[str, Any], joint_dof_count: int | None = None) -> bool:
	ja = np.asarray(candidate.get("joint_angles", []), dtype=np.float64).reshape(-1)
	if ja.size == 0:
		return False
	if joint_dof_count is None or joint_dof_count <= 0:
		return True
	return ja.shape[0] == int(joint_dof_count)


def _move_robot_to_candidate(
	world: World,
	robot: SingleArticulation,
	controller,
	arm_joint_ids: list[int],
	candidate: dict[str, Any],	
) -> tuple[bool, float]:
	return _move_robot_to_joint_angles(world, robot, controller, arm_joint_ids, candidate.get("joint_angles", []))


def _move_robot_to_joint_angles(
	world: World,
	robot: SingleArticulation,
	controller,
	arm_joint_ids: list[int],
	joint_angles: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[bool, float]:
	current_q = robot.get_joint_positions()
	if current_q is None:
		return False, float("inf")
	goal_q = np.asarray(current_q, dtype=np.float64).copy()
	ja = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
	if ja.shape[0] == len(arm_joint_ids):
		for local_i, joint_i in enumerate(arm_joint_ids):
			goal_q[joint_i] = ja[local_i]
	elif ja.shape[0] == goal_q.shape[0]:
		goal_q[:] = ja
	else:
		return False, float("inf")

	try:
		robot.set_joint_positions(goal_q)
		robot.set_joint_velocities(np.zeros_like(goal_q))
	except Exception:
		pass

	max_err = float("inf")
	for _ in range(JOINT_REACH_MAX_STEPS):
		controller.apply_action(ArticulationAction(joint_positions=goal_q))
		world.step(render=True)
		cur = robot.get_joint_positions()
		if cur is None:
			continue
		cur = np.asarray(cur, dtype=np.float64)
		err = np.abs(cur[arm_joint_ids] - goal_q[arm_joint_ids]) if arm_joint_ids else np.abs(cur - goal_q)
		max_err = float(np.max(err)) if err.size else 0.0
		if max_err <= JOINT_REACH_TOL_RAD:
			return True, max_err
	return False, max_err


def _move_camera_to_candidate(camera: Camera, candidate: dict[str, Any]) -> bool:
	cam_pos, cam_quat = _candidate_pose(candidate)
	if cam_pos is None or cam_quat is None:
		return False
	return _set_camera_sensor_world_pose(camera, cam_pos, cam_quat, camera_axes="ros")


def _safe_tag(tag: str) -> str:
	out = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tag).strip("_")
	return out or "step"


def _record_view_outputs(
	output_dir: Path,
	tag: str,
	rgba: np.ndarray | None,
	depth: np.ndarray | None,
	accumulator: VoxelAccumulator,
	components: dict[str, Any],
	max_depth_m: float,
) -> dict[str, str | None]:
	safe = _safe_tag(tag)
	rgb_path = output_dir / "rgb" / f"{safe}.png"
	depth_path = output_dir / "depth" / f"{safe}.png"
	depth_preview_path = output_dir / "depth_preview" / f"{safe}.png"
	ply_path = output_dir / "occupancy_ply" / f"{safe}.ply"
	bbox_path = output_dir / "occluded_bboxes" / f"{safe}.json"
	_save_rgb(rgba, rgb_path)
	_save_depth_mm(depth, depth_path)
	_save_depth_preview(depth, depth_preview_path, max_depth_m=max_depth_m)
	accumulator.write_ply(ply_path, include_unknown=False)
	bbox_path.parent.mkdir(parents=True, exist_ok=True)
	bbox_path.write_text(json.dumps(components, indent=2), encoding="utf-8")
	return {
		"rgb": str(rgb_path),
		"depth": str(depth_path),
		"depth_preview": str(depth_preview_path),
		"occupancy_ply": str(ply_path),
		"occluded_bboxes": str(bbox_path),
	}


def _min_output_file_placeholders() -> dict[str, None]:
	return {
		"rgb": None,
		"depth": None,
		"depth_preview": None,
		"occupancy_ply": None,
		"occluded_bboxes": None,
	}


def _rgb_preview_pil(rgba: np.ndarray | None) -> PIL.Image.Image | None:
	if rgba is None:
		return None
	arr = np.asarray(rgba)
	if arr.ndim == 3 and arr.shape[2] == 4:
		arr = arr[:, :, :3]
	if arr.ndim != 3 or arr.shape[2] != 3:
		return None
	return PIL.Image.fromarray(arr.astype(np.uint8))


def _depth_preview_pil(depth_m: np.ndarray | None, max_depth_m: float) -> PIL.Image.Image | None:
	if depth_m is None:
		return None
	depth = np.asarray(depth_m, dtype=np.float32)
	if depth.ndim != 2:
		return None
	valid = np.isfinite(depth) & (depth > 0.0)
	gray = np.zeros(depth.shape, dtype=np.uint8)
	gray[valid] = np.clip(255.0 * (1.0 - depth[valid] / max(max_depth_m, 1e-6)), 0.0, 255.0).astype(np.uint8)
	return PIL.Image.fromarray(gray).convert("RGB")


def _save_preview_contact_sheet(
	frames: list[tuple[str, PIL.Image.Image]],
	output_path: Path,
	tile_size: tuple[int, int] = (320, 180),
	cols: int = 4,
) -> str | None:
	if not frames:
		return None
	tw, th = tile_size
	cols = max(1, int(cols))
	rows = int(math.ceil(len(frames) / float(cols)))
	pad = 8
	label_h = 28
	canvas = PIL.Image.new(
		"RGB",
		(cols * (tw + pad) + pad, rows * (th + label_h + pad) + pad),
		color=(24, 24, 24),
	)
	draw = PIL.ImageDraw.Draw(canvas)
	for i, (label, img) in enumerate(frames):
		row = i // cols
		col = i % cols
		x0 = pad + col * (tw + pad)
		y0 = pad + row * (th + label_h + pad)
		thumb = img.convert("RGB").resize((tw, th), resample=_bilinear_resize_filter())
		canvas.paste(thumb, (x0, y0))
		draw.rectangle([x0, y0, x0 + tw - 1, y0 + th - 1], outline=(120, 120, 120), width=1)
		draw.text((x0 + 4, y0 + th + 5), label[:42], fill=(230, 230, 230))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	canvas.save(str(output_path), optimize=True)
	return str(output_path)


# ---------------------------------------------------------------------------
# Lightweight live visualization
# ---------------------------------------------------------------------------


class _QuietSimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
	def log_message(self, format: str, *args: Any) -> None:
		return


def _nearest_resize_filter() -> Any:
	resampling = getattr(PIL.Image, "Resampling", None)
	return resampling.NEAREST if resampling is not None else PIL.Image.NEAREST


def _bilinear_resize_filter() -> Any:
	resampling = getattr(PIL.Image, "Resampling", None)
	return resampling.BILINEAR if resampling is not None else PIL.Image.BILINEAR


def _write_live_viz_index(output_dir: Path, refresh_ms: int) -> Path:
	live_dir = output_dir / "live_viz"
	live_dir.mkdir(parents=True, exist_ok=True)
	index_path = live_dir / "index.html"
	index_path.write_text(
		f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Active Perception Live Viz</title>
  <style>
    :root {{ --bg:#10151e; --panel:#182233; --line:#2a3850; --text:#edf4ff; --muted:#8fa2bd; --hot:#ffb052; --cool:#49d5ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 20% 10%, #21354e 0, transparent 34rem), var(--bg); color:var(--text); font-family:ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif; }}
    header {{ padding:18px 24px 8px; display:flex; justify-content:space-between; gap:16px; align-items:flex-end; }}
    h1 {{ margin:0; font-size:24px; letter-spacing:-0.03em; }}
    .sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:14px 24px 24px; }}
    .card {{ background:rgba(24,34,51,.9); border:1px solid rgba(255,255,255,.08); border-radius:18px; overflow:hidden; box-shadow:0 16px 48px rgba(0,0,0,.22); }}
    .card h2 {{ margin:0; padding:12px 14px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; border-bottom:1px solid var(--line); color:#d7e4f6; }}
    .imgbox {{ padding:12px; }}
    img {{ width:100%; display:block; background:#080b10; border:1px solid rgba(255,255,255,.08); border-radius:12px; image-rendering:pixelated; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; padding:12px; }}
    .metric {{ background:rgba(8,13,20,.55); border:1px solid rgba(255,255,255,.06); border-radius:13px; padding:10px; }}
    .metric b {{ display:block; font-size:21px; font-variant-numeric:tabular-nums; }}
    .metric span {{ color:var(--muted); font-size:12px; }}
    .status {{ border:1px solid rgba(255,255,255,.1); border-radius:999px; padding:7px 10px; color:#c9d8ec; font-size:13px; font-variant-numeric:tabular-nums; }}
    .wide {{ grid-column:1 / -1; }}
    @media (max-width:960px) {{ .grid {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Active Perception Live Viz</h1>
      <div class="sub" id="subtitle">waiting for state...</div>
    </div>
    <div class="status" id="updated">idle</div>
  </header>
  <main class="grid">
    <section class="card"><h2>Occupancy Top-Down</h2><div class="imgbox"><img id="occ"></div></section>
    <section class="card"><h2>Unknown Score Top-Down</h2><div class="imgbox"><img id="score"></div></section>
    <section class="card"><h2>RGB Preview</h2><div class="imgbox"><img id="rgb"></div></section>
    <section class="card"><h2>Depth Preview</h2><div class="imgbox"><img id="depth"></div></section>
    <section class="card wide">
      <h2>Counts</h2>
      <div class="metrics">
        <div class="metric"><b id="occCount">-</b><span>occupied</span></div>
        <div class="metric"><b id="freeCount">-</b><span>free</span></div>
        <div class="metric"><b id="unkCount">-</b><span>unknown</span></div>
        <div class="metric"><b id="scoreCount">-</b><span>scored unknown</span></div>
      </div>
    </section>
  </main>
  <script>
    const refreshMs = {max(500, int(refresh_ms))};
    let lastVersion = null;
    const el = id => document.getElementById(id);
    function img(id, path, version) {{
      if (!path) return;
      el(id).src = path + "?v=" + version;
    }}
    async function tick() {{
      try {{
        const res = await fetch("state.json?t=" + Date.now(), {{cache:"no-store"}});
        if (!res.ok) throw new Error("state not ready");
        const s = await res.json();
        el("subtitle").textContent = `${{s.tag || "unknown"}} | selected=${{s.selected_candidate ?? "-"}} | raycast=${{s.ig_with_raycast}}`;
        el("updated").textContent = new Date((s.updated_at || 0) * 1000).toLocaleTimeString();
        el("occCount").textContent = s.counts?.occupied ?? "-";
        el("freeCount").textContent = s.counts?.free ?? "-";
        el("unkCount").textContent = s.counts?.unknown ?? "-";
        el("scoreCount").textContent = s.scored_unknown_voxels ?? "-";
        if (s.version !== lastVersion) {{
          lastVersion = s.version;
          img("occ", s.images?.occupancy, s.version);
          img("score", s.images?.score, s.version);
          img("rgb", s.images?.rgb_preview, s.version);
          img("depth", s.images?.depth_preview, s.version);
        }}
      }} catch (err) {{
        el("subtitle").textContent = "waiting for live_viz/state.json...";
      }}
    }}
    tick();
    setInterval(tick, refreshMs);
  </script>
</body>
</html>
""",
		encoding="utf-8",
	)
	return index_path


def _start_live_viz_server(output_dir: Path, port: int) -> tuple[http.server.ThreadingHTTPServer, str]:
	handler = functools.partial(_QuietSimpleHTTPRequestHandler, directory=str(output_dir))
	try:
		server = http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), handler)
	except OSError:
		server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	url = f"http://127.0.0.1:{server.server_address[1]}/live_viz/index.html"
	return server, url


def _draw_grid_cross(img: np.ndarray, ix: int, iy: int, color: tuple[int, int, int]) -> None:
	h, w = img.shape[:2]
	if not (0 <= ix < w and 0 <= iy < h):
		return
	for dx in range(-2, 3):
		x = ix + dx
		if 0 <= x < w:
			img[iy, x] = color
	for dy in range(-2, 3):
		y = iy + dy
		if 0 <= y < h:
			img[y, ix] = color


def _save_small_rgb(img: np.ndarray, path: Path, scale: int = 8) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	pil_img = PIL.Image.fromarray(np.asarray(img, dtype=np.uint8))
	if scale > 1:
		pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), resample=_nearest_resize_filter())
	pil_img.save(str(path), optimize=True)


def _resize_preview(pil_img: PIL.Image.Image, max_width: int = 480) -> PIL.Image.Image:
	if pil_img.width <= max_width:
		return pil_img
	height = max(1, int(round(pil_img.height * max_width / pil_img.width)))
	return pil_img.resize((max_width, height), resample=_bilinear_resize_filter())


def _save_live_camera_previews(
	live_dir: Path,
	rgba: np.ndarray | None,
	depth_m: np.ndarray | None,
	max_depth_m: float,
) -> dict[str, str | None]:
	live_dir.mkdir(parents=True, exist_ok=True)
	rgb_path = live_dir / "rgb_preview.jpg"
	depth_path = live_dir / "depth_preview.jpg"
	if rgba is not None:
		arr = np.asarray(rgba)
		if arr.ndim == 3 and arr.shape[2] == 4:
			arr = arr[:, :, :3]
		if arr.ndim == 3 and arr.shape[2] == 3:
			_resize_preview(PIL.Image.fromarray(arr.astype(np.uint8))).save(str(rgb_path), quality=72, optimize=True)
	if depth_m is not None:
		depth = np.asarray(depth_m, dtype=np.float32)
		valid = np.isfinite(depth) & (depth > 0.0)
		gray = np.zeros(depth.shape, dtype=np.uint8)
		gray[valid] = np.clip(255.0 * (1.0 - depth[valid] / max(max_depth_m, 1e-6)), 0.0, 255.0).astype(np.uint8)
		_resize_preview(PIL.Image.fromarray(gray)).save(str(depth_path), quality=72, optimize=True)
	return {
		"rgb_preview": "/live_viz/rgb_preview.jpg" if rgb_path.exists() else None,
		"depth_preview": "/live_viz/depth_preview.jpg" if depth_path.exists() else None,
	}


def _occupancy_topdown_rgb(accumulator: VoxelAccumulator) -> np.ndarray:
	start_z = (accumulator.ground_layer_index + 1) if accumulator.ground_layer_index is not None else 0
	grid = accumulator.grid[start_z:] if start_z < accumulator.grid.shape[0] else accumulator.grid
	occupied = np.any(grid == 1, axis=0)
	free = np.any(grid == -1, axis=0)
	unknown = np.any(grid == 0, axis=0)
	img = np.zeros((accumulator.ny, accumulator.nx, 3), dtype=np.uint8)
	img[:] = (18, 22, 30)
	img[unknown] = (70, 78, 90)
	img[free] = (66, 151, 255)
	img[occupied] = (255, 159, 56)
	if accumulator.last_camera_position is not None:
		idxs, valid = accumulator.world_to_voxel(np.asarray(accumulator.last_camera_position).reshape(1, 3))
		if valid[0]:
			ix, iy, _ = [int(v) for v in idxs[0]]
			_draw_grid_cross(img, ix, iy, (255, 255, 255))
	return np.flipud(img)


def _score_topdown_rgb(accumulator: VoxelAccumulator, unknown_points: np.ndarray, unknown_scores: np.ndarray) -> np.ndarray:
	score_grid = np.zeros((accumulator.ny, accumulator.nx), dtype=np.float32)
	if unknown_points.shape[0] > 0 and unknown_scores.shape[0] == unknown_points.shape[0]:
		idxs, valid = accumulator.world_to_voxel(unknown_points)
		for (ix, iy, _), score in zip(idxs[valid], unknown_scores[valid]):
			ix, iy = int(ix), int(iy)
			score_grid[iy, ix] = max(float(score_grid[iy, ix]), float(score))
	v = np.clip(score_grid, 0.0, 1.0)
	img = np.zeros((accumulator.ny, accumulator.nx, 3), dtype=np.uint8)
	img[..., 0] = np.clip(255.0 * v, 0, 255).astype(np.uint8)
	img[..., 1] = np.clip(210.0 * np.sqrt(v), 0, 255).astype(np.uint8)
	img[..., 2] = np.clip(255.0 * (1.0 - v) * (v > 0), 0, 255).astype(np.uint8)
	img[v <= 0.0] = (18, 22, 30)
	if accumulator.last_camera_position is not None:
		idxs, valid = accumulator.world_to_voxel(np.asarray(accumulator.last_camera_position).reshape(1, 3))
		if valid[0]:
			ix, iy, _ = [int(vv) for vv in idxs[0]]
			_draw_grid_cross(img, ix, iy, (255, 255, 255))
	return np.flipud(img)


def _write_live_viz_snapshot(
	output_dir: Path,
	version: int,
	tag: str,
	accumulator: VoxelAccumulator,
	unknown_points: np.ndarray,
	unknown_scores: np.ndarray,
	rgba: np.ndarray | None,
	depth: np.ndarray | None,
	selected_candidate: int | None = None,
) -> None:
	live_dir = output_dir / "live_viz"
	live_dir.mkdir(parents=True, exist_ok=True)
	occ_path = live_dir / "occupancy_topdown.png"
	score_path = live_dir / "score_topdown.png"
	_save_small_rgb(_occupancy_topdown_rgb(accumulator), occ_path, scale=8)
	_save_small_rgb(_score_topdown_rgb(accumulator, unknown_points, unknown_scores), score_path, scale=8)
	camera_images = _save_live_camera_previews(live_dir, rgba, depth, max_depth_m=float(ARGS.max_depth))
	state = {
		"version": int(version),
		"updated_at": time.time(),
		"tag": tag,
		"selected_candidate": int(selected_candidate) if selected_candidate is not None else None,
		"ig_with_raycast": bool(ARGS.ig_with_raycast),
		"counts": accumulator.counts(),
		"scored_unknown_voxels": int(unknown_points.shape[0]),
		"unknown_score_mean": float(np.mean(unknown_scores)) if unknown_scores.size else 0.0,
		"unknown_score_max": float(np.max(unknown_scores)) if unknown_scores.size else 0.0,
		"update_stats": dict(accumulator.last_update_stats),
		"images": {
			"occupancy": "/live_viz/occupancy_topdown.png",
			"score": "/live_viz/score_topdown.png",
			**camera_images,
		},
	}
	tmp_path = live_dir / "state.json.tmp"
	state_path = live_dir / "state.json"
	tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
	tmp_path.replace(state_path)


def main() -> None:
	global _raycast_unblocked_mask_numba
	if scene_json_path is None:
		raise RuntimeError("Provide --scene and --scene-num, or --scene-json.")
	if not scene_json_path.is_file():
		raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")
	if ARGS.base_usd is None:
		raise RuntimeError("Base USD not specified. Pass --scene or --base-usd.")

	scene_data = _load_json_loose(scene_json_path)
	object_root = scene_data.get("object_root", OBJECT_ROOT)
	all_objects_data = scene_data.get("objects", [])
	objects_usda = scene_json_path.with_name("isaac_objects.usda")
	if not objects_usda.is_file():
		raise FileNotFoundError(f"Objects USDA not found: {objects_usda}")

	candidates_json = _resolve_repo_path(ARGS.candidates_json)
	if not candidates_json.is_file():
		raise FileNotFoundError(f"Candidates JSON not found: {candidates_json}")
	candidates, candidates_doc = _load_candidates(candidates_json, ARGS.top_k)
	if not 0 <= int(ARGS.initial_candidate) < len(candidates):
		raise ValueError(f"--initial-candidate must be in [0, {len(candidates) - 1}]")

	scene_dir = scene_json_path.parent
	output_dir = _resolve_repo_path(ARGS.output_dir) if ARGS.output_dir else scene_dir / "active_perception"
	output_dir.mkdir(parents=True, exist_ok=True)
	live_enabled = bool(ARGS.live_viz)
	live_server = None
	if live_enabled:
		_write_live_viz_index(output_dir, int(ARGS.live_viz_refresh_ms))
		try:
			live_server, live_url = _start_live_viz_server(output_dir, int(ARGS.live_viz_port))
			print(f"[ActivePerception] Live viz: {live_url}", flush=True)
			if ARGS.live_viz_open:
				webbrowser.open(live_url)
		except Exception as exc:
			live_enabled = False
			print(f"[ActivePerception] Live viz disabled: {exc}", flush=True)

	x_range = _range_from_args_or_doc(ARGS.x_range, candidates_doc, "x_range", DEFAULT_ROI_X_RANGE)
	y_range = _range_from_args_or_doc(ARGS.y_range, candidates_doc, "y_range", DEFAULT_ROI_Y_RANGE)
	z_range = _range_from_args_or_doc(ARGS.z_range, candidates_doc, "z_range", DEFAULT_ROI_Z_RANGE)
	voxel_size = _voxel_size_from_args_or_doc(ARGS.voxel_size, candidates_doc)

	base_usd = Path(ARGS.base_usd).resolve()
	if not base_usd.is_file():
		raise FileNotFoundError(f"Base USD not found: {base_usd}")

	usd_context = omni.usd.get_context()
	if not usd_context.open_stage(str(base_usd)):
		raise RuntimeError(f"Failed to open stage: {base_usd}")
	while usd_context.get_stage_loading_status()[2] > 0:
		simulation_app.update()
	stage = usd_context.get_stage()
	if stage is None:
		raise RuntimeError("Failed to get USD stage")

	_reload_objects_from_usda(stage, objects_usda, object_root, all_objects_data)

	world = World(stage_units_in_meters=1.0)
	robot = None
	controller = None
	arm_joint_ids: list[int] = []
	robot_prim = stage.GetPrimAtPath(ARGS.robot_prim_path)
	if robot_prim.IsValid():
		robot = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))
	elif not ARGS.camera_only:
		raise RuntimeError(f"Robot prim not found: {ARGS.robot_prim_path}")

	if ARGS.camera_only and ARGS.camera_prim_path != DEFAULT_ROBOT_CAMERA_PRIM_PATH:
		_ensure_camera_prim_like_source(stage, DEFAULT_ROBOT_CAMERA_PRIM_PATH, ARGS.camera_prim_path)

	cam_w, cam_h = (ARGS.resolution if ARGS.resolution is not None else (DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT))
	camera = Camera(
		prim_path=ARGS.camera_prim_path,
		name="active_perception_camera",
		resolution=(int(cam_w), int(cam_h)),
		render_product_path=None,
	)
	world.scene.add(camera)

	world.reset()
	camera.initialize()
	camera.add_distance_to_image_plane_to_frame()
	world.play()
	_step(world, 10)

	if robot is not None:
		controller = robot.get_articulation_controller()
		arm_joint_ids = [i for i, name in enumerate(robot.dof_names) if "finger" not in name.lower() and "gripper" not in name.lower()]
		print(f"[ActivePerception] Arm DOF indices: {arm_joint_ids}", flush=True)

	intrinsics = _camera_intrinsics(camera, int(cam_w), int(cam_h))
	accumulator = VoxelAccumulator(
		x_range=x_range,
		y_range=y_range,
		z_range=z_range,
		voxel_size=voxel_size,
		z_plane=float(ARGS.z_plane),
		min_depth_m=float(ARGS.min_depth),
		max_depth_m=float(ARGS.max_depth),
		pixel_stride=int(ARGS.pixel_stride),
		min_component_voxels=int(ARGS.min_component_voxels),
	)
	scorer = UnknownVoxelScorer(
		min_threshold=voxel_size,
		max_threshold=DEFAULT_UNKNOWN_SCORE_MAX_DISTANCE_M,
		unresolved_target_penalty=float(ARGS.unresolved_target_penalty),
	)
	candidate_ranker = CandidateScoreManager(
		intrinsics=intrinsics,
		width=int(cam_w),
		height=int(cam_h),
		min_depth_m=float(ARGS.min_depth),
		max_depth_m=float(ARGS.max_depth),
		ig_with_raycast=bool(ARGS.ig_with_raycast),
		max_scoring_voxels=int(ARGS.max_scoring_voxels),
	)
	raycast_backend = "off"
	if ARGS.ig_with_raycast:
		raycast_backend = "numpy"
		if _raycast_unblocked_mask_numba is not None:
			try:
				_warmup_raycast_kernel()
				raycast_backend = "numba"
			except Exception as exc:
				print(f"[ActivePerception] Numba raycast warmup failed; using numpy fallback: {exc}", flush=True)
				_raycast_unblocked_mask_numba = None

	print(f"[ActivePerception] Loaded {len(candidates)} candidates from {candidates_json}", flush=True)
	print(f"[ActivePerception] Output: {output_dir}", flush=True)
	if ARGS.min_output:
		print(
			"[ActivePerception] Min output: saving summary.json and combined RGB/depth-preview sheets only",
			flush=True,
		)
	if ARGS.camera_only:
		print(
			f"[ActivePerception] Motion mode: camera-only debug; moving {ARGS.camera_prim_path} while robot joints stay fixed. "
			"Omit --camera-only to move Franka joints and capture from the wrist camera.",
			flush=True,
		)
	else:
		print(
			f"[ActivePerception] Motion mode: robot joints; moving {ARGS.robot_prim_path} candidate joints "
			f"and capturing from {ARGS.camera_prim_path}",
			flush=True,
		)
	print(
		f"[ActivePerception] Candidate IG raycast: {bool(ARGS.ig_with_raycast)} "
		f"backend={raycast_backend} max_scoring_voxels={candidate_ranker.max_scoring_voxels}",
		flush=True,
	)
	print(
		f"[ActivePerception] Unknown score min_dist={scorer.min_threshold:.4f}m "
		f"max_dist={scorer.max_threshold:.4f}m unresolved_penalty={scorer.unresolved_target_penalty:.3f} "
		"center_weighted=True",
		flush=True,
	)
	print(
		f"[ActivePerception] ROI x={x_range} y={y_range} z={z_range} voxel={voxel_size:.4f}m grid={accumulator.grid.shape}",
		flush=True,
	)

	used_indices: set[int] = set()
	view_records: list[dict[str, Any]] = []
	min_output_rgb_frames: list[tuple[str, PIL.Image.Image]] = []
	min_output_depth_frames: list[tuple[str, PIL.Image.Image]] = []
	live_version = 0
	live_selected_candidate: int | None = None

	def capture_current_view(
		tag: str,
		candidate: dict[str, Any] | None = None,
		candidate_idx: int | None = None,
		expected_gain: dict[str, Any] | None = None,
		max_err: float = 0.0,
	) -> bool:
		nonlocal live_version
		rgba, depth = _capture_rgbd_with_retry(camera, world, expected_hw=(int(cam_h), int(cam_w)))
		cam_pos, cam_quat = _get_camera_cv_sensor_world_pose(camera)
		if cam_pos is None or cam_quat is None:
			cam_pos, cam_quat = _get_camera_cv_world_pose(stage, ARGS.camera_prim_path)
		if (cam_pos is None or cam_quat is None) and candidate is not None:
			cand_pos, cand_quat = _candidate_pose(candidate)
			cam_pos = cand_pos
			cam_quat = cand_quat
		if cam_pos is None or cam_quat is None or depth is None:
			print(f"[{tag}] missing camera pose/depth; skipping occupancy update", flush=True)
			return False

		cam_to_world = _pose_to_matrix(cam_pos, cam_quat)
		counts = accumulator.update(depth, cam_to_world, intrinsics)
		unknown_points, unknown_scores = scorer.compute(accumulator)
		components = _estimate_occluded_components(accumulator, int(ARGS.min_component_voxels))
		if ARGS.min_output:
			label = f"{len(view_records):02d} {tag}"
			if candidate_idx is not None:
				label += f" idx={candidate_idx:04d}"
			rgb_img = _rgb_preview_pil(rgba)
			depth_img = _depth_preview_pil(depth, max_depth_m=float(ARGS.max_depth))
			if rgb_img is not None:
				min_output_rgb_frames.append((label, rgb_img))
			if depth_img is not None:
				min_output_depth_frames.append((label, depth_img))
			files = _min_output_file_placeholders()
		else:
			files = _record_view_outputs(output_dir, tag, rgba, depth, accumulator, components, max_depth_m=float(ARGS.max_depth))

		record = {
			"tag": tag,
			"candidate_index": int(candidate_idx) if candidate_idx is not None else None,
			"candidate_rank": (
				int(candidate.get("rank", candidate_idx + 1))
				if candidate is not None and candidate_idx is not None
				else None
			),
			"move_success": True,
			"max_joint_error_rad": float(max_err),
			"camera_position": [float(v) for v in cam_pos],
			"camera_quaternion_xyzw": [float(v) for v in cam_quat],
			"camera_forward_xyz": [float(v) for v in _camera_forward_xyzw(cam_quat)],
			"camera_matrix": cam_to_world.tolist(),
			"expected_gain": expected_gain,
			"grid_counts": counts,
			"occupancy_update_stats": dict(accumulator.last_update_stats),
			"scored_unknown_voxels": int(unknown_points.shape[0]),
			"score_mean": float(np.mean(unknown_scores)) if unknown_scores.size else 0.0,
			"occluded_component_count": int(len(components.get("components", []))),
			"files": files,
		}
		view_records.append(record)
		update_stats = accumulator.last_update_stats
		print(
			f"[{tag}] grid occ={counts['occupied']} free={counts['free']} unknown={counts['unknown']} "
			f"scored_unknown={unknown_points.shape[0]} components={record['occluded_component_count']} "
			f"depth_valid={update_stats.get('valid_depth_samples', 0)} "
			f"roi_samples={update_stats.get('inside_roi_samples', 0)} "
			f"roi_voxels={update_stats.get('unique_endpoint_voxels', 0)}",
			flush=True,
		)
		if live_enabled:
			live_version += 1
			_write_live_viz_snapshot(
				output_dir=output_dir,
				version=live_version,
				tag=tag,
				accumulator=accumulator,
				unknown_points=unknown_points,
				unknown_scores=unknown_scores,
				rgba=rgba,
				depth=depth,
				selected_candidate=live_selected_candidate,
			)
		return True

	def observe_candidate(candidate_idx: int, tag: str, expected_gain: dict[str, Any] | None = None) -> bool:
		candidate = candidates[candidate_idx]
		print(f"[{tag}] move/capture candidate_idx={candidate_idx}", flush=True)
		if ARGS.camera_only:
			moved = _move_camera_to_candidate(camera, candidate)
			max_err = 0.0
			_step(world, SETTLE_FRAMES)
		else:
			if robot is None or controller is None:
				moved = False
				max_err = float("inf")
			else:
				moved, max_err = _move_robot_to_candidate(world, robot, controller, arm_joint_ids, candidate)
				_step(world, SETTLE_FRAMES)
		if not moved:
			print(f"[{tag}] failed to reach candidate_idx={candidate_idx} max_joint_err={max_err:.6f}", flush=True)
			return False
		if ARGS.camera_only:
			actual_pos, actual_quat = _get_camera_cv_sensor_world_pose(camera)
			target = _candidate_look_at_target(candidate)
			if actual_pos is not None and actual_quat is not None and target is not None:
				align = _camera_target_alignment(actual_pos, actual_quat, target)
				print(f"[{tag}] camera +Z target alignment cos={align:.4f}", flush=True)
		return capture_current_view(
			tag=tag,
			candidate=candidate,
			candidate_idx=candidate_idx,
			expected_gain=expected_gain,
			max_err=max_err,
		)

	def observe_joint_view(joint_angles: list[float], tag: str) -> bool:
		print(f"[{tag}] move/capture preset joint view", flush=True)
		if robot is None or controller is None:
			print(f"[{tag}] robot articulation is unavailable", flush=True)
			return False
		moved, max_err = _move_robot_to_joint_angles(world, robot, controller, arm_joint_ids, joint_angles)
		_step(world, SETTLE_FRAMES)
		if not moved:
			print(f"[{tag}] failed to reach preset joint view max_joint_err={max_err:.6f}", flush=True)
			return False
		if ARGS.camera_only:
			robot_cam_pos, robot_cam_quat = _get_camera_cv_world_pose(stage, DEFAULT_ROBOT_CAMERA_PRIM_PATH)
			if robot_cam_pos is None or robot_cam_quat is None:
				print(f"[{tag}] robot wrist camera pose unavailable: {DEFAULT_ROBOT_CAMERA_PRIM_PATH}", flush=True)
				return False
			if not _set_camera_sensor_world_pose(camera, robot_cam_pos, robot_cam_quat, camera_axes="ros"):
				print(f"[{tag}] failed to copy robot wrist camera pose to {ARGS.camera_prim_path}", flush=True)
				return False
			_step(world, SETTLE_FRAMES)
		return capture_current_view(tag=tag, max_err=max_err)

	def observe_fixed_camera_view(cam_pos: np.ndarray, cam_quat: np.ndarray, tag: str, source: str, camera_axes: str = "ros") -> bool:
		print(f"[{tag}] move/capture fixed camera view source={source}", flush=True)
		if not _set_camera_sensor_world_pose(camera, cam_pos, cam_quat, camera_axes=camera_axes):
			print(f"[{tag}] failed to set fixed camera pose", flush=True)
			return False
		_step(world, SETTLE_FRAMES)
		return capture_current_view(tag=tag, max_err=0.0)

	if ARGS.camera_only:
		seed_pos, seed_quat, seed_axes, seed_source = _camera_only_seed_pose(scene_dir, stage, scene_data)
		if seed_pos is None or seed_quat is None:
			initial_idx = int(ARGS.initial_candidate)
			used_indices.add(initial_idx)
			print("[init] Camera-only seed pose unavailable; falling back to initial candidate view", flush=True)
			if not observe_candidate(initial_idx, "initial", expected_gain=None):
				raise RuntimeError("Initial active perception observation failed")
		else:
			if not observe_fixed_camera_view(seed_pos, seed_quat, "initial", seed_source, camera_axes=seed_axes):
				raise RuntimeError("Initial active perception observation failed at camera-only seed pose")
			above_pos = seed_pos.astype(np.float64, copy=True)
			above_pos[2] += CAMERA_ONLY_ABOVE_OFFSET_M
			if not observe_fixed_camera_view(
				above_pos,
				seed_quat,
				"above_view",
				f"{seed_source}+{CAMERA_ONLY_ABOVE_OFFSET_M:.2f}m_z",
				camera_axes=seed_axes,
			):
				raise RuntimeError("Initial active perception observation failed at camera-only above pose")
	elif robot is not None and controller is not None:
		if not observe_joint_view(HOME_JOINTS, "initial"):
			raise RuntimeError("Initial active perception observation failed at HOME_JOINTS")
		if not observe_joint_view(ABOVE_JOINTS, "above_view"):
			raise RuntimeError("Initial active perception observation failed at ABOVE_JOINTS")
	else:
		initial_idx = int(ARGS.initial_candidate)
		used_indices.add(initial_idx)
		print("[init] Robot unavailable; falling back to initial candidate view", flush=True)
		if not observe_candidate(initial_idx, "initial", expected_gain=None):
			raise RuntimeError("Initial active perception observation failed")

	for step_idx in range(1, max(0, int(ARGS.max_steps)) + 1):
		unknown_points, unknown_scores = scorer.compute(accumulator)
		if unknown_points.shape[0] == 0:
			print(f"[stop] no unknown voxels with positive score remain at step {step_idx}", flush=True)
			break
		best_idx, best_meta = candidate_ranker.select_best(
			candidates,
			unknown_points,
			unknown_scores,
			used_indices,
			accumulator,
			require_joint_angles=not ARGS.camera_only,
			joint_dof_count=len(arm_joint_ids),
		)
		if best_idx is None or best_meta is None:
			print(f"[stop] no unused candidate remains at step {step_idx}", flush=True)
			break
		gain = float(best_meta["weighted_sum"])
		print(
			f"[select {step_idx:02d}] idx={best_idx} gain={gain:.6f} "
			f"visible_unknown={best_meta['visible_unknown_voxels']} mean_dist={best_meta['mean_distance']:.3f}m",
			flush=True,
		)
		if gain <= float(ARGS.min_gain):
			print(f"[stop] best gain {gain:.6f} <= min_gain {float(ARGS.min_gain):.6f}", flush=True)
			break

		live_selected_candidate = int(best_idx)
		cam_pos, cam_quat = _candidate_pose(candidates[best_idx])
		visible, _, _ = (
			candidate_ranker.candidate_visible_mask(candidates[best_idx], unknown_points.astype(np.float64), accumulator)
			if cam_pos is not None and cam_quat is not None
			else (np.zeros((unknown_points.shape[0],), dtype=bool), None, None)
		)
		center_weights = (
			candidate_ranker.image_center_weights(unknown_points[visible], cam_pos, cam_quat)
			if cam_pos is not None and cam_quat is not None and np.any(visible)
			else np.zeros((0,), dtype=np.float32)
		)
		scorer.record_targeted_voxels(accumulator, unknown_points[visible], center_weights)
		used_indices.add(best_idx)
		if not observe_candidate(best_idx, f"step_{step_idx:03d}_cand_{best_idx:03d}", expected_gain=best_meta):
			continue

	final_counts = accumulator.counts()
	combined_preview_files: dict[str, str | None] = {}
	if ARGS.min_output:
		combined_preview_files = {
			"rgb_contact_sheet": _save_preview_contact_sheet(
				min_output_rgb_frames,
				output_dir / "previews" / "rgb_contact_sheet.png",
			),
			"depth_preview_contact_sheet": _save_preview_contact_sheet(
				min_output_depth_frames,
				output_dir / "previews" / "depth_preview_contact_sheet.png",
			),
		}
	summary = {
		"scene_json": str(scene_json_path),
		"base_usd": str(base_usd),
		"objects_usda": str(objects_usda),
		"min_output": bool(ARGS.min_output),
		"combined_preview_files": combined_preview_files,
		"camera_only": bool(ARGS.camera_only),
		"motion_mode": "camera_only" if ARGS.camera_only else "robot_joints",
		"ig_with_raycast": bool(ARGS.ig_with_raycast),
		"robot_prim_path": ARGS.robot_prim_path,
		"camera_prim_path": ARGS.camera_prim_path,
		"candidates_json": str(candidates_json),
		"num_candidates_loaded": len(candidates),
		"used_candidate_indices": sorted(int(v) for v in used_indices),
		"roi": {
			"x_range": list(x_range),
			"y_range": list(y_range),
			"z_range": list(z_range),
			"voxel_size_m": float(voxel_size),
			"grid_shape_zyx": list(accumulator.grid.shape),
		},
		"depth_limits_m": {"min": float(ARGS.min_depth), "max": float(ARGS.max_depth)},
		"unknown_scoring": {
			"min_distance_m": float(scorer.min_threshold),
			"max_distance_m": float(scorer.max_threshold),
			"unresolved_target_penalty": float(scorer.unresolved_target_penalty),
			"penalty_image_center_weighted": True,
			"max_scoring_voxels": int(candidate_ranker.max_scoring_voxels),
		},
		"intrinsics": {
			"width": int(cam_w),
			"height": int(cam_h),
			"fx": float(intrinsics[0, 0]),
			"fy": float(intrinsics[1, 1]),
			"cx": float(intrinsics[0, 2]),
			"cy": float(intrinsics[1, 2]),
			"K": intrinsics.tolist(),
		},
		"final_grid_counts": final_counts,
		"views": view_records,
	}
	summary_path = output_dir / "summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(f"[ActivePerception] Saved summary: {summary_path}", flush=True)
	if ARGS.min_output:
		print(f"[ActivePerception] Saved min-output previews: {combined_preview_files}", flush=True)
	else:
		accumulator.write_ply(output_dir / "occupancy_final.ply", include_unknown=True)
		print(f"[ActivePerception] Saved final occupancy: {output_dir / 'occupancy_final.ply'}", flush=True)
	if live_enabled and live_server is not None:
		print(f"[ActivePerception] Live viz files: {output_dir / 'live_viz'}", flush=True)
	simulation_app.close()


if __name__ == "__main__":
	main()
