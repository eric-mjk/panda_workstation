import math
import os

import cv2
import numpy as np
import open3d as o3d

try:
	from tqdm import tqdm
except Exception:
	def tqdm(it, **kwargs):
		return it

from data_utils import (
	build_image_path,
	generate_sphere_points,
	get_camera_intrinsics,
	read_class_map,
	read_depth_image,
	write_ply_xyzrgb,
	write_ply_xyzrgb_scalar,
)
from viz_utils import score_to_bgr


TSDF_VOXEL_SIZE = 0.01
TSDF_TRUNCATION = 0.04


def collect_target_object_points(
	sorted_idxs,
	pose_map,
	target_class_id,
	intrinsics_json_path,
	rgb_dir,
	depth_dir,
	class_dir,
	max_points,
	stride,
	seed=0,
):
	fx, fy, cx, cy, _, _ = get_camera_intrinsics(intrinsics_json_path)
	rng = np.random.default_rng(seed)
	all_points = []	
	all_colors = []

	for idx in sorted_idxs:
		depth = read_depth_image(build_image_path(depth_dir, idx))
		rgb = cv2.imread(build_image_path(rgb_dir, idx), cv2.IMREAD_COLOR)
		cls = read_class_map(build_image_path(class_dir, idx))
		if rgb is None:
			raise FileNotFoundError(f"Failed to read image: {build_image_path(rgb_dir, idx)}")

		ys, xs = np.where((depth > 0) & (cls == int(target_class_id)))
		if len(xs) == 0:
			continue
		if int(stride) > 1:
			keep = ((ys % int(stride)) == 0) & ((xs % int(stride)) == 0)
			ys = ys[keep]
			xs = xs[keep]
		if len(xs) == 0:
			continue
		if xs.size > int(max_points):
			choice = rng.choice(xs.size, size=int(max_points), replace=False)
			ys = ys[choice]
			xs = xs[choice]

		z = depth[ys, xs].astype(np.float32)
		x = (xs.astype(np.float32) - cx) * z / fx
		y = -(ys.astype(np.float32) - cy) * z / fy
		p_cam = np.stack([x, y, -z, np.ones_like(z)], axis=1)
		c2w = pose_map[idx]["c2w"]
		p_world = (c2w @ p_cam.T).T[:, :3]
		all_points.append(p_world)
		all_colors.append(rgb[ys, xs, :])

	if len(all_points) == 0:
		return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
	points = np.concatenate(all_points, axis=0).astype(np.float32)
	colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
	if points.shape[0] > int(max_points):
		choice = rng.choice(points.shape[0], size=int(max_points), replace=False)
		points = points[choice]
		colors = colors[choice]
	return points, colors


def compute_volume_bounds(points, center_world=None, margin_m=0.08):
	points = np.asarray(points, dtype=np.float32)
	if points.size == 0:
		raise ValueError("Cannot build TSDF volume from an empty point set.")
	mins = np.min(points, axis=0)
	maxs = np.max(points, axis=0)
	margin = float(margin_m)
	if center_world is not None:
		center = np.asarray(center_world, dtype=np.float32)
		mins = np.minimum(mins, center - margin)
		maxs = np.maximum(maxs, center + margin)
	else:
		mins = mins - margin
		maxs = maxs + margin
	return mins, maxs


def build_tsdf_volume(
	sorted_idxs,
	pose_map,
	intrinsics_json_path,
	rgb_dir,
	depth_dir,
	class_dir,
	volume_min,
	volume_max,
	voxel_size=TSDF_VOXEL_SIZE,
	truncation=TSDF_TRUNCATION,
	target_class_id=None,
	background_class_id=None,
	exclude_background=False,
	mask_dilate_px=0,
	integration_stride=4,
):
	fx, fy, cx, cy, _, _ = get_camera_intrinsics(intrinsics_json_path)
	_ = np.asarray(volume_min, dtype=np.float32)
	_ = np.asarray(volume_max, dtype=np.float32)

	volume = o3d.pipelines.integration.ScalableTSDFVolume(
		voxel_length=float(voxel_size),
		sdf_trunc=float(truncation),
		color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
	)

	for idx in tqdm(sorted_idxs, desc="TSDF fuse", leave=False):
		depth = read_depth_image(build_image_path(depth_dir, idx))
		cls = read_class_map(build_image_path(class_dir, idx))
		if target_class_id is not None:
			exclude_mask = cls == int(target_class_id)
			if exclude_background and (background_class_id is not None):
				exclude_mask = exclude_mask | (cls == int(background_class_id))
			if int(mask_dilate_px) > 0:
				k = int(mask_dilate_px) * 2 + 1
				kernel = np.ones((k, k), dtype=np.uint8)
				exclude_mask = cv2.dilate(exclude_mask.astype(np.uint8), kernel, iterations=1) > 0
			depth = depth.copy()
			depth[exclude_mask] = 0.0

		stride = max(1, int(integration_stride))
		depth = depth[::stride, ::stride]
		height, width = depth.shape[:2]
		if width <= 0 or height <= 0:
			continue

		this_fx = fx / float(stride)
		this_fy = fy / float(stride)
		this_cx = cx / float(stride)
		this_cy = cy / float(stride)

		depth_img = o3d.geometry.Image(depth.astype(np.float32))
		color_img = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
			color_img,
			depth_img,
			depth_scale=1.0,
			depth_trunc=100.0,
			convert_rgb_to_intensity=False,
		)
		intrinsic = o3d.camera.PinholeCameraIntrinsic(
			int(width),
			int(height),
			float(this_fx),
			float(this_fy),
			float(this_cx),
			float(this_cy),
		)
		# pose_map uses camera frame x-right, y-up, z-backward.
		# Open3D expects x-right, y-down, z-forward.
		# Convert camera frame by F=diag(1,-1,-1) before TSDF integration.
		frame_flip = np.eye(4, dtype=np.float64)
		frame_flip[1, 1] = -1.0
		frame_flip[2, 2] = -1.0
		extrinsic = frame_flip @ np.asarray(pose_map[idx]["w2c"], dtype=np.float64)
		volume.integrate(rgbd, intrinsic, extrinsic)

	mesh = volume.extract_triangle_mesh()
	if len(mesh.vertices) == 0:
		raise ValueError("Open3D TSDF produced an empty mesh.")
	tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
	scene = o3d.t.geometry.RaycastingScene()
	_ = scene.add_triangles(tmesh)

	return {
		"backend": "open3d",
		"mesh": mesh,
		"scene": scene,
		"voxel_size": float(voxel_size),
		"truncation": float(truncation),
	}


def query_tsdf_trilinear(tsdf_data, points):
	if tsdf_data.get("backend") != "open3d":
		raise ValueError("TSDF backend must be open3d.")
	scene = tsdf_data["scene"]
	points = np.asarray(points, dtype=np.float32)
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("points must have shape (N, 3)")
	tensor_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
	# Use unsigned distance to nearest reconstructed surface.
	# Signed distance is unstable on non-watertight/open meshes and produced monotonic artifacts.
	dist = scene.compute_distance(tensor_points).numpy()
	return dist.astype(np.float32)


def map_clearance_to_geometry_score(clearance_m, max_clearance=0.08):
	clearance_m = np.asarray(clearance_m, dtype=np.float32)
	return np.clip(1.0 - clearance_m / float(max_clearance), 0.0, 1.0)


def build_swept_points(center_world, direction_world, step_m=0.01, num_steps=15):
	center_world = np.asarray(center_world, dtype=np.float32)
	direction_world = np.asarray(direction_world, dtype=np.float32)
	direction_world = direction_world / max(np.linalg.norm(direction_world), 1e-12)
	offsets = np.arange(0, int(num_steps) + 1, dtype=np.float32) * float(step_m)
	return center_world[None, :] + offsets[:, None] * direction_world[None, :]


def compute_geometry_scores(
	candidate_dirs,
	tsdf_data,
	sweep_start_m=0.00,
	sweep_end_m=0.15,
	sweep_num_steps=16,
	coll_clearance_m=0.008,
	safe_clearance_m=0.04,
	quantile=0.10,
	adaptive_thresholds=True,
	adaptive_low_q=0.20,
	adaptive_high_q=0.80,
	target_points=None,
	center_world=None,
	**kwargs
):
	candidate_dirs = np.asarray(candidate_dirs, dtype=np.float32)
	if candidate_dirs.ndim != 2 or candidate_dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	# Accept multiple points for robust scoring
	# target_points: (M,3)
	if target_points is None:
		if center_world is None:
			raise ValueError("Must provide target_points or center_world")
		target_points = np.asarray(center_world, dtype=np.float32)[None, :]
	else:
		target_points = np.asarray(target_points, dtype=np.float32)
	if target_points.ndim == 1:
		target_points = target_points[None, :]
	M = target_points.shape[0]
	deltas = np.linspace(float(sweep_start_m), float(sweep_end_m), int(sweep_num_steps), dtype=np.float32)
	all_quantiles = []
	for direction in tqdm(candidate_dirs, desc="Geometry score", leave=False):
		direction = direction / max(np.linalg.norm(direction), 1e-12)
		# For each point, sweep along direction
		all_swept = []
		for pt in target_points:
			swept = pt[None, :] + deltas[:, None] * direction[None, :]
			all_swept.append(swept)
		all_swept = np.concatenate(all_swept, axis=0)  # (M*sweep_num_steps, 3)
		sdfs = query_tsdf_trilinear(tsdf_data, all_swept)
		all_quantiles.append(float(np.quantile(sdfs, float(quantile))))
	all_quantiles = np.asarray(all_quantiles, dtype=np.float32)
	if adaptive_thresholds:
		coll_thr = float(np.quantile(all_quantiles, float(adaptive_low_q)))
		safe_thr = float(np.quantile(all_quantiles, float(adaptive_high_q)))
		if safe_thr <= coll_thr + 1e-6:
			coll_thr = float(np.min(all_quantiles))
			safe_thr = float(np.max(all_quantiles))
	else:
		# For unsigned TSDF, set more appropriate thresholds
		coll_thr = float(coll_clearance_m)
		safe_thr = float(safe_clearance_m)
	if safe_thr <= coll_thr + 1e-6:
		scores = np.zeros_like(all_quantiles)
	else:
		scores = (all_quantiles - coll_thr) / (safe_thr - coll_thr + 1e-8)
		scores = np.clip(scores, 0.0, 1.0)
	return scores.astype(np.float32), all_quantiles, int(np.argmax(scores))


def probe_tsdf_along_directions(
	candidate_dirs,
	center_world,
	tsdf_data,
	sweep_start_m=0.00,
	sweep_end_m=0.15,
	sweep_num_steps=16,
	abs_quantile=0.10,
):
	candidate_dirs = np.asarray(candidate_dirs, dtype=np.float32)
	if candidate_dirs.ndim != 2 or candidate_dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	center_world = np.asarray(center_world, dtype=np.float32)
	deltas = np.linspace(float(sweep_start_m), float(sweep_end_m), int(sweep_num_steps), dtype=np.float32)

	signed_rows = []
	for direction in tqdm(candidate_dirs, desc="TSDF probe", leave=False):
		direction = direction / max(np.linalg.norm(direction), 1e-12)
		swept = center_world[None, :] + deltas[:, None] * direction[None, :]
		signed_rows.append(query_tsdf_trilinear(tsdf_data, swept))

	signed_sdf = np.asarray(signed_rows, dtype=np.float32)

	return {
		"signed_sdf": signed_sdf,
		"signed_mean": np.mean(signed_sdf, axis=1).astype(np.float32),
		"signed_min": np.min(signed_sdf, axis=1).astype(np.float32),
		"signed_max": np.max(signed_sdf, axis=1).astype(np.float32),
		"signed_q10": np.quantile(signed_sdf, float(abs_quantile), axis=1).astype(np.float32),
		"signed_flat": signed_sdf.reshape(-1).astype(np.float32),
	}


def fibonacci_upper_hemisphere(num_directions=1000, up_axis=2):
	n = int(num_directions)
	phi = (1.0 + math.sqrt(5.0)) * 0.5
	ga = 2.0 * math.pi * (1.0 - 1.0 / phi)
	points = []
	i = 0
	while len(points) < n:
		z = 1.0 - 2.0 * ((i + 0.5) / (2 * n))
		r = math.sqrt(max(0.0, 1.0 - z * z))
		theta = ga * i
		if z >= 0.0:
			points.append([r * math.cos(theta), r * math.sin(theta), z])
		i += 1
	pts = np.asarray(points[:n], dtype=np.float32)
	norms = np.linalg.norm(pts, axis=1, keepdims=True)
	pts = pts / (norms + 1e-12)
	up_axis = int(up_axis)
	if up_axis == 2:
		return pts
	if up_axis == 1:
		return np.stack([pts[:, 0], pts[:, 2], pts[:, 1]], axis=1).astype(np.float32)
	if up_axis == 0:
		return np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=1).astype(np.float32)
	raise ValueError("up_axis must be 0, 1, or 2")


def clock_bin_to_q_index(clock_bin):
	return int(clock_bin) % 12


def interpolate_clock_score(clock_scores, direction):
	clock_scores = np.asarray(clock_scores, dtype=np.float32)
	if clock_scores.shape[0] != 12:
		raise ValueError("clock_scores must have length 12")
	direction = np.asarray(direction, dtype=np.float32)
	angle = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
	# Continuous bin position (bin centers at 0°, 30°, 60°, ...)
	pos = angle / 30.0
	lo = int(math.floor(pos)) % 12
	hi = (lo + 1) % 12
	t = pos - math.floor(pos)  # fractional part in [0, 1)
	return float((1.0 - t) * clock_scores[lo] + t * clock_scores[hi])


def build_view_weights(candidate_dirs):
	candidate_dirs = np.asarray(candidate_dirs, dtype=np.float32)
	return np.ones(candidate_dirs.shape[0], dtype=np.float32)


def compute_distilled_direction_scores(candidate_dirs, gemini_scores, geometry_scores, alpha=0.5):
	candidate_dirs = np.asarray(candidate_dirs, dtype=np.float32)
	gemini_scores = np.asarray(gemini_scores, dtype=np.float32)
	geometry_scores = np.asarray(geometry_scores, dtype=np.float32)
	# If gemini_scores are clock-binned (12), project onto each candidate direction
	if gemini_scores.shape[0] == 12 and candidate_dirs.shape[0] != 12:
		gemini_scores = np.array(
			[interpolate_clock_score(gemini_scores, d[:2]) for d in candidate_dirs],
			dtype=np.float32,
		)
	if gemini_scores.shape != geometry_scores.shape:
		raise ValueError("gemini_scores and geometry_scores must have the same shape")
	weights = build_view_weights(candidate_dirs)
	final = alpha * gemini_scores + (1.0 - alpha) * geometry_scores
	final = final * weights
	final = (final - float(np.min(final))) / (float(np.max(final) - np.min(final)) + 1e-8)
	best_idx = int(np.argmax(final))
	return final.astype(np.float32), best_idx


def create_arrow_points(center, direction, length=0.12, head_length=0.03):
	center = np.asarray(center, dtype=np.float32)
	direction = np.asarray(direction, dtype=np.float32)
	direction = direction / max(np.linalg.norm(direction), 1e-12)
	end = center + direction * float(length)
	head = end - direction * float(head_length)
	return np.stack([center, head, end], axis=0)


def build_distill_plys(out_dir, directions, scores, center_world):
	os.makedirs(out_dir, exist_ok=True)
	for i, (direction, score) in enumerate(zip(directions, scores)):
		pts = create_arrow_points(center_world, direction)
		colors = np.tile(np.asarray(score_to_bgr(score), dtype=np.uint8)[None, :], (pts.shape[0], 1))
		write_ply_xyzrgb(os.path.join(out_dir, f"distill_dir_{i:02d}.ply"), pts, colors)


def build_scoring_plys(out_dir, directions, scores, center_world):
	os.makedirs(out_dir, exist_ok=True)
	for i, (direction, score) in enumerate(zip(directions, scores)):
		pts = create_arrow_points(center_world, direction)
		colors = np.tile(np.asarray(score_to_bgr(score), dtype=np.uint8)[None, :], (pts.shape[0], 1))
		write_ply_xyzrgb(os.path.join(out_dir, f"score_dir_{i:02d}.ply"), pts, colors)


def create_arrow_point_cloud(center, direction, length=0.18, head_length=0.05, shaft_points=18, head_points=6):
	center = np.asarray(center, dtype=np.float32)
	direction = np.asarray(direction, dtype=np.float32)
	direction = direction / max(np.linalg.norm(direction), 1e-12)
	length = float(length)
	head_length = min(float(head_length), max(length * 0.25, 1e-6))
	head_base = center + direction * (length - head_length)
	tip = center + direction * length

	shaft_t = np.linspace(0.0, max(length - head_length, 0.0), max(2, int(shaft_points)), dtype=np.float32)
	shaft_pts = center[None, :] + shaft_t[:, None] * direction[None, :]

	base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	if abs(float(np.dot(base, direction))) > 0.85:
		base = np.array([0.0, 1.0, 0.0], dtype=np.float32)
	ortho1 = np.cross(direction, base)
	ortho1 = ortho1 / max(np.linalg.norm(ortho1), 1e-12)
	ortho2 = np.cross(direction, ortho1)
	ortho2 = ortho2 / max(np.linalg.norm(ortho2), 1e-12)

	head_t = np.linspace(0.0, 1.0, max(2, int(head_points)), dtype=np.float32)
	head_left = head_base[None, :] + head_t[:, None] * (tip - head_base)[None, :] * 0.5 + (1.0 - head_t)[:, None] * (head_length * 0.35) * ortho1[None, :]
	head_right = head_base[None, :] + head_t[:, None] * (tip - head_base)[None, :] * 0.5 - (1.0 - head_t)[:, None] * (head_length * 0.35) * ortho2[None, :]

	arrow_points = np.concatenate([shaft_pts, head_left, head_right, tip[None, :]], axis=0).astype(np.float32)
	return arrow_points


def build_scene_visualization_plys(
	out_dir,
	prefix,
	local_points,
	local_colors,
	directions,
	scores,
	center_world,
	best_idx,
	sphere_radius=0.15,
	arrow_length=0.18,
	gray_value=160,
):
	os.makedirs(out_dir, exist_ok=True)
	directions = np.asarray(directions, dtype=np.float32)
	scores = np.asarray(scores, dtype=np.float32)
	local_points = np.asarray(local_points, dtype=np.float32)
	local_colors = np.asarray(local_colors, dtype=np.uint8)
	center_world = np.asarray(center_world, dtype=np.float32)

	if local_points.shape[0] != local_colors.shape[0]:
		raise ValueError("local_points and local_colors must have the same length")
	if directions.ndim != 2 or directions.shape[1] != 3:
		raise ValueError("directions must have shape (N, 3)")
	if scores.ndim != 1 or scores.shape[0] != directions.shape[0]:
		raise ValueError("scores must have shape (N,)")

	local_gray = np.full_like(local_colors, int(gray_value), dtype=np.uint8)
	sphere_points = center_world[None, :] + directions * float(sphere_radius)
	sphere_colors = np.tile(np.asarray([score_to_bgr(s) for s in scores], dtype=np.uint8), (1, 1))

	if 0 <= int(best_idx) < directions.shape[0]:
		arrow_points = create_arrow_point_cloud(center_world, directions[int(best_idx)], length=arrow_length)
		arrow_colors = np.tile(np.asarray([255, 255, 255], dtype=np.uint8)[None, :], (arrow_points.shape[0], 1))
	else:
		arrow_points = np.zeros((0, 3), dtype=np.float32)
		arrow_colors = np.zeros((0, 3), dtype=np.uint8)

	variants = [
		("rgb", local_colors),
		("gray", local_gray),
	]
	paths = {}
	for variant_name, variant_local_colors in variants:
		points = np.concatenate([local_points, sphere_points, arrow_points], axis=0)
		colors = np.concatenate([variant_local_colors, sphere_colors, arrow_colors], axis=0)
		path = os.path.join(out_dir, f"{prefix}_{variant_name}.ply")
		write_ply_xyzrgb(path, points, colors)
		paths[variant_name] = path

	return paths


def select_nearest_latitude_indices(candidate_dirs, target_latitudes_deg, up_axis=2):
	candidate_dirs = np.asarray(candidate_dirs, dtype=np.float32)
	if candidate_dirs.ndim != 2 or candidate_dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	up_axis = int(up_axis)
	if up_axis == 2:
		lat_component = candidate_dirs[:, 2]
	elif up_axis == 1:
		lat_component = candidate_dirs[:, 1]
	elif up_axis == 0:
		lat_component = candidate_dirs[:, 0]
	else:
		raise ValueError("up_axis must be 0, 1, or 2")
	lats = np.degrees(np.arcsin(np.clip(lat_component, -1.0, 1.0)))
	selected = []
	for target_lat in target_latitudes_deg:
		idx = int(np.argmin(np.abs(lats - float(target_lat))))
		selected.append((float(target_lat), idx, float(lats[idx])))
	return selected


def sample_tsdf_sweep_points(center_world, direction_world, tsdf_data, sweep_start_m=0.00, sweep_end_m=0.15, sweep_num_steps=32):
	center_world = np.asarray(center_world, dtype=np.float32)
	direction_world = np.asarray(direction_world, dtype=np.float32)
	direction_world = direction_world / max(np.linalg.norm(direction_world), 1e-12)
	deltas = np.linspace(float(sweep_start_m), float(sweep_end_m), int(sweep_num_steps), dtype=np.float32)
	sweep_points = center_world[None, :] + deltas[:, None] * direction_world[None, :]
	sweep_sdfs = query_tsdf_trilinear(tsdf_data, sweep_points).astype(np.float32)
	return sweep_points.astype(np.float32), sweep_sdfs, deltas.astype(np.float32)


def build_latitude_debug_plys(
	out_dir,
	candidate_dirs,
	center_world,
	tsdf_data,
	local_points,
	local_colors,
	target_latitudes_deg=(15, 30, 45, 60, 75),
	sweep_start_m=0.00,
	sweep_end_m=0.15,
	sweep_num_steps=32,
	query_gray=170,
	low10_sphere_radius=0.008,
	low10_sphere_points=32,
	low10_sphere_color=(0, 0, 255),
	scalar_name="tsdf",
	up_axis=2,
):
	os.makedirs(out_dir, exist_ok=True)
	local_points = np.asarray(local_points, dtype=np.float32)
	local_colors = np.asarray(local_colors, dtype=np.uint8)
	selected = select_nearest_latitude_indices(candidate_dirs, target_latitudes_deg, up_axis=up_axis)
	paths = []

	for target_lat, idx, actual_lat in selected:
		direction = np.asarray(candidate_dirs[idx], dtype=np.float32)
		sweep_points, sweep_sdfs, _ = sample_tsdf_sweep_points(
			center_world,
			direction,
			tsdf_data,
			sweep_start_m=sweep_start_m,
			sweep_end_m=sweep_end_m,
			sweep_num_steps=sweep_num_steps,
		)
		threshold = float(np.quantile(sweep_sdfs, 0.10))
		low_mask = sweep_sdfs <= threshold

		query_colors = np.full((sweep_points.shape[0], 3), int(query_gray), dtype=np.uint8)
		query_scalars = sweep_sdfs.astype(np.float32)
		local_scalars = np.zeros((local_points.shape[0],), dtype=np.float32)

		low_sphere_points = []
		low_sphere_colors = []
		low_sphere_scalars = []
		for point in sweep_points[low_mask]:
			sphere_points, sphere_colors = generate_sphere_points(point, low10_sphere_radius, low10_sphere_points, low10_sphere_color)
			low_sphere_points.append(sphere_points)
			low_sphere_colors.append(sphere_colors)
			low_sphere_scalars.append(np.full((sphere_points.shape[0],), threshold, dtype=np.float32))

		if len(low_sphere_points) > 0:
			low_sphere_points = np.concatenate(low_sphere_points, axis=0)
			low_sphere_colors = np.concatenate(low_sphere_colors, axis=0)
			low_sphere_scalars = np.concatenate(low_sphere_scalars, axis=0)
		else:
			low_sphere_points = np.zeros((0, 3), dtype=np.float32)
			low_sphere_colors = np.zeros((0, 3), dtype=np.uint8)
			low_sphere_scalars = np.zeros((0,), dtype=np.float32)

		points = np.concatenate([local_points, sweep_points, low_sphere_points], axis=0)
		colors = np.concatenate([local_colors, query_colors, low_sphere_colors], axis=0)
		scalars = np.concatenate([local_scalars, query_scalars, low_sphere_scalars], axis=0)
		lat_tag = int(np.round(target_lat))
		path = os.path.join(out_dir, f"lat_{lat_tag:02d}_query_debug.ply")
		write_ply_xyzrgb_scalar(path, points, colors, scalars, scalar_name=scalar_name)
		paths.append({"target_latitude": target_lat, "selected_idx": idx, "actual_latitude": actual_lat, "path": path})

	return paths


def build_tsdf_grid_scalar_ply(
	out_path,
	center_world,
	tsdf_data,
	local_points,
	local_colors,
	half_extent_m=0.12,
	step_m=0.01,
	grid_gray=150,
	scalar_name="tsdf",
):
	center_world = np.asarray(center_world, dtype=np.float32)
	local_points = np.asarray(local_points, dtype=np.float32)
	local_colors = np.asarray(local_colors, dtype=np.uint8)
	half_extent_m = float(half_extent_m)
	step_m = float(step_m)

	axis_vals = np.arange(-half_extent_m, half_extent_m + 0.5 * step_m, step_m, dtype=np.float32)
	grid_x, grid_y, grid_z = np.meshgrid(axis_vals, axis_vals, axis_vals, indexing="xy")
	offsets = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], axis=1)
	grid_points = center_world[None, :] + offsets
	grid_sdf = query_tsdf_trilinear(tsdf_data, grid_points).astype(np.float32)
	grid_colors = np.full((grid_points.shape[0], 3), int(grid_gray), dtype=np.uint8)

	local_scalars = np.zeros((local_points.shape[0],), dtype=np.float32)
	all_points = np.concatenate([local_points, grid_points], axis=0)
	all_colors = np.concatenate([local_colors, grid_colors], axis=0)
	all_scalars = np.concatenate([local_scalars, grid_sdf], axis=0)
	write_ply_xyzrgb_scalar(out_path, all_points, all_colors, all_scalars, scalar_name=scalar_name)

	return {
		"path": out_path,
		"grid_point_count": int(grid_points.shape[0]),
		"tsdf_min": float(np.min(grid_sdf)),
		"tsdf_max": float(np.max(grid_sdf)),
		"tsdf_mean": float(np.mean(grid_sdf)),
	}
