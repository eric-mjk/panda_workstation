import json
import os

import cv2
import numpy as np


def build_image_path(folder, index):
	return os.path.join(folder, f"{index:04d}.png")


def read_depth_image(depth_path):
	depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
	if depth is None:
		raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
	if depth.dtype == np.uint16:
		return depth.astype(np.float32) / 1000.0
	return depth.astype(np.float32)


def read_class_map(class_path):
	cls = cv2.imread(class_path, cv2.IMREAD_UNCHANGED)
	if cls is None:
		raise FileNotFoundError(f"Failed to read class image: {class_path}")
	if cls.ndim == 3:
		cls = cls[..., 0]
	return cls.astype(np.int32)


def get_target_class_id(pose_json_path):
	with open(pose_json_path, "r", encoding="utf-8") as f:
		pose_data = json.load(f)
	return int(pose_data.get("target_class_id", 1))


def get_background_class_id(pose_json_path):
	with open(pose_json_path, "r", encoding="utf-8") as f:
		pose_data = json.load(f)
	return int(pose_data.get("background_class_id", 0))


def get_pose_reached_indices(pose_json_path):
	with open(pose_json_path, "r", encoding="utf-8") as f:
		pose_data = json.load(f)
	return [int(i) for i in pose_data.get("reached_indices", [])]


def get_pose_index_to_transforms(pose_json_path):
	with open(pose_json_path, "r", encoding="utf-8") as f:
		pose_data = json.load(f)

	pose_map = {}
	for p in pose_data["poses"]:
		idx = int(p["index"])
		c2w = np.asarray(p["cam_matrix"], dtype=np.float32)
		pose_map[idx] = {
			"c2w": c2w,
			"w2c": np.linalg.inv(c2w),
		}
	return pose_map


def validate_pose_convention(pose_json_path, pose_map):
	with open(pose_json_path, "r", encoding="utf-8") as f:
		pose_data = json.load(f)

	first_pose = pose_data["poses"][0]
	first_idx = int(first_pose["index"])
	cam_position = np.asarray(first_pose["cam_position"], dtype=np.float32)
	c2w = pose_map[first_idx]["c2w"]
	translation_error = float(np.linalg.norm(c2w[:3, 3] - cam_position))
	if translation_error > 1e-4:
		raise ValueError(
			f"Pose convention mismatch: cam_matrix translation differs from cam_position by {translation_error:.6f}. "
			"This dataset is not being read as c2w."
		)
	return translation_error


def should_rotate_image_180(c2w):
	return float(np.asarray(c2w, dtype=np.float32)[2, 1]) < 0.0


def rotate_image_180(image):
	return cv2.rotate(image, cv2.ROTATE_180)


def rotate_point_180(point, width, height):
	x, y = point
	return (int(width - 1 - x), int(height - 1 - y))


def rotate_direction_scores_180(scores):
	return np.roll(np.asarray(scores, dtype=np.float32), -6)


def get_pose_index_to_position(pose_json_path):
	with open(pose_json_path, "r", encoding="utf-8") as f:
		pose_data = json.load(f)
	return {int(pose["index"]): pos for pos, pose in enumerate(pose_data["poses"])}


def get_camera_intrinsics(intrinsics_json_path):
	with open(intrinsics_json_path, "r", encoding="utf-8") as f:
		intrinsics = json.load(f)

	fx = float(intrinsics["fx"])
	fy = float(intrinsics["fy"])
	cx = float(intrinsics["cx"])
	cy = float(intrinsics["cy"])
	width = int(intrinsics["width"])
	height = int(intrinsics["height"])
	return fx, fy, cx, cy, width, height


def unproject_pixel_to_world(u, v, z, c2w, fx, fy, cx, cy):
	x = (float(u) - cx) * z / fx
	y = -(float(v) - cy) * z / fy
	p_cam = np.array([x, y, -z, 1.0], dtype=np.float32)
	p_world = c2w @ p_cam
	return p_world[:3]


def project_world_to_pixel(p_world, c2w, fx, fy, cx, cy):
	w2c = np.linalg.inv(c2w)
	p_world_h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32)
	p_cam = w2c @ p_world_h
	x, y, z = p_cam[:3]

	if z >= 0:
		return None, z

	z_dist = -z
	u = fx * (x / z_dist) + cx
	v = fy * (-y / z_dist) + cy
	return (float(u), float(v)), float(z)


def get_query_point_from_class_map(class_map, target_class_id):
	ys, xs = np.where(class_map == int(target_class_id))
	h, w = class_map.shape[:2]
	if len(xs) == 0:
		return (w // 2, h // 2)
	return (int(np.round(xs.mean())), int(np.round(ys.mean())))


def draw_yellow_circle(image_path, query_point, radius=18, outline=4):
	img = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"Failed to read image: {image_path}")

	x, y = map(int, query_point)
	drawn = img.copy()
	yellow = (0, 255, 255)
	black = (0, 0, 0)

	cv2.circle(drawn, (x, y), radius=radius, color=black, thickness=-1, lineType=cv2.LINE_AA)
	inner_radius = max(1, radius - outline)
	cv2.circle(drawn, (x, y), radius=inner_radius, color=yellow, thickness=-1, lineType=cv2.LINE_AA)
	return drawn


def random_downsample_points(points, colors, max_points, seed=0):
	if points.shape[0] <= int(max_points):
		return points, colors
	rng = np.random.default_rng(seed)
	idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
	return points[idx], colors[idx]


def write_ply_xyzrgb(save_path, points, colors):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	with open(save_path, "w", encoding="utf-8") as f:
		f.write("ply\n")
		f.write("format ascii 1.0\n")
		f.write(f"element vertex {points.shape[0]}\n")
		f.write("property float x\n")
		f.write("property float y\n")
		f.write("property float z\n")
		f.write("property uchar red\n")
		f.write("property uchar green\n")
		f.write("property uchar blue\n")
		f.write("end_header\n")
		for p, c in zip(points, colors):
			f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[2])} {int(c[1])} {int(c[0])}\n")


def write_ply_xyzrgb_scalar(save_path, points, colors, scalars, scalar_name="tsdf"):
	points = np.asarray(points, dtype=np.float32)
	colors = np.asarray(colors, dtype=np.uint8)
	scalars = np.asarray(scalars, dtype=np.float32)
	if points.shape[0] != colors.shape[0] or points.shape[0] != scalars.shape[0]:
		raise ValueError("points, colors, and scalars must have the same length")
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	with open(save_path, "w", encoding="utf-8") as f:
		f.write("ply\n")
		f.write("format ascii 1.0\n")
		f.write(f"element vertex {points.shape[0]}\n")
		f.write("property float x\n")
		f.write("property float y\n")
		f.write("property float z\n")
		f.write("property uchar red\n")
		f.write("property uchar green\n")
		f.write("property uchar blue\n")
		f.write(f"property float {scalar_name}\n")
		f.write("end_header\n")
		for p, c, s in zip(points, colors, scalars):
			f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[2])} {int(c[1])} {int(c[0])} {float(s):.6f}\n")


def collect_points_near_center(sorted_idxs, pose_map, center_world, max_radius_m, intrinsics_json_path, rgb_dir, depth_dir):
	all_points = []
	all_colors = []

	fx, fy, cx, cy, _, _ = get_camera_intrinsics(intrinsics_json_path)
	for idx in sorted_idxs:
		rgb_path = build_image_path(rgb_dir, idx)
		depth_path = build_image_path(depth_dir, idx)
		depth = read_depth_image(depth_path)
		rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
		if rgb is None:
			raise FileNotFoundError(f"Failed to read image: {rgb_path}")

		ys, xs = np.where(depth > 0)
		if len(xs) == 0:
			continue

		z = depth[ys, xs].astype(np.float32)
		x = (xs.astype(np.float32) - cx) * z / fx
		y = -(ys.astype(np.float32) - cy) * z / fy
		p_cam = np.stack([x, y, -z, np.ones_like(z)], axis=1)
		c2w = pose_map[idx]["c2w"]
		p_world = (c2w @ p_cam.T).T[:, :3]

		dist = np.linalg.norm(p_world - center_world[None, :], axis=1)
		mask = dist <= float(max_radius_m)
		if not np.any(mask):
			continue

		all_points.append(p_world[mask])
		all_colors.append(rgb[ys[mask], xs[mask], :])

	if len(all_points) == 0:
		return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

	points = np.concatenate(all_points, axis=0).astype(np.float32)
	colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
	return points, colors


def generate_sphere_points(center, radius, num_points, color):
	center = np.asarray(center, dtype=np.float32)
	num_points = max(20, int(num_points))
	phi = np.arccos(1.0 - 2.0 * (np.arange(num_points, dtype=np.float32) + 0.5) / num_points)
	theta = np.pi * (1.0 + 5.0**0.5) * np.arange(num_points, dtype=np.float32)
	x = radius * np.sin(phi) * np.cos(theta)
	y = radius * np.sin(phi) * np.sin(theta)
	z = radius * np.cos(phi)
	points = np.stack([x, y, z], axis=1) + center[None, :]
	colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (num_points, 1))
	return points.astype(np.float32), colors


def collect_strided_full_point_cloud(sorted_idxs, pose_map, stride, intrinsics_json_path, rgb_dir, depth_dir):
	fx, fy, cx, cy, _, _ = get_camera_intrinsics(intrinsics_json_path)
	all_points = []
	all_colors = []

	for idx in sorted_idxs:
		rgb_path = build_image_path(rgb_dir, idx)
		depth_path = build_image_path(depth_dir, idx)
		depth = read_depth_image(depth_path)
		rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
		if rgb is None:
			raise FileNotFoundError(f"Failed to read image: {rgb_path}")

		h, w = depth.shape[:2]
		ys = np.arange(0, h, int(stride), dtype=np.int32)
		xs = np.arange(0, w, int(stride), dtype=np.int32)
		xv, yv = np.meshgrid(xs, ys)
		xv = xv.reshape(-1)
		yv = yv.reshape(-1)
		z = depth[yv, xv].astype(np.float32)
		mask = z > 0
		if not np.any(mask):
			continue

		xv = xv[mask].astype(np.float32)
		yv = yv[mask].astype(np.float32)
		z = z[mask]
		x = (xv - cx) * z / fx
		y = -(yv - cy) * z / fy
		p_cam = np.stack([x, y, -z, np.ones_like(z)], axis=1)
		c2w = pose_map[idx]["c2w"]
		p_world = (c2w @ p_cam.T).T[:, :3]
		colors = rgb[yv.astype(np.int32), xv.astype(np.int32), :]
		all_points.append(p_world)
		all_colors.append(colors)

	if len(all_points) == 0:
		return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

	points = np.concatenate(all_points, axis=0).astype(np.float32)
	colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
	return points, colors
