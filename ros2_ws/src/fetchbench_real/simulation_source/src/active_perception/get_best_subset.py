#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DATASET_ROOT = Path("/isaac-sim/corl2025/dataset")
NUM_TARGET_LINE_POINTS = 10
DEPTH_OCCLUSION_THRESHOLD_MM = 50.0
DEPTH_PATCH_RADIUS = 1
CENTER_SCORE_WEIGHT = 1.0
PROJECTED_SIZE_WEIGHT = 0.3
OCCLUSION_PENALTY_WEIGHT = 1.0
OUT_OF_FRAME_PENALTY_WEIGHT = 1.0
PROJECTION_SCORE_MARGIN_RATIO = 0.03
MIN_RAY_GROUND_ANGLE_DEG = 30.0
CENTER_REGION_RATIO = 0.60
MIN_CENTER_REGION_POINTS = 7


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Select best subset of view images for VLM input")
	parser.add_argument("--scene", required=True, help="Scene id, e.g. 01")
	parser.add_argument("--scene-num", required=True, help="Scene number, e.g. 000")
	parser.add_argument("--num-views", type=int, required=True, help="Number of views to select")
	parser.add_argument(
		"--sim-weight",
		type=float,
		default=1.0,
		help="Penalty weight for pairwise ray-direction cosine similarity",
	)
	parser.add_argument(
		"--num-starts",
		type=int,
		default=16,
		help="Number of greedy restarts",
	)
	parser.add_argument(
		"--ap-only",
		action="store_true",
		help="Select the subset only from active_perception/summary.json used_candidate_indices.",
	)
	return parser.parse_args()


def _load_json(path: Path) -> dict:
	with path.open(encoding="utf-8") as f:
		return json.load(f)


def _normalize(v: np.ndarray) -> np.ndarray:
	n = float(np.linalg.norm(v))
	if n < 1e-12:
		return np.zeros_like(v)
	return v / n


def _project_point_to_image(
	point_world: np.ndarray,
	cam_matrix: np.ndarray,
	K: np.ndarray,
	img_width: int,
	img_height: int,
) -> tuple[float, float, float, bool]:
	"""
	Project 3D point to 2D image plane. Returns (u, v, depth_mm, is_valid).
	cam_matrix: 4x4 camera-to-world transformation from pose.json.
	K: 3x3 camera intrinsic matrix.
	"""
	point_h = np.concatenate([point_world, [1.0]])
	# pose.json cam_matrix stores camera->world; projection needs world->camera.
	world_to_cam = np.linalg.inv(cam_matrix)
	point_cam = world_to_cam @ point_h
	z = float(point_cam[2])
	if abs(z) <= 0.01:
		return 0.0, 0.0, 0.0, False

	# Different camera pipelines use +Z or -Z as forward; support both conventions.
	denom = z if z > 0.01 else -z
	u = float(K[0, 0] * point_cam[0] / denom + K[0, 2])
	# Image v-axis points downward, so camera y should be flipped.
	v = float(K[1, 1] * (-point_cam[1]) / denom + K[1, 2])
	depth_mm = float(denom * 1000.0)
	valid = 0.0 <= u < float(img_width) and 0.0 <= v < float(img_height)
	return u, v, depth_mm, valid


def _is_inside_score_margin(u: float, v: float, img_width: int, img_height: int) -> bool:
	margin_u = PROJECTION_SCORE_MARGIN_RATIO * img_width
	margin_v = PROJECTION_SCORE_MARGIN_RATIO * img_height
	return margin_u <= u < (img_width - margin_u) and margin_v <= v < (img_height - margin_v)


def _load_depth_mm(depth_path: Path) -> np.ndarray:
	depth_img = Image.open(depth_path)
	depth = np.array(depth_img, dtype=np.float32)
	if depth.ndim != 2:
		raise ValueError(f"Expected single-channel depth image, got shape={depth.shape}")
	return depth


def _sample_depth_mm(depth_mm: np.ndarray, u: float, v: float, patch_radius: int = DEPTH_PATCH_RADIUS) -> float:
	ui = int(round(u))
	vi = int(round(v))
	h, w = depth_mm.shape
	x0 = max(0, ui - patch_radius)
	x1 = min(w, ui + patch_radius + 1)
	y0 = max(0, vi - patch_radius)
	y1 = min(h, vi + patch_radius + 1)
	patch = depth_mm[y0:y1, x0:x1]
	valid = patch[patch > 0.0]
	if valid.size == 0:
		return 0.0
	return float(np.median(valid))


def _compute_centering_score(u: float, v: float, center_u: float, center_v: float, max_dist: float) -> float:
	if max_dist <= 1e-12:
		return 0.0
	center_dist = math.hypot(u - center_u, v - center_v)
	# Linear center preference: 1 at image center, 0 near image corners.
	return max(0.0, 1.0 - (center_dist / max_dist))


def _is_inside_center_region(u: float, v: float, img_width: int, img_height: int) -> bool:
	margin_ratio = max(0.0, (1.0 - CENTER_REGION_RATIO) * 0.5)
	margin_u = margin_ratio * img_width
	margin_v = margin_ratio * img_height
	return margin_u <= u < (img_width - margin_u) and margin_v <= v < (img_height - margin_v)


def _objective(indices: list[int], view_scores: np.ndarray, cos_mat: np.ndarray, sim_weight: float) -> float:
	if not indices:
		return -1e18
	score = float(np.sum(view_scores[indices]))
	sim_penalty = 0.0
	for i in range(len(indices)):
		a = indices[i]
		for j in range(i + 1, len(indices)):
			b = indices[j]
			sim_penalty += float(cos_mat[a, b])
	return score - sim_weight * sim_penalty


def _greedy_select(
	candidate_count: int,
	k: int,
	view_scores: np.ndarray,
	cos_mat: np.ndarray,
	sim_weight: float,
	start_idx: int,
) -> list[int]:
	selected = [start_idx]
	remaining = set(range(candidate_count))
	remaining.remove(start_idx)

	while len(selected) < k and remaining:
		best_i = None
		best_gain = -1e18
		for i in remaining:
			sim_sum = float(np.sum(cos_mat[i, selected]))
			gain = float(view_scores[i]) - sim_weight * sim_sum
			if gain > best_gain:
				best_gain = gain
				best_i = i
		if best_i is None:
			break
		selected.append(best_i)
		remaining.remove(best_i)
	return selected


def _local_swap_improve(
	selected: list[int],
	candidate_count: int,
	view_scores: np.ndarray,
	cos_mat: np.ndarray,
	sim_weight: float,
) -> list[int]:
	current = selected[:]
	current_obj = _objective(current, view_scores, cos_mat, sim_weight)
	improved = True

	while improved:
		improved = False
		current_set = set(current)
		for s in range(len(current)):
			out_idx = current[s]
			for in_idx in range(candidate_count):
				if in_idx in current_set:
					continue
				trial = current[:]
				trial[s] = in_idx
				trial_obj = _objective(trial, view_scores, cos_mat, sim_weight)
				if trial_obj > current_obj + 1e-9:
					current = trial
					current_obj = trial_obj
					improved = True
					break
			if improved:
				break
	return current


def _build_contact_sheet(
	images: list[Image.Image],
	labels: list[str],
	selected_pos: set[int],
	tile_size: tuple[int, int] = (256, 144),
	cols: int = 10,
) -> Image.Image:
	tw, th = tile_size
	rows = int(math.ceil(len(images) / float(cols))) if images else 1
	pad = 6
	label_h = 42
	canvas_w = cols * (tw + pad) + pad
	canvas_h = rows * (th + label_h + pad) + pad
	canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 24))
	draw = ImageDraw.Draw(canvas)

	for i, img in enumerate(images):
		r = i // cols
		c = i % cols
		x0 = pad + c * (tw + pad)
		y0 = pad + r * (th + label_h + pad)

		thumb = img.resize((tw, th), Image.Resampling.BILINEAR)
		canvas.paste(thumb, (x0, y0))

		border_color = (255, 90, 90) if i in selected_pos else (110, 110, 110)
		border_w = 4 if i in selected_pos else 1
		for b in range(border_w):
			draw.rectangle([x0 + b, y0 + b, x0 + tw - 1 - b, y0 + th - 1 - b], outline=border_color)

		txt = labels[i]
		txt_color = (255, 190, 190) if i in selected_pos else (220, 220, 220)
		draw.text((x0 + 4, y0 + th + 4), txt, fill=txt_color)

	return canvas


def _draw_projection_points(
	image: Image.Image,
	projection_points: list[dict],
	point_radius: int = 6,
) -> Image.Image:
	img = image.copy()
	draw = ImageDraw.Draw(img)
	width, height = img.size
	for p in projection_points:
		u = float(p["u"])
		v = float(p["v"])
		status = str(p.get("status", "invalid"))
		if status == "valid":
			color = "lime"
		elif status == "occluded":
			color = "red"
		else:
			color = "orange"
		if not (0.0 <= u < float(width) and 0.0 <= v < float(height)):
			continue
		draw.ellipse([u - point_radius, v - point_radius, u + point_radius, v + point_radius], fill=color, outline=color)
	return img


def _concat_selected(images: list[Image.Image], labels: list[str]) -> Image.Image:
	if not images:
		return Image.new("RGB", (640, 360), color=(0, 0, 0))
	h = min(360, min(img.height for img in images))
	resized = []
	for img in images:
		w = int(round(img.width * (h / float(img.height))))
		resized.append(img.resize((w, h), Image.Resampling.BILINEAR))
	pad = 6
	label_h = 28
	total_w = sum(img.width for img in resized) + pad * (len(resized) + 1)
	canvas_h = h + label_h + 2 * pad
	canvas = Image.new("RGB", (total_w, canvas_h), color=(20, 20, 20))
	draw = ImageDraw.Draw(canvas)

	x = pad
	for i, img in enumerate(resized):
		canvas.paste(img, (x, pad))
		draw.rectangle([x, pad, x + img.width - 1, pad + h - 1], outline=(255, 80, 80), width=3)
		draw.text((x + 4, pad + h + 4), labels[i], fill=(255, 210, 210))
		x += img.width + pad
	return canvas


def main() -> None:
	args = _parse_args()
	scene_num_int = int(args.scene_num)

	scene_dir = DATASET_ROOT / f"{args.scene}_robot" / f"scene_{scene_num_int:03d}"
	views_dir = scene_dir / "views"
	rgb_dir = views_dir / "rgb"
	depth_dir = views_dir / "depth"
	intrinsics_path = views_dir / "intrinsics.json"
	pose_path = views_dir / "pose.json"
	target_point_path = scene_dir / "bev" / "target_point.json"
	subset_dir = scene_dir / "active_perception" / "subset" if args.ap_only else scene_dir / "subset"
	subset_dir.mkdir(parents=True, exist_ok=True)

	if not scene_dir.is_dir():
		raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
	if not rgb_dir.is_dir():
		raise FileNotFoundError(f"RGB views directory not found: {rgb_dir}")
	if not depth_dir.is_dir():
		raise FileNotFoundError(f"Depth views directory not found: {depth_dir}")
	if not intrinsics_path.is_file():
		raise FileNotFoundError(f"intrinsics.json not found: {intrinsics_path}")
	if not pose_path.is_file():
		raise FileNotFoundError(f"pose.json not found: {pose_path}")
	if not target_point_path.is_file():
		raise FileNotFoundError(f"target_point.json not found: {target_point_path}")

	intrinsics = _load_json(intrinsics_path)
	pose_data = _load_json(pose_path)
	target_point = _load_json(target_point_path)

	target_world = np.asarray(target_point["grasp_position_world"], dtype=np.float64)
	target_world_z0 = target_world.copy()
	target_world_z0[2] = 0.0
	target_line_points = [
		target_world + t * (target_world_z0 - target_world)
		for t in np.linspace(0.0, 1.0, NUM_TARGET_LINE_POINTS)
	]

	K = np.asarray([
		[intrinsics["fx"], 0.0, intrinsics["cx"]],
		[0.0, intrinsics["fy"], intrinsics["cy"]],
		[0.0, 0.0, 1.0],
	], dtype=np.float64)
	img_w = int(intrinsics["width"])
	img_h = int(intrinsics["height"])

	poses = pose_data.get("poses", [])
	reached = set(int(i) for i in pose_data.get("reached_indices", []))
	pose_by_idx = {int(p["index"]): p for p in poses if "index" in p}
	ap_summary_path = scene_dir / "active_perception" / "summary.json"
	ap_allowed_indices: set[int] | None = None
	ap_missing_indices: list[int] = []
	if args.ap_only:
		if not ap_summary_path.is_file():
			raise FileNotFoundError(f"Active perception summary not found: {ap_summary_path}")
		ap_summary = _load_json(ap_summary_path)
		ap_indices = {int(i) for i in ap_summary.get("used_candidate_indices", [])}
		for view in ap_summary.get("views", []):
			idx = view.get("candidate_index")
			if idx is not None:
				ap_indices.add(int(idx))
		if not ap_indices:
			raise RuntimeError(f"No AP candidate indices found in {ap_summary_path}")
		valid_ap_indices = set()
		for idx in sorted(ap_indices):
			if idx not in pose_by_idx or not (rgb_dir / f"{idx:04d}.png").is_file() or not (depth_dir / f"{idx:04d}.png").is_file():
				ap_missing_indices.append(idx)
				continue
			valid_ap_indices.add(idx)
		if not valid_ap_indices:
			raise RuntimeError(
				f"AP selected indices exist in {ap_summary_path}, but none have matching data in {views_dir}"
			)
		ap_allowed_indices = valid_ap_indices
		print(
			f"[AP-only] Restricting subset candidates to {len(ap_allowed_indices)} AP-selected views "
			f"from {ap_summary_path}",
			flush=True,
		)
		if ap_missing_indices:
			print(f"[AP-only] Missing AP view data skipped: {ap_missing_indices}", flush=True)

	all_rgb_files = sorted(rgb_dir.glob("*.png"))
	candidates = []
	filtered_low_angle_count = 0
	filtered_center_region_count = 0
	for rgb_path in all_rgb_files:
		idx = int(rgb_path.stem)
		if ap_allowed_indices is not None and idx not in ap_allowed_indices:
			continue
		if reached and idx not in reached:
			continue
		pose = pose_by_idx.get(idx)
		if pose is None:
			continue

		cam_pos = np.asarray(pose["cam_position"], dtype=np.float64)
		ray = target_world - cam_pos
		ray_n = _normalize(ray)
		if float(np.linalg.norm(ray_n)) < 1e-8:
			continue
		# theta: angle between camera ray and z=0 plane.
		# For near-vertical views, cos(theta) gets small, so we relax occlusion threshold.
		cos_theta = float(np.linalg.norm(ray_n[:2]))
		ray_ground_angle_deg = float(math.degrees(math.atan2(abs(float(ray_n[2])), max(cos_theta, 1e-12))))
		if ray_ground_angle_deg < MIN_RAY_GROUND_ANGLE_DEG:
			filtered_low_angle_count += 1
			continue
		occlusion_threshold_mm = (
			DEPTH_OCCLUSION_THRESHOLD_MM / cos_theta
			if cos_theta > 1e-12
			else float("inf")
		)

		depth_path = depth_dir / rgb_path.name
		if not depth_path.is_file():
			continue
		depth_mm = _load_depth_mm(depth_path)

		# Projection and visibility score over sampled points on grasp->z0 segment.
		cam_matrix = np.asarray(pose["cam_matrix"], dtype=np.float64)
		grasp_u, grasp_v, grasp_expected_depth_mm, grasp_proj_valid = _project_point_to_image(
			target_world,
			cam_matrix,
			K,
			img_w,
			img_h,
		)
		if not grasp_proj_valid:
			continue
		if not _is_inside_score_margin(grasp_u, grasp_v, img_w, img_h):
			continue
		center_u, center_v = float(img_w) / 2.0, float(img_h) / 2.0
		max_dist = math.hypot(center_u, center_v)
		center_score_sum = 0.0
		size_score_sum = 0.0
		occlusion_penalty_sum = 0.0
		out_of_frame_penalty_sum = 0.0
		center_region_points = 0
		valid_count = 0
		occluded_count = 0
		out_of_frame_count = 0
		valid_us = []
		valid_vs = []
		projection_points = []

		for target_p in target_line_points:
			u, v, expected_depth_mm, proj_valid = _project_point_to_image(target_p, cam_matrix, K, img_w, img_h)
			if proj_valid and _is_inside_center_region(u, v, img_w, img_h):
				center_region_points += 1
			score_valid = proj_valid and _is_inside_score_margin(u, v, img_w, img_h)
			if not proj_valid:
				out_of_frame_count += 1
				out_of_frame_penalty_sum += 1.0
			status = "invalid"
			if score_valid:
				valid_count += 1
				valid_us.append(u)
				valid_vs.append(v)
				observed_depth_mm = _sample_depth_mm(depth_mm, u, v)
				is_occluded = (
					observed_depth_mm > 0.0
					and (expected_depth_mm - observed_depth_mm) > occlusion_threshold_mm
				)
				if is_occluded:
					occluded_count += 1
					# Foreground object closer than expected target depth: penalize this sample negatively.
					occlusion_penalty_sum += 1.0
					status = "occluded"
					projection_points.append({
						"u": float(u),
						"v": float(v),
						"status": status,
					})
					continue
				center_score_sum += _compute_centering_score(u, v, center_u, center_v, max_dist)
				status = "valid"
				dist_to_target = float(np.linalg.norm(target_p - cam_pos))
				if dist_to_target < 0.2:
					size_score_sum += 0.5  # Too close, may be out-of-focus
				elif dist_to_target > 1.5:
					size_score_sum += 0.3  # Too far, target too small
				else:
					size_score_sum += 1.0 / (1.0 + (dist_to_target - 0.5) ** 2)
			projection_points.append({
				"u": float(u),
				"v": float(v),
				"status": status,
			})

		if center_region_points < MIN_CENTER_REGION_POINTS:
			filtered_center_region_count += 1
			continue

		# Require at least one sampled point visible in-frame.
		if valid_count == 0:
			continue

		# Average scores over all sampled points (invisible points contribute zero,
		# occluded points contribute negative penalty).
		center_score = center_score_sum / float(NUM_TARGET_LINE_POINTS)
		size_score = size_score_sum / float(NUM_TARGET_LINE_POINTS)
		occlusion_penalty = occlusion_penalty_sum / float(NUM_TARGET_LINE_POINTS)
		out_of_frame_penalty = out_of_frame_penalty_sum / float(NUM_TARGET_LINE_POINTS)
		u = float(np.mean(valid_us))
		v = float(np.mean(valid_vs))

		# Composite score: weighted sum
		view_score = (
			CENTER_SCORE_WEIGHT * center_score +
			PROJECTED_SIZE_WEIGHT * size_score
		) - (OCCLUSION_PENALTY_WEIGHT * occlusion_penalty) - (OUT_OF_FRAME_PENALTY_WEIGHT * out_of_frame_penalty)

		candidates.append(
			{
				"idx": idx,
				"rgb_path": rgb_path,
				"cam_pos": cam_pos,
				"ray": ray_n,
				"view_score": float(view_score),
				"center_score": float(center_score),
				"size_score": float(size_score),
				"occlusion_penalty": float(occlusion_penalty),
				"out_of_frame_penalty": float(out_of_frame_penalty),
				"line_points_inside_center_region": int(center_region_points),
				"line_points_visible": int(valid_count),
				"line_points_occluded": int(occluded_count),
				"line_points_out_of_frame": int(out_of_frame_count),
				"cos_theta_ray_to_z0": float(cos_theta),
				"ray_ground_angle_deg": float(ray_ground_angle_deg),
				"occlusion_threshold_mm": float(occlusion_threshold_mm),
				"projection_points": projection_points,
				"grasp_proj_valid": bool(grasp_proj_valid),
				"grasp_proj_u": float(grasp_u),
				"grasp_proj_v": float(grasp_v),
				"grasp_expected_depth_mm": float(grasp_expected_depth_mm),
			}
		)

	if not candidates:
		raise RuntimeError(
			f"No valid view candidates found. Filtered out {filtered_low_angle_count} views "
			f"with ray-ground angle < {MIN_RAY_GROUND_ANGLE_DEG:.1f} deg and "
			f"{filtered_center_region_count} views with center-region points < {MIN_CENTER_REGION_POINTS}."
		)

	n = len(candidates)
	k = min(max(1, int(args.num_views)), n)
	if k != args.num_views:
		print(f"[WARN] Requested num_views={args.num_views}, adjusted to {k} (available={n})")

	view_scores = np.asarray([c["view_score"] for c in candidates], dtype=np.float64)
	rays = np.asarray([c["ray"] for c in candidates], dtype=np.float64)
	cos_mat = rays @ rays.T
	# Use non-negative similarity only: opposite directions are treated as 0 similarity.
	cos_mat = np.clip(cos_mat, 0.0, 1.0)
	np.fill_diagonal(cos_mat, 0.0)

	starts_by_score = np.argsort(-view_scores)
	num_starts = min(max(1, int(args.num_starts)), n)
	start_indices = [int(i) for i in starts_by_score[:num_starts]]

	best_sel = None
	best_obj = -1e18
	for s in start_indices:
		sel = _greedy_select(
			candidate_count=n,
			k=k,
			view_scores=view_scores,
			cos_mat=cos_mat,
			sim_weight=float(args.sim_weight),
			start_idx=s,
		)
		if len(sel) < k:
			continue
		sel = _local_swap_improve(sel, n, view_scores, cos_mat, float(args.sim_weight))
		obj = _objective(sel, view_scores, cos_mat, float(args.sim_weight))
		if obj > best_obj:
			best_obj = obj
			best_sel = sel

	if best_sel is None:
		raise RuntimeError("Failed to compute a valid subset.")

	selected = [candidates[i] for i in best_sel]
	selected_sorted = sorted(selected, key=lambda x: x["idx"])
	selected_indices = [int(c["idx"]) for c in selected_sorted]
	idx_to_candidate_pos = {int(c["idx"]): i for i, c in enumerate(candidates)}

	pair_penalty = 0.0
	for i in range(len(best_sel)):
		for j in range(i + 1, len(best_sel)):
			pair_penalty += float(cos_mat[best_sel[i], best_sel[j]])

	selected_pair_similarities = []
	for i in range(len(selected_indices)):
		for j in range(i + 1, len(selected_indices)):
			idx_i = int(selected_indices[i])
			idx_j = int(selected_indices[j])
			pos_i = idx_to_candidate_pos[idx_i]
			pos_j = idx_to_candidate_pos[idx_j]
			sim_ij = float(cos_mat[pos_i, pos_j])
			selected_pair_similarities.append(
				{
					"pair": [idx_i, idx_j],
					"cosine_similarity": sim_ij,
				}
			)

	out = {
		"scene": args.scene,
		"scene_num": f"{scene_num_int:03d}",
		"ap_only": bool(args.ap_only),
		"active_perception_summary_path": str(ap_summary_path) if args.ap_only else None,
		"active_perception_candidate_indices": sorted(int(i) for i in ap_allowed_indices) if ap_allowed_indices is not None else None,
		"active_perception_missing_indices": ap_missing_indices if args.ap_only else [],
		"num_views_requested": int(args.num_views),
		"num_views_selected": int(k),
		"num_candidates_after_filter": int(n),
		"num_views_filtered_low_ray_angle": int(filtered_low_angle_count),
		"num_views_filtered_center_region": int(filtered_center_region_count),
		"min_ray_ground_angle_deg": float(MIN_RAY_GROUND_ANGLE_DEG),
		"center_region_ratio": float(CENTER_REGION_RATIO),
		"min_center_region_points": int(MIN_CENTER_REGION_POINTS),
		"sim_weight": float(args.sim_weight),
		"num_starts": int(num_starts),
		"num_target_line_points": int(NUM_TARGET_LINE_POINTS),
		"target_grasp_position_world": [float(x) for x in target_world.tolist()],
		"target_grasp_z0_position_world": [float(x) for x in target_world_z0.tolist()],
		"intrinsics": {
			"width": intrinsics.get("width"),
			"height": intrinsics.get("height"),
			"fx": intrinsics.get("fx"),
			"fy": intrinsics.get("fy"),
			"cx": intrinsics.get("cx"),
			"cy": intrinsics.get("cy"),
		},
		"objective": {
			"sum_view_scores": float(np.sum([c["view_score"] for c in selected])),
			"sum_pair_cosine": float(pair_penalty),
			"value": float(best_obj),
			"formula": "sum(view_score_i) - sim_weight * sum(cos(ray_i, ray_j), i<j)",
			"weights": {
				"center_score": float(CENTER_SCORE_WEIGHT),
				"projected_size": float(PROJECTED_SIZE_WEIGHT),
				"occlusion_penalty": float(OCCLUSION_PENALTY_WEIGHT),
				"out_of_frame_penalty": float(OUT_OF_FRAME_PENALTY_WEIGHT),
			},
		},
		"selected_pair_similarities": selected_pair_similarities,
		"selected_indices": selected_indices,
		"selected_views": [
			{
				"index": int(c["idx"]),
				"rgb_path": str(c["rgb_path"]),
				"view_score": float(c["view_score"]),
				"components": {
					"center_framing": float(c["center_score"]),
					"projected_size": float(c["size_score"]),
					"occlusion_penalty": float(c["occlusion_penalty"]),
					"out_of_frame_penalty": float(c["out_of_frame_penalty"]),
					"line_points_inside_center_region": int(c["line_points_inside_center_region"]),
					"line_points_visible": int(c["line_points_visible"]),
					"line_points_occluded": int(c["line_points_occluded"]),
					"line_points_out_of_frame": int(c["line_points_out_of_frame"]),
					"cos_theta_ray_to_z0": float(c["cos_theta_ray_to_z0"]),
					"ray_ground_angle_deg": float(c["ray_ground_angle_deg"]),
					"occlusion_threshold_mm": float(c["occlusion_threshold_mm"]),
				},
				"ray_to_target": [float(v) for v in c["ray"].tolist()],
				"camera_position": [float(v) for v in c["cam_pos"].tolist()],
				"grasp_point_projection_uv": [float(c["grasp_proj_u"]), float(c["grasp_proj_v"])],
				"grasp_point_projection_valid": bool(c["grasp_proj_valid"]),
				"grasp_point_expected_depth_mm": float(c["grasp_expected_depth_mm"]),
			}
			for c in selected_sorted
		],
	}

	json_path = subset_dir / f"best_subset_{k}.json"
	with json_path.open("w", encoding="utf-8") as f:
		json.dump(out, f, ensure_ascii=False, indent=2)

	# Visualization: all candidates contact sheet with selected ones highlighted.
	candidates_by_score = sorted(candidates, key=lambda c: c["view_score"], reverse=True)
	all_images = [
		_draw_projection_points(Image.open(c["rgb_path"]).convert("RGB"), c["projection_points"])
		for c in candidates_by_score
	]
	all_labels = [
		f"{c['idx']:04d}  s={c['view_score']:.2f}\ncf={c['center_score']:.2f}  cp={c['line_points_inside_center_region']}/{NUM_TARGET_LINE_POINTS}  ps={c['size_score']:.2f}  ang={c['ray_ground_angle_deg']:.1f}"
		for c in candidates_by_score
	]
	selected_pos = {i for i, c in enumerate(candidates_by_score) if int(c["idx"]) in set(selected_indices)}
	sheet = _build_contact_sheet(all_images, all_labels, selected_pos=selected_pos, cols=10)
	sheet_path = subset_dir / f"all_views_contact_sheet_k{k}.png"
	sheet.save(sheet_path)

	# Visualization: selected views concatenated.
	sel_images = [Image.open(c["rgb_path"]).convert("RGB") for c in selected_sorted]
	sel_labels = [f"{c['idx']:04d}" for c in selected_sorted]
	concat_img = _concat_selected(sel_images, sel_labels)
	concat_path = subset_dir / f"selected_subset_concat_k{k}.png"
	concat_img.save(concat_path)

	print("=== Best subset selection done ===")
	print(f"scene: {args.scene} / scene_num: {scene_num_int:03d}")
	print(f"selected indices: {selected_indices}")
	print(
		f"filtered low-angle views (< {MIN_RAY_GROUND_ANGLE_DEG:.1f} deg): {filtered_low_angle_count}"
	)
	print(
		f"filtered center-region views (< {MIN_CENTER_REGION_POINTS}/{NUM_TARGET_LINE_POINTS} points in center {int(CENTER_REGION_RATIO * 100)}%): {filtered_center_region_count}"
	)
	print(f"objective: {best_obj:.4f}")
	print("selected pair similarities (cosine):")
	for row in selected_pair_similarities:
		pair = row["pair"]
		sim = row["cosine_similarity"]
		print(f"  {pair[0]:04d}-{pair[1]:04d}: {sim:+.4f}")
	print(f"saved: {json_path}")
	print(f"saved: {sheet_path}")
	print(f"saved: {concat_path}")


if __name__ == "__main__":
	main()
