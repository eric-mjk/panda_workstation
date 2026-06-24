import argparse
import contextlib
import json
import multiprocessing as mp
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from data_utils import (
	build_image_path,
	collect_points_near_center,
	get_background_class_id,
	get_camera_intrinsics,
	get_pose_index_to_position,
	get_pose_index_to_transforms,
	get_pose_reached_indices,
	get_target_class_id,
	random_downsample_points,
	should_rotate_image_180,
	unproject_pixel_to_world,
	validate_pose_convention,
	write_ply_xyzrgb,
)
from gemini_utils import process_single_index
from prompt import prompt_single
from tsdf_utils import (
	build_tsdf_volume,
	build_latitude_debug_plys,
	build_scene_visualization_plys,
	build_tsdf_grid_scalar_ply,
	collect_target_object_points,
	compute_distilled_direction_scores,
	compute_geometry_scores,
	compute_volume_bounds,
	fibonacci_upper_hemisphere,
	interpolate_clock_score,
	probe_tsdf_along_directions,
)
from viz_utils import (
	concatenate_2x2,
	plot_candidate_scores_contour_like_vis,
	plot_candidate_scores_like_vis,
	plot_score_vs_latitude,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
load_dotenv()

DATASET_DIR = os.path.join(SCRIPT_DIR, "ex_dataset", "scene_000")
POSE_PATH = os.path.join(DATASET_DIR, "views", "pose.json")
INTRINSICS_PATH = os.path.join(DATASET_DIR, "views", "intrinsics.json")
RGB_DIR = os.path.join(DATASET_DIR, "views", "rgb")
DEPTH_DIR = os.path.join(DATASET_DIR, "views", "depth")
CLASS_DIR = os.path.join(DATASET_DIR, "views", "class")

IDXS = [39, 59, 72, 75]
RADIUS_M = 0.30
MAX_POINTS_30CM = 100000
DISTILL_NUM_DIRECTIONS = 1000
GEO_LAMBDA = 0.5

TSDF_VOXEL_SIZE_M = 0.01
TSDF_TRUNCATION_M = 0.04
TSDF_VOLUME_MARGIN_M = 0.20
TSDF_INTEGRATION_STRIDE = 1
GEO_TARGET_POINT_STRIDE = 4
GEO_TARGET_POINT_MAX_POINTS = 2500
GEO_SWEEP_START_M = 0.00
GEO_SWEEP_END_M = 0.15
GEO_SWEEP_NUM_STEPS = 16
GEO_COLLISION_CLEARANCE_M = 0.01
GEO_SAFE_CLEARANCE_M = 0.20
GEO_ADAPTIVE_THRESHOLDS = False
GEO_ADAPTIVE_LOW_Q = 0.20
GEO_ADAPTIVE_HIGH_Q = 0.80
TSDF_TARGET_MASK_DILATE_PX = 0
GEOMETRY_CACHE_VERSION = 6
LATITUDE_DEBUG_TARGETS_DEG = (15, 30, 45, 60, 75, 90)
LATITUDE_DEBUG_SWEEP_NUM_STEPS = 32
LATITUDE_UP_AXIS = 2
TSDF_GRID_HALF_EXTENT_M = 0.12
TSDF_GRID_STEP_M = 0.01


def save_tsdf_probe_visuals(output_dir, candidate_dirs, tsdf_signed_mean, tsdf_signed_q10, tsdf_signed_flat):
	tsdf_mean_vis_path = os.path.join(output_dir, "tsdf_signed_mean_contour_vis.png")
	tsdf_signed_q10_vis_path = os.path.join(output_dir, "tsdf_signed_q10_contour_vis.png")
	tsdf_hist_path = os.path.join(output_dir, "tsdf_signed_histogram.png")

	best_mean_idx = int(np.argmax(tsdf_signed_mean))
	best_signed_q10_idx = int(np.argmax(tsdf_signed_q10))
	plot_candidate_scores_contour_like_vis(
		candidate_dirs,
		tsdf_signed_mean,
		best_mean_idx,
		"TSDF distance mean over sweep",
		tsdf_mean_vis_path,
		level_step=0.01,
		cbar_label="distance to surface (m)",
	)
	plot_candidate_scores_contour_like_vis(
		candidate_dirs,
		tsdf_signed_q10,
		best_signed_q10_idx,
		"TSDF distance q10 over sweep",
		tsdf_signed_q10_vis_path,
		level_step=0.005,
		cbar_label="distance q10 (m)",
	)

	fig = plt.figure(figsize=(9, 5.5))
	ax = fig.add_subplot(111)
	ax.hist(np.asarray(tsdf_signed_flat, dtype=np.float32), bins=80, color="#3465a4", alpha=0.9, edgecolor="white", linewidth=0.3)
	ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
	ax.set_title("TSDF distance histogram")
	ax.set_xlabel("distance to surface (m)")
	ax.set_ylabel("count")
	ax.grid(True, alpha=0.25)
	fig.tight_layout()
	fig.savefig(tsdf_hist_path, dpi=180)
	plt.close(fig)

	return {
		"tsdf_signed_mean_contour_vis_path": tsdf_mean_vis_path,
		"tsdf_signed_q10_contour_vis_path": tsdf_signed_q10_vis_path,
		"tsdf_signed_histogram_path": tsdf_hist_path,
	}


@contextlib.contextmanager
def stage(name):
	start = time.time()
	print(f"[START] {name}")
	try:
		yield
	finally:
		elapsed = time.time() - start
		print(f"[DONE] {name} ({elapsed:.2f}s)")


def main(skip_gemini):
	with stage("Load environment"):
		api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env or export it.")
	os.environ["GEMINI_API_KEY"] = api_key

	sorted_idxs = sorted(IDXS)
	set_name = "set_" + "".join(f"{i:04d}" for i in sorted_idxs)
	output_dir = os.path.join(SCRIPT_DIR, "output", "multi_view", set_name)
	idx_output_dir = os.path.join(output_dir, "idx_outputs")
	ply_output_dir = os.path.join(output_dir, "ply")
	geometry_cache_path = os.path.join(output_dir, "geometry_cache.json")
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(idx_output_dir, exist_ok=True)
	os.makedirs(ply_output_dir, exist_ok=True)

	with stage("Load pose/intrinsics metadata"):
		reached_indices = set(get_pose_reached_indices(POSE_PATH))
		pose_index_to_position = get_pose_index_to_position(POSE_PATH)
		pose_map = get_pose_index_to_transforms(POSE_PATH)
		pose_translation_error = validate_pose_convention(POSE_PATH, pose_map)
		rotate_flags = {idx: should_rotate_image_180(pose_map[idx]["c2w"]) for idx in sorted_idxs}
	for idx in sorted_idxs:
		if idx not in reached_indices:
			raise ValueError(f"Requested frame idx {idx} is not present in pose.json reached_indices.")

	target_class_id = get_target_class_id(POSE_PATH)
	background_class_id = get_background_class_id(POSE_PATH)
	candidate_dirs = fibonacci_upper_hemisphere(DISTILL_NUM_DIRECTIONS, up_axis=LATITUDE_UP_AXIS)
	worker_args = [
		(
			idx,
			target_class_id,
			output_dir,
			idx_output_dir,
			skip_gemini,
			rotate_flags[idx],
			build_image_path(RGB_DIR, idx),
			build_image_path(CLASS_DIR, idx),
			prompt_single,
		)
		for idx in sorted_idxs
	]

	with stage(f"Gemini per-view processing (N={len(sorted_idxs)})"):
		print(f"Processing {len(sorted_idxs)} images in parallel...")
		with mp.Pool(processes=min(4, len(sorted_idxs))) as pool:
			results = pool.map(process_single_index, worker_args)

	results = sorted(results, key=lambda item: item["idx"])
	vis_images = []
	gemini_scores_per_view = []
	for item in results:
		img = cv2.imread(item["dir_vis_path"], cv2.IMREAD_COLOR)
		if img is None:
			raise FileNotFoundError(f"Failed to read visualization: {item['dir_vis_path']}")
		vis_images.append(img)
		with open(item["response_json_path"], "r", encoding="utf-8") as f:
			response = json.load(f)
		gemini_scores_per_view.append(np.asarray(response["Direction scores"], dtype=np.float32))
		if item["used_cached_response"]:
			print(f"[idx {item['idx']:04d}] Reused cached Gemini JSON")
		else:
			print(f"[idx {item['idx']:04d}] Gemini response time: {item['elapsed']:.2f} seconds")
		best_reason = response.get("Best direction reason", "N/A")
		worst_reason = response.get("Worst direction reason", "N/A")
		print(f"[idx {item['idx']:04d}]  Best : {best_reason}")
		print(f"[idx {item['idx']:04d}]  Worst: {worst_reason}")

	with stage("Save 2x2 direction visualization"):
		concatenate_2x2(vis_images, os.path.join(output_dir, "direction_scores_vis_2x2.png"))

	with stage("Backproject yellow query points"):
		yellow_world_points = []
		for item in results:
			idx = item["idx"]
			u, v = item["query_point"]
			depth = cv2.imread(build_image_path(DEPTH_DIR, idx), cv2.IMREAD_UNCHANGED)
			if depth is None:
				raise FileNotFoundError(f"Failed to read depth image: {build_image_path(DEPTH_DIR, idx)}")
			if depth.dtype == np.uint16:
				depth = depth.astype(np.float32) / 1000.0
			else:
				depth = depth.astype(np.float32)
			h, w = depth.shape[:2]
			if not (0 <= u < w and 0 <= v < h):
				raise ValueError(f"Query point out of bounds for idx={idx}: {(u, v)}")
			z = float(depth[v, u])
			if z <= 0:
				raise ValueError(f"Invalid depth at yellow point for idx={idx}: {z}")
			fx, fy, cx, cy, _, _ = get_camera_intrinsics(INTRINSICS_PATH)
			p_world = unproject_pixel_to_world(u, v, z, pose_map[idx]["c2w"], fx, fy, cx, cy)
			yellow_world_points.append(p_world)

	center_world = np.mean(np.asarray(yellow_world_points, dtype=np.float32), axis=0)
	with stage("Build local 30cm point cloud"):
		local_points, local_colors = collect_points_near_center(
			sorted_idxs,
			pose_map,
			center_world,
			RADIUS_M,
			INTRINSICS_PATH,
			RGB_DIR,
			DEPTH_DIR,
		)
		local_points, local_colors = random_downsample_points(local_points, local_colors, MAX_POINTS_30CM, seed=0)
		write_ply_xyzrgb(os.path.join(ply_output_dir, "points_within_30cm.ply"), local_points, local_colors)

	geometry_cache = None
	tsdf_signed_mean = None
	tsdf_signed_q10 = None
	tsdf_signed_flat = None
	tsdf_probe_vis_paths = None
	tsdf_volume_for_debug = None
	latitude_debug_paths = []
	if skip_gemini and os.path.exists(geometry_cache_path):
		try:
			with open(geometry_cache_path, "r", encoding="utf-8") as f:
				loaded = json.load(f)
			if (
					loaded.get("cache_version") == GEOMETRY_CACHE_VERSION
					and loaded.get("idxs") == sorted_idxs
					and len(loaded.get("geometry_scores", [])) == DISTILL_NUM_DIRECTIONS
					and len(loaded.get("tsdf_signed_mean", [])) == DISTILL_NUM_DIRECTIONS
					and len(loaded.get("tsdf_signed_q10", [])) == DISTILL_NUM_DIRECTIONS
					and len(loaded.get("tsdf_signed_flat", [])) > 0
				):
				geometry_cache = loaded
				print(f"[CACHE] Reusing geometry cache: {geometry_cache_path}")
		except Exception:
			geometry_cache = None

	if geometry_cache is None:
		with stage("Collect target points for TSDF"):
			target_points_world, _target_colors = collect_target_object_points(
				sorted_idxs,
				pose_map,
				target_class_id,
				INTRINSICS_PATH,
				RGB_DIR,
				DEPTH_DIR,
				CLASS_DIR,
				GEO_TARGET_POINT_MAX_POINTS,
				GEO_TARGET_POINT_STRIDE,
			)
			volume_min, volume_max = compute_volume_bounds(target_points_world, center_world, TSDF_VOLUME_MARGIN_M)

		with stage("Build TSDF volume"):
			tsdf_volume = build_tsdf_volume(
				sorted_idxs,
				pose_map,
				INTRINSICS_PATH,
				RGB_DIR,
				DEPTH_DIR,
				CLASS_DIR,
				volume_min,
				volume_max,
				TSDF_VOXEL_SIZE_M,
				TSDF_TRUNCATION_M,
				target_class_id=target_class_id,
				background_class_id=background_class_id,
				exclude_background=True,
				mask_dilate_px=TSDF_TARGET_MASK_DILATE_PX,
				integration_stride=TSDF_INTEGRATION_STRIDE,
			)
			tsdf_volume_for_debug = tsdf_volume

		with stage("Probe TSDF raw distribution"):
			tsdf_probe = probe_tsdf_along_directions(
				candidate_dirs,
				center_world,
				tsdf_volume,
				GEO_SWEEP_START_M,
				GEO_SWEEP_END_M,
				GEO_SWEEP_NUM_STEPS,
				0.10,
			)
			tsdf_signed_mean = tsdf_probe["signed_mean"]
			tsdf_signed_q10 = tsdf_probe["signed_q10"]
			tsdf_signed_flat = tsdf_probe["signed_flat"]
			tsdf_probe_vis_paths = save_tsdf_probe_visuals(
				output_dir,
				candidate_dirs,
				tsdf_signed_mean,
				tsdf_signed_q10,
				tsdf_signed_flat,
			)

		with stage("Export latitude TSDF debug PLYs"):
			latitude_debug_paths = build_latitude_debug_plys(
				ply_output_dir,
				candidate_dirs,
				center_world,
				tsdf_volume,
				local_points,
				local_colors,
				target_latitudes_deg=LATITUDE_DEBUG_TARGETS_DEG,
				sweep_start_m=GEO_SWEEP_START_M,
				sweep_end_m=GEO_SWEEP_END_M,
				sweep_num_steps=LATITUDE_DEBUG_SWEEP_NUM_STEPS,
				query_gray=170,
				low10_sphere_radius=0.008,
				low10_sphere_points=32,
				low10_sphere_color=(0, 0, 255),
				scalar_name="tsdf",
				up_axis=LATITUDE_UP_AXIS,
			)

		with stage("Compute geometry scores"):
			   # Use robust geometry scoring: sample ~100 points from target class
			   num_sampled_points = 100
			   if target_points_world.shape[0] > num_sampled_points:
				   rng = np.random.default_rng(0)
				   choice = rng.choice(target_points_world.shape[0], size=num_sampled_points, replace=False)
				   sampled_points = target_points_world[choice]
			   else:
				   sampled_points = target_points_world
			   geometry_scores, geometry_quantiles, geometry_best_idx = compute_geometry_scores(
				   candidate_dirs,
				   tsdf_volume,
				   sweep_start_m=GEO_SWEEP_START_M,
				   sweep_end_m=GEO_SWEEP_END_M,
				   sweep_num_steps=GEO_SWEEP_NUM_STEPS,
				   coll_clearance_m=GEO_COLLISION_CLEARANCE_M,
				   safe_clearance_m=GEO_SAFE_CLEARANCE_M,
				   adaptive_thresholds=GEO_ADAPTIVE_THRESHOLDS,
				   adaptive_low_q=GEO_ADAPTIVE_LOW_Q,
				   adaptive_high_q=GEO_ADAPTIVE_HIGH_Q,
				   target_points=sampled_points,
			   )

		geometry_cache = {
			"cache_version": GEOMETRY_CACHE_VERSION,
			"idxs": sorted_idxs,
			"geometry_scores": geometry_scores.tolist(),
			"geometry_quantiles": geometry_quantiles.tolist(),
			"geometry_best_idx": int(geometry_best_idx),
			"tsdf_signed_mean": tsdf_signed_mean.tolist(),
			"tsdf_signed_q10": tsdf_signed_q10.tolist(),
			"tsdf_signed_flat": tsdf_signed_flat.tolist(),
			"tsdf_probe_vis_paths": tsdf_probe_vis_paths,
		}
		with open(geometry_cache_path, "w", encoding="utf-8") as f:
			json.dump(geometry_cache, f, indent=2)
		print(f"[CACHE] Saved geometry cache: {geometry_cache_path}")
	else:
		geometry_scores = np.asarray(geometry_cache["geometry_scores"], dtype=np.float32)
		geometry_quantiles = np.asarray(geometry_cache["geometry_quantiles"], dtype=np.float32)
		geometry_best_idx = int(geometry_cache["geometry_best_idx"])
		if "tsdf_signed_mean" in geometry_cache and "tsdf_signed_q10" in geometry_cache and "tsdf_signed_flat" in geometry_cache:
			tsdf_signed_mean = np.asarray(geometry_cache["tsdf_signed_mean"], dtype=np.float32)
			tsdf_signed_q10 = np.asarray(geometry_cache["tsdf_signed_q10"], dtype=np.float32)
			tsdf_signed_flat = np.asarray(geometry_cache["tsdf_signed_flat"], dtype=np.float32)
			with stage("Render cached TSDF raw visualization"):
				tsdf_probe_vis_paths = save_tsdf_probe_visuals(
					output_dir,
					candidate_dirs,
					tsdf_signed_mean,
					tsdf_signed_q10,
					tsdf_signed_flat,
				)
		else:
			print("[CACHE] TSDF raw probe data missing in cache. Re-run once without cache reuse to generate TSDF distribution visualizations.")

	if not latitude_debug_paths:
		with stage("Collect target points for latitude TSDF debug"):
			target_points_world, _target_colors = collect_target_object_points(
				sorted_idxs,
				pose_map,
				target_class_id,
				INTRINSICS_PATH,
				RGB_DIR,
				DEPTH_DIR,
				CLASS_DIR,
				GEO_TARGET_POINT_MAX_POINTS,
				GEO_TARGET_POINT_STRIDE,
			)
			volume_min, volume_max = compute_volume_bounds(target_points_world, center_world, TSDF_VOLUME_MARGIN_M)

		with stage("Build TSDF volume for latitude debug"):
			tsdf_volume_for_latitude = build_tsdf_volume(
				sorted_idxs,
				pose_map,
				INTRINSICS_PATH,
				RGB_DIR,
				DEPTH_DIR,
				CLASS_DIR,
				volume_min,
				volume_max,
				TSDF_VOXEL_SIZE_M,
				TSDF_TRUNCATION_M,
				target_class_id=target_class_id,
				background_class_id=background_class_id,
				exclude_background=True,
				mask_dilate_px=TSDF_TARGET_MASK_DILATE_PX,
				integration_stride=TSDF_INTEGRATION_STRIDE,
			)
			tsdf_volume_for_debug = tsdf_volume_for_latitude

		with stage("Export latitude TSDF debug PLYs"):
			latitude_debug_paths = build_latitude_debug_plys(
				ply_output_dir,
				candidate_dirs,
				center_world,
				tsdf_volume_for_latitude,
				local_points,
				local_colors,
				target_latitudes_deg=LATITUDE_DEBUG_TARGETS_DEG,
				sweep_start_m=GEO_SWEEP_START_M,
				sweep_end_m=GEO_SWEEP_END_M,
				sweep_num_steps=LATITUDE_DEBUG_SWEEP_NUM_STEPS,
				query_gray=170,
				low10_sphere_radius=0.008,
				low10_sphere_points=32,
				low10_sphere_color=(0, 0, 255),
				scalar_name="tsdf",
				up_axis=LATITUDE_UP_AXIS,
			)

	if tsdf_volume_for_debug is not None:
		with stage("Export regular TSDF grid PLY"):
			tsdf_grid_info = build_tsdf_grid_scalar_ply(
				os.path.join(ply_output_dir, "tsdf_grid_scalar.ply"),
				center_world,
				tsdf_volume_for_debug,
				local_points,
				local_colors,
				half_extent_m=TSDF_GRID_HALF_EXTENT_M,
				step_m=TSDF_GRID_STEP_M,
				grid_gray=150,
				scalar_name="tsdf",
			)
	else:
		tsdf_grid_info = None

	gemini_scores = np.mean(np.stack(gemini_scores_per_view, axis=0), axis=0)
	distill_scores_clock = np.asarray(gemini_scores, dtype=np.float32)  # shape (12,)
	with stage("Fuse distill + geometry scores"):
		final_scores, final_best_idx = compute_distilled_direction_scores(
			candidate_dirs,
			distill_scores_clock,
			geometry_scores,
			alpha=GEO_LAMBDA,
		)
	# Project clock scores onto candidate dirs for visualization
	distill_scores = np.array(
		[interpolate_clock_score(distill_scores_clock, d[:2]) for d in candidate_dirs],
		dtype=np.float32,
	)
	distill_best_idx = int(np.argmax(distill_scores))

	distill_vis_path = os.path.join(output_dir, "distill_score_vis.png")
	distill_contour_path = os.path.join(output_dir, "distill_score_contour_vis.png")
	geometry_vis_path = os.path.join(output_dir, "geometry_score_vis.png")
	geometry_contour_path = os.path.join(output_dir, "geometry_score_contour_vis.png")
	geometry_lat_path = os.path.join(output_dir, "geometry_score_vs_latitude.png")
	final_vis_path = os.path.join(output_dir, "final_score_vis.png")
	final_contour_path = os.path.join(output_dir, "final_score_contour_vis.png")

	with stage("Render score visualizations"):
			plot_candidate_scores_like_vis(candidate_dirs, distill_scores, distill_best_idx, "Distill score", distill_vis_path, up_axis=LATITUDE_UP_AXIS)
			plot_candidate_scores_contour_like_vis(candidate_dirs, distill_scores, distill_best_idx, "Distill score", distill_contour_path, up_axis=LATITUDE_UP_AXIS)
			plot_candidate_scores_like_vis(candidate_dirs, geometry_scores, geometry_best_idx, "Geometry score", geometry_vis_path, up_axis=LATITUDE_UP_AXIS)
			plot_candidate_scores_contour_like_vis(candidate_dirs, geometry_scores, geometry_best_idx, "Geometry score", geometry_contour_path, up_axis=LATITUDE_UP_AXIS)
			plot_score_vs_latitude(candidate_dirs, geometry_scores, geometry_lat_path, "Geometry score vs latitude", up_axis=LATITUDE_UP_AXIS)
			plot_candidate_scores_like_vis(candidate_dirs, final_scores, final_best_idx, "Final score", final_vis_path, up_axis=LATITUDE_UP_AXIS)
			plot_candidate_scores_contour_like_vis(candidate_dirs, final_scores, final_best_idx, "Final score", final_contour_path, up_axis=LATITUDE_UP_AXIS)

	with stage("Export scoring PLYs"):
		distill_ply_paths = build_scene_visualization_plys(
			ply_output_dir,
			"distill_scene",
			local_points,
			local_colors,
			candidate_dirs,
			distill_scores,
			center_world,
			distill_best_idx,
			sphere_radius=0.15,
			arrow_length=0.18,
		)
		geometry_ply_paths = build_scene_visualization_plys(
			ply_output_dir,
			"geometry_scene",
			local_points,
			local_colors,
			candidate_dirs,
			geometry_scores,
			center_world,
			geometry_best_idx,
			sphere_radius=0.15,
			arrow_length=0.18,
		)
		final_ply_paths = build_scene_visualization_plys(
			ply_output_dir,
			"final_scene",
			local_points,
			local_colors,
			candidate_dirs,
			final_scores,
			center_world,
			final_best_idx,
			sphere_radius=0.15,
			arrow_length=0.18,
		)

	distill_json_path = os.path.join(output_dir, "distill_3d_direction.json")
	geometry_json_path = os.path.join(output_dir, "geometry_3d_direction.json")
	final_json_path = os.path.join(output_dir, "final_3d_direction.json")

	with stage("Write result JSONs"):
		with open(distill_json_path, "w", encoding="utf-8") as f:
			json.dump(
				{
					"best_idx": distill_best_idx,
					"best_direction": candidate_dirs[distill_best_idx].tolist(),
					"scores": distill_scores.tolist(),
					"center_world": center_world.tolist(),
					"yellow_world_points": [p.tolist() for p in yellow_world_points],
					"pose_translation_error": pose_translation_error,
				},
				f,
				indent=2,
			)

		with open(geometry_json_path, "w", encoding="utf-8") as f:
			json.dump(
				{
					"best_idx": geometry_best_idx,
					"best_direction": candidate_dirs[geometry_best_idx].tolist(),
					"scores": geometry_scores.tolist(),
					"quantiles": geometry_quantiles.tolist(),
					"geometry_method": "tsdf_quantile10_over_swept_points",
					"geometry_score_vis_path": geometry_vis_path,
					"geometry_score_contour_vis_path": geometry_contour_path,
					"geometry_score_vs_latitude_path": geometry_lat_path,
				},
				f,
				indent=2,
			)

		with open(final_json_path, "w", encoding="utf-8") as f:
			json.dump(
				{
					"best_idx": final_best_idx,
					"best_direction": candidate_dirs[final_best_idx].tolist(),
					"scores": final_scores.tolist(),
					"distill_score_vis_path": distill_vis_path,
					"distill_score_contour_vis_path": distill_contour_path,
					"geometry_score_vis_path": geometry_vis_path,
					"geometry_score_contour_vis_path": geometry_contour_path,
					"final_score_vis_path": final_vis_path,
					"final_score_contour_vis_path": final_contour_path,
				},
				f,
				indent=2,
			)

	print("Done.")
	print(f"- {distill_vis_path}")
	print(f"- {geometry_vis_path}")
	print(f"- {final_vis_path}")
	print(f"- {distill_ply_paths['rgb']}")
	print(f"- {distill_ply_paths['gray']}")
	print(f"- {geometry_ply_paths['rgb']}")
	print(f"- {geometry_ply_paths['gray']}")
	print(f"- {final_ply_paths['rgb']}")
	print(f"- {final_ply_paths['gray']}")
	if tsdf_grid_info is not None:
		print(f"- {tsdf_grid_info['path']} (N={tsdf_grid_info['grid_point_count']}, tsdf_min={tsdf_grid_info['tsdf_min']:.4f}, tsdf_max={tsdf_grid_info['tsdf_max']:.4f})")
	for item in latitude_debug_paths:
		print(f"- {item['path']} (target lat {item['target_latitude']:.0f}, selected idx {item['selected_idx']}, actual lat {item['actual_latitude']:.2f})")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Multi-view Gemini + point cloud pipeline")
	parser.add_argument(
		"--skip-gemini",
		action="store_true",
		help="If response JSON already exists for an index, skip Gemini API call and reuse cached JSON.",
	)
	args = parser.parse_args()
	main(args.skip_gemini)
