import math
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def projection_components(direction, up_axis=2):
	direction = np.asarray(direction, dtype=np.float64)
	up_axis = int(up_axis)
	if up_axis == 2:
		return float(direction[0]), float(direction[1]), "x", "y", "z"
	if up_axis == 1:
		return float(direction[0]), float(direction[2]), "x", "z", "y"
	if up_axis == 0:
		return float(direction[1]), float(direction[2]), "y", "z", "x"
	raise ValueError("up_axis must be 0, 1, or 2")


def score_to_bgr(score):
	score = float(np.clip(score, 0.0, 1.0))
	blue = np.array([255, 0, 0], dtype=np.float32)
	red = np.array([0, 0, 255], dtype=np.float32)
	color = (1.0 - score) * blue + score * red
	return tuple(int(c) for c in color)


def direction_to_lon_lat_deg(direction, up_axis=2):
	direction = np.asarray(direction, dtype=np.float64)
	if direction.shape != (3,):
		raise ValueError("direction must have shape (3,)")
	up_axis = int(up_axis)
	if up_axis == 2:
		horizontal = (direction[0], direction[1])
		up_value = direction[2]
	elif up_axis == 1:
		horizontal = (direction[0], direction[2])
		up_value = direction[1]
	elif up_axis == 0:
		horizontal = (direction[1], direction[2])
		up_value = direction[0]
	else:
		raise ValueError("up_axis must be 0, 1, or 2")
	lon = math.degrees(math.atan2(horizontal[1], horizontal[0]))
	lat = math.degrees(math.asin(max(-1.0, min(1.0, float(up_value)))))
	return lon, lat


def draw_direction_arrows_on_image(image, direction_scores, center, arrow_length=70, circle_radius=90, thickness=3, tip_length=0.22):
	drawn = image.copy()
	h, w = drawn.shape[:2]
	cx, cy = center

	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.9
	text_thickness = 2
	pad_x = 8
	pad_y = 6

	for i in range(12):
		score = float(direction_scores[i])
		angle_deg = -60 + i * 30
		angle_rad = math.radians(angle_deg)

		x1 = int(cx + circle_radius * math.cos(angle_rad))
		y1 = int(cy + circle_radius * math.sin(angle_rad))
		x2 = int(cx + (circle_radius + arrow_length) * math.cos(angle_rad))
		y2 = int(cy + (circle_radius + arrow_length) * math.sin(angle_rad))
		color = score_to_bgr(score)

		cv2.arrowedLine(drawn, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=thickness + 2, line_type=cv2.LINE_AA, tipLength=tip_length)
		cv2.arrowedLine(drawn, (x1, y1), (x2, y2), color=color, thickness=thickness, line_type=cv2.LINE_AA, tipLength=tip_length)

		score_text = f"{score:.2f}"
		tx = int(cx + (circle_radius + arrow_length + 42) * math.cos(angle_rad))
		ty = int(cy + (circle_radius + arrow_length + 42) * math.sin(angle_rad))
		(tw, th), baseline = cv2.getTextSize(score_text, font, font_scale, text_thickness)

		x_min = max(0, tx - tw // 2 - pad_x)
		x_max = min(w - 1, tx + tw // 2 + pad_x)
		y_min = max(0, ty - th // 2 - pad_y)
		y_max = min(h - 1, ty + th // 2 + pad_y + baseline)

		cv2.rectangle(drawn, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
		cv2.rectangle(drawn, (x_min + 1, y_min + 1), (x_max - 1, y_max - 1), (255, 255, 255), thickness=-1)
		cv2.putText(drawn, score_text, (tx - tw // 2, ty + th // 2), font, font_scale, color, text_thickness, cv2.LINE_AA)

	return drawn


def plot_candidate_scores_like_vis(candidate_dirs, candidate_scores, best_idx, title, out_png, cbar_label="score", up_axis=2, best_edge_color="white"):
	dirs = np.asarray(candidate_dirs, dtype=np.float64)
	scores = np.asarray(candidate_scores, dtype=np.float64)
	if dirs.ndim != 2 or dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	if scores.ndim != 1 or scores.shape[0] != dirs.shape[0]:
		raise ValueError("candidate_scores must have shape (N,)")

	norms = np.linalg.norm(dirs, axis=1, keepdims=True)
	dirs = dirs / np.maximum(norms, 1e-12)
	lons = np.array([direction_to_lon_lat_deg(d, up_axis=up_axis)[0] for d in dirs], dtype=np.float64)
	lats = np.array([direction_to_lon_lat_deg(d, up_axis=up_axis)[1] for d in dirs], dtype=np.float64)
	proj_base_x = np.array([projection_components(d, up_axis=up_axis)[0] for d in dirs], dtype=np.float64)
	proj_base_y = np.array([projection_components(d, up_axis=up_axis)[1] for d in dirs], dtype=np.float64)
	# Upper-hemisphere panel convention: x-axis uses y-values, y-axis uses x-values.
	proj_x = proj_base_y
	proj_y = proj_base_x
	axis_x_label = projection_components(dirs[0], up_axis=up_axis)[3]
	axis_y_label = projection_components(dirs[0], up_axis=up_axis)[2]
	up_axis_label = projection_components(dirs[0], up_axis=up_axis)[4]
	vals = (scores - float(np.min(scores))) / (float(np.max(scores) - np.min(scores)) + 1e-8)

	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
	ax_ll = fig.add_subplot(gs[0, 0])
	ax_xy = fig.add_subplot(gs[0, 1])
	fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.10, wspace=0.28)

	marker_sizes = np.full(vals.shape[0], 18.0, dtype=np.float64)
	sc1 = ax_ll.scatter(lons, lats, c=vals, s=marker_sizes, cmap="plasma", vmin=0.0, vmax=max(float(np.max(vals)), 0.1), edgecolors="black", linewidths=0.25)
	ax_ll.set_title("Direction (Longitude/Latitude)")
	ax_ll.set_xlabel("Longitude (deg)")
	ax_ll.set_ylabel("Latitude (deg)")
	ax_ll.set_xlim(-180, 180)
	ax_ll.set_ylim(0, 90)
	ax_ll.grid(True, alpha=0.3)

	ax_xy.scatter(proj_x, proj_y, c=vals, s=marker_sizes, cmap="plasma", vmin=0.0, vmax=max(float(np.max(vals)), 0.1), edgecolors="black", linewidths=0.25)
	ax_xy.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1.0))
	ax_xy.set_aspect("equal", "box")
	ax_xy.set_xlim(-1.05, 1.05)
	ax_xy.set_ylim(-1.05, 1.05)
	ax_xy.invert_xaxis()  # Left side is positive on x-axis.
	ax_xy.set_xlabel(axis_x_label)
	ax_xy.set_ylabel(axis_y_label)
	ax_xy.set_title(f"Upper Hemisphere Projection ({up_axis_label} > 0)")
	ax_xy.grid(True, alpha=0.3)

	if 0 <= int(best_idx) < dirs.shape[0]:
		best_dir = dirs[int(best_idx)]
		best_lon, best_lat = direction_to_lon_lat_deg(best_dir, up_axis=up_axis)
		ax_ll.scatter([best_lon], [best_lat], s=220, facecolors="none", edgecolors=best_edge_color, linewidths=1.8)
		base_x, base_y, _, _, _ = projection_components(best_dir, up_axis=up_axis)
		best_x, best_y = base_y, base_x
		ax_xy.scatter([best_x], [best_y], s=220, facecolors="none", edgecolors=best_edge_color, linewidths=1.8)

	cax = fig.add_axes([0.492, 0.13, 0.016, 0.70])
	cbar = fig.colorbar(sc1, cax=cax)
	cbar.set_label(cbar_label)

	fig.suptitle(title)
	os.makedirs(os.path.dirname(out_png), exist_ok=True)
	fig.savefig(out_png, dpi=180)
	plt.close(fig)


def plot_best_directions_comparison(candidate_dirs, background_scores, best_specs, title, out_png, cbar_label="score", up_axis=2):
	dirs = np.asarray(candidate_dirs, dtype=np.float64)
	scores = np.asarray(background_scores, dtype=np.float64)
	if dirs.ndim != 2 or dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	if scores.ndim != 1 or scores.shape[0] != dirs.shape[0]:
		raise ValueError("background_scores must have shape (N,)")

	norms = np.linalg.norm(dirs, axis=1, keepdims=True)
	dirs = dirs / np.maximum(norms, 1e-12)
	lons = np.array([direction_to_lon_lat_deg(d, up_axis=up_axis)[0] for d in dirs], dtype=np.float64)
	lats = np.array([direction_to_lon_lat_deg(d, up_axis=up_axis)[1] for d in dirs], dtype=np.float64)
	proj_base_x = np.array([projection_components(d, up_axis=up_axis)[0] for d in dirs], dtype=np.float64)
	proj_base_y = np.array([projection_components(d, up_axis=up_axis)[1] for d in dirs], dtype=np.float64)
	# Upper-hemisphere panel convention: x-axis uses y-values, y-axis uses x-values.
	proj_x = proj_base_y
	proj_y = proj_base_x
	axis_x_label = projection_components(dirs[0], up_axis=up_axis)[3]
	axis_y_label = projection_components(dirs[0], up_axis=up_axis)[2]
	up_axis_label = projection_components(dirs[0], up_axis=up_axis)[4]
	vals = (scores - float(np.min(scores))) / (float(np.max(scores) - np.min(scores)) + 1e-8)

	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
	ax_ll = fig.add_subplot(gs[0, 0])
	ax_xy = fig.add_subplot(gs[0, 1])
	fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.10, wspace=0.28)

	marker_sizes = np.full(vals.shape[0], 18.0, dtype=np.float64)
	sc1 = ax_ll.scatter(lons, lats, c=vals, s=marker_sizes, cmap="plasma", vmin=0.0, vmax=max(float(np.max(vals)), 0.1), edgecolors="black", linewidths=0.25)
	ax_ll.set_title("Direction (Longitude/Latitude)")
	ax_ll.set_xlabel("Longitude (deg)")
	ax_ll.set_ylabel("Latitude (deg)")
	ax_ll.set_xlim(-180, 180)
	ax_ll.set_ylim(0, 90)
	ax_ll.grid(True, alpha=0.3)

	ax_xy.scatter(proj_x, proj_y, c=vals, s=marker_sizes, cmap="plasma", vmin=0.0, vmax=max(float(np.max(vals)), 0.1), edgecolors="black", linewidths=0.25)
	ax_xy.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1.0))
	ax_xy.set_aspect("equal", "box")
	ax_xy.set_xlim(-1.05, 1.05)
	ax_xy.set_ylim(-1.05, 1.05)
	ax_xy.invert_xaxis()  # Left side is positive on x-axis.
	ax_xy.set_xlabel(axis_x_label)
	ax_xy.set_ylabel(axis_y_label)
	ax_xy.set_title(f"Upper Hemisphere Projection ({up_axis_label} > 0)")
	ax_xy.grid(True, alpha=0.3)

	legend_handles = []
	for spec in best_specs:
		label = str(spec["label"])
		best_idx = int(spec["idx"])
		color = spec.get("color", "white")
		if not (0 <= best_idx < dirs.shape[0]):
			continue
		best_dir = dirs[best_idx]
		best_lon, best_lat = direction_to_lon_lat_deg(best_dir, up_axis=up_axis)
		base_x, base_y, _, _, _ = projection_components(best_dir, up_axis=up_axis)
		best_x, best_y = base_y, base_x
		ax_ll.scatter([best_lon], [best_lat], s=240, facecolors="none", edgecolors=color, linewidths=2.2, label=label)
		ax_xy.scatter([best_x], [best_y], s=240, facecolors="none", edgecolors=color, linewidths=2.2, label=label)
		ax_ll.annotate(label, (best_lon, best_lat), xytext=(6, 6), textcoords="offset points", color=color, fontsize=9, weight="bold")
		ax_xy.annotate(label, (best_x, best_y), xytext=(6, 6), textcoords="offset points", color=color, fontsize=9, weight="bold")
		legend_handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=color, markeredgewidth=2.0, markersize=9, label=label))

	if legend_handles:
		ax_xy.legend(handles=legend_handles, loc="lower left", framealpha=0.9)

	cax = fig.add_axes([0.492, 0.13, 0.016, 0.70])
	cbar = fig.colorbar(sc1, cax=cax)
	cbar.set_label(cbar_label)

	fig.suptitle(title)
	os.makedirs(os.path.dirname(out_png), exist_ok=True)
	fig.savefig(out_png, dpi=180)
	plt.close(fig)


def plot_candidate_scores_contour_like_vis(
	candidate_dirs,
	candidate_scores,
	best_idx,
	title,
	out_png,
	level_step=0.2,
	cbar_label="score",
	up_axis=2,
	marker_direction=None,
	marker_color="blue",
):
	dirs = np.asarray(candidate_dirs, dtype=np.float64)
	scores = np.asarray(candidate_scores, dtype=np.float64)
	if dirs.ndim != 2 or dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	if scores.ndim != 1 or scores.shape[0] != dirs.shape[0]:
		raise ValueError("candidate_scores must have shape (N,)")

	norms = np.linalg.norm(dirs, axis=1, keepdims=True)
	dirs = dirs / np.maximum(norms, 1e-12)
	lons = np.array([direction_to_lon_lat_deg(d, up_axis=up_axis)[0] for d in dirs], dtype=np.float64)
	lats = np.array([direction_to_lon_lat_deg(d, up_axis=up_axis)[1] for d in dirs], dtype=np.float64)
	proj_base_x = np.array([projection_components(d, up_axis=up_axis)[0] for d in dirs], dtype=np.float64)
	proj_base_y = np.array([projection_components(d, up_axis=up_axis)[1] for d in dirs], dtype=np.float64)
	# Upper-hemisphere panel convention: x-axis uses y-values, y-axis uses x-values.
	proj_x = proj_base_y
	proj_y = proj_base_x
	axis_x_label = projection_components(dirs[0], up_axis=up_axis)[3]
	axis_y_label = projection_components(dirs[0], up_axis=up_axis)[2]
	up_axis_label = projection_components(dirs[0], up_axis=up_axis)[4]
	vals = np.asarray(scores, dtype=np.float64)

	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
	ax_ll = fig.add_subplot(gs[0, 0])
	ax_xy = fig.add_subplot(gs[0, 1])
	fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.10, wspace=0.28)

	level_min = float(np.floor(np.min(vals) / level_step) * level_step)
	level_max = float(np.ceil(np.max(vals) / level_step) * level_step)
	if level_max <= level_min:
		level_max = level_min + level_step
	levels = np.arange(level_min, level_max + 0.5 * level_step, level_step, dtype=np.float64)
	if levels.size < 2:
		levels = np.array([level_min, level_min + level_step], dtype=np.float64)

	tri_ll = mtri.Triangulation(lons, lats)
	tri_xy = mtri.Triangulation(proj_x, proj_y)

	cont_ll = ax_ll.tricontourf(tri_ll, vals, levels=levels, cmap="plasma", alpha=0.95)
	ax_ll.tricontour(tri_ll, vals, levels=levels, colors="black", linewidths=0.45, alpha=0.65)
	ax_ll.scatter(lons, lats, s=8.0, c="white", alpha=0.75, linewidths=0)
	ax_ll.set_title("Direction (Longitude/Latitude)")
	ax_ll.set_xlabel("Longitude (deg)")
	ax_ll.set_ylabel("Latitude (deg)")
	ax_ll.set_xlim(-180, 180)
	ax_ll.set_ylim(0, 90)
	ax_ll.grid(True, alpha=0.3)

	ax_xy.tricontourf(tri_xy, vals, levels=levels, cmap="plasma", alpha=0.95)
	ax_xy.tricontour(tri_xy, vals, levels=levels, colors="black", linewidths=0.45, alpha=0.65)
	ax_xy.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1.0))
	ax_xy.set_aspect("equal", "box")
	ax_xy.set_xlim(-1.05, 1.05)
	ax_xy.set_ylim(-1.05, 1.05)
	ax_xy.invert_xaxis()  # Left side is positive on x-axis.
	ax_xy.set_xlabel(axis_x_label)
	ax_xy.set_ylabel(axis_y_label)
	ax_xy.set_title(f"Upper Hemisphere Projection ({up_axis_label} > 0)")
	ax_xy.grid(True, alpha=0.3)

	if 0 <= int(best_idx) < dirs.shape[0]:
		best_dir = dirs[int(best_idx)]
		best_lon, best_lat = direction_to_lon_lat_deg(best_dir, up_axis=up_axis)
		ax_ll.scatter([best_lon], [best_lat], s=240, facecolors="none", edgecolors="white", linewidths=1.8)
		base_x, base_y, _, _, _ = projection_components(best_dir, up_axis=up_axis)
		best_x, best_y = base_y, base_x
		ax_xy.scatter([best_x], [best_y], s=240, facecolors="none", edgecolors="white", linewidths=1.8)

	if marker_direction is not None:
		m_dir = np.asarray(marker_direction, dtype=np.float64).reshape(-1)
		if m_dir.shape[0] == 3:
			norm = float(np.linalg.norm(m_dir))
			if norm > 1e-12:
				m_dir = m_dir / norm
				m_lon, m_lat = direction_to_lon_lat_deg(m_dir, up_axis=up_axis)
				m_base_x, m_base_y, _, _, _ = projection_components(m_dir, up_axis=up_axis)
				m_x, m_y = m_base_y, m_base_x
				ax_ll.scatter([m_lon], [m_lat], s=180, facecolors="none", edgecolors=marker_color, linewidths=2.2)
				ax_xy.scatter([m_x], [m_y], s=180, facecolors="none", edgecolors=marker_color, linewidths=2.2)

	cax = fig.add_axes([0.492, 0.13, 0.016, 0.70])
	cbar = fig.colorbar(cont_ll, cax=cax)
	cbar.set_label(cbar_label)

	fig.suptitle(title)
	os.makedirs(os.path.dirname(out_png), exist_ok=True)
	fig.savefig(out_png, dpi=180)
	plt.close(fig)


def plot_score_vs_latitude(candidate_dirs, candidate_scores, out_png, title, up_axis=2):
	dirs = np.asarray(candidate_dirs, dtype=np.float64)
	scores = np.asarray(candidate_scores, dtype=np.float64)
	if dirs.ndim != 2 or dirs.shape[1] != 3:
		raise ValueError("candidate_dirs must have shape (N, 3)")
	if scores.ndim != 1 or scores.shape[0] != dirs.shape[0]:
		raise ValueError("candidate_scores must have shape (N,)")

	norms = np.linalg.norm(dirs, axis=1, keepdims=True)
	dirs = dirs / np.maximum(norms, 1e-12)
	up_axis = int(up_axis)
	if up_axis == 2:
		lat_component = dirs[:, 2]
	elif up_axis == 1:
		lat_component = dirs[:, 1]
	elif up_axis == 0:
		lat_component = dirs[:, 0]
	else:
		raise ValueError("up_axis must be 0, 1, or 2")
	lats = np.degrees(np.arcsin(np.clip(lat_component, -1.0, 1.0)))

	fig = plt.figure(figsize=(9, 5.5))
	ax = fig.add_subplot(111)
	ax.scatter(lats, scores, s=12, c=scores, cmap="plasma", vmin=0.0, vmax=1.0, alpha=0.9, linewidths=0)

	bins = np.linspace(0.0, 90.0, 10)
	bin_centers = 0.5 * (bins[:-1] + bins[1:])
	bin_means = []
	for i in range(len(bins) - 1):
		m = (lats >= bins[i]) & (lats < bins[i + 1])
		if np.any(m):
			bin_means.append(float(np.mean(scores[m])))
		else:
			bin_means.append(np.nan)
	ax.plot(bin_centers, np.asarray(bin_means, dtype=np.float64), color="black", linewidth=2.0, marker="o", markersize=4)

	ax.set_xlim(0.0, 90.0)
	ax.set_ylim(0.0, 1.02)
	ax.set_xlabel("Latitude (deg)")
	ax.set_ylabel("Score")
	ax.grid(True, alpha=0.3)
	ax.set_title("Score vs Latitude")

	fig.suptitle(title)
	os.makedirs(os.path.dirname(out_png), exist_ok=True)
	fig.savefig(out_png, dpi=180)
	plt.close(fig)


def concatenate_2x2(images, save_path, padding=12, bg_color=(0, 0, 0)):
	if len(images) != 4:
		raise ValueError("concatenate_2x2 expects exactly 4 images.")

	heights = [img.shape[0] for img in images]
	widths = [img.shape[1] for img in images]
	cell_h = max(heights)
	cell_w = max(widths)

	rows, cols = 2, 2
	canvas_h = rows * cell_h + (rows + 1) * padding
	canvas_w = cols * cell_w + (cols + 1) * padding
	canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

	for i, img in enumerate(images):
		r = i // cols
		c = i % cols
		y0 = padding + r * (cell_h + padding)
		x0 = padding + c * (cell_w + padding)
		h, w = img.shape[:2]
		y_off = (cell_h - h) // 2
		x_off = (cell_w - w) // 2
		canvas[y0 + y_off : y0 + y_off + h, x0 + x_off : x0 + x_off + w] = img

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	cv2.imwrite(save_path, canvas)
	return canvas
