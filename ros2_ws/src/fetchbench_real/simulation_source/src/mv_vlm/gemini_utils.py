import json
import os
import time

import cv2
from google import genai
from google.genai import types

from data_utils import draw_yellow_circle, get_query_point_from_class_map, read_class_map, rotate_image_180, rotate_point_180
from viz_utils import draw_direction_arrows_on_image


gemini_version = "gemini-3.1-pro-preview"


def get_gemini_response(prompt_text, image_bgr):
	client = genai.Client()
	frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	success, encoded = cv2.imencode(".png", frame_rgb)
	if not success:
		raise ValueError("Failed to encode frame as PNG.")

	parts = [
		prompt_text,
		types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/png"),
	]

	response = client.models.generate_content(
		model=gemini_version,
		contents=parts,
	)
	return response


def extract_json_text(text):
	stripped = text.strip()
	if stripped.startswith("```"):
		lines = stripped.splitlines()
		if len(lines) >= 3 and lines[-1].startswith("```"):
			stripped = "\n".join(lines[1:-1]).strip()
			if stripped.lower().startswith("json"):
				stripped = stripped[4:].strip()

	left = stripped.find("{")
	right = stripped.rfind("}")
	if left == -1 or right == -1 or right <= left:
		raise ValueError("Could not find JSON object in Gemini response text.")
	return stripped[left : right + 1]


def parse_single_view_response(text):
	response = json.loads(extract_json_text(text))

	if "Direction scores" in response:
		scores = response["Direction scores"]
	elif "direction_scores" in response:
		scores = response["direction_scores"]
	else:
		raise KeyError("Direction scores field not found in Gemini response.")

	if not isinstance(scores, list) or len(scores) != 12:
		raise ValueError("Direction scores must be a list of 12 values.")

	response["Direction scores"] = [float(s) for s in scores]

	return response


def process_single_index(args):
	idx, target_class_id, output_dir, idx_output_dir, skip_gemini, rotate_180, rgb_path, class_path, prompt_text = args

	class_map = read_class_map(class_path)
	query_point = get_query_point_from_class_map(class_map, target_class_id)
	marked_image = draw_yellow_circle(rgb_path, query_point)
	if rotate_180:
		marked_image = rotate_image_180(marked_image)

	h, w = class_map.shape[:2]
	display_query_point = rotate_point_180(query_point, w, h) if rotate_180 else query_point

	idx_prefix = f"idx_{idx:04d}"
	input_vis_path = os.path.join(idx_output_dir, f"{idx_prefix}_input_with_query.png")
	raw_text_path = os.path.join(idx_output_dir, f"{idx_prefix}_gemini_response_raw.txt")
	response_json_path = os.path.join(idx_output_dir, f"{idx_prefix}_gemini_response.json")
	legacy_response_json_path = os.path.join(output_dir, f"{idx_prefix}_gemini_response.json")
	dir_vis_path = os.path.join(output_dir, f"{idx_prefix}_direction_scores_vis.png")

	os.makedirs(idx_output_dir, exist_ok=True)
	cv2.imwrite(input_vis_path, marked_image)

	used_cached_response = False
	if skip_gemini:
		try:
			cached_json_path = response_json_path
			if not os.path.exists(cached_json_path) and os.path.exists(legacy_response_json_path):
				cached_json_path = legacy_response_json_path

			with open(cached_json_path, "r", encoding="utf-8") as f:
				response_dict = json.load(f)
			if "Direction scores" not in response_dict and "direction_scores" in response_dict:
				response_dict["Direction scores"] = response_dict["direction_scores"]
			if not isinstance(response_dict.get("Direction scores", None), list):
				raise ValueError("Cached JSON does not contain a valid Direction scores list.")
			cached_rotation = bool(response_dict.get("image_rotated_180", False))
			if cached_rotation != bool(rotate_180):
				raise ValueError("Cached JSON rotation flag does not match current pose-based rotation.")
			if cached_json_path != response_json_path:
				with open(response_json_path, "w", encoding="utf-8") as f:
					json.dump(response_dict, f, indent=2)
			used_cached_response = True
		except Exception as e:
			raise RuntimeError(
				f"--skip-gemini is enabled but cached response is unavailable/invalid for idx {idx:04d}: {e}"
			) from e

	if not used_cached_response:
		print(f"[idx {idx:04d}] Getting Gemini response...")
		start_time = time.time()
		response = get_gemini_response(prompt_text, marked_image)
		end_time = time.time()
		with open(raw_text_path, "w", encoding="utf-8") as f:
			f.write(response.text)
		response_dict = parse_single_view_response(response.text)
		response_dict["image_rotated_180"] = bool(rotate_180)
		with open(response_json_path, "w", encoding="utf-8") as f:
			json.dump(response_dict, f, indent=2)
	else:
		start_time = time.time()
		end_time = start_time

	dir_scores = response_dict["Direction scores"]
	dir_vis = draw_direction_arrows_on_image(marked_image, dir_scores, center=display_query_point)
	cv2.imwrite(dir_vis_path, dir_vis)

	return {
		"idx": idx,
		"query_point": [int(query_point[0]), int(query_point[1])],
		"display_query_point": [int(display_query_point[0]), int(display_query_point[1])],
		"elapsed": end_time - start_time,
		"used_cached_response": used_cached_response,
		"image_rotated_180": bool(rotate_180),
		"input_vis_path": input_vis_path,
		"raw_text_path": raw_text_path,
		"response_json_path": response_json_path,
		"dir_vis_path": dir_vis_path,
	}
