import json
import math
import os
import time

import cv2
from dotenv import load_dotenv
import numpy as np
from google import genai
from google.genai import types

from prompt import prompt_single

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
	raise RuntimeError("GEMINI_API_KEY is not set. Put it in a .env file or export it.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
load_dotenv()

dataset_dir = os.path.join(SCRIPT_DIR, "ex_dataset", "scene_000")
pose_path = os.path.join(dataset_dir, "views", "pose.json")
rgb_dir = os.path.join(dataset_dir, "views", "rgb")
depth_dir = os.path.join(dataset_dir, "views", "depth")
class_dir = os.path.join(dataset_dir, "views", "class")

idx = 39


def build_image_path(folder, index):
	return os.path.join(folder, f"{index:04d}.png")


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

	cv2.circle(
		drawn,
		(x, y),
		radius=radius,
		color=black,
		thickness=-1,
		lineType=cv2.LINE_AA,
	)

	inner_radius = max(1, radius - outline)
	cv2.circle(
		drawn,
		(x, y),
		radius=inner_radius,
		color=yellow,
		thickness=-1,
		lineType=cv2.LINE_AA,
	)
	return drawn


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
		model="gemini-2.5-pro",
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


def score_to_bgr(score):
	score = float(np.clip(score, 0.0, 1.0))
	blue = np.array([255, 0, 0], dtype=np.float32)
	red = np.array([0, 0, 255], dtype=np.float32)
	color = (1.0 - score) * blue + score * red
	return tuple(int(c) for c in color)


def draw_direction_arrows_on_image(
	image,
	direction_scores,
	center,
	arrow_length=70,
	circle_radius=90,
	thickness=3,
	tip_length=0.22,
):
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

		cv2.arrowedLine(
			drawn,
			(x1, y1),
			(x2, y2),
			color=(0, 0, 0),
			thickness=thickness + 2,
			line_type=cv2.LINE_AA,
			tipLength=tip_length,
		)
		cv2.arrowedLine(
			drawn,
			(x1, y1),
			(x2, y2),
			color=color,
			thickness=thickness,
			line_type=cv2.LINE_AA,
			tipLength=tip_length,
		)

		score_text = f"{score:.2f}"
		tx = int(cx + (circle_radius + arrow_length + 42) * math.cos(angle_rad))
		ty = int(cy + (circle_radius + arrow_length + 42) * math.sin(angle_rad))
		(tw, th), baseline = cv2.getTextSize(
			score_text,
			font,
			font_scale,
			text_thickness,
		)

		x_min = max(0, tx - tw // 2 - pad_x)
		x_max = min(w - 1, tx + tw // 2 + pad_x)
		y_min = max(0, ty - th // 2 - pad_y)
		y_max = min(h - 1, ty + th // 2 + pad_y + baseline)

		cv2.rectangle(drawn, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
		cv2.rectangle(
			drawn,
			(x_min + 1, y_min + 1),
			(x_max - 1, y_max - 1),
			(255, 255, 255),
			thickness=-1,
		)

		cv2.putText(
			drawn,
			score_text,
			(tx - tw // 2, ty + th // 2),
			font,
			font_scale,
			color,
			text_thickness,
			cv2.LINE_AA,
		)

	return drawn


if __name__ == "__main__":
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env or export it.")

	os.environ["GEMINI_API_KEY"] = api_key

	rgb_path = build_image_path(rgb_dir, idx)
	class_path = build_image_path(class_dir, idx)

	target_class_id = get_target_class_id(pose_path)
	class_map = read_class_map(class_path)
	query_point = get_query_point_from_class_map(class_map, target_class_id)

	marked_image = draw_yellow_circle(rgb_path, query_point)

	output_dir = os.path.join(SCRIPT_DIR, "output", "single_view", f"idx_{idx:04d}")
	os.makedirs(output_dir, exist_ok=True)

	input_vis_path = os.path.join(output_dir, "input_with_query.png")
	cv2.imwrite(input_vis_path, marked_image)

	print("Getting Gemini response...")
	start_time = time.time()
	response = get_gemini_response(prompt_single, marked_image)
	end_time = time.time()
	print(f"Gemini response time: {end_time - start_time:.2f} seconds")

	raw_text_path = os.path.join(output_dir, "gemini_response_raw.txt")
	with open(raw_text_path, "w", encoding="utf-8") as f:
		f.write(response.text)

	response_dict = parse_single_view_response(response.text)
	response_json_path = os.path.join(output_dir, "gemini_response.json")
	with open(response_json_path, "w", encoding="utf-8") as f:
		json.dump(response_dict, f, indent=2)

	dir_scores = response_dict["Direction scores"]
	dir_vis = draw_direction_arrows_on_image(
		marked_image,
		dir_scores,
		center=query_point,
	)
	dir_vis_path = os.path.join(output_dir, "direction_scores_vis.png")
	cv2.imwrite(dir_vis_path, dir_vis)

	print("Saved files:")
	print(f"- {input_vis_path}")
	print(f"- {raw_text_path}")
	print(f"- {response_json_path}")
	print(f"- {dir_vis_path}")

