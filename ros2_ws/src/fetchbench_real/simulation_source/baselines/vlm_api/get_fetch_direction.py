#!/isaac-sim/kit/python/bin/python3
import argparse
import base64
import json
import math
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from prompt import prompt_single


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv()

SUPPORTED_MODELS = [
	"gpt-5.4",
	"gpt-5.5",
	"gemini-3.1-pro-preview",
	"gemini-robotics-er-1.6-preview",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Get fetch direction from OpenAI/Gemini VLM")
	parser.add_argument("--scene", required=True, help="Scene id, e.g. 01")
	parser.add_argument("--scene-num", required=True, help="Scene number, e.g. 000")
	parser.add_argument(
		"--model",
		required=True,
		choices=["all"] + SUPPORTED_MODELS,
		help="Model name or all",
	)
	return parser.parse_args()


def build_scene_paths(scene: str, scene_num: str) -> dict[str, Path]:
	scene_dir = DATASET_ROOT / f"{scene}_robot" / f"scene_{scene_num}"
	bev_dir = scene_dir / "bev"
	return {
		"scene_dir": scene_dir,
		"bev_dir": bev_dir,
		"rgb": bev_dir / "rgb_bev.png",
		"target_json": bev_dir / "target_point.json",
		"vlm_result_dir": scene_dir / "vlm_result",
	}


def extract_json_text(text: str) -> str:
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
		raise ValueError("Could not find JSON object in model response text")
	return stripped[left : right + 1]


def parse_direction_scores(response_text: str) -> dict:
	parsed = json.loads(extract_json_text(response_text))
	scores = parsed.get("Direction scores")
	if scores is None:
		scores = parsed.get("direction_scores")
	if not isinstance(scores, list) or len(scores) != 12:
		raise ValueError("Direction scores must be a list of 12 values")
	scores = [float(x) for x in scores]
	parsed["Direction scores"] = scores
	return parsed


def get_query_point(target_json_path: Path, image_shape: tuple[int, int]) -> tuple[int, int]:
	with target_json_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	uv = data.get("grasp_pixel_uv")
	if not isinstance(uv, list) or len(uv) < 2:
		raise KeyError(f"grasp_pixel_uv not found in {target_json_path}")

	h, w = image_shape[:2]
	x = int(round(float(uv[0])))
	y = int(round(float(uv[1])))
	x = max(0, min(w - 1, x))
	y = max(0, min(h - 1, y))
	return x, y


def draw_yellow_circle(image_bgr: np.ndarray, center: tuple[int, int], radius: int = 18, outline: int = 4) -> np.ndarray:
	x, y = center
	img = image_bgr.copy()
	yellow = (0, 255, 255)
	black = (0, 0, 0)
	cv2.circle(img, (x, y), radius=radius, color=black, thickness=-1, lineType=cv2.LINE_AA)
	cv2.circle(img, (x, y), radius=max(1, radius - outline), color=yellow, thickness=-1, lineType=cv2.LINE_AA)
	return img


def score_to_bgr(score: float) -> tuple[int, int, int]:
	s = float(np.clip(score, 0.0, 1.0))
	blue = np.array([255, 0, 0], dtype=np.float32)
	red = np.array([0, 0, 255], dtype=np.float32)
	mix = (1.0 - s) * blue + s * red
	return int(mix[0]), int(mix[1]), int(mix[2])


def draw_direction_scores(
	image_bgr: np.ndarray,
	direction_scores: list[float],
	center: tuple[int, int],
	selected_clock: int,
	model_label: str,
) -> np.ndarray:
	out = image_bgr.copy()
	h, w = out.shape[:2]
	cx, cy = center

	arrow_length = 70
	circle_radius = 90
	thickness = 3
	tip_length = 0.22
	font = cv2.FONT_HERSHEY_SIMPLEX
	score_font_scale = 0.72
	score_thickness = 2
	pad_x = 10
	pad_y = 8

	def draw_text_with_white_box(
		img: np.ndarray,
		text: str,
		anchor_x: int,
		anchor_y: int,
		text_color: tuple[int, int, int],
		font_scale: float,
		text_thickness: int,
	) -> None:
		(tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
		x_min = max(0, anchor_x - tw // 2 - pad_x)
		x_max = min(w - 1, anchor_x + tw // 2 + pad_x)
		y_min = max(0, anchor_y - th // 2 - pad_y)
		y_max = min(h - 1, anchor_y + th // 2 + baseline + pad_y)

		cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), thickness=-1)
		cv2.putText(
			img,
			text,
			(anchor_x - tw // 2, anchor_y + th // 2),
			font,
			font_scale,
			text_color,
			text_thickness,
			cv2.LINE_AA,
		)

	for i in range(12):
		clock = i + 1
		score = float(direction_scores[i])
		angle_deg = -60 + i * 30
		angle_rad = math.radians(angle_deg)

		x1 = int(cx + circle_radius * math.cos(angle_rad))
		y1 = int(cy + circle_radius * math.sin(angle_rad))
		x2 = int(cx + (circle_radius + arrow_length) * math.cos(angle_rad))
		y2 = int(cy + (circle_radius + arrow_length) * math.sin(angle_rad))

		if clock == selected_clock:
			color = (0, 255, 0)
			thick = thickness + 1
		else:
			color = score_to_bgr(score)
			thick = thickness

		cv2.arrowedLine(
			out,
			(x1, y1),
			(x2, y2),
			color=(0, 0, 0),
			thickness=thick + 2,
			line_type=cv2.LINE_AA,
			tipLength=tip_length,
		)
		cv2.arrowedLine(
			out,
			(x1, y1),
			(x2, y2),
			color=color,
			thickness=thick,
			line_type=cv2.LINE_AA,
			tipLength=tip_length,
		)

		txt = f"{score:.2f}"
		tx = int(cx + (circle_radius + arrow_length + 48) * math.cos(angle_rad))
		ty = int(cy + (circle_radius + arrow_length + 48) * math.sin(angle_rad))
		draw_text_with_white_box(
			out,
			txt,
			tx,
			ty,
			color,
			score_font_scale,
			score_thickness,
		)

	draw_text_with_white_box(out, model_label, 150, 28, (0, 0, 0), 0.8, 2)
	draw_text_with_white_box(out, f"best: {selected_clock} o'clock", 130, 62, (0, 170, 0), 0.78, 2)
	return out


def encode_image_to_data_url(image_bgr: np.ndarray) -> str:
	ok, encoded = cv2.imencode(".png", image_bgr)
	if not ok:
		raise ValueError("Failed to encode image")
	b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
	return f"data:image/png;base64,{b64}"


def run_openai(model_name: str, image_bgr: np.ndarray) -> tuple[str, float]:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY is not set")

	from openai import OpenAI

	client = OpenAI(api_key=api_key)
	data_url = encode_image_to_data_url(image_bgr)
	messages = [
		{"role": "system", "content": prompt_single},
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "Return only valid JSON."},
				{"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
			],
		},
	]

	t0 = time.time()
	if model_name.startswith("gpt-5"):
		try:
			resp = client.chat.completions.create(
				model=model_name,
				messages=messages,
				max_completion_tokens=4000,
				reasoning_effort="low",
			)
		except Exception:
			resp = client.chat.completions.create(
				model=model_name,
				messages=messages,
				max_completion_tokens=4000,
			)
	else:
		resp = client.chat.completions.create(
			model=model_name,
			messages=messages,
			temperature=0.0,
			max_tokens=1500,
		)
	elapsed = time.time() - t0

	text = resp.choices[0].message.content or ""
	return text, elapsed


def run_gemini(model_name: str, image_bgr: np.ndarray) -> tuple[str, float]:
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is not set")

	from google import genai
	from google.genai import types

	client = genai.Client(api_key=api_key)
	frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	ok, encoded = cv2.imencode(".png", frame_rgb)
	if not ok:
		raise ValueError("Failed to encode image")

	parts = [
		prompt_single,
		"Return only valid JSON.",
		types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/png"),
	]

	t0 = time.time()
	resp = client.models.generate_content(model=model_name, contents=parts)
	elapsed = time.time() - t0

	text = getattr(resp, "text", None)
	if text:
		return text, elapsed

	candidates = getattr(resp, "candidates", None)
	if candidates:
		for cand in candidates:
			content = getattr(cand, "content", None)
			if content and getattr(content, "parts", None):
				chunks = []
				for p in content.parts:
					part_text = getattr(p, "text", None)
					if part_text:
						chunks.append(part_text)
				if chunks:
					return "\n".join(chunks), elapsed

	raise RuntimeError("Failed to read Gemini response text")


def provider_for_model(model_name: str) -> str:
	if model_name.startswith("gpt-"):
		return "openai"
	if model_name.startswith("gemini-"):
		return "gemini"
	raise ValueError(f"Unsupported model: {model_name}")


def run_model(model_name: str, marked_image: np.ndarray, center: tuple[int, int], out_root: Path, scene: str, scene_num: str, paths: dict[str, Path]) -> dict:
	model_dir = out_root / model_name
	model_dir.mkdir(parents=True, exist_ok=True)

	input_path = model_dir / "input_with_query.png"
	raw_txt_path = model_dir / "vlm_response_raw.txt"
	out_json_path = model_dir / "vlm_result.json"
	out_vis_path = model_dir / "direction_scores_vis.png"

	cv2.imwrite(str(input_path), marked_image)

	provider = provider_for_model(model_name)
	if provider == "openai":
		raw_text, elapsed = run_openai(model_name, marked_image)
	else:
		raw_text, elapsed = run_gemini(model_name, marked_image)

	raw_txt_path.write_text(raw_text, encoding="utf-8")

	parsed = parse_direction_scores(raw_text)
	scores = [float(v) for v in parsed["Direction scores"]]
	best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
	best_clock = best_idx + 1

	vis = draw_direction_scores(marked_image, scores, center, best_clock, model_name)
	cv2.imwrite(str(out_vis_path), vis)

	result = {
		"scene": scene,
		"scene_num": scene_num,
		"provider": provider,
		"model": model_name,
		"prompt_source": str((Path(__file__).parent / "prompt.py").resolve()),
		"input_image_path": str(paths["rgb"]),
		"target_point_json": str(paths["target_json"]),
		"query_point_uv": [int(center[0]), int(center[1])],
		"inference_sec": float(elapsed),
		"selected_clock_direction": int(best_clock),
		"selected_score": float(scores[best_idx]),
		"direction_scores": scores,
		"parsed_response": parsed,
		"raw_response_path": str(raw_txt_path),
		"vis_path": str(out_vis_path),
	}
	out_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
	return result


def concat_2x2(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
	if len(images) != 4:
		raise ValueError("Need exactly 4 images for 2x2 concat")

	h = min(img.shape[0] for img in images)
	w = min(img.shape[1] for img in images)
	resized = [cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) for img in images]

	top = np.concatenate([resized[0], resized[1]], axis=1)
	bot = np.concatenate([resized[2], resized[3]], axis=1)
	return np.concatenate([top, bot], axis=0)


def _run_model_worker(args_tuple: tuple) -> dict | None:
	"""Worker function for multiprocessing pool"""
	model_name, marked, query_xy, out_root, scene, scene_num, paths = args_tuple
	try:
		return run_model(model_name, marked, query_xy, out_root, scene, scene_num, paths)
	except Exception as exc:
		error_dir = out_root / model_name
		error_dir.mkdir(parents=True, exist_ok=True)
		error_json = {
			"scene": scene,
			"scene_num": scene_num,
			"model": model_name,
			"error": str(exc),
		}
		(error_dir / "vlm_result.json").write_text(json.dumps(error_json, ensure_ascii=False, indent=2), encoding="utf-8")
		return None


def main() -> None:
	args = parse_args()
	paths = build_scene_paths(args.scene, args.scene_num)

	if not paths["scene_dir"].is_dir():
		raise FileNotFoundError(f"Scene directory not found: {paths['scene_dir']}")
	if not paths["rgb"].is_file():
		raise FileNotFoundError(f"RGB image not found: {paths['rgb']}")
	if not paths["target_json"].is_file():
		raise FileNotFoundError(f"target_point.json not found: {paths['target_json']}")

	image = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Failed to read image: {paths['rgb']}")

	query_xy = get_query_point(paths["target_json"], image.shape)
	marked = draw_yellow_circle(image, query_xy)

	out_root = paths["vlm_result_dir"]
	out_root.mkdir(parents=True, exist_ok=True)

	if args.model == "all":
		# Prepare worker arguments
		worker_args = [
			(model_name, marked, query_xy, out_root, args.scene, args.scene_num, paths)
			for model_name in SUPPORTED_MODELS
		]

		# Run models in parallel using multiprocessing
		print(f"Running {len(SUPPORTED_MODELS)} models in parallel...")
		t_start = time.time()
		with Pool(processes=min(4, len(SUPPORTED_MODELS))) as pool:
			results = pool.map(_run_model_worker, worker_args)
		elapsed_total = time.time() - t_start

		# Collect successful results and generate visualizations
		all_results = [r for r in results if r is not None]
		vis_images = []

		for model_name, result in zip(SUPPORTED_MODELS, results):
			if result is not None:
				vis = cv2.imread(str(Path(result["vis_path"])), cv2.IMREAD_COLOR)
				if vis is not None:
					vis_images.append(vis)
			else:
				# Use error image for failed models
				err_img = marked.copy()
				cv2.putText(err_img, f"ERROR: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
				vis_images.append(err_img)

		all_dir = out_root / "all"
		all_dir.mkdir(parents=True, exist_ok=True)

		if len(vis_images) >= 4:
			grid = concat_2x2(vis_images[:4], SUPPORTED_MODELS)
			grid_path = all_dir / "direction_scores_2x2.png"
			cv2.imwrite(str(grid_path), grid)
		else:
			grid_path = None

		summary = {
			"scene": args.scene,
			"scene_num": args.scene_num,
			"models": SUPPORTED_MODELS,
			"results": all_results,
			"grid_vis_path": str(grid_path) if grid_path is not None else None,
			"parallel_execution_sec": float(elapsed_total),
		}
		(all_dir / "vlm_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

		print(f"✓ Completed {len(all_results)}/{len(SUPPORTED_MODELS)} models in {elapsed_total:.2f}s")
		print(f"Saved all-model results under: {out_root}")
		if grid_path is not None:
			print(f"Saved 2x2 vis: {grid_path}")
	else:
		result = run_model(args.model, marked, query_xy, out_root, args.scene, args.scene_num, paths)
		print(f"Saved result: {out_root / args.model / 'vlm_result.json'}")
		print(f"Saved vis: {result['vis_path']}")


if __name__ == "__main__":
	main()
