from pathlib import Path
from .pose_io import save_json


def create_trial_dir(root, scene_id, trial_id) -> Path:
    trial_dir = Path(root) / str(scene_id) / str(trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def save_trial_log(trial_dir, data):
    save_json(Path(trial_dir) / "trial_log.json", data)
