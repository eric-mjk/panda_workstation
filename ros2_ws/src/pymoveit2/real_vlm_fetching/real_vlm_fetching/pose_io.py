import json
from pathlib import Path


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())
