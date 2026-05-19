import json
from pathlib import Path
from datetime import datetime

STAGES = []

def log_stage(stage: str):
    STAGES.append(stage)
    print(f"STAGE: {stage}")


def ensure_dirs():
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp():
    return datetime.utcnow().isoformat() + "Z"
