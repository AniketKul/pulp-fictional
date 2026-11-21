"""Read the offline sentence-embedding evidence bundled with the demo."""
import json
from pathlib import Path


def load_scores() -> dict:
    path = Path(__file__).with_name("data") / "semantic.json"
    return json.loads(path.read_text(encoding="utf-8"))["queries"]
