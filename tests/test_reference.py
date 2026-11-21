"""Check that the shipped experiment report describes the weights and fixtures in this checkout."""
from hashlib import sha256
import json
from pathlib import Path
import unittest


class ReferenceTest(unittest.TestCase):
    def test_reference_file_identity(self):
        root = Path(__file__).resolve().parents[1]
        reference = root / "experiments/lora/reference"
        report = json.loads((reference / "evaluation.json").read_text())
        for key, path in [
            ("curriculum_sha256", root / "src/pulp_fictional/data/curriculum.json"),
            ("gold_sha256", root / "evaluation/tail.jsonl"),
            ("adapter_sha256", reference / "adapter/adapter_model.safetensors"),
        ]:
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), report[key])
        gold = [json.loads(line)["query"] for line in (root / "evaluation/tail.jsonl").read_text().splitlines()]
        teacher = json.loads((reference / "teacher.json").read_text())
        for result in (teacher, report["base"], report["merged_lora"]):
            self.assertEqual([r["query"] for r in result["predictions"]], gold)
