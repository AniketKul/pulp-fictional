"""Verify public catalog provenance and model input/output identities."""

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


class MarketTests(unittest.TestCase):
    def test_every_result_has_its_original_catalog_record(self):
        source = read("live/data/source.json")
        market = read("live/data/market.json")
        original = {p["id"]: p for p in source["products"]}
        self.assertEqual({p["id"] for p in market["products"]}, set(original))
        for p in market["products"]:
            for key in ("title", "category", "brand"):
                self.assertEqual(p[key], original[p["id"]][key])
            self.assertRegex(p["english_title"], "[A-Za-z]")
        queries = {q["query"]: q for q in source["queries"]}
        self.assertEqual(len(queries), len(source["queries"]))
        for row in market["cache"]:
            self.assertGreater(queries[row["query"]]["sessions"], 0)
            self.assertEqual(row["sessions"], queries[row["query"]]["sessions"])
            self.assertTrue(
                all(s in original for s in queries[row["query"]]["purchases"])
            )

    def test_current_artifacts_match_training_and_generation_receipts(self):
        teacher = read("experiments/market/teacher.json")
        eval_report = read("experiments/market/evaluation.json")
        inputs = [
            ("live/data/source.json", teacher["source_sha256"]),
            ("scripts/real_teacher.py", teacher["builder_sha256"]),
            ("live/data/market.json", eval_report["market_sha256"]),
            ("live/data/prompt.json", eval_report["prompt_sha256"]),
            ("experiments/market/teacher.json", eval_report["teacher_sha256"]),
            (
                "experiments/market/adapter/adapter_model.safetensors",
                eval_report["adapter_sha256"],
            ),
        ]
        for name, digest in inputs:
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name
            )
