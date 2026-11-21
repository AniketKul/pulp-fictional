from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pulp_fictional.learning import IntentRouter, train


class LearningTests(TestCase):
    def test_trained_tail_and_cache(self):
        rows = []
        for query in ("oat drink", "original oat", "oat unsweetened", "freezer bags", "gallon freezer bags", "freezer safe bags"):
            product = "oat milk" if "oat" in query else "freezer bags"
            rows.append(dict(query=query, head=query == "oat drink", proposal={
                "roles": {"product": product, "attributes": [], "brand": ""}
            }))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps({"rows": rows, "student": train(rows)}))
            router = IntentRouter(path)
        route, result = router.predict(" OAT   drink ")
        self.assertEqual(route, "head-cache")
        result["roles"]["product"] = "changed"
        self.assertEqual(router.predict("oat drink")[1]["roles"]["product"], "oat milk")
        route, result = router.predict("a carton of oat")
        self.assertEqual((route, result["roles"]["product"]), ("student", "oat milk"))
        self.assertEqual(router.predict("star chart")[1]["roles"]["product"], "")

    def test_custom_tail_model_is_only_used_on_a_miss(self):
        calls = []
        def predict(query):
            calls.append(query)
            return {"product": "tofu", "brand": "", "attributes": []}

        router = IntentRouter(tail_predictor=predict)
        router.predict("bread")
        self.assertEqual(calls, [])
        self.assertEqual(router.predict("tofu for dinner")[1]["roles"]["product"], "tofu")
        self.assertEqual(calls, ["tofu for dinner"])

    def test_shipped_curriculum_rebuilds_student(self):
        from pulp_fictional.learning import DATA, key

        bundle = json.loads(DATA.read_text(encoding="utf-8"))
        rows = bundle["rows"]
        self.assertGreater(len(rows), 80)
        self.assertEqual(len({key(row["query"]) for row in rows}), len(rows))
        self.assertEqual(train([r for r in rows if r["trainable"]]), bundle["student"])
        router = IntentRouter()
        self.assertEqual(router.predict("oat milk")[0], "head-cache")
        self.assertEqual(router.predict("please get me some oat drink")[0], "student")


    def test_teacher_receipt_matches_the_input_files(self):
        from pulp_fictional.engine import DATA as catalog
        from pulp_fictional.learning import DATA

        receipt = json.loads(DATA.read_text(encoding="utf-8"))["teacher"]
        for name, digest in receipt["catalog_sha256"].items():
            self.assertEqual(sha256((catalog / name).read_bytes()).hexdigest(), digest, name)
        for name in ("engine", "semantic"):
            key = "engine_sha256" if name == "engine" else "semantic_py_sha256"
            self.assertEqual(sha256((catalog.parent / f"{name}.py").read_bytes()).hexdigest(), receipt[key])
        builder = Path(__file__).resolve().parents[1] / "scripts/build_curriculum.py"
        self.assertEqual(sha256(builder.read_bytes()).hexdigest(), receipt["teacher_prompt_sha256"])
