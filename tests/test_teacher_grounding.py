"""Verify offline teacher decisions against the published purchase records and immutable vectors."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


class TeacherGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = read("live/data/source.json")
        cls.source = {row["query"]: row["purchases"] for row in source["queries"]}
        cls.catalog = {p["id"]: p for p in source["products"]}
        cls.market = read("live/data/market.json")
        cls.teacher = read("experiments/market/teacher.json")
        cls.rows = {r["query"]: r for r in cls.teacher["rows"]}

    def test_all_context_counts_come_from_real_query_item_incidences(self):
        self.assertEqual(set(self.rows), set(self.source))
        for query, logged in self.source.items():
            evidence = self.rows[query]["evidence"]
            self.assertEqual({x["item_id"]: x["purchase_occurrences"] for x in evidence["purchased_products"]}, logged)
            brands, paths = Counter(), Counter()
            for sku, count in logged.items():
                brands[self.catalog[sku]["brand"]] += count
                paths[self.catalog[sku]["category"]] += count
            self.assertEqual({b["brand"]: b["purchase_occurrences"] for b in evidence["converted_brands"]}, brands)
            expected = sorted(paths.items(), key=lambda x: (-x[1], x[0]))[:8]
            self.assertEqual([(c["source_path"], c["purchase_occurrences"]) for c in evidence["converted_categories"]], expected)
            scores = [row["cosine"] for row in evidence["similar_brands"]]
            self.assertGreater(len(scores), 0)
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bilingual_brand_aliases_survive_catalog_validation(self):
        for source_query, brand in (("德芙巧克力", "Dove"), ("士力架", "Snickers"), ("咖啡雀巢瓶装", "Nestlé")):
            with self.subTest(query=source_query):
                self.assertEqual(self.rows[source_query]["roles"]["brand"], brand)

    def test_accepted_cache_paths_and_rewrites_pass_the_offline_guardrails(self):
        threshold = self.teacher["category_min_cosine"]
        for row in self.rows.values():
            candidates = {c["number"]: c for c in row["evidence"]["converted_categories"]}
            reranked = [candidates[n] for n in row["reranked_candidate_numbers"]]
            expected = [c["path"] for c in reranked if c["cosine"] >= threshold]
            self.assertEqual(expected, row["categories"])
            for key, audit in row["rewrite_validation"].items():
                self.assertEqual(bool(row[key]), audit["accepted"])
                if row[key]:
                    self.assertGreaterEqual(audit["query_cosine"] + 1e-6, self.teacher["rewrite_min_cosine"][key])
                    self.assertGreaterEqual(audit["best_item_cosine"] + 1e-6, self.teacher["rewrite_item_min_cosine"])
        for cached in self.market["cache"]:
            self.assertEqual(cached["categories"], self.rows[cached["query"]]["categories"])
        for key, name in (("embeddings_sha256", "live/data/embeddings.json"),
                          ("embeddings_builder_sha256", "scripts/build_embeddings.py"),
                          ("ranker_sha256", "live/retrieval.js"),
                          ("rewrite_filter_sha256", "scripts/filter_teacher_rewrites.cjs")):
            self.assertEqual(self.teacher[key], sha256((ROOT / name).read_bytes()).hexdigest())
