import unittest

from pulp_fictional.engine import Engine


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()

    def test_rejects_noisy_top_conversion(self):
        result = self.engine.search("vegan roast")
        beef, plant = result["candidates"][:2]
        self.assertGreater(beef["conversions"], plant["conversions"])
        self.assertFalse(beef["accepted"])
        self.assertTrue(plant["accepted"])
        self.assertEqual([p["id"] for p in result["products"]], ["roast-01"])
        self.assertEqual(result["roles"]["brand"], "")

    def test_rewrite_types_drive_distinct_searches(self):
        result = self.engine.search("flat leaf parsley")
        rewrites = result["rewrites"]
        self.assertIn("italian parsley", rewrites["synonyms"])
        self.assertIn("curly parsley", rewrites["substitutes"])
        self.assertIn("parsley", rewrites["broader"])
        for q, ids in [("italian parsley", {"herb-01"}), ("curly parsley", {"herb-02"}),
                       ("parsley", {"herb-01", "herb-02"})]:
            self.assertEqual({p["id"] for p in self.engine.search(q)["products"]}, ids)

    def test_constraints_survive_rewrites_and_model_errors(self):
        q = "bread no gluten"
        result = self.engine.search(q, roles={"product": "bananas", "brand": "Orchard Row", "attributes": []})
        self.assertEqual(result["roles"], {"product": "bread", "attributes": ["gluten-free"], "brand": ""})
        self.assertEqual({p["id"] for p in result["products"]}, {"bread-01", "bread-02"})
        for group in result["rewrites"].values():
            self.assertTrue(group)
            for rewrite in group:
                hits = self.engine.search(rewrite)["products"]
                self.assertTrue(hits)
                self.assertTrue(all("gluten-free" in p["attributes"] for p in hits))

    def test_dairy_constraints_are_distinct(self):
        for q in ["no dairy milk", "lactose free milk"]:
            r = self.engine.search(q)
            ids = {p["id"] for p in r["products"]}
            self.assertTrue(ids)
            self.assertEqual("dairy-02" in ids, q == "lactose free milk")

    def test_unknown_queries_and_unsupported_exclusions_abstain(self):
        for q in ["lamp", "bread without raisins", "milk no nuts", "!!!"]:
            self.assertEqual(self.engine.search(q, roles={"product": "bread"})["products"], [])

    def test_recovers_sku_nickname_and_brand(self):
        r = self.engine.search("verdant tide")
        self.assertEqual(r["roles"]["product"], "smoothie")
        self.assertEqual(r["roles"]["brand"], "Daybreak Press")
        self.assertEqual([p["id"] for p in r["products"]], ["smoothie-01"])
        self.assertEqual(r["categories"], [])
        path = r["products"][0]["category"]
        rejected = next(c for c in r["candidates"] if c["category"] == path)
        self.assertEqual(rejected["guard"], "below cosine threshold")

    def test_category_rerank_and_rewrite_allowlist(self):
        paths = self.engine.search("milk")["categories"]
        result = self.engine.search("milk", categories=["Imaginary", paths[-1], paths[-1]])
        self.assertEqual(result["categories"][0], paths[-1])
        self.assertEqual(result["products"][0]["category"], paths[-1])
        self.assertNotIn("Imaginary", result["categories"])
        filtered = self.engine.search("oat milk", rewrites={"synonyms": ["oat drink", 3, "soap"],
                                                           "broader": "everything", "substitutes": ["oat drink"]})
        self.assertEqual(filtered["rewrites"]["synonyms"], ["oat drink"])
        self.assertEqual(filtered["rewrites"]["substitutes"], [])

    def test_embeddings_filter_after_category_ranking(self):
        e = self.engine
        good = e.by_id["oat-01"]["category"]
        blocked = e.by_id["almond-01"]["category"]
        e.semantic = {"milk": {"categories": {good: 0.62, blocked: 0.08}, "products": {"oat-01": 0.78}}}
        r = e.search("milk", categories=[blocked, good])
        self.assertEqual(r["categories"], [good])
        self.assertTrue(all(p["category"] == good for p in r["products"]))
        self.assertEqual(next(c["guard"] for c in r["candidates"] if c["category"] == blocked), "below cosine threshold")
        self.assertEqual(e.context("milk")["retrieved_products"][0]["id"], "oat-01")

    def test_student_can_recover_supported_tail_and_preserve_filters(self):
        query = "unsweetened oat drink for my morning latte"
        self.assertFalse(self.engine.search(query)["products"])
        recovered = self.engine.search(query, roles={"product": "oat milk"})
        self.assertEqual(recovered["retrieval"], "model-grounded recovery")
        self.assertEqual([p["id"] for p in recovered["products"]], ["oat-01"])
        for conflict in ["soy drink for my morning latte", "oat milk for coffee without nuts",
                         "lantern mill oat milk for coffee", "77 oz oat milk for coffee", "telescope for coffee"]:
            self.assertFalse(self.engine.search(conflict, roles={"product": "oat milk"})["products"])

    def test_typos_and_longest_attribute_phrase(self):
        self.assertEqual(self.engine.search("flat leaf parsly")["roles"]["product"], "flat leaf parsley")
        self.assertEqual(self.engine.search("whole wheat loaf")["roles"]["attributes"], ["whole wheat"])

    def test_unsupported_exclusions_abstain_even_with_model_proposals(self):
        for q in ["milk -soy", "bread -gluten-free", "milk −organic for coffee",
                  "fat-free milk", "fat free lactose-free milk", "milk not organic",
                  "milk except soy", "anything but beef roast"]:
            with self.subTest(q=q):
                r = self.engine.search(q, roles={"product": "milk", "attributes": ["organic"]})
                self.assertFalse(r["products"])
                self.assertFalse(r["roles"]["attributes"])
                self.assertFalse(any(r["rewrites"].values()))

    def test_alternatives_do_not_leak_across_taxonomy_siblings(self):
        for q in ["milk alternatives", "dairy alternative", "meat alternatives"]:
            with self.subTest(q=q):
                r = self.engine.search(q)
                self.assertTrue(r["products"])
                self.assertTrue(all("vegan" in p["attributes"] for p in r["products"]))
        self.assertEqual({p["product"] for p in self.engine.search("meat alternatives")["products"]}, {"tofu", "roast"})

    def test_generic_roast_covers_beef_and_plant_roasts(self):
        for proposed in [None, {"product": "roast"}]:
            result = self.engine.search("roast", roles=proposed)
            self.assertEqual({p["id"] for p in result["products"]}, {"beef-01", "roast-01"})
            self.assertEqual(result["roles"]["product"], "roast")
            self.assertFalse(result["rewrites"]["synonyms"])
        self.assertEqual([p["id"] for p in self.engine.search("beef roast")["products"]], ["beef-01"])

    def test_size_specific_queries_keep_results_and_skip_rewrites(self):
        for q, ids in [("18 oz bread", {"bread-01"}), ("24-count freezer bags", {"freezer-01"}),
                       ("32 fl oz oat milk", {"oat-01", "oat-02"})]:
            with self.subTest(q=q):
                result = self.engine.search(q)
                self.assertEqual({p["id"] for p in result["products"]}, ids)
                self.assertFalse(any(result["rewrites"].values()))
        self.assertTrue(self.engine.search("2% dairy milk")["rewrites"]["synonyms"])

    def test_unhandled_script_and_unicode_minus_never_become_positive_matches(self):
        for q in ["milk 牛乳アレルギー", "٢٪ milk", "milk ‑organic", "milk ‐organic",
                  "豆乳 oat milk for coffee", "milk ‑organic for coffee"]:
            with self.subTest(q=q):
                result = self.engine.search(q, roles={"product": "oat milk"})
                self.assertFalse(result["products"])
                self.assertFalse(any(result["roles"].values()))
        self.assertEqual(self.engine.search("ｏａｔ ｍｉｌｋ")["roles"]["product"], "oat milk")

    def test_input_boundary(self):
        for invalid in [None, "", "  ", "q" * 201]:
            with self.assertRaises(ValueError):
                self.engine.search(invalid)


if __name__ == "__main__":
    unittest.main()
