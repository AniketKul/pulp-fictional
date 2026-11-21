import json
import threading
import unittest
from http.client import HTTPConnection

from pulp_fictional.server import make_server


class StorefrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = make_server(0)
        cls.thread = threading.Thread(target=cls.http.serve_forever)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.thread.join()
        cls.http.server_close()

    def get(self, path):
        conn = HTTPConnection("127.0.0.1", self.http.server_port, timeout=4)
        try:
            conn.request("GET", path)
            res = conn.getresponse()
            return res.status, res.read()
        finally:
            conn.close()

    def test_serves_real_catalog_and_model_ui(self):
        status, html = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Pulp Fictional", html)
        self.assertIn(b"window.PULP_GPU=false", html)
        for path in (
            "/assets/market.jpg",
            "/llm.js",
            "/search.js",
            "/store.js",
            "/retrieval.js",
            "/data/embeddings.json",
        ):
            self.assertEqual(self.get(path)[0], 200, path)
        _, raw = self.get("/data/source.json")
        data = json.loads(raw)
        self.assertGreater(len(data["products"]), 100)
        self.assertTrue(
            all(c > 0 for q in data["queries"] for c in q["purchases"].values())
        )

    def test_cached_search_api_on_real_record(self):
        status, body = self.get("/api/search?q=Mooncakes")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["source_query"], "月饼")
        self.assertEqual(result["route"], "teacher-cache")
        self.assertIn("64021", [p["id"] for p in result["products"]])
        self.assertEqual(self.get("/api/search?q=an+unseen+query+without+gpu")[0], 503)

    def test_no_directory_index(self):
        self.assertEqual(self.get("/data/")[0], 404)
