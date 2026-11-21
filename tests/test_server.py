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

    def test_serves_real_catalog_and_storefront(self):
        status, html = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Pulp Fictional", html)
        for path in (
            "/assets/market.jpg",
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

    def test_invalid_model_outputs_and_gpu_errors_fail_the_request(self):
        def broken(query):
            if query == "incomplete":
                return '{"attributes":[]}'
            raise RuntimeError("device unavailable")

        with make_server(0, predict=broken) as http:
            thread = threading.Thread(target=http.serve_forever)
            thread.start()
            try:
                for query in ("incomplete", "fail"):
                    conn = HTTPConnection("127.0.0.1", http.server_port, timeout=4)
                    try:
                        conn.request("GET", "/api/search?q=" + query)
                        self.assertEqual(conn.getresponse().status, 502)
                    finally:
                        conn.close()
            finally:
                http.shutdown()
                thread.join()

    def test_no_directory_index(self):
        self.assertEqual(self.get("/data/")[0], 404)


class SearchApiTests(unittest.TestCase):
    def start(self, **options):
        http = make_server(0, **options)
        thread = threading.Thread(target=http.serve_forever)
        thread.start()

        def stop():
            http.shutdown()
            thread.join()
            http.server_close()

        self.addCleanup(stop)
        return http.server_port

    def request(self, port, method, path, headers=None, body=None):
        conn = HTTPConnection("127.0.0.1", port, timeout=6)
        try:
            conn.request(method, path, headers=headers or {}, body=body)
            res = conn.getresponse()
            return res.status, dict(res.getheaders()), res.read()
        finally:
            conn.close()

    def test_rejects_invalid_parameters(self):
        port = self.start()
        for tail in (
            "",
            "q=",
            "q=a&q=b",
            "q=%FF",
            "q=%0Aabc",
            "q=%xy",
            "q=%3C%7Cim_start%7C%3E",
            "q=tea&debug=1",
            "q=tea&force=2",
            "q=tea&force=",
            "q=" + "z" * 201,
        ):
            with self.subTest(tail=tail):
                self.assertEqual(
                    self.request(port, "GET", "/api/search?" + tail)[0], 400
                )

    def test_forced_cached_query_keeps_history_and_private_intent_remains_compatible(
        self,
    ):
        calls = []

        def predict(query):
            calls.append(query)
            return json.dumps(
                dict(
                    product="mooncakes",
                    brand="",
                    attributes=[],
                    synonym="",
                    substitute="",
                    broader="",
                )
            )

        port = self.start(predict=predict)
        _, _, raw = self.request(port, "GET", "/api/search?q=Mooncakes")
        normal = json.loads(raw)
        self.assertEqual(normal["route"], "teacher-cache")
        self.assertFalse(calls)
        status, _, raw = self.request(port, "GET", "/api/search?q=Mooncakes&force=1")
        self.assertEqual(status, 200)
        forced = json.loads(raw)
        self.assertEqual(forced["source_query"], normal["source_query"])
        self.assertEqual(forced["route"], "live-student")
        self.assertEqual(forced["evidence"], normal["evidence"])
        status, _, raw = self.request(
            port,
            "POST",
            "/api/intent",
            {"Content-Type": "application/json"},
            json.dumps({"query": "Mooncakes"}),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(json.loads(raw)["completion"])["product"], "mooncakes"
        )
        self.assertEqual(calls, ["Mooncakes", "Mooncakes"])

    def test_recovery_runs_once_only_for_malformed_generations(self):
        from pulp_fictional.service import recovery_schema

        valid = json.dumps(
            dict(
                product="chocolate",
                brand="",
                attributes=["不要花生"],
                synonym="",
                substitute="",
                broader="",
            ),
            ensure_ascii=False,
        )
        calls = []

        def predict(query):
            calls.append(("normal", query))
            return (
                valid if query == "valid" else '{"product":"chocolate",["synonym":""}'
            )

        def retry(query):
            calls.append(("retry", query))
            return valid if query == "recoverable" else '{"product":"chocolate"}'

        predict.retry_invalid_intent = retry
        port = self.start(predict=predict)
        for query, expected in (
            ("valid", 200),
            ("recoverable", 200),
            ("still-invalid", 502),
        ):
            status, _, raw = self.request(
                port,
                "POST",
                "/api/intent",
                {"Content-Type": "application/json"},
                json.dumps({"query": query}),
            )
            self.assertEqual(status, expected)
            if status == 200:
                self.assertEqual(
                    json.loads(json.loads(raw)["completion"]), json.loads(valid)
                )
        self.assertEqual(
            calls,
            [
                ("normal", "valid"),
                ("normal", "recoverable"),
                ("retry", "recoverable"),
                ("normal", "still-invalid"),
                ("retry", "still-invalid"),
            ],
        )
        allowed = recovery_schema("不要花生的巧克力")["properties"]["attributes"][
            "items"
        ]["enum"]
        self.assertIn("不要花生", allowed)
        self.assertNotIn("花生", allowed)

    def test_inference_backpressure_leaves_head_cache_available(self):
        entered = threading.Event()
        release = threading.Event()

        def predict(query):
            entered.set()
            release.wait(6)
            return json.dumps(
                dict(
                    product="tea",
                    brand="",
                    attributes=[],
                    synonym="",
                    substitute="",
                    broader="",
                )
            )

        port = self.start(predict=predict, inference_capacity=1)
        statuses = []
        worker = threading.Thread(
            target=lambda: statuses.append(
                self.request(port, "GET", "/api/search?q=first+tail")[0]
            )
        )
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            status, headers, _ = self.request(port, "GET", "/api/search?q=second+tail")
            self.assertEqual(status, 503)
            self.assertIn("Retry-After", headers)
            self.assertEqual(
                self.request(port, "GET", "/api/search?q=Mooncakes")[0], 200
            )
        finally:
            release.set()
            worker.join()
        self.assertEqual(statuses, [200])
