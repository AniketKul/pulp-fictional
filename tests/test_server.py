import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from pulp_fictional.server import Search, handler


class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Search()))
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_serves_cache_and_student(self):
        for q, route in [("bread+no+gluten", "head-cache"), ("unsweetened+oat+drink+please", "student")]:
            code, data = self.get("/api/search?q=" + q)
            self.assertEqual(code, 200)
            result = json.loads(data)
            self.assertEqual(result["route"], route)
            self.assertTrue(result["products"])
            self.assertIn("proposed_roles", result)

    def test_input_boundary(self):
        for suffix in ["", "?q=", "?q=x&q=y", "?q=" + "a" * 201]:
            self.assertEqual(self.get("/api/search" + suffix)[0], 400)
        self.assertEqual(self.get("/static/../server.py")[0], 404)
        self.assertIn(b"<!", self.get("/")[1][:20])
