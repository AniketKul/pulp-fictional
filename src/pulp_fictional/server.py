"""Local grocery storefront and JSON search API."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .engine import Engine
from .learning import IntentRouter


class Search:
    def __init__(self, adapter=None):
        self.engine = Engine()
        self.backend = "linear-cpu"
        predictor = None
        if adapter:
            from .lora import LoraStudent
            predictor = LoraStudent(adapter)
            self.backend = "qwen3-lora"
        self.router = IntentRouter(tail_predictor=predictor)

    def __call__(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            raise ValueError("Enter a query between 1 and 200 characters.")
        route, proposal = self.router.predict(query)
        result = self.engine.search(query, **proposal)
        return {**result, "route": route, "backend": "teacher-cache" if route == "head-cache" else self.backend,
                "proposed_roles": proposal.get("roles", {})}


def handler(search):
    static = Path(__file__).with_name("static")
    assets = {"/": ("text/html; charset=utf-8", (static / "index.html").read_bytes())}
    mime = {".js": "application/javascript", ".css": "text/css", ".jpg": "image/jpeg"}
    for path in static.iterdir():
        if path.suffix in mime:
            assets["/static/" + path.name] = mime[path.suffix], path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, content_type, content):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                if url.path in assets:
                    self.respond(200, *assets[url.path])
                    return
                if url.path != "/api/search":
                    self.send_error(404)
                    return
                params = parse_qs(url.query, keep_blank_values=True, max_num_fields=8)
                if len(params.get("q", [])) != 1:
                    raise ValueError("Supply exactly one q parameter.")
                data = search(params["q"][0])
                status = 200
            except ValueError as error:
                data, status = {"error": str(error)}, 400
            self.respond(status, "application/json; charset=utf-8", json.dumps(data).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--adapter", type=Path, help="Use a merged Qwen LoRA student for cache misses (CUDA)")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler(Search(args.adapter))) as server:
        print(f"Storefront listening at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
