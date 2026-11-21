"""Serve the Pulp Fictional market locally, optionally with a resident CUDA student."""

import argparse
import functools
import json
import shutil
import subprocess
import unicodedata
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PUBLIC = "https://aniketkul.github.io/pulp-fictional/"


def site_root():
    installed = Path(__file__).with_name("live")
    return (
        installed
        if installed.is_dir()
        else Path(__file__).resolve().parents[2] / "live"
    )


def gpu_predictor(adapter, system):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU mode requires a CUDA device with bfloat16 support.")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval()
    lock = Lock()

    def predict(q):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        with lock, torch.inference_mode():
            answer = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(answer[0, ids.shape[1] :], skip_special_tokens=True)

    return predict


def make_server(port=8000, root=None, predict=None):
    root = (root or site_root()).resolve()

    def key(query):
        return " ".join(unicodedata.normalize("NFKC", query).lower().split())

    market = json.loads((root / "data/market.json").read_text())
    cache = {
        key(term): r
        for r in market["cache"]
        for term in (r["query"], r["english"])
        if term
    }

    class Handler(SimpleHTTPRequestHandler):
        def generate(self, query):
            try:
                raw = predict(query)
                payload = json.loads(
                    raw.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                fields = ("product", "brand", "synonym", "substitute", "broader")
                if (
                    not isinstance(payload, dict)
                    or not all(isinstance(payload.get(f), str) for f in fields)
                    or not isinstance(payload.get("attributes"), list)
                    or not all(isinstance(a, str) for a in payload["attributes"])
                ):
                    raise ValueError
                return payload
            except (ValueError, TypeError, AttributeError):
                self.send_bytes(
                    502,
                    b'{"error":"The student returned an incomplete intent."}',
                    "application/json",
                )
            except RuntimeError:
                self.send_bytes(
                    502, b'{"error":"GPU inference failed."}', "application/json"
                )
            return None

        def search_api(self):
            try:
                q = parse_qs(urlsplit(self.path).query, max_num_fields=2).get("q", [])
            except ValueError:
                q = []
            if len(q) != 1 or not q[0].strip() or len(q[0]) > 200 or "<|" in q[0]:
                self.send_bytes(
                    400,
                    b'{"error":"Enter one query of 1 to 200 characters."}',
                    "application/json",
                )
                return
            query = q[0].strip()
            head = cache.get(key(query))
            if not head and predict is None:
                self.send_bytes(
                    503,
                    b'{"error":"The browser runs this model locally. Use the storefront or start the server with --adapter for live API inference."}',
                    "application/json",
                )
                return
            if not shutil.which("node"):
                self.send_bytes(
                    503,
                    b'{"error":"The local search API requires Node.js for catalog ranking."}',
                    "application/json",
                )
                return
            if head:
                proposal = {
                    **head["roles"],
                    **{
                        k: head[k]
                        for k in ("synonym", "substitute", "broader", "categories")
                    },
                }
            else:
                proposal = self.generate(query)
                if proposal is None:
                    return
            req = {
                "query": query,
                "source_query": head["query"] if head else query,
                "proposal": proposal,
                "route": "teacher-cache" if head else "live-student",
                "backend": "Qwen2.5-7B-Instruct" if head else "Qwen3-0.6B LoRA CUDA",
            }
            result = subprocess.run(
                ["node", str(Path(__file__).with_name("rank.cjs")), str(root)],
                input=json.dumps(req).encode(),
                capture_output=True,
                timeout=15,
            )
            if result.returncode:
                self.send_bytes(
                    500, b'{"error":"Catalog ranking failed."}', "application/json"
                )
                return
            self.send_bytes(200, result.stdout, "application/json")

        def send_bytes(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/search":
                self.search_api()
            elif path in ("/", "/index.html"):
                html = (root / "index.html").read_text()
                settings = f"window.PULP_GPU={json.dumps(bool(predict))};"
                if not predict and not (root / "model.json").exists():
                    settings += f"window.PULP_MODEL_ASSETS={json.dumps(PUBLIC)};"
                self.send_bytes(
                    200,
                    html.replace(
                        "<head>", f"<head><script>{settings}</script>"
                    ).encode(),
                    "text/html; charset=utf-8",
                )
            else:
                super().do_GET()

        def list_directory(self, path):
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/intent" or predict is None:
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 2048:
                    raise ValueError
                obj = json.loads(self.rfile.read(n))
                query = obj.get("query", "") if isinstance(obj, dict) else ""
                if (
                    not isinstance(query, str)
                    or not 0 < len(query.strip()) <= 200
                    or "<|" in query
                ):
                    raise ValueError
            except (ValueError, UnicodeError):
                self.send_error(400)
                return
            proposal = self.generate(query.strip())
            if proposal is None:
                return
            self.send_bytes(
                200,
                json.dumps(
                    {"completion": json.dumps(proposal, ensure_ascii=False)}
                ).encode(),
                "application/json",
            )

    return ThreadingHTTPServer(
        ("127.0.0.1", port), functools.partial(Handler, directory=str(root))
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument(
        "--adapter", type=Path, help="Serve this adapter on a local CUDA GPU"
    )
    cli.add_argument(
        "--site",
        type=Path,
        default=site_root(),
        help="Directory with the live storefront",
    )
    args = cli.parse_args()
    system = json.loads((args.site / "data/prompt.json").read_text())["system"]
    predictor = gpu_predictor(args.adapter, system) if args.adapter else None
    with make_server(args.port, args.site, predictor) as http:
        print(
            f"Storefront listening at http://127.0.0.1:{http.server_port}", flush=True
        )
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
