"""Serve the resident Pulp Fictional search engine on loopback."""

import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore

from .service import (
    RequestError,
    SearchService,
    recovery_schema,
    search_parameters,
    validate_query,
)

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def site_root():
    installed = Path(__file__).with_name("live")
    return (
        installed
        if installed.is_dir()
        else Path(__file__).resolve().parents[2] / "live"
    )


def gpu_predictor(adapter, system):
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU mode requires a CUDA device with bfloat16 support.")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval()

    tokenizer_data = build_token_enforcer_tokenizer_data(tok)

    def generate(q, *, instructions=system, prefix_allowed_tokens_fn=None):
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": q},
        ]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            answer = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        return tok.decode(answer[0, ids.shape[1] :], skip_special_tokens=True)

    def retry_invalid_intent(query):
        parser = JsonSchemaParser(recovery_schema(query))
        allowed = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
        return generate(
            query,
            instructions=system
            + " Copy brand and attributes directly from the shopper's original text, preserving exclusions.",
            prefix_allowed_tokens_fn=allowed,
        )

    generate.retry_invalid_intent = retry_invalid_intent
    return generate


class LocalServer(ThreadingHTTPServer):
    """Cap active clients, including slow or incomplete HTTP requests."""

    def __init__(self, address, handler, engine, *, connections=16):
        self.slots = BoundedSemaphore(connections)
        self.engine = engine
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        request.settimeout(10)
        if self.slots.acquire(blocking=False):
            try:
                return super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise
        try:
            request.settimeout(0.2)
            request.sendall(
                b"HTTP/1.0 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Length: 0\r\n\r\n"
            )
        except OSError:
            pass
        finally:
            self.shutdown_request(request)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def server_close(self):
        super().server_close()
        self.engine.close()


def make_server(port=8000, root=None, predict=None, *, inference_capacity=3):
    root = (root or site_root()).resolve()
    engine = SearchService(root, predict, inference_capacity)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not getattr(self, "path", "").startswith("/api/"):
                super().log_message(fmt, *args)

        def send_json(self, status, data, *, retry=None):
            content = json.dumps(
                data, ensure_ascii=False, separators=(",", ":")
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            if retry:
                self.send_header("Retry-After", str(retry))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path, _, params = self.path.partition("?")
            if path == "/api/search":
                try:
                    query, force = search_parameters(params)
                    self.send_json(200, engine.search(query, force))
                except RequestError as error:
                    self.send_json(
                        error.status, {"error": str(error)}, retry=error.retry
                    )
            else:
                super().do_GET()

        def list_directory(self, path):
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/intent" or engine.predict is None:
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
                query = validate_query(
                    obj.get("query") if isinstance(obj, dict) else None
                )
                proposal = engine.generate(query)
            except RequestError as error:
                self.send_json(error.status, {"error": str(error)}, retry=error.retry)
                return
            except (ValueError, UnicodeError):
                self.send_json(400, {"error": "Supply a JSON object with a query."})
                return
            self.send_json(
                200, {"completion": json.dumps(proposal, ensure_ascii=False)}
            )

    return LocalServer(
        ("127.0.0.1", port),
        functools.partial(Handler, directory=str(root)),
        engine,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument(
        "--adapter",
        type=Path,
        help="Merge and serve this LoRA adapter on a local CUDA GPU",
    )
    cli.add_argument(
        "--site",
        type=Path,
        default=site_root(),
        help="Directory with the catalog and storefront",
    )
    args = cli.parse_args()
    if not 0 <= args.port <= 65535:
        cli.error("Port must be between 0 and 65535.")
    system = json.loads((args.site / "data/prompt.json").read_bytes())["system"]
    predict = gpu_predictor(args.adapter, system) if args.adapter else None
    with make_server(args.port, args.site, predict) as server:
        print(
            f"Storefront listening at http://127.0.0.1:{server.server_port}", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
