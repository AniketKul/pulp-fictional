"""Resident model inference and catalog ranking for the local storefront."""

import json
import os
import re
import selectors
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path
from threading import BoundedSemaphore, Lock
from urllib.parse import parse_qs


class RequestError(Exception):
    def __init__(self, status, message, retry=None):
        super().__init__(message)
        self.status = status
        self.retry = retry


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def validate_query(query):
    if not isinstance(query, str) or not 0 < len(query) <= 200 or not query.strip():
        raise RequestError(400, "Enter one query of 1 to 200 characters.")
    if "<|" in unicodedata.normalize("NFKC", query) or any(
        unicodedata.category(char) in {"Cc", "Cs"} for char in query
    ):
        raise RequestError(400, "The query contains unsupported characters.")
    return query.strip()


def search_parameters(raw):
    try:
        if len(raw) > 4096 or re.search(r"%(?![0-9a-fA-F]{2})", raw):
            raise ValueError
        fields = parse_qs(
            raw,
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
            max_num_fields=2,
        )
        if set(fields) - {"q", "force"} or len(fields.get("q", [])) != 1:
            raise ValueError
        force = fields.get("force", ["0"])
        if len(force) != 1 or force[0] not in ("0", "1"):
            raise ValueError
    except ValueError:
        raise RequestError(
            400, "Supply one q and an optional force value of 0 or 1."
        ) from None
    return validate_query(fields["q"][0]), force[0] == "1"


INTENT_STRINGS = ("product", "brand", "synonym", "substitute", "broader")


def parse_intent(raw):
    if not isinstance(raw, str):
        raise ValueError("Expected JSON text.")

    def unique_keys(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("Duplicate intent fields.")
        return obj

    payload = json.loads(
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip(),
        object_pairs_hook=unique_keys,
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {*INTENT_STRINGS, "attributes"}
        or not all(isinstance(payload[f], str) for f in INTENT_STRINGS)
        or not isinstance(payload["attributes"], list)
        or not all(isinstance(a, str) for a in payload["attributes"])
    ):
        raise ValueError("Invalid intent fields.")
    return payload


def recovery_schema(query):
    # Offer whole source clauses to the decoder so an ingredient is not detached
    # from its exclusion, e.g. "without peanuts" or "不要花生".
    clauses = re.split(
        r"[的,;，；、。!?]+|(?=\b(?:without|including|with|excluding|except)\b)",
        query,
        flags=re.IGNORECASE,
    )
    choices = sorted({part.strip() for part in [query, *clauses] if part.strip()})
    fields = {
        key: {"type": "string"}
        for key in (
            "product",
            "brand",
            "attributes",
            "synonym",
            "substitute",
            "broader",
        )
    }
    fields["brand"] = {"type": "string", "enum": ["", *choices]}
    fields["attributes"] = {
        "type": "array",
        "items": {"type": "string", "enum": choices},
        "maxItems": 4,
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


class CatalogRanker:
    """Keep one Node process with the published index in memory; serialize its NDJSON pipe."""

    def __init__(self, root):
        self.root = root
        self.node = shutil.which("node")
        self.worker = None
        self.closed = False
        self.lock = Lock()

    def _stop(self):
        worker = self.worker
        self.worker = None
        if worker:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
            worker.stdin.close()
            worker.stdout.close()

    def close(self):
        with self.lock:
            self.closed = True
            self._stop()

    def rank(self, request):
        if not self.node:
            raise RequestError(503, "Catalog ranking requires Node.js.")
        with self.lock:
            if self.closed:
                raise RequestError(503, "Catalog ranking is shutting down.")
            try:
                if self.worker is None or self.worker.poll() is not None:
                    self._stop()
                    self.worker = subprocess.Popen(
                        [
                            self.node,
                            str(Path(__file__).with_name("rank.cjs")),
                            str(self.root),
                            "--serve",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        bufsize=0,
                    )
                body = json.dumps(request, ensure_ascii=False).encode() + b"\n"
                self.worker.stdin.write(body)
                deadline = time.monotonic() + 15
                result = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(self.worker.stdout, selectors.EVENT_READ)
                    while not result.endswith(b"\n"):
                        if not selector.select(max(0, deadline - time.monotonic())):
                            raise TimeoutError
                        chunk = os.read(self.worker.stdout.fileno(), 65536)
                        if not chunk or len(result) + len(chunk) > 4_000_000:
                            raise ValueError
                        result.extend(chunk)
                data = json.loads(result)
                if "products" not in data:
                    raise ValueError
                return data
            except (OSError, ValueError):
                self._stop()
                raise RequestError(
                    503, "Catalog ranking is unavailable.", retry=2
                ) from None


class SearchService:
    def __init__(self, root, predict=None, inference_capacity=3):
        self.predict = predict
        self.gpu_slots = BoundedSemaphore(inference_capacity)
        self.gpu_lock = Lock()
        self.ranker = CatalogRanker(root)
        market = json.loads((root / "data/market.json").read_bytes())
        self.heads = {
            normalize(term): row
            for row in market["cache"]
            for term in (row["query"], row["english"])
            if term
        }
        source = json.loads((root / "data/source.json").read_bytes())
        self.observed = {
            normalize(row["query"]): row["query"] for row in source["queries"]
        }
        # Only the 12 offline head families may be memoized. A tail always runs the model.
        self.head_results = {}
        self.head_lock = Lock()

    def close(self):
        self.ranker.close()

    def generate(self, query):
        if self.predict is None:
            raise RequestError(503, "The resident language model is unavailable.")
        if not self.gpu_slots.acquire(blocking=False):
            raise RequestError(
                503, "All inference slots are busy. Try again shortly.", retry=2
            )
        try:
            with self.gpu_lock:
                raw = self.predict(query)
                try:
                    return parse_intent(raw)
                except ValueError:
                    retry = getattr(self.predict, "retry_invalid_intent", None)
                    if retry is None:
                        raise
                    return parse_intent(retry(query))
        except (ValueError, TypeError, AttributeError):
            raise RequestError(
                502, "The student returned an incomplete intent."
            ) from None
        except RuntimeError:
            raise RequestError(502, "GPU inference failed.") from None
        finally:
            self.gpu_slots.release()

    def search(self, query, force=False):
        qkey = normalize(query)
        head = self.heads.get(qkey)
        source_query = head["query"] if head else self.observed.get(qkey)
        if head and not force:
            # The canonical spelling of each head alias is stable across casing/spacing.
            with self.head_lock:
                if qkey not in self.head_results:
                    canonical = next(
                        term
                        for term in (head["query"], head["english"])
                        if normalize(term) == qkey
                    )
                    proposal = {
                        **head["roles"],
                        **{
                            field: head[field]
                            for field in (
                                "synonym",
                                "substitute",
                                "broader",
                                "categories",
                            )
                        },
                    }
                    self.head_results[qkey] = self.ranker.rank(
                        {
                            "query": canonical,
                            "proposal": proposal,
                            "source_query": source_query,
                            "route": "teacher-cache",
                            "backend": "Qwen2.5-7B-Instruct · offline",
                        }
                    )
                return {**self.head_results[qkey], "query": query}
        if not self.ranker.node:
            raise RequestError(503, "Catalog ranking requires Node.js.")
        proposal = self.generate(query)
        return self.ranker.rank(
            {
                "query": query,
                "proposal": proposal,
                "source_query": source_query,
                "route": "live-student",
                "backend": "Qwen3-0.6B · merged LoRA · GPU",
            }
        )
