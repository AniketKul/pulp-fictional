"""Precompute cosine scores for head queries so serving needs no embedding runtime."""
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DATA = Path(__file__).resolve().parents[2] / "src/pulp_fictional/data"


@torch.inference_mode()
def main():
    def read(name):
        return json.loads((DATA / name).read_text())

    queries = sorted(read("conversions.json"))
    products = read("catalog.json")
    paths = sorted({item["category"] for item in read("concepts.json").values()})
    names = [f"{p['brand']} {p['name']}" for p in products]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    encoder = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
    batch = tokenizer(queries + paths + names, padding=True, return_tensors="pt")
    hidden = encoder(**batch).last_hidden_state
    mask = batch.attention_mask.unsqueeze(-1)
    vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1))
    scores = vectors[:len(queries)] @ vectors[len(queries):].T
    output = {}
    for query, values in zip(queries, scores.tolist()):
        categories = dict(zip(paths, values[:len(paths)]))
        catalog = dict(zip([p["id"] for p in products], values[len(paths):]))
        output[query] = {key: {k: round(v, 6) for k, v in section.items()}
                         for key, section in [("categories", categories), ("products", catalog)]}
    artifact = {"model": MODEL, "revision": REVISION, "queries": output}
    (DATA / "semantic.json").write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
