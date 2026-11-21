# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6"]
# ///
"""Fit bilingual latent semantic indexing vectors on published listings and query-item pairs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

TERMS = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
STOP = set("a an and are at by can could do for from get how i in is it me my of on or please some that the to want with would you looking look find show buy need like one have something shop shopping".split())


def tokens(text):
    out = []
    for word in TERMS.findall(unicodedata.normalize("NFKC", text).lower()):
        if "\u3400" <= word[0] <= "\u9fff":
            out.extend(word)
            out.extend(word[i:i+2] for i in range(len(word) - 1))
        else:
            if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
                word = word[:-1]
            if word not in STOP:
                out.append(word)
    return out


def weighted(parts):
    counts = Counter()
    for text, weight in parts:
        for term in tokens(text or ""):
            counts[term] += weight
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("market", "source", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--glossary", type=Path, help="Bilingual catalog glossary (defaults to glossary.json beside the market)")
    args = parser.parse_args()
    glossary = args.glossary or args.market.with_name("glossary.json")
    market = json.loads(args.market.read_text())
    products = market["products"]
    source = json.loads(args.source.read_text())
    ids = [p["id"] for p in products]
    if not products or len(ids) != len(set(ids)) or set(ids) != {p["id"] for p in source["products"]}:
        raise ValueError("Catalog IDs must match the source snapshot")
    by_id = {p["id"]: p for p in products}
    docs = []
    for p in products:
        docs.append(weighted([(p["title"], 2), (p.get("english_title"), 2), (p["brand"], 2),
                              (p["category"], 1), (p.get("english_category"), 1)]))
    # Each observed query–item pair contributes one document. Purchase frequency is reserved for runtime evidence.
    translated = json.loads(glossary.read_text())["queries"]
    if set(translated) != {q["query"] for q in source["queries"]}:
        raise ValueError("Every source query requires an English gloss")
    for q in source["queries"]:
        for sku in q["purchases"]:
            p = by_id[sku]
            docs.append(weighted([(q["query"], 3), (translated.get(q["query"]), 3), (p["title"], 1), (p.get("english_title"), 1),
                                  (p["category"], 1), (p.get("english_category"), 1)]))
    vocab = sorted(set().union(*(d.keys() for d in docs)))
    positions = {word: i for i, word in enumerate(vocab)}
    df = Counter(word for d in docs for word in d)
    idf = np.array([math.log((len(docs)+1)/(df[w]+1)) + 1 for w in vocab])

    def dense(counts):
        row = np.zeros(len(vocab))
        for w, count in counts.items():
            if w in positions:
                col = positions[w]
                row[col] = math.log1p(count) * idf[col]
        return row

    X = np.stack([dense(d) for d in docs])
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    # The document Gram matrix avoids computing a full vocabulary-sized SVD.
    values, left = np.linalg.eigh(X @ X.T)
    keep = np.flatnonzero(values > 1e-8)[-min(64, len(products)):]
    basis = (X.T @ left[:, keep]) / np.sqrt(values[keep])
    vectors = X[:len(ids)] @ basis
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    grouped = defaultdict(list)
    for i, p in enumerate(products):
        grouped[p["category"]].append(i)
    categories = {}
    for category, indices in grouped.items():
        english = " ".join(sorted({products[i].get("english_category", "") for i in indices}))
        path = dense(weighted([(category, 1), (english, 1)])) @ basis
        centroid = vectors[indices].mean(axis=0) + path / max(np.linalg.norm(path), 1e-12)
        categories[category] = (centroid / max(np.linalg.norm(centroid), 1e-12)).round(5).tolist()
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    output = {"method": "bilingual-tfidf-lsi", "rank": int(len(keep)), "documents": len(docs),
              "numpy": np.__version__, "builder_sha256": digest(Path(__file__)),
              "source_sha256": digest(args.source), "market_sha256": digest(args.market), "glossary_sha256": digest(glossary),
              "vocabulary": vocab, "idf": idf.round(5).tolist(), "basis": basis.round(5).tolist(),
              "items": dict(zip(ids, vectors.round(5).tolist())), "categories": categories}
    args.output.write_text(json.dumps(output, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"{len(vocab)} terms, {len(keep)} dimensions, {len(ids)} items, {len(categories)} catalog paths")


if __name__ == "__main__":
    main()
