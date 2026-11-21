"""Build a grocery showcase from real KuaiSearch search and purchase records."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "16f555af360f6a86209f60eb1f12462f9e25be2c"
DATASET = "https://huggingface.co/datasets/benchen4395/KuaiSearch"
RECALL = "recall_lite/train.jsonl"
ITEMS = "items_lite/train.jsonl"
PREFIX_BYTES = 256 * 1024 * 1024
FILES = {
    RECALL: (257794992, "8949ed6b5cf685bb69067710ac70da12fd7999f66e66bafea4b9d8d78bfefe8a"),
    ITEMS: (PREFIX_BYTES, "52575dbd226fff8b9724faa14a4d102ef9a924921d8b28fdb71628b4ac67a104"),
}
DEPARTMENTS = {"零食/坚果/特产", "生鲜", "速食/干货", "粮油调味", "咖啡/麦片/冲饮", "茶"}
PRODUCT_LIMIT = 144
CONTACT = re.compile(r"https?://|www\.|@|\d{7,}", re.IGNORECASE)


def artifact(key, directory):
    length, expected = FILES[key]
    path = directory / (key.split("/")[0] + ".jsonl")
    if not path.exists():
        url = f"{DATASET}/resolve/{REVISION}/{key}"
        headers = {"Range": f"bytes=0-{length-1}"} if key == ITEMS else {}
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=directory, delete=False) as f:
            temporary = Path(f.name)
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response, temporary.open("wb") as out:
                remaining = length
                while remaining:
                    chunk = response.read(min(2**20, remaining))
                    if not chunk:
                        raise ValueError(f"Incomplete download of {key}")
                    out.write(chunk)
                    remaining -= len(chunk)
            verify(temporary, length, expected)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        verify(path, length, expected)
    return path


def verify(path, length, expected):
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    if path.stat().st_size != length or digest.hexdigest() != expected:
        raise ValueError(f"Size or SHA-256 mismatch for {path}")


def rows(path):
    with path.open("rb") as f:
        for line in f:
            if line.endswith(b"\n"):
                yield json.loads(line)


def build(catalog_path, recall_path):
    products = {}
    catalog_rows = 0
    for item in rows(catalog_path):
        catalog_rows += 1
        top = item["category_level1_name"]
        if top not in DEPARTMENTS or CONTACT.search(item["item_title"]):
            continue
        path = [item[f"category_level{i}_name"] for i in (1, 2, 3)]
        products[item["item_id"]] = {
            "id": str(item["item_id"]), "title": item["item_title"], "brand": item["brand_name"],
            "category": " > ".join(segment for segment in path if segment != "UNKNOWN"),
        }

    splits = Counter()
    session_keys = set()
    query_sessions = Counter()
    counts = defaultdict(Counter)
    for row in rows(recall_path):
        split = row["split"]
        splits[split] += 1
        query = row["query"]
        if split != "train" or not isinstance(query, str) or not 1 <= len(query.strip()) <= 45 or CONTACT.search(query):
            continue
        session = (row["user_id"], row["session_id"])
        if session in session_keys:
            raise ValueError("Duplicate search session in the training split")
        session_keys.add(session)
        query_sessions[query] += 1
        for sku in set(row["purchased_item_ids"]):
            if sku in products:
                counts[sku][query] += 1

    # Round robin across second-level categories avoids a storefront dominated by one department.
    groups = defaultdict(list)
    for sku in counts:
        groups[" > ".join(products[sku]["category"].split(" > ")[:2])].append(sku)
    for group in groups.values():
        group.sort(key=lambda sku: (-sum(counts[sku].values()), sku))
    ordered = [skus for _, skus in sorted(groups.items())]
    selected = []
    for index in range(max(map(len, ordered), default=0)):
        selected.extend(group[index] for group in ordered if index < len(group))
        if len(selected) >= PRODUCT_LIMIT:
            break
    selected = sorted(selected[:PRODUCT_LIMIT])
    queries = defaultdict(dict)
    for sku in selected:
        for query, count in counts[sku].items():
            queries[query][str(sku)] = count
    source = {
        "dataset": "KuaiSearch-Lite", "url": DATASET, "revision": REVISION, "license": "MIT",
        "query_language": "zh", "catalog_source": ITEMS, "catalog_prefix_bytes": PREFIX_BYTES,
        "catalog_prefix_sha256": FILES[ITEMS][1], "complete_catalog_rows_scanned": catalog_rows,
        "eligible_grocery_products_in_prefix": len(products), "recall_source": RECALL,
        "recall_sha256": FILES[RECALL][1], "recall_sessions_by_split": dict(sorted(splits.items())),
        "eligible_train_search_sessions": len(session_keys), "grocery_skus_with_observed_purchases": len(counts),
        "exported_products": len(selected), "exported_distinct_queries": len(queries),
        "exported_query_product_pairs": sum(len(pairs) for pairs in queries.values()),
        "exported_purchase_session_item_pairs": sum(sum(pairs.values()) for pairs in queries.values()),
        "count_unit": "Distinct training search sessions with the exact source query and purchased item ID.",
    }
    return {"products": [products[sku] for sku in selected],
            "queries": [{"query": query, "sessions": query_sessions[query], "purchases": queries[query]} for query in sorted(queries)],
            "source": source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Directory for verified source downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "live/data/source.json")
    args = parser.parse_args()
    result = build(artifact(ITEMS, args.cache), artifact(RECALL, args.cache))
    if len(result["products"]) != PRODUCT_LIMIT or len(result["queries"]) < 60:
        raise ValueError("The verified corpus did not yield the expected grocery slice")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["source"], indent=2))


if __name__ == "__main__":
    main()
