"""Compare retrieval and SRL on fixed, manually labeled queries outside the curriculum."""
import json
from pathlib import Path

from pulp_fictional.engine import Engine, normalize
from pulp_fictional.learning import DATA
from pulp_fictional.server import Search


def main():
    gold = [json.loads(line) for line in Path(__file__).with_name("tail.jsonl").read_text().splitlines()]
    seen = {normalize(row["query"]) for row in json.loads(DATA.read_text())["rows"]}
    if any(normalize(row["query"]) in seen for row in gold):
        raise ValueError("Training and evaluation queries overlap")
    engine = Engine()
    search = Search()

    def raw(query):
        words = query.casefold().split()
        return {"products": [p for p in engine.products if all(w in p["name"].casefold().split() for w in words)]}

    for name, function in [("Literal product-name search", raw), ("Catalog rules", engine.search),
                           ("CPU student + catalog rules", search)]:
        exact = positives = top1 = roles = 0
        for row in gold:
            result = function(row["query"])
            ids = [p["id"] for p in result["products"]]
            exact += set(ids) == set(row["ids"])
            positives += bool(row["ids"])
            top1 += bool(ids) and ids[0] in row["ids"]
            if "roles" in result:
                r = result["roles"]
                roles += (r["product"] == row["product"] and r["brand"] == row["brand"]
                          and set(r["attributes"]) == set(row["attributes"]))
        print(f"{name}: exact SKU sets {exact}/{len(gold)}, top-1 on positive queries {top1}/{positives}"
              + (f", exact grounded roles {roles}/{len(gold)}" if function != raw else ""))


if __name__ == "__main__":
    main()
