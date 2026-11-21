"""Ground a Qwen teacher in public purchase evidence and bilingual grocery catalog embeddings."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import unicodedata

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_embeddings import digest, fit_index, tokens, write_index

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
CATEGORY_MIN_COSINE = 0.15
REWRITE_MIN_COSINE = {"synonym": 0.30, "substitute": 0.15, "broader": 0.20}
RELEVANT_ITEM_MIN_COSINE = 0.15
UNKNOWN_BRANDS = {"", "无品牌", "无", "unknown"}
ROLE_TASK = """Interpret the English grocery query. Return one JSON object {"product":"","brand":"","attributes":[]}.
Product is a short English noun phrase. Use the catalog and logged purchases to disambiguate opaque words.
Only include a brand or attributes actually requested by the shopper. Never infer an attribute from a purchase.
Broad queries keep their broad meaning. Keep attributes as exact phrases from the English query."""
CATEGORY_TASK = """Classify the query using only the supplied numbered category paths. Select a path if it includes the
requested product, even if the path is a parent category. A past purchase alone cannot make an unrelated path relevant.
Output a JSON array of relevant integer candidate numbers, most relevant first. Return [] only if none apply."""
# Prompt demonstrations illustrate each relation. They never become shopper events or supervised training rows.
RELATIONS = {
    "synonym": ("Name one different search phrase for the very same product. Preserve all requested constraints.",
                [("chickpeas", "garbanzo beans"), ("sweet bell pepper", "sweet capsicum")]),
    "substitute": ("Name one different grocery product that can directly substitute for the query in the same use. "
                   "Preserve explicit dietary or brand constraints. Do not simply reword the original.",
                   [("pinto beans", "kidney beans"), ("plant-based oat creamer", "plant-based soy creamer")]),
    "broader": ("Name a genuine parent product concept that includes the original and other products. "
                "Keep all explicit dietary or brand constraints. Do not narrow the query or add features.",
                [("Fusilli pasta", "pasta"), ("gluten-free penne", "gluten-free pasta")]),
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def english_ok(value):
    return isinstance(value, str) and bool(re.search(r"[A-Za-z]", value)) and not re.search(r"[\u3400-\u9fff]", value)


def clean(value, limit):
    if not isinstance(value, str):
        return ""
    value = value.strip().strip('"').strip()
    return value if len(value) <= limit and not re.search(r"[\n{}<>]", value) else ""


def parse(raw, default):
    try:
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip())
    except ValueError:
        return default


def normalized(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class Grounding:
    def __init__(self, catalog, index):
        self.products = {p["id"]: p for p in catalog}
        self.index = index
        self.positions = {word: i for i, word in enumerate(index["vocabulary"])}
        self.basis = np.asarray(index["basis"])
        self.idf = np.asarray(index["idf"])
        self.items = {i: np.asarray(v) for i, v in index["items"].items()}
        self.categories = index["categories"]
        self.category_names = {p["category"]: p["english_category"] for p in catalog}
        by_brand = defaultdict(list)
        for p in catalog:
            if p["brand"].casefold() not in UNKNOWN_BRANDS:
                by_brand[p["brand"]].append(self.items[p["id"]])
        # A brand vector combines the spelling on the source record with the centroid of its actual listings.
        self.brands = {brand: normalized(self.encode(brand) + np.mean(items, axis=0))
                       for brand, items in by_brand.items()}

    def encode(self, text):
        result = np.zeros(self.index["rank"])
        for word, n in Counter(tokens(text)).items():
            pos = self.positions.get(word)
            if pos is not None:
                result += np.log1p(n) * self.idf[pos] * self.basis[pos]
        return normalized(result)

    def evidence(self, query, english):
        vector = self.encode(query["query"] + " " + english)
        categories, brands = Counter(), Counter()
        purchased = []
        for sku, count in sorted(query["purchases"].items(), key=lambda kv: (-kv[1], kv[0])):
            p = self.products[sku]
            categories[p["category"]] += count
            # Count each query/SKU purchase incidence once. Unknown brand remains a visible evidence bucket.
            brands[p["brand"]] += count
            purchased.append({"item_id": sku, "title": p["english_title"], "brand": p["brand"],
                              "purchase_occurrences": count})
        top_categories = sorted(categories, key=lambda path: (-categories[path], path))[:8]
        converted = [{"number": i, "path": self.category_names[key], "source_path": key,
                      "purchase_occurrences": categories[key], "cosine": round(float(vector @ self.categories[key]), 6)}
                     for i, key in enumerate(top_categories, 1)]
        top_brands = sorted(self.brands, key=lambda b: (-float(vector @ self.brands[b]), b))[:5]
        top_items = sorted(self.items, key=lambda sku: (-float(vector @ self.items[sku]), sku))[:4]
        return {"query": english, "original_query": query["query"], "purchased_products": purchased,
                "converted_categories": converted,
                "converted_brands": [{"brand": b, "purchase_occurrences": n}
                                     for b, n in sorted(brands.items(), key=lambda kv: (-kv[1], kv[0]))],
                "similar_brands": [{"brand": b, "cosine": round(float(vector @ self.brands[b]), 6)} for b in top_brands],
                "similar_products": [{"item_id": s, "title": self.products[s]["english_title"],
                                      "cosine": round(float(vector @ self.items[s]), 6)} for s in top_items]}

    def rewrite(self, query_vector, english, raw, relation):
        text = clean(raw, 90)
        lexical = (english_ok(text) and text.casefold() != english.casefold() and
                   not re.search(r"\b(?:none|null|n/a|unknown)\b", text, re.I))
        vec = self.encode(text) if lexical else np.zeros(self.index["rank"])
        similarity = float(vec @ query_vector)
        best_item = max(float(vec @ v) for v in self.items.values())
        allowed = bool(lexical and similarity >= REWRITE_MIN_COSINE[relation] and best_item >= RELEVANT_ITEM_MIN_COSINE)
        return (text if allowed else ""), {"query_cosine": round(similarity, 6), "best_item_cosine": round(best_item, 6),
                                           "accepted": allowed}


def make_generator():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16).to("cuda").eval()

    def generate(task, contexts, limit, demos=()):
        answers = []
        for start in range(0, len(contexts), 16):
            chats = []
            for context in contexts[start:start + 16]:
                chat = [{"role": "system", "content": task}]
                for query, answer in demos:
                    chat.extend([{"role": "user", "content": json.dumps({"query": query} if isinstance(query, str) else query)},
                                 {"role": "assistant", "content": answer}])
                chat.append({"role": "user", "content": json.dumps(context, ensure_ascii=False)})
                chats.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
            inputs = tok(chats, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=limit, do_sample=False, pad_token_id=tok.pad_token_id)
            answers.extend(x.strip() for x in tok.batch_decode(result[:, inputs.input_ids.shape[1]:], skip_special_tokens=True))
        print(f"Completed {len(answers)} teacher responses for {task[:36]}", flush=True)
        return answers
    return generate


def fold_brand(text):
    return "".join(ch for ch in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(ch))


def confirmed_brand(proposal, evidence, titles):
    brand_sources = [b["brand"] for b in evidence["converted_brands"] + evidence["similar_brands"]]
    catalog = fold_brand(" ".join(brand_sources + titles))
    english = fold_brand(evidence["query"])
    source = fold_brand(evidence["original_query"])
    aliases = [alias.strip() for alias in re.split(r"[/／]", proposal) if alias.strip()]
    # Source names can join two scripts. Prefer the actual query spelling, e.g. Dove in 德芙/DOVE.
    for query, displayed in ((english, evidence["query"]), (source, evidence["original_query"])):
        for alias in aliases:
            key = fold_brand(alias)
            if re.search(r"[a-z0-9]", key):
                pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
            else:
                pattern = re.escape(key)
            match = re.search(pattern, query)
            if match and re.search(pattern, catalog):
                return displayed[match.start():match.end()] if len(query) == len(displayed) else alias
    return ""


def validated_roles(raw, evidence):
    proposal = parse(raw, {})
    if not isinstance(proposal, dict):
        proposal = {}
    product = clean(proposal.get("product"), 80)
    brand = clean(proposal.get("brand"), 70)
    attrs = proposal.get("attributes")
    eligible = (english_ok(product) and isinstance(proposal.get("brand"), str) and isinstance(attrs, list)
                and all(isinstance(a, str) for a in attrs))
    titles = [p["title"] for p in evidence["purchased_products"] + evidence["similar_products"]]
    context_words = set(tokens(" ".join([evidence["query"], *titles, *(c["path"] for c in evidence["converted_categories"])])))
    eligible = bool(eligible and context_words.intersection(tokens(product)))
    if not eligible:
        return {"product": "", "brand": "", "attributes": []}, False
    # A frequent purchased brand is evidence for a role, never an automatic brand constraint for a broad query.
    brand = confirmed_brand(brand, evidence, titles)
    attributes = [a for value in attrs if (a := clean(value, 50)) and a.casefold() in evidence["query"].casefold()]
    return {"product": product, "brand": brand, "attributes": list(dict.fromkeys(attributes))}, True


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=ROOT / "live/data/source.json")
    cli.add_argument("--glossary", type=Path, default=ROOT / "live/data/glossary.json")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())
    gloss = json.loads(args.glossary.read_text())
    if not all(english_ok(s) for part in gloss.values() for s in part.values()):
        raise ValueError("Incomplete English glossary")
    catalog = [{**p, "english_title": gloss["titles"][p["id"]], "english_category":
                " > ".join(gloss["levels"][level.strip()] for level in p["category"].split(">"))}
               for p in source["products"]]
    index = fit_index(catalog, source, gloss["queries"])
    grounding = Grounding(catalog, index)
    queries = source["queries"]
    contexts = [grounding.evidence(q, gloss["queries"][q["query"]]) for q in queries]
    generate = make_generator()
    tagged = generate(ROLE_TASK, contexts, 80)
    category_prompts = [{"query": c["query"], "converted_categories": [
        {k: opt[k] for k in ("number", "path", "purchase_occurrences")} for opt in c["converted_categories"]],
        "purchased_products": c["purchased_products"]} for c in contexts]
    reranked = generate(CATEGORY_TASK, category_prompts, 64, [
        ({"query": "plain yogurt", "converted_categories": [{"number": 1, "path": "Dairy > Yogurt"},
                                                           {"number": 2, "path": "Cleaning > Dish soap"}]}, "[1]"),
        ({"query": "fresh radishes", "converted_categories": [{"number": 1, "path": "Drinks > Cola"}]}, "[]"),
    ])
    phrase_contexts = [{"query": c["query"], "catalog_context": [p["title"] for p in c["similar_products"]]} for c in contexts]
    rewrites = {key: generate(task + " Return only one short English phrase, or NONE.", phrase_contexts, 32, demos)
                for key, (task, demos) in RELATIONS.items()}
    rows = []
    for i, (q, evidence) in enumerate(zip(queries, contexts)):
        roles, valid = validated_roles(tagged[i], evidence)
        proposed = parse(reranked[i], [])
        options = {c["number"]: c for c in evidence["converted_categories"]}
        numbers = []
        if isinstance(proposed, list):
            for value in proposed:
                if type(value) is int or (isinstance(value, str) and re.fullmatch(r"[0-9]{1,2}", value)):
                    number = int(value)
                    if number in options and number not in numbers:
                        numbers.append(number)
        accepted = [options[n]["path"] for n in numbers if options[n]["cosine"] >= CATEGORY_MIN_COSINE]
        vector = grounding.encode(q["query"] + " " + evidence["query"])
        phrases, checks = {}, {}
        for key, texts in rewrites.items():
            phrases[key], checks[key] = grounding.rewrite(vector, evidence["query"], texts[i], key)
        rows.append({"query": q["query"], "english": evidence["query"], "sessions": q["sessions"],
                     "roles": roles, "categories": accepted, **phrases, "valid": valid,
                     "evidence": evidence, "reranked_candidate_numbers": numbers, "rewrite_validation": checks,
                     "raw": {"roles": tagged[i], "categories": reranked[i], **{k: v[i] for k, v in rewrites.items()}}})
    filter_script = Path(__file__).with_name("filter_teacher_rewrites.cjs")
    response = subprocess.run(["node", str(filter_script)], input=json.dumps(
        {"products": catalog, "source": source, "index": index, "rows": rows}, ensure_ascii=False),
        text=True, capture_output=True, check=True)
    verified = json.loads(response.stdout)
    if len(verified) != len(rows):
        raise ValueError("Catalog rewrite filter omitted source queries")
    for row, matches in zip(rows, verified):
        for key in RELATIONS:
            audit = row["rewrite_validation"][key]
            audit["cosine_passed"] = audit["accepted"]
            audit["accepted"] = bool(matches[key])
            row[key] = matches[key]
    frequencies = Counter(r["english"].casefold() for r in rows)
    ordered = sorted((r for r in rows if r["valid"] and r["categories"] and frequencies[r["english"].casefold()] == 1),
                     key=lambda r: (-r["sessions"], r["query"]))
    heads = ordered[:12]
    if len(heads) != 12 or sum(r["valid"] for r in rows) < 32:
        raise ValueError("Insufficient grounded teacher labels")
    public_fields = ("query", "english", "sessions", "roles", "categories", *RELATIONS)
    market = args.out / "market.json"
    save(market, {"products": catalog, "cache": [{k: r[k] for k in public_fields} for r in heads]})
    embeddings = args.out / "embeddings.json"
    write_index(index, args.source, args.glossary, market, embeddings)
    save(args.out / "teacher.json", {"model": MODEL, "revision": REV, "source_sha256": digest(args.source),
         "glossary_sha256": digest(args.glossary), "builder_sha256": digest(Path(__file__)),
         "embeddings_sha256": digest(embeddings), "embeddings_builder_sha256": index["builder_sha256"],
         "market_sha256": digest(market), "ranker_sha256": digest(ROOT / "live/retrieval.js"),
         "rewrite_filter_sha256": digest(filter_script), "torch": torch.__version__, "transformers": transformers.__version__,
         "gpu": torch.cuda.get_device_name(), "category_min_cosine": CATEGORY_MIN_COSINE,
         "rewrite_min_cosine": REWRITE_MIN_COSINE, "rewrite_item_min_cosine": RELEVANT_ITEM_MIN_COSINE, "rows": rows})
    print(f"Published {len(rows)} source queries, {sum(r['valid'] for r in rows)} valid labels and {len(heads)} cache rows.", flush=True)


if __name__ == "__main__":
    main()
