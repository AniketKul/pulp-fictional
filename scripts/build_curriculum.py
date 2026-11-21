# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch==2.6.0", "transformers==4.52.4"]
# ///
"""Generate the bundled teacher cache and fit the CPU student with a CUDA teacher."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pulp_fictional.engine import DATA as CATALOG, Engine, normalize
from pulp_fictional.learning import DATA, train


MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TYPES = {
    "synonyms": "Select differently worded searches that mean the same thing and preserve every constraint.",
    "substitutes": "Select distinct products that can fill the same shopping need. Keep dietary and brand requirements.",
    "broader": "Select less specific searches that include the original product. Keep explicit shopping requirements.",
}
SLOT_TASK = (
    'Read the shopper query and return exactly {"product":"", "brand":"", "attributes":[]} with filled values. '
    "Use the supplied catalog spelling. Extract the requested product, an explicitly named brand (or the brand of a "
    "uniquely named item), and ONLY the features the shopper asks for. An item's other features are irrelevant. "
    'For "no gluten sandwich bread", output {"product":"bread","brand":"","attributes":["gluten-free"]}. '
    'For a generic product like "milk", preserve that broad term. Empty slots are allowed.'
)
TRAIN_PHRASES = (
    "no gluten sandwich loaf", "sliced bread without gluten", "gf multigrain bread", "bread gluten free",
    "extra large resealable bags", "zip lock bags quart", "freezer safe bags xl", "wholewheat sandwich loaf",
    "2 percent dairy milk", "cow's milk no lactose", "reduced fat chocolate milk", "milk no dairy",
    "barista oat drink", "organic vanilla oat beverage", "unsweetened almond drink", "plant based holiday roast",
    "animal free plant roast", "firm bean curd", "organic coriander leaves", "fresh curly parsley",
    "flat parsley organic", "organic blended fruit drink", "galaxy telescope", "bicycle wheels",
    "laptop battery", "concert tickets", "purple satellite dish", "bagels cinnamon", "sourdough sandwich loaf",
)


def training_queries(engine):
    """Use synthetic conversion-log queries as heads and catalog-derived examples as extra supervision."""
    heads = set(engine.conversions)
    queries = list(engine.conversions) + list(TRAIN_PHRASES)
    for concept, fields in engine.concepts.items():
        queries.extend([concept, *fields["aliases"]])
    for sku in engine.products:
        queries.extend([sku["name"], f"{sku['brand']} {sku['product']}"])
        if sku["attributes"]:
            queries.append(f"{sku['attributes'][0]} {sku['product']}")
    unique = {}
    for query in queries:
        unique.setdefault(normalize(query), query)
    return [(query, normalized in heads) for normalized, query in unique.items()]


def request(task, query, evidence):
    return [
        {"role": "system", "content": "You label grocery searches. Return a single JSON object without commentary."},
        {"role": "user", "content": task + "\n" + json.dumps({"shopper_query": query, "evidence": evidence})},
    ]


def tasks(engine, query, head):
    context = engine.context(query)
    skus = [{f: p[f] for f in ("name", "product", "brand", "attributes")} for p in context["products"][:6]]
    paths = [{"id": i, **{k: c[k] for k in ("category", "conversions", "cosine")}}
             for i, c in enumerate(context["candidates"], start=1)]
    neighbors = [{k: p[k] for k in ("name", "brand", "product", "cosine")}
                 for p in context["retrieved_products"]]
    evidence = {"matching_skus": skus, "embedding_neighbors": neighbors, "categories": paths}
    messages = request(SLOT_TASK, query, evidence)
    messages[1:1] = [
        {"role": "user", "content": 'Shopper query: "vanilla soy drink". Matching SKU is Vanilla Soy Milk, '
         'brand Green Farm, attributes vanilla, vegan, organic, dairy-free.'},
        {"role": "assistant", "content": '{"product":"soy milk","brand":"","attributes":["vanilla"]}'},
        {"role": "user", "content": 'Shopper query: "bagels". Matching SKU is Cinnamon Breakfast Bagels, '
         'brand Little Baker, attributes cinnamon and organic.'},
        {"role": "assistant", "content": '{"product":"bagels","brand":"","attributes":[]}'},
        {"role": "user", "content": 'Shopper query: "bicycle seat". No matching grocery SKUs.'},
        {"role": "assistant", "content": '{"product":"","brand":"","attributes":[]}'},
    ]
    messages[-1]["content"] += (
        "\nReport the shopper's own words as catalog labels. A SKU being sold by a brand does not mean "
        "the shopper asked for that brand. If there are no matching SKUs, return empty slots."
    )
    jobs = {"roles": messages}
    if head:
        jobs["categories"] = request(
            'Rank the relevant shopping categories by their integer id. '
            'A purchase count or a high cosine alone cannot establish relevance. '
            'Example: for "oranges" with 1=crackers and 2=citrus, answer {"ids":[2]}. '
            'Copy only integer ids for relevant categories into the ids array.',
            query, evidence,
        )
        for field, instruction in TYPES.items():
            jobs[field] = request(
                instruction + ' Choose up to three from the supplied candidates. Return {"values":[]} when none qualify.',
                query, {"candidates": context["rewrites"][field]},
            )
    return jobs


def parse(text):
    # Small instruction models sometimes surround a valid JSON object with a Markdown fence.
    try:
        start = text.index("{")
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def strings(items):
    return [s for s in items if isinstance(s, str)] if isinstance(items, list) else []


def ground(engine, query, raw, head):
    roles = parse(raw["roles"])
    rewrites = {k: strings(parse(raw.get(k, "")).get("values")) for k in TYPES}
    context = engine.context(query)
    candidates = {i: c["category"] for i, c in enumerate(context["candidates"], 1)}
    ids = parse(raw.get("categories", "")).get("ids", [])
    cats = [candidates[i] for i in ids if type(i) is int and i in candidates] if isinstance(ids, list) else []
    checked = engine.search(query, roles=roles, categories=cats, rewrites=rewrites)
    accepted = {}
    # Never silently turn a missing or rejected teacher answer into a rule-generated training label.
    for field in ("product", "brand"):
        value = roles.get(field)
        supported = checked["roles"][field]
        normalized = normalize(value) if isinstance(value, str) else ""
        if field == "product":
            normalized = normalize(engine.aliases.get(normalized, normalized))
        accepted[field] = supported if normalized == normalize(supported) else ""
    proposed = {engine.attributes.get(normalize(a), a) for a in strings(roles.get("attributes"))}
    accepted["attributes"] = [a for a in checked["roles"]["attributes"] if a in proposed]
    proposal = {"roles": accepted}
    if head:
        proposal.update(categories=[c for c in checked["categories"] if c in cats], rewrites=checked["rewrites"])
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("Building the teacher curriculum requires a CUDA GPU with bfloat16 support.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).to("cuda").eval()
    engine = Engine()
    examples = training_queries(engine)
    raw = [{} for _ in examples]
    jobs = [(i, field, messages) for i, (q, head) in enumerate(examples) for field, messages in tasks(engine, q, head).items()]
    for start in range(0, len(jobs), 8):
        batch = jobs[start:start + 8]
        prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for _, _, messages in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            tokens = model.generate(**inputs, do_sample=False, max_new_tokens=120)
        answers = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for (i, field, _), answer in zip(batch, answers, strict=True):
            raw[i][field] = answer
        print(f"Generated {min(start + 8, len(jobs))}/{len(jobs)} teacher responses", flush=True)

    rows = [{"query": q, "head": head, "raw": answer, "proposal": ground(engine, q, answer, head)}
            for (q, head), answer in zip(examples, raw, strict=True)]
    for row in rows:
        slots = parse(row["raw"]["roles"])
        roles = row["proposal"]["roles"]
        expected = engine.context(row["query"])["roles"]
        well_formed = (isinstance(slots.get("product"), str) and isinstance(slots.get("brand"), str)
                       and isinstance(slots.get("attributes"), list)
                       and all(isinstance(a, str) for a in slots["attributes"]))
        # Unknown examples teach abstention only if the teacher itself returned an empty label.
        trustworthy_empty = any(expected.values()) or not any(slots.get(k) for k in expected)
        row["trainable"] = bool(well_formed and trustworthy_empty and roles == expected)
    recognized = sum(bool(parse(row["raw"]["roles"])) for row in rows)
    if recognized < 0.9 * len(rows):
        raise ValueError(f"Teacher returned parseable role JSON for only {recognized}/{len(rows)} queries")
    sources = {name: sha256((CATALOG / name).read_bytes()).hexdigest()
               for name in ("catalog.json", "concepts.json", "attributes.json", "conversions.json", "semantic.json")}
    metadata = {"model": MODEL, "revision": REVISION, "torch": torch.__version__, "transformers": version("transformers"),
                "device": torch.cuda.get_device_name(), "catalog_sha256": sources,
                "engine_sha256": sha256((CATALOG.parent / "engine.py").read_bytes()).hexdigest(),
                "semantic_py_sha256": sha256((CATALOG.parent / "semantic.py").read_bytes()).hexdigest(),
                "decoding": "greedy", "teacher_prompt_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    training = [r for r in rows if r["trainable"]]
    if len(training) < 80:
        raise ValueError(f"Only {len(training)} role rows passed the training gate")
    student = train(training)
    # Keep one JSON row per query and one row per learned label for reviewable diffs.
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False, encoding="utf-8") as out:
        out.write('{\n"teacher":' + encode(metadata) + ',\n"rows":[\n')
        out.write(",\n".join(map(encode, rows)))
        out.write('\n],\n"student":{\n')
        out.write(",\n".join(encode(k) + ":" + encode(v) for k, v in student.items()))
        out.write("\n}\n}\n")
    os.replace(out.name, args.output)
    print(f"Saved {len(training)} accepted training rows from {len(rows)} queries, {len(engine.conversions)} cache entries and {len(student)} learned labels.")


if __name__ == "__main__":
    main()
