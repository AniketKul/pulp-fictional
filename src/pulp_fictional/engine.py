"""Catalog-grounded query understanding for a small, synthetic grocery store."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from .semantic import load_scores

MIN_CATEGORY_COSINE = 0.16
QUANTITY = re.compile(r"\b\d+(?: \d+)? (?:fl oz|oz|lbs?|g|kg|ml|liters?|gallons?|quarts?|count|ct|packs?)\b")
PURPOSE_WORDS = {"my", "our", "morning", "coffee", "lattes", "latte", "breakfast", "lunch", "dinner",
                 "baking", "cooking", "sandwich", "sandwiches", "toast", "a", "the", "making", "meal", "prep"}
DATA = Path(__file__).with_name("data")
REWRITE_TYPES = ("synonyms", "substitutes", "broader")
STOP = {"i", "a", "an", "the", "some", "please", "looking", "for", "want", "need", "buy", "find", "me", "of"}


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    return " ".join(re.findall(r"[a-z0-9]+%?", text.replace("'", "")))


def _phrases(text: str, aliases: dict[str, str]) -> tuple[str, list[str]]:
    found = []
    padded = f" {text} "
    for alias, label in aliases.items():  # Aliases are ordered longest first.
        if f" {alias} " in padded:
            padded = padded.replace(f" {alias} ", " ")
            if label not in found:
                found.append(label)
    return padded.strip(), found


def _load(filename: str) -> Any:
    return json.loads((DATA / filename).read_text(encoding="utf-8"))


def _strings(values: object) -> list[str]:
    return [v for v in values if isinstance(v, str)] if isinstance(values, list) else []


class Engine:
    def __init__(self) -> None:
        self.semantic = load_scores()
        self.concepts = _load("concepts.json")
        self.products = _load("catalog.json")
        self.conversions = _load("conversions.json")
        attributes = _load("attributes.json")
        self.attributes = {normalize(v): k for k, vs in attributes.items() for v in [k, *vs]}
        self.attributes = dict(sorted(self.attributes.items(), key=lambda pair: -len(pair[0])))
        self.brands = {normalize(p["brand"]): p["brand"] for p in self.products}
        self.by_id = {p["id"]: p for p in self.products}
        self.aliases = {}
        self.words = {}
        for kind, concept in self.concepts.items():
            for alias in [kind, *concept["aliases"]]:
                self.aliases[normalize(alias)] = kind
        for p in self.products:
            concept = self.concepts[p["product"]]
            p["category"] = concept["category"]
            # A combined department like "Meat & Alternatives" cannot label beef as an alternative.
            segments = [part for part in p["category"].split(" > ") if "&" not in part]
            terms = [p["product"], p["name"], p["size"], *segments, concept["broader"], *concept["aliases"]]
            self.words[p["id"]] = set(normalize(" ".join(terms)).split())
        self.vocabulary = sorted(set.union(*self.words.values()))

    def _analyze(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            raise ValueError("Search text must contain 1 to 200 characters.")
        key = normalize(query)
        remaining, brands = _phrases(key, self.brands)
        remaining, attrs = _phrases(remaining, self.attributes)
        folded = unicodedata.normalize("NFKC", query)
        unsupported_script = any(ch.isalnum() and not ch.isascii() for ch in folded)
        unary_minus = any((i == 0 or folded[i - 1].isspace()) and (ch == "−" or unicodedata.category(ch) == "Pd")
                          for i, ch in enumerate(folded))
        if unsupported_script or unary_minus or re.search(r"\b(no|not|without|except|excluding|free|anything but)\b", remaining):
            guard = "unsupported script" if unsupported_script else "unsupported exclusion"
            return {"query": query.strip(), "products": [], "candidates": [], "guard_mode": guard,
                    "roles": {"product": "", "brand": "", "attributes": []}, "phrase": "", "exact_kind": None}
        remaining = re.sub(r"\b(?:dairy|milk) alternatives?\b", "plant milk", remaining)
        tokens = [w for w in remaining.split() if w not in STOP]
        corrected = []
        for w in tokens:
            if len(w) >= 4 and w not in self.vocabulary:
                corrected.extend(get_close_matches(w, self.vocabulary, n=1, cutoff=0.84) or [w])
            else:
                corrected.append(w)
        terms = set(corrected)
        phrase = " ".join(corrected)
        # The catalog's plant roast shares its noun with the beef roast.
        exact_kind = self.aliases.get(phrase) if phrase != "roast" else None
        historical = self.conversions.get(key, {})
        counts: Counter[str] = Counter()
        for sku, count in historical.items():
            counts[self.by_id[sku]["category"]] += count
        embedding = self.semantic.get(key)
        cosine = embedding["categories"] if embedding else {}
        converted = dict(counts.most_common(5))
        ranked = []
        for p in self.products:
            if not (terms or attrs or brands):
                continue
            if brands and p["brand"] not in brands or not set(attrs) <= set(p["attributes"]):
                continue
            exact_segment = phrase in [normalize(level.replace("&", "and")) for level in p["category"].split(" > ")]
            if (not exact_segment and not terms <= self.words[p["id"]]) or (exact_kind and p["product"] != exact_kind):
                continue
            concept = self.concepts[p["product"]]
            semantic = 3 if exact_kind else 2 if phrase and phrase in normalize(p["name"]) else 1
            if phrase and phrase == normalize(concept["broader"]):
                semantic = max(semantic, 2)
            # A conversion can break a lexical tie, but can never admit an unrelated SKU.
            ranked.append((semantic, historical.get(p["id"], 0), p))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]["id"]))
        relevance: dict[str, int] = {}
        for score, _, p in ranked:
            relevance[p["category"]] = max(score, relevance.get(p["category"], 0))
        candidates = sorted(converted.keys() | relevance.keys(),
                            key=lambda c: (-counts[c], -relevance.get(c, 0), c))
        evidence = []
        valid = set()
        for path in candidates:
            similarity = cosine.get(path)
            guard = "accepted"
            if not relevance.get(path):
                guard = "no matching SKU"
            elif embedding and (similarity is None or similarity < MIN_CATEGORY_COSINE):
                guard = "below cosine threshold"
            if guard == "accepted":
                valid.add(path)
            evidence.append({"category": path, "conversions": counts[path], "relevance": relevance.get(path, 0),
                             "cosine": similarity, "accepted": path in valid, "guard": guard})
        # A catalog name match identifies a SKU independently of its category's embedding.
        products = []
        for _, _, p in ranked:
            name = normalize(p["name"])
            identified = (key == name or (len(ranked) == 1 and len(terms) >= 2 and
                          not exact_kind and historical.get(p["id"], 0) and f" {phrase} " in f" {name} "))
            if p["category"] in valid or identified:
                products.append(p)
        kinds = {p["product"] for p in products}
        kind = next(iter(kinds)) if len(kinds) == 1 else phrase if products else ""
        brand = brands[0] if brands else ""
        # A distinctive SKU name plus repeated conversions supports an implicit brand.
        if not brand and len(products) == 1 and len(terms) >= 2 and historical.get(products[0]["id"], 0) and not exact_kind:
            if phrase and phrase in normalize(products[0]["name"]):
                brand = products[0]["brand"]
        return {"query": query.strip(), "products": products,
                "candidates": evidence, "guard_mode": "embedding+catalog" if embedding else "catalog",
                "roles": {"product": kind, "brand": brand, "attributes": attrs},
                "phrase": phrase, "exact_kind": exact_kind}

    def _rewrites(self, context: dict, roles: dict) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {kind: [] for kind in REWRITE_TYPES}
        kind = roles["product"]
        # Size is searchable in the SKU text but has no rewrite slot.
        if QUANTITY.search(normalize(context["query"])):
            return output
        if kind not in self.concepts or not context["products"] or any(p["product"] != kind for p in context["products"]):
            return output
        concept = self.concepts[kind]
        choices = {"synonyms": [kind, *concept["aliases"]],
                   "substitutes": concept["substitutes"], "broader": [concept["broader"]]}
        # An individual SKU nickname has no equivalent at the generic product level.
        is_specific = context["phrase"] and not context["exact_kind"] and context["phrase"] != normalize(kind)
        for relation, terms in choices.items():
            if relation == "synonyms" and is_specific:
                continue
            for term in terms:
                filters = [roles["brand"], *roles["attributes"]]
                rewrite = " ".join(filter(None, [*filters, term]))
                if normalize(rewrite) == normalize(context["query"]):
                    continue
                found = self._analyze(rewrite)["products"]
                if not found:
                    continue
                original_ids = {p["id"] for p in context["products"]}
                target_ids = {p["id"] for p in found}
                if relation == "synonyms" and target_ids != original_ids:
                    continue
                if relation == "broader" and not original_ids < target_ids:
                    continue
                output[relation].append(rewrite)
        return output

    def context(self, query: str) -> dict:
        """Retrieve top converted category candidates (including noise) and matching products."""
        context = self._analyze(query)
        embeddings = self.semantic.get(normalize(query), {}).get("products", {})
        context["retrieved_products"] = [
            {**self.by_id[sku], "cosine": embeddings[sku]}
            for sku in sorted(embeddings, key=embeddings.get, reverse=True)[:5] if sku in self.by_id
        ]
        context["rewrites"] = self._rewrites(context, context["roles"])
        return {k: v for k, v in context.items() if k not in {"phrase", "exact_kind"}}

    def search(self, query: str, roles: dict | None = None, categories: list[str] | None = None,
               rewrites: dict | None = None) -> dict:
        context = self._analyze(query)
        role_recovery = False
        if (not context["products"] and not context["candidates"] and isinstance(roles, dict)
                and context["guard_mode"] == "catalog"):
            # A purpose clause can obscure a known product. Only drop supported usage words.
            core, separator, usage = normalize(query).partition(" for ")
            suggested = roles.get("product")
            if separator and usage and set(usage.split()) <= PURPOSE_WORDS and isinstance(suggested, str):
                anchor = self._analyze(core)
                hypothesis = self.aliases.get(normalize(suggested), normalize(suggested))
                if anchor["products"] and hypothesis == anchor["roles"]["product"]:
                    context = {**anchor, "query": query.strip()}
                    role_recovery = True
        parsed = context["roles"].copy()
        products = context["products"]
        # Ground proposed slots jointly against matching SKUs. Explicit constraints always survive.
        if isinstance(roles, dict) and products:
            suggested = roles.get("product")
            proposed_brand = roles.get("brand")
            proposed_attrs = _strings(roles.get("attributes"))
            # Ambiguous hints may refine a role only when the original query contains supporting text.
            query_text = f" {normalize(query)} "
            for field, value in [("product", suggested), ("brand", proposed_brand)]:
                if not isinstance(value, str) or not value:
                    continue
                canonical = self.aliases.get(normalize(value)) if field == "product" else self.brands.get(normalize(value))
                if canonical and canonical != parsed[field] and f" {normalize(value)} " in query_text:
                    matches = [p for p in products if p[field] == canonical]
                    if matches:
                        parsed[field] = canonical
                        products = matches
            for attr in proposed_attrs:
                canonical = self.attributes.get(normalize(attr))
                if canonical and canonical not in parsed["attributes"] and f" {normalize(attr)} " in query_text:
                    matches = [p for p in products if canonical in p["attributes"]]
                    if matches:
                        parsed["attributes"].append(canonical)
                        products = matches
        passed = {candidate["category"] for candidate in context["candidates"] if candidate["accepted"]}
        valid_paths = list(dict.fromkeys(p["category"] for p in products if p["category"] in passed))
        preferred = [c for c in _strings(categories) if c in valid_paths]
        ordered = list(dict.fromkeys([*preferred, *valid_paths]))
        if preferred:
            priority = {path: i for i, path in enumerate(ordered)}
            products = sorted(products, key=lambda p: priority.get(p["category"], len(ordered)))
        suggestions = self._rewrites(context, parsed)
        if isinstance(rewrites, dict):
            for kind in REWRITE_TYPES:
                options = {normalize(v): v for v in suggestions[kind]}
                if isinstance(rewrites.get(kind), list):
                    suggestions[kind] = list(dict.fromkeys(options[normalize(v)] for v in _strings(rewrites[kind])
                                                            if normalize(v) in options))
        return {"query": context["query"], "roles": parsed, "categories": ordered, "rewrites": suggestions,
                "products": products, "candidates": context["candidates"], "guard_mode": context["guard_mode"],
                "retrieval": "model-grounded recovery" if role_recovery else "catalog match"}
