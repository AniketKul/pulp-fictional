"""A warmed head cache and a supervised, sparse linear role tagger for CPU serving."""

from collections import Counter
from copy import deepcopy
from math import exp
import json
from pathlib import Path
import re


DATA = Path(__file__).with_name("data") / "curriculum.json"


def key(query):
    return " ".join(query.casefold().split())


def features(query):
    words = re.findall(r"[a-z0-9]+%?", key(query))
    return sorted(set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])})


def train(rows):
    """Fit one logistic classifier per observed product, brand, or attribute."""
    examples = []
    counts = Counter()
    for row in rows:
        roles = row["proposal"]["roles"]
        labels = {f"attributes/{value}" for value in roles["attributes"]}
        labels.update(f"{field}/{roles[field]}" for field in ("product", "brand") if roles[field])
        counts.update(labels)
        examples.append((features(row["query"]), labels))

    model = {}
    for label, positives in sorted(counts.items()):
        weights = Counter()
        bias = -1.0
        # Balance rare labels without making a single mislabeled row dominate.
        positive_weight = min(6.0, len(rows) / positives)
        for _ in range(70):
            gradient = Counter()
            intercept_gradient = 0.0
            for terms, labels in examples:
                score = bias + sum(weights[t] for t in terms)
                error = (label in labels) - 1 / (1 + exp(-score))
                if label in labels:
                    error *= positive_weight
                intercept_gradient += error
                for term in terms:
                    gradient[term] += error
            bias += 2 * intercept_gradient / len(rows)
            for term, derivative in gradient.items():
                weights[term] += 2 * (derivative / len(rows) - 0.004 * weights[term])
        model[label] = [round(bias, 5), {t: round(w, 5) for t, w in sorted(weights.items()) if abs(w) >= 0.04}]
    return model


def tag(model, query):
    terms = features(query)
    scored = {"product": [], "brand": [], "attributes": []}
    for label, (bias, weights) in model.items():
        score = bias + sum(weights.get(term, 0) for term in terms)
        if score > max(0, bias):
            field, value = label.split("/", 1)
            scored[field].append((score, value))
    roles = {f: max(scored[f])[1] if scored[f] else "" for f in ("product", "brand")}
    roles["attributes"] = sorted(value for _, value in scored["attributes"])
    return roles


class IntentRouter:
    def __init__(self, path=DATA, tail_predictor=None):
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        self.cache = {key(row["query"]): row["proposal"] for row in bundle["rows"] if row["head"]}
        self.student = bundle["student"]
        self.tail_predictor = tail_predictor

    def predict(self, query):
        if key(query) in self.cache:
            return "head-cache", deepcopy(self.cache[key(query)])
        roles = self.tail_predictor(query) if self.tail_predictor else tag(self.student, query)
        return "student", {"roles": roles}
