"""Score raw role predictions from the bundled experiments without loading a GPU model."""
import json
from pathlib import Path

from pulp_fictional.learning import IntentRouter

DATA = Path(__file__).resolve().parent
REFERENCE = DATA.parent / "experiments/lora/reference"


def slots(row):
    values = {("attributes", value) for value in row.get("attributes", [])}
    values.update((field, row[field]) for field in ("product", "brand") if row.get(field))
    return values


def score(gold, predictions):
    if [r["query"] for r in predictions] != [r["query"] for r in gold]:
        raise ValueError("The report and gold query order differ")
    correct = proposed = expected = exact = 0
    for reference, row in zip(gold, predictions):
        prediction = row["prediction"]
        valid = (isinstance(prediction, dict) and all(isinstance(prediction.get(k), str) for k in ("product", "brand"))
                 and isinstance(prediction.get("attributes"), list)
                 and all(isinstance(v, str) for v in prediction["attributes"]))
        wanted, actual = slots(reference), slots(prediction) if valid else set()
        correct += len(wanted & actual)
        proposed += len(actual)
        expected += len(wanted)
        exact += valid and wanted == actual
    return correct / max(1, proposed), correct / expected, 2 * correct / (expected + proposed), exact


def main():
    gold = [json.loads(line) for line in (DATA / "tail.jsonl").read_text().splitlines()]
    student = json.loads((REFERENCE / "evaluation.json").read_text())
    teacher = json.loads((REFERENCE / "teacher.json").read_text())
    router = IntentRouter()
    cpu = []
    for example in gold:
        route, proposal = router.predict(example["query"])
        if route != "student":
            raise ValueError("A held-out query hit the head cache")
        cpu.append({"query": example["query"], "prediction": proposal["roles"]})
    print("Model                           Precision  Recall     F1     Exact")
    for name, predictions in [("7B teacher with evidence", teacher["predictions"]),
                              ("Linear CPU student", cpu),
                              ("0.6B base", student["base"]["predictions"]),
                              ("0.6B merged LoRA", student["merged_lora"]["predictions"])]:
        p, r, f1, exact = score(gold, predictions)
        print(f"{name:<32}{p:8.3f} {r:7.3f} {f1:6.3f}    {exact}/{len(gold)}")


if __name__ == "__main__":
    main()
