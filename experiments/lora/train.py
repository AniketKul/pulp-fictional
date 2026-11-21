"""Fine-tune on accepted teacher labels and compare base and merged adapter on held-out queries."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model

from pulp_fictional.lora import MODEL, REVISION, load_base, predict_roles, prompt_ids

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src/pulp_fictional/data/curriculum.json"
GOLD = ROOT / "evaluation/tail.jsonl"
SEED = 31


def slots(roles):
    return {(k, s) for k, v in roles.items() if k in ("product", "brand", "attributes")
            for s in (v if isinstance(v, list) else [v]) if s}


def evaluate(model, tokenizer, examples):
    predictions, correct, proposed, expected, exact, latencies = [], 0, 0, 0, 0, []
    model.eval()
    predict_roles(model, tokenizer, "warmup")
    for example in examples:
        query = example["query"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = predict_roles(model, tokenizer, query)
        torch.cuda.synchronize()
        latencies.append(1000 * (time.perf_counter() - start))
        gold, actual = slots(example), slots(result)
        correct += len(actual & gold)
        proposed += len(actual)
        expected += len(gold)
        exact += int(bool(result) and actual == gold)
        predictions.append({"query": query, "prediction": result, "valid_json_schema": bool(result)})
    return {"micro_precision": correct / max(proposed, 1), "micro_recall": correct / expected,
            "micro_f1": 2 * correct / max(expected + proposed, 1), "exact_match": exact / len(examples),
            "valid_outputs": sum(p["valid_json_schema"] for p in predictions),
            "warm_single_request_p50_ms": round(statistics.median(latencies), 2),
            "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    curriculum = json.loads(DATA.read_text())["rows"]
    trainable = [row for row in curriculum if row["trainable"]]
    examples = [json.loads(line) for line in GOLD.read_text().splitlines()]
    from pulp_fictional.engine import normalize
    if {normalize(r["query"]) for r in examples} & {normalize(r["query"]) for r in curriculum}:
        raise ValueError("An evaluation query appears in the training curriculum")
    model, tokenizer = load_base()
    baseline = evaluate(model, tokenizer, examples)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0004)
    samples = []
    for row in trainable:
        prefix = prompt_ids(tokenizer, row["query"])
        roles = row["proposal"]["roles"]
        target = json.dumps({k: roles[k] for k in ("product", "brand", "attributes")}, ensure_ascii=False)
        answer = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        samples.append((prefix + answer, [-100] * len(prefix) + answer))
    if not samples:
        raise ValueError("No accepted teacher labels for training")
    model.train()
    model.config.use_cache = False
    steps = 0
    for epoch in range(8):
        rng.shuffle(samples)
        for start in range(0, len(samples), 4):
            batch = samples[start:start + 4]
            width = max(len(ids) for ids, _ in batch)
            def pad(rows, value):
                return torch.tensor([r + [value] * (width - len(r)) for r in rows], device="cuda")
            loss = model(input_ids=pad([x for x, _ in batch], tokenizer.pad_token_id),
                         attention_mask=pad([[1] * len(x) for x, _ in batch], 0),
                         labels=pad([y for _, y in batch], -100)).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            steps += 1
        print(f"Epoch {epoch + 1}, last batch loss {loss.item():.4f}", flush=True)
    model.save_pretrained(args.output / "adapter")
    model.config.use_cache = True
    adapted = evaluate(model.merge_and_unload(), tokenizer, examples)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
              "seed": SEED, "dtype": "bfloat16", "train_rows": len(samples), "eval_rows": len(examples),
              "steps": steps, "curriculum_sha256": digest(DATA), "gold_sha256": digest(GOLD),
              "adapter_sha256": digest(args.output / "adapter/adapter_model.safetensors"),
              "base": baseline, "merged_lora": adapted}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
