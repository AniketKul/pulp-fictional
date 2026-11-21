"""Distill grounded KuaiSearch teacher labels into a bilingual Qwen3 query interpreter."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import unicodedata

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from merge_market_student import MODEL, REVISION, merge

SEED = 47
FIELDS = ("product", "brand", "attributes", "synonym", "substitute", "broader")


def normalize(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def split_examples(rows, heads):
    # Both spellings of a query stay together. Ambiguous cross-query translations supply no labels.
    def variants(row):
        return {normalize(row[k]) for k in ("query", "english")} - {""}
    counts = Counter(key for row in rows for key in variants(row))
    eligible = [r for r in rows if r["valid"] and r["english"].strip() and
                all(counts[key] == 1 for key in variants(r))]
    cached = {key for head in heads for key in variants(head)}
    cold = sorted((r for r in eligible if variants(r).isdisjoint(cached)),
                  key=lambda r: sha256(f'{SEED}:{r["query"]}'.encode()).hexdigest())
    if len(cold) < 32:
        raise ValueError("Too few distinct tail queries for a disjoint evaluation")
    held = {r["query"] for r in cold[:16]}
    train, test = [], []
    for row in eligible:
        target = {**{k: row["roles"][k] for k in FIELDS[:3]}, **{k: row[k] for k in FIELDS[3:]}}
        entries = test if row["query"] in held else train
        entries.append({"query": row["query"], "source_query": row["query"], "form": "original", "target": target})
        if normalize(row["english"]) != normalize(row["query"]):
            entries.append({"query": row["english"], "source_query": row["query"], "form": "translation", "target": target})
    train_keys, test_keys = ({normalize(r["query"]) for r in group} for group in (train, test))
    if train_keys & test_keys or cached & test_keys:
        raise ValueError("Cache or training query leaked into evaluation")
    return train, test


def prompt(tokenizer, system, query):
    return tokenizer.apply_chat_template([{"role":"system", "content":system}, {"role":"user", "content":query}],
                                         tokenize=True, add_generation_prompt=True, enable_thinking=False)


def parse(raw):
    def unique_keys(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("Repeated JSON field")
        return obj
    try:
        data = json.loads(raw, object_pairs_hook=unique_keys)
    except ValueError:
        return None
    if (not isinstance(data, dict) or set(data) != set(FIELDS)
            or not all(isinstance(data[k], str) for k in FIELDS if k != "attributes")
            or not isinstance(data["attributes"], list) or not all(isinstance(a,str) for a in data["attributes"])):
        return None
    return data


def slots(intent):
    return {(key, item) for key in FIELDS[:3]
            for item in (intent[key] if key == "attributes" else [intent[key]]) if item}


@torch.inference_mode()
def evaluate(model, tokenizer, system, examples):
    records, tp, predicted, gold, schema, exact = [], 0, 0, 0, 0, 0
    model.eval()
    for start in range(0, len(examples), 8):
        batch = examples[start:start+8]
        inputs = tokenizer.pad([{"input_ids":prompt(tokenizer,system,r["query"])} for r in batch],
                               padding=True, return_tensors="pt").to("cuda")
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=144, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(tokens[:,inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for row, raw in zip(batch, texts):
            target, actual = row["target"], parse(raw.strip())
            expected_slots, actual_slots = slots(target), slots(actual) if actual is not None else set()
            tp += len(expected_slots & actual_slots)
            predicted += len(actual_slots)
            gold += len(expected_slots)
            schema += actual is not None
            exact += actual == target
            records.append({**row,"raw":raw,"prediction":actual})
    return {"examples":len(records),"valid_schema":schema,"exact_all_fields":exact,
            "role_micro_precision":tp/max(predicted,1),"role_micro_recall":tp/max(gold,1),
            "role_micro_f1":2*tp/max(1,predicted+gold),"predictions":records}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--teacher", type=Path, required=True)
    cli.add_argument("--market", type=Path, required=True)
    cli.add_argument("--prompt", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    teacher = json.loads(args.teacher.read_text())
    system = json.loads(args.prompt.read_text())["system"]
    train, test = split_examples(teacher["rows"], json.loads(args.market.read_text())["cache"])
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                           target_modules=["q_proj","k_proj","v_proj","o_proj"], revision=REVISION))
    model.peft_config["default"].revision = REVISION
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.00025)
    samples = []
    for row in train:
        prefix = prompt(tokenizer,system,row["query"])
        answer = tokenizer.encode(json.dumps(row["target"],ensure_ascii=False,separators=(",",":")),
                                  add_special_tokens=False) + [tokenizer.eos_token_id]
        samples.append((prefix+answer, [-100]*len(prefix)+answer))
    model.train()
    model.config.use_cache = False
    steps, batch_size, epochs = 0, 6, 5
    for epoch in range(epochs):
        rng.shuffle(samples)
        for start in range(0,len(samples),batch_size):
            batch = samples[start:start+batch_size]
            length = max(len(x) for x,_ in batch)
            def pad(rows, fill):
                return torch.tensor([r+[fill]*(length-len(r)) for r in rows],device="cuda")
            loss = model(input_ids=pad([x for x,_ in batch],tokenizer.pad_token_id),
                         attention_mask=pad([[1]*len(x) for x,_ in batch],0),
                         labels=pad([y for _,y in batch],-100)).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            steps += 1
        print(f"Epoch {epoch+1}/{epochs}, final batch loss {loss.item():.4f}",flush=True)
    adapter, merged = args.output/"adapter", args.output/"merged"
    model.save_pretrained(adapter)
    del model, optimizer, base
    torch.cuda.empty_cache()
    merge(adapter,merged)
    print(f"Merged model ready at {merged.resolve()}",flush=True)
    report = {"model":MODEL,"revision":REVISION,"gpu":torch.cuda.get_device_name(),
              "torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,
              "teacher":teacher["model"],"teacher_revision":teacher["revision"],"source_sha256":teacher["source_sha256"],
              "glossary_sha256":teacher["glossary_sha256"],"teacher_sha256":digest(args.teacher),"market_sha256":digest(args.market),"prompt_sha256":digest(args.prompt),
              "training_script_sha256":digest(Path(__file__)), "merge_script_sha256":digest(Path(__file__).with_name("merge_market_student.py")),
              "adapter_sha256":digest(adapter/"adapter_model.safetensors"),"seed":SEED,"epochs":epochs,"steps":steps,
              "training_examples":len(train),"training_source_queries":sorted({r["source_query"] for r in train}),
              "metric_reference":"Agreement with the grounded offline teacher on reserved bilingual query families."}
    for name, path, opts in [("base",MODEL,{"revision":REVISION}),("merged_lora",merged,{})]:
        eval_model = AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,**opts).to("cuda")
        report[name] = evaluate(eval_model,tokenizer,system,test)
        del eval_model
    (args.output/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
