"""Run the production teacher prompt on held-out queries, before catalog post-processing."""
import argparse
import json
from pathlib import Path

from build_curriculum import Engine, MODEL, REVISION, tasks, parse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    root = Path(__file__).resolve().parents[1]
    examples = [json.loads(line) for line in (root / "evaluation/tail.jsonl").read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).to("cuda").eval()
    engine = Engine()
    predictions = []
    for start in range(0, len(examples), 8):
        queries = [x["query"] for x in examples[start:start + 8]]
        prompts = [tokenizer.apply_chat_template(tasks(engine, q, False)["roles"], tokenize=False,
                  add_generation_prompt=True) for q in queries]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=120)
        raw = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predictions.extend({"query": q, "raw": r, "prediction": parse(r)} for q, r in zip(queries, raw))
    result = {"model": MODEL, "revision": REVISION, "predictions": predictions}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
