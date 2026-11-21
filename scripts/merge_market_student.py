"""Merge the Pulp Fictional adapter into the pinned Qwen3-0.6B base on CPU."""
import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def merge(adapter, output):
    config = PeftConfig.from_pretrained(str(adapter))
    if config.base_model_name_or_path != MODEL or config.revision != REVISION:
        raise ValueError("Adapter base model or revision does not match the pinned model")
    output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL, revision=REVISION).save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    merge(args.adapter, args.output)
