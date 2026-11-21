"""Optional CUDA student. The default CPU route has no third-party dependencies."""
import json
from pathlib import Path
from threading import Lock

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def messages(query):
    return [{"role": "system", "content": (
        'Extract grocery search intent. Return only JSON with keys product, brand, attributes. '
        'Use canonical grocery product names, exact brand casing, and lowercase attributes. '
        'Use empty strings for unknown product or brand and a list for attributes. '
        'Label only requested attributes. For a completely unrelated query return empty fields.'
    )}, {"role": "user", "content": query}]


def prompt_ids(tokenizer, query):
    return tokenizer.apply_chat_template(messages(query), tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)


def load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The LoRA student requires a CUDA GPU with bfloat16 support.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                 torch_dtype=torch.bfloat16).to("cuda")
    return model, tokenizer


def predict_roles(model, tokenizer, query):
    import torch

    inputs = torch.tensor([prompt_ids(tokenizer, query)], device=model.device)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip()
    try:
        roles = json.loads(response)
    except ValueError:
        return {}
    if (not isinstance(roles, dict) or not isinstance(roles.get("product"), str)
            or not isinstance(roles.get("brand"), str) or not isinstance(roles.get("attributes"), list)
            or not all(isinstance(v, str) for v in roles["attributes"])):
        return {}
    return {k: roles[k] for k in ("product", "brand", "attributes")}


class LoraStudent:
    def __init__(self, adapter: Path):
        from peft import PeftModel

        model, self.tokenizer = load_base()
        self.model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
        self.lock = Lock()

    def __call__(self, query):
        with self.lock:
            return predict_roles(self.model, self.tokenizer, query)
