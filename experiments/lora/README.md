# Reproduce the GPU experiment

The checked-in adapter specializes **Qwen3-0.6B** for grocery role extraction. The offline teacher is **Qwen2.5-7B-Instruct**. Both models run locally on the GPU. The default storefront instead loads the small CPU model in the curriculum bundle.

## Set up

Run from the repository root on Linux with a CUDA GPU that supports bfloat16. The reference run used an H100 PCIe, Python 3.10 and PyTorch 2.6.0 with CUDA 12.4. The model loader pins the public Hugging Face revision for each model. Allow disk space for both base models if rebuilding the teacher. To serve the checked-in adapter, only the student base is required.

```sh
uv sync --frozen --python 3.10
uv pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install -r experiments/lora/requirements.txt
```

## Serve the reference adapter

```sh
uv run --no-sync pulp-fictional --adapter experiments/lora/reference/adapter
```

The server merges the adapter into the base weights at startup. Multiple requests can read the head cache concurrently. GPU generations on misses are serialized. The intent trace identifies the serving backend for each query.

## Rebuild from the catalog

```sh
uv run --no-sync python experiments/grounding/encode.py
uv run --no-sync python scripts/build_curriculum.py
uv run --no-sync python experiments/lora/train.py --output experiments/lora/my-run
uv run --no-sync pulp-fictional --adapter experiments/lora/my-run/adapter
```

The first two commands regenerate `semantic.json` and `curriculum.json` in the source checkout. Start with the third command to reuse the bundled teacher labels. The training command requires a fresh output directory to preserve earlier runs.

1. MiniLM embeds the head queries, taxonomy paths, and brand-plus-product names. The offline encoder uses masked mean pooling, L2 normalization, and cosine similarity.
2. The 7B teacher labels synthetic queries using retrieved store evidence. Head queries also get a category reranking call and three separate rewrite calls. The bundle keeps the raw answers, filtered proposals, eligibility flags, source hashes, and CPU student weights.
3. The LoRA trainer uses eligible role labels with loss on the answer tokens. It trains rank-8 adapters on query and value projections for eight epochs. Training and evaluation queries must be disjoint after normalization.
4. The evaluator compares the base and merged-adapter predictions on the same handwritten queries, with identical prompts and greedy decoding. Inspect each prediction, precision, recall, exact match and batch-one warm median latency in the [reference report](reference/evaluation.json). The reference adapter uses the [Qwen license](reference/adapter/LICENSE).

To rerun the teacher on the held-out set, use `uv run --no-sync python scripts/evaluate_teacher.py --output experiments/lora/my-run/teacher.json`. To score all four bundled sets of predictions, run `uv run python evaluation/roles.py`.

The comparison evaluates role extraction before the catalog validator. The measured time includes tokenization, generation and parsing, and excludes model loading, HTTP, and catalog lookup. These timings describe single requests on one GPU. [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) · [Qwen2.5 teacher model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
