# Grocery intent student

The public demo's student starts from Qwen3-0.6B and learns to extract a product, brand, attributes, and three types of rewrite. Training examples pair actual KuaiSearch query strings and the published English glosses with catalog-grounded teacher proposals. The adapter uses rank 16 on the attention query, key, value, and output projections. The training objective masks the prompt and supervises the JSON completion.

The [evaluation report](evaluation.json) contains each reserved query, teacher target, raw response, parsed prediction and the hashes for the artifacts used. It compares greedy predictions before and after fine-tuning using the same prompt. The role micro-F1 counts exact product, brand and attribute values. Exact match requires the entire six-field JSON object to agree. An invalid schema counts as a miss. These numbers measure agreement with the offline teacher. Semantic rewrite quality and actual conversion rates require other evidence.

The split groups each original search text with its translated form. It drops translation collisions, excludes cached queries from evaluation, and reserves 16 distinct tail queries. Some related products and phrases can occur in both subsets. Evaluation runs on bfloat16 model weights on the H100. The browser runs a further quantized export of the merged model, which can change predictions.

With the GPU dependencies installed in the project's uv environment, reproduce the run from the repository root. Supply the teacher and market files produced by the matching teacher build. Choose a new output directory.

```sh
uv run --no-sync python scripts/train_market_student.py \
  --teacher path/to/teacher.json --market path/to/market.json \
  --prompt live/data/prompt.json --output /tmp/pulp-student-training
```

To rebuild the merged weights used for browser quantization, run the merge on a machine with sufficient RAM for the base model. The script loads the pinned Hugging Face revision on CPU.

```sh
uv run --no-sync python scripts/merge_market_student.py \
  experiments/market/adapter /tmp/pulp-merged-weights
```

The [model license](adapter/LICENSE) applies to the derivative adapter and merged model. [Base model](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) · [Data provenance](../../live/data/README.md).

## Regenerate the teacher and search index

The source importer downloads the pinned purchase-session split and a bounded item-catalog prefix. English labels are fixed translations of the corresponding Chinese fields in the repository glossary. Reproducing the original selection instructions is documented in the [data notes](../../live/data/README.md). On a bfloat16-capable GPU with the training dependencies above:

```sh
uv run --no-sync python scripts/real_teacher.py --out /tmp/pulp-teacher
uv run scripts/build_embeddings.py --source live/data/source.json --market /tmp/pulp-teacher/market.json --glossary live/data/glossary.json --output /tmp/pulp-embeddings.json
```

The teacher run saves the market and the full raw-generation audit to the specified new directory. Supply those two files to the training command above. The Pages workflow merges the checked-in adapter, converts it to GGUF and quantizes the model before publishing the WebAssembly application.
