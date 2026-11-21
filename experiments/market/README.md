# Grocery intent student

The real-time student starts from Qwen3-0.6B and learns to extract a product, brand, attributes, and three types of rewrite. Training examples pair actual KuaiSearch query strings and the published English glosses with catalog-grounded teacher proposals. The adapter uses rank 16 on the attention query, key, value, and output projections. The training objective masks the prompt and supervises the JSON completion.

The [evaluation report](evaluation.json) contains each reserved query, teacher target, raw response, parsed prediction and the hashes for the artifacts used. It compares greedy predictions before and after fine-tuning using the same prompt. The role micro-F1 counts exact product, brand and attribute values. Exact match requires the entire six-field JSON object to agree. An invalid schema counts as a miss. These numbers measure agreement with the offline teacher. Semantic rewrite quality and actual conversion rates require other evidence.

The split groups each original search text with its translated form. It drops translation collisions, excludes cached queries from evaluation, and reserves 16 distinct tail queries. Some related products and phrases can occur in both subsets. Evaluation runs on bfloat16 model weights on the H100. The private inference service uses the same precision and a merged adapter.

The teacher accepted 183 of 185 original queries. Fine-tuning used 329 original and translated examples from 165 query families across five epochs. On the 32 reserved query forms the tuned student produced valid six-field JSON 32 times and matched all six teacher fields five times. Strict role micro-F1 was 0.373. The untuned base produced valid JSON zero times under the same prompt. Product names matched the teacher on nine of 16 English forms and three of 16 Chinese forms. These results describe this small catalog and evaluation set.

With the GPU dependencies installed in the project's uv environment, run training from the repository root using the checked-in teacher labels and catalog. Choose a new output directory.

```sh
uv run --no-sync python scripts/train_market_student.py \
  --teacher experiments/market/teacher.json --market live/data/market.json \
  --prompt live/data/prompt.json --output /tmp/pulp-student-training
```

To export a standalone copy of the merged student, run the merge on a machine with sufficient RAM for the base model. The script loads the pinned Hugging Face revision on CPU.

```sh
uv run --no-sync python scripts/merge_market_student.py \
  experiments/market/adapter /tmp/pulp-merged-weights
```

The [model license](adapter/LICENSE) applies to the derivative adapter and merged model. [Base model](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) · [Data provenance](../../live/data/README.md).

## Regenerate the teacher and search index

The source importer downloads the pinned purchase-session split and a bounded item-catalog prefix. English labels are fixed translations of the corresponding Chinese fields in the repository glossary. Reproducing the original selection instructions is documented in the [data notes](../../live/data/README.md). On a bfloat16-capable GPU with the training dependencies above:

```sh
uv run --no-sync python scripts/real_teacher.py --out /tmp/pulp-teacher
```

The teacher run saves the market, fitted embedding index and full raw-generation audit to the specified new directory. Supply its teacher and market files to the student training command above. At startup the search server merges the checked-in adapter and holds the model on the GPU.
