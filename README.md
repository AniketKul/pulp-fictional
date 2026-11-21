# Pulp Fictional

[Open the live grocery search](https://aniketkul.github.io/pulp-fictional/)

Search 144 real grocery listings in English or Chinese. Each product card shows its original title, English translation, category and source item ID. The intent trace shows which model handled your query, what it understood, and which catalog evidence supports the results.

![Live grocery search returning a Dove chocolate gift box and its model interpretation](docs/storefront.png)

## Try it

1. Choose a suggested search. Open **Intent trace** to see the cached teacher interpretation and the original shopper query behind the recorded purchases.
2. Click **Run live model on this query**, or type your own search, such as **dark chocolate gift box**. On the public site, approve the initial download of approximately 400 MB of weights plus the runtime. A quantized, fine-tuned Qwen3-0.6B runs on your browser's CPU using WebAssembly. The first search can take longer while the model loads.
3. Try a related product, an explicit constraint, or something outside the catalog. The same model handles new text. Catalog checks can discard a suggested rewrite or return zero products.

## Where the data comes from

The KuaiSearch research release contains real queries, purchase-session item IDs and product listings. Pulp Fictional joins those records by source item ID. The selected catalog spans six grocery departments. Its 185 original queries link to 209 distinct query–product pairs, totaling 225 session–product purchase occurrences. For a query such as 月饼, the search count includes all eligible sessions for that phrase. The purchase count displayed for a category covers the selected catalog slice. An English translation points back to the original Chinese query. A new English phrase has no historical count of its own. [Data, selection rules and license](live/data/README.md) · [KuaiSearch research release](https://benchen4395.github.io/KuaiSearch/).

These are historical listings. The storefront has no retailer connection for current inventory, prices or checkout.

## How the search works

| Stage | Execution |
|---|---|
| Offline teacher | Qwen2.5-7B-Instruct receives real query and catalog evidence, reranks candidate category paths, extracts structured roles, and runs separate instructions for synonyms, substitutes and broader searches. Its raw answers and filtered labels are available for inspection. |
| Frequent queries | The engine caches eligible teacher results for frequent source queries. An exact query or its unique English gloss can use the cache. The **Run live model** button bypasses it. |
| New queries | A Qwen3-0.6B student fine-tuned on the teacher labels generates a structured intent from the submitted text. The public site uses the merged Q4_K_M weights locally. The optional GPU endpoint runs the merged bfloat16 model. |
| Retrieval | A bilingual TF-IDF index with a 64-dimensional latent semantic projection ranks actual product and taxonomy records. The ranker combines query similarity, model concepts, and available purchase evidence, then filters category and rewrite suggestions against the catalog. |

The public deployment serves the model, runtime and catalog from GitHub Pages. After downloading the files, the browser computes cache misses locally. If inference fails, the storefront shows the error. [Runtime](live/search.js) · [Retrieval](live/retrieval.js) · [Teacher pipeline](scripts/real_teacher.py) · [Student training and evaluation](experiments/market/README.md).

## What we measured

On the H100, the teacher returned 181 eligible role labels for 185 real source queries. The student trained on 325 original and translated examples after reserving distinct query families and removing translation collisions. On 32 held-out query forms from 16 families, the fine-tuned student produced the expected six-field JSON schema 32 times. The base model did so zero times with the same prompt. Exact role-value micro-F1 against the teacher was 0.299 for the student. Neither model matched the teacher on all six fields for any held-out query. These are teacher-agreement scores on bfloat16 weights. Product relevance and browser-quantized model quality need separate evaluation. [Full predictions](experiments/market/evaluation.json).

## Run locally

```sh
git clone https://github.com/AniketKul/pulp-fictional.git
cd pulp-fictional
uv run pulp-fictional
```

Open [the local storefront](http://127.0.0.1:8000). Default local browser inference loads the published model assets on first use. To serve a self-contained built site, pass its directory with `--site`.

For CUDA inference on a compatible GPU, from the repository root:

```sh
uv sync --frozen --python 3.10
uv pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install 'transformers==4.52.4' 'peft==0.15.2' 'accelerate==1.7.0'
uv run --no-sync pulp-fictional --adapter experiments/market/adapter
```

The local JSON search API is available at `/api/search?q=Mooncakes` with Node.js installed. Cache misses on this endpoint require GPU mode. The local server binds to loopback. In GPU mode it sends the query to the local model endpoint. A GPU cache miss runs inference without a browser model download.

```sh
uv run --frozen python -m unittest discover -s tests
node --test tests/*.cjs
```
