# Pulp Fictional

A grocery search prototype served from a private GPU host.

Search 144 real grocery listings in English or Chinese. Type into the search box to update the results. The **Intent trace** shows which model interpreted the query, the catalog categories it matched, and the purchase evidence available for that phrase.

![The internal storefront showing the Dove chocolate gift box and the live model's interpretation](docs/storefront.png)

## Try it

1. Open the internal storefront. It starts with **Mooncakes**. This frequent query uses a validated teacher interpretation from the server's cache. Open **Purchase evidence** to see the count and original Chinese query.
2. Type **a sugar-free mooncake**. The server sends this new phrase to the resident language model, validates its proposed rewrites and ranks catalog listings. Each result includes its original source item ID.
3. Refresh or share the URL to run the same search automatically. The browser sends a short search request and receives JSON. Model weights stay on the server.

## Architecture

```mermaid
flowchart LR
    H[Observed search and purchase records] --> T[Offline teacher with catalog context]
    C[Bilingual catalog and taxonomy] --> T
    T --> V[Validate categories, labels and rewrites]
    V --> K[Frequent-query cache]
    V --> D[Distillation curriculum]
    D --> L[LoRA fine-tuning and adapter merge]
    L --> G[Resident GPU student]
    Q[Shopper query] --> A[Search API]
    A -->|Head hit| K
    A -->|Miss| G
    K --> R[Catalog retrieval and guardrails]
    G --> R
    R --> U[Products and intent trace]
```

- **Offline teacher.** Qwen2.5-7B-Instruct uses purchase-derived brand and category counts, embedding-ranked catalog evidence and the bilingual taxonomy. It reranks converted category paths, extracts product, brand and attributes, and answers separate prompts for synonyms, substitutes and broader searches. The pipeline saves the raw generations alongside the validated curriculum. [Teacher pipeline](scripts/real_teacher.py) · [Teacher records](experiments/market/teacher.json)
- **Online routing.** A single search API checks the frequent-query cache. A miss runs Qwen3-0.6B, fine-tuned on the teacher's labels and merged with its LoRA adapter, in bfloat16 on the H100. The model stays loaded across visitors and page reloads. The trace offers a forced live run on cached queries for inspection. [Search service](src/pulp_fictional/service.py) · [Training and held-out evaluation](experiments/market/README.md)
- **Retrieval.** A resident ranker combines bilingual semantic vectors, model concepts and available historical purchase evidence. It checks explicit brands against the catalog's brands and listing titles, checks dietary constraints against source labels, and validates rewrites against the products they retrieve. The trace omits model tags that lack support in the query or matching listings. [Catalog ranker](live/retrieval.js) · [Serving checks](src/pulp_fictional/rank.cjs) · [Embedding builder](scripts/build_embeddings.py)

The storefront and search API share a loopback address on the inference host. Remote users with SSH access can forward that port to their own machine. [Run the private demo](docs/serving.md)

## Data and scope

KuaiSearch-Lite supplies the original queries, product listings and purchased-item IDs. The catalog covers six grocery departments. Its 185 original queries link to 209 distinct query–product pairs and 225 session–product purchase occurrences. Counts belong to the historical Chinese query. A new English phrase has no purchase history of its own. The storefront displays original titles alongside editorial English translations. This research catalog has no current retail prices, stock feed or checkout. [Data selection and license](live/data/README.md) · [KuaiSearch research release](https://benchen4395.github.io/KuaiSearch/)

## Run and test

For the complete storefront, start the service on a compatible CUDA GPU with Node.js, the `uv` environment and the checked-in adapter. [Setup and launch commands](docs/serving.md)

```sh
uv run --frozen python -m unittest discover -s tests
node --test tests/*.cjs
```
