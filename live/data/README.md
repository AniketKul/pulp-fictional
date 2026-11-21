# Grocery data provenance

Pulp Fictional uses a reproducible slice of KuaiSearch-Lite. Its query field contains the Chinese text shoppers entered on Kuaishou. The product IDs, titles, brands and category paths come from the same release. Catalog titles reflect the seller's description at collection time. The storefront data carries no current price or stock feed. [Pinned dataset and license declaration](https://huggingface.co/datasets/benchen4395/KuaiSearch/tree/16f555af360f6a86209f60eb1f12462f9e25be2c) · [KuaiSearch paper](https://arxiv.org/abs/2602.11518) · [License](LICENSE-KuaiSearch)

The importer reads complete lines from the first 256 MiB of the catalog file and the entire Lite recall file. It joins on the publisher's item IDs, using the training split. Six grocery departments cover snacks, fresh foods, instant and dried foods, pantry staples and seasoning, coffee and cereal drinks, and tea. Exact source categories and input hashes appear in the [importer](../../scripts/import_grocery_data.py). Queries longer than 45 trimmed characters, blank queries, and text matching the importer's contact-pattern filter are excluded.

For each SKU, the importer counts distinct search sessions that named that SKU in the purchased-item list. Each SKU counts once per session. A session with two purchased products can contribute to two counts. It keeps up to 144 purchased items by cycling across second-level grocery categories, ordering each category by purchase-session count and then item ID. All eligible queries leading to those selected SKUs are retained. Each query's `sessions` counts its occurrences across the eligible training searches, including sessions with no purchase. The retained item counts cover the displayed catalog subset. These event counts do not record units, order revenue, or present-day demand.

The English titles, category labels and query glosses in [glossary.json](glossary.json) are editorial translations for this demo. Product IDs and original Chinese text preserve the link to each source record. The English strings were never counted as original shopper queries.

The published aggregate omits shopper identifiers and individual session records. The counts, corpus sizes and selection denominators are recorded in [source.json](source.json).

Rebuild from the repository root with a local directory for the download cache.

```sh
uv run --no-project python scripts/import_grocery_data.py --cache /tmp/pulp-kuaisearch-cache
```
