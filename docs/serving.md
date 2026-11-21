# Serving grocery search

The server reads the catalog and offline teacher cache, loads the pinned Qwen3 base model, merges the trained adapter and keeps the resulting bfloat16 model on the GPU. Install Node.js for the catalog ranker and use a CUDA GPU with bfloat16 support. The first start needs network access to fetch the base weights.

```sh
uv sync --frozen --python 3.10
uv pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install 'transformers==4.52.4' 'peft==0.15.2' 'accelerate==1.7.0' 'lm-format-enforcer==0.11.3'
uv run --no-sync pulp-fictional --adapter experiments/market/adapter --port 8000
```

On the machine running the service, open `http://127.0.0.1:8000/`. To use a browser on a different machine, follow the SSH instructions below. Both the UI and the JSON API use the same cache, student, catalog and guardrails. A server launched without an adapter can inspect teacher-cache hits. A tail query on that server reports that the resident model is unavailable.

The service validates the model's six output fields. If the first generation is malformed, the same model gets one retry with JSON-schema-constrained decoding. Retrieval still checks the original shopper query for dietary exclusions.

```sh
curl -Gs http://127.0.0.1:8000/api/search --data-urlencode 'q=Mooncakes'
curl -Gs http://127.0.0.1:8000/api/search --data-urlencode 'q=a sugar-free mooncake'
```

## Open the remote service on your own machine

Start the server on the GPU host. From your own machine, forward a local port to the server's loopback port over SSH. Replace the SSH alias and remote port with your deployment's values.

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8000 YOUR-GPU-HOST
```

With the SSH connection running, open `http://127.0.0.1:8765/` on your own machine. The browser sends a query to `/api/search` and receives products and an intent trace. The optional `force=1` parameter runs the GPU student on a cached query. Stopping the SSH connection closes access through the local port.

## Check the shopper flow

With the forwarded storefront running, this test opens a fresh Chromium profile, types an uncached search, reloads the page and verifies the actual catalog items returned by both serving routes. It also checks browser requests for model downloads.

```sh
uvx --from playwright==1.63.0 playwright install chromium
uv run scripts/smoke_storefront.py --url http://127.0.0.1:8765/
```
