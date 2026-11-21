"""Stage a self-hosted WebAssembly runtime and a content-addressed Qwen GGUF in a Pages artifact."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = "https://registry.npmjs.org/@wllama/wllama/-/wllama-3.13.0.tgz"
RUNTIME_SHA256 = "a9eaac205ff9a1d6fdbea395a48652bd5febc56511a7b66b736d862bf726d2d5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--sha256", required=True, help="Expected GGUF digest from the build receipt")
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.model.open("rb") as source:
        if source.read(4) != b"GGUF":
            parser.error("Model is not a GGUF file")
        source.seek(0)
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != args.sha256:
        parser.error("GGUF digest differs from the expected SHA-256")
    out = args.site
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RUNTIME, timeout=60) as response:
        runtime = response.read()
    if hashlib.sha256(runtime).hexdigest() != RUNTIME_SHA256:
        parser.error("Runtime archive differs from the pinned release")
    with tarfile.open(fileobj=io.BytesIO(runtime), mode="r:gz") as archive:
        for member, name in [
            ("package/esm/index.min.js", "wllama.js"),
            ("package/esm/wasm/wllama.wasm", "wllama.wasm"),
            ("package/LICENCE", "wllama-LICENSE"),
        ]:
            with archive.extractfile(member) as source, (out / name).open("wb") as dest:
                shutil.copyfileobj(source, dest)
    name = f"student-{args.sha256[:16]}.gguf"
    shutil.copyfile(args.model, out / name)
    shutil.copyfile(ROOT / "live/llm.js", out / "llm.js")
    shutil.copyfile(ROOT / "experiments/market/adapter/LICENSE", out / "model-LICENSE")
    (out / "model.json").write_text(json.dumps({"file": name, "sha256": args.sha256,
                                              "bytes": args.model.stat().st_size}) + "\n")


if __name__ == "__main__":
    main()
