#!/usr/bin/env bash
# Convert a merged Hugging Face Qwen3 checkpoint and its tokenizer to a browser GGUF.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MERGED_MODEL_DIRECTORY BUILD_DIRECTORY" >&2
  exit 2
fi
model_dir=$(cd "$1" && pwd)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd)
cd "$build_dir"
revision=a7a98e0fffed794396b3fbad4dcdbbc184963645
if [[ ! -d llama.cpp/.git ]]; then
  git init -q llama.cpp
fi
if ! git -C llama.cpp cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C llama.cpp fetch --depth=1 https://github.com/ggml-org/llama.cpp.git "$revision"
fi
git -C llama.cpp checkout -q --detach "$revision"
cmake -S llama.cpp -B build -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-quantize --parallel 4
if [[ ! -d .venv ]]; then
  uv venv --python 3.10 .venv
fi
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
uv pip install --python .venv/bin/python 'transformers==4.52.4' 'mistral-common[sentencepiece]==1.8.5' 'sentencepiece==0.2.1' 'protobuf==4.25.8'
uv run --no-project --python .venv/bin/python python llama.cpp/convert_hf_to_gguf.py "$model_dir" --outfile student.f16.gguf --outtype f16
build/bin/llama-quantize student.f16.gguf student-Q4_K_M.gguf Q4_K_M 4
sha256sum student-Q4_K_M.gguf
