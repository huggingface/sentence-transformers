#!/usr/bin/env bash
#
# Run the compiled-vs-regular SentenceTransformer benchmark on a GPU machine.
#
#   ./run.sh                                          # benchmark the default model list
#   ./run.sh --model_names sentence-transformers/all-MiniLM-L6-v2   # a subset
#
# benchmark.py declares its own dependencies via PEP 723 inline metadata, so
# `uv run` builds an isolated environment for it (published sentence-transformers
# + CUDA-enabled torch). Nothing to pip install beyond uv itself.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: no GPU detected (nvidia-smi not found). The compiled path uses CUDA graphs." >&2
    exit 1
fi
nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv run "$script_dir/benchmark.py" "$@"
