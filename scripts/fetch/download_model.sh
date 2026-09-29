#!/bin/bash
# Reproducibility script: downloads the exact Qwen3-8B Q4_K_M GGUF used here.
# (The repo already ships it in llm/qwen3-8b-parts/ - you do NOT need this.)
set -e
URL="https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf"
OUT="${1:-qwen3-8b-q4km.gguf}"
echo "downloading $URL -> $OUT"
curl -L -C - --retry 10 -o "$OUT" "$URL"
sha256sum "$OUT"
echo "expected sha256: see llm/qwen3-8b-parts/SHA256SUMS.txt (FULL line)"
