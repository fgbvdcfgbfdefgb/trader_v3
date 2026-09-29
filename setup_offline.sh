#!/bin/bash
# One-time offline setup for trader_v3 (no internet needed).
set -e
cd "$(dirname "$0")"

echo "== [1/4] python dependencies =="
python3 -c "import numpy, matplotlib; print('numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)" 2>/dev/null || {
  echo "ERROR: numpy + matplotlib are required."
  echo "The Snowflake ML image ships them; elsewhere: pip install -r requirements.txt"
  exit 1
}

echo "== [2/4] making llama.cpp binaries executable =="
chmod +x llm/llama-b11249/llama-server llm/llama-b11249/llama-cli 2>/dev/null || true
bash scripts/check_llm.sh || true

echo "== [3/4] assembling Qwen3-8B GGUF from repo parts (needs ~5 GB free disk) =="
python3 scripts/assemble_model.py

echo "== [4/4] building the data cache from bundled Binance minute bars =="
python3 scripts/prepare_data.py

echo ""
echo "Setup complete. Start training with:"
echo "   python3 run_training.py                 # auto: 1 agent/GPU (or CPU agents), LLM on"
echo "   python3 run_training.py --agents 2      # force 2 parallel agents"
echo "   python3 run_training.py --no-llm        # quick test without Qwen"
echo ""
echo "Optional: pre-warm the LLM advice cache (recommended before long runs):"
echo "   python3 run_training.py --no-llm --epochs 0 &   # not needed, just:"
echo "   # (start server manually and run prefill - see README.md)"
