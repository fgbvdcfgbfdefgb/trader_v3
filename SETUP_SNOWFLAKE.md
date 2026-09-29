# Running trader_v3 on a Snowflake workspace (offline)

Your situation: the workspace was created by syncing this GitHub repo
**once**, and the training machine has **no internet** afterwards. This
repo is designed exactly for that — no step below needs the network.

## 1. Open a shell on the workspace

If your workspace has a terminal, use it. From a Snowflake **notebook**
you can use a `bash` cell, or run everything through `subprocess`:

```python
# notebook cell 1 - one-time setup (~5-15 min)
import subprocess
r = subprocess.run(["bash", "setup_offline.sh"], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
```

`setup_offline.sh` will:
1. verify numpy/matplotlib (preinstalled in the Snowflake ML image),
2. make the llama.cpp binaries executable,
3. stitch `llm/qwen3-8b-parts/*` back into the 4.7 GB GGUF (sha256-checked),
4. build the data cache from the bundled Binance minute bars.

Needs ~11 GB free disk and ~7 GB RAM for the 8B model. If RAM is tight,
the trainer still works (it falls back to neutral advisor features) — or
ask for a smaller quant to be swapped into the repo.

## 2. Start training

```python
# notebook cell 2 - start training in the background (survives the cell)
import subprocess, os
os.makedirs("runs", exist_ok=True)
log = open("runs/train.log", "a")
p = subprocess.Popen(
    ["python3", "run_training.py"],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("started, pid", p.pid)
```

From a terminal it's simply:

```bash
python3 run_training.py 2>&1 | tee -a runs/train.log
```

The orchestrator detects GPUs (`nvidia-smi`) and runs **one agent per
GPU**, or several CPU agents when no GPU exists. It also starts the
local llama.cpp Qwen3-8B server on CPU (port auto-picked, log at
`runs/llama_server.log`) and waits for it to load.

## 3. Watch progress

```python
# notebook cell 3 - status anytime
import subprocess
print(subprocess.run(["python3", "status.py"], capture_output=True,
                     text=True).stdout)
```

or look at the files:
- `runs/overview.png` — all agents compared (refreshed every minute)
- `runs/agent_i/latest.png` — newest epoch report for agent i
- `runs/agent_i/epochs/epoch_NNNNNN_DAY.png` — one PNG per epoch
- `runs/agent_i/progress.png` — that agent's whole history
- `runs/agent_i/metrics.jsonl` — raw per-epoch numbers

To render the latest PNG inside a notebook:

```python
from IPython.display import Image, display
display(Image(filename="runs/overview.png"))
display(Image(filename="runs/agent_0/latest.png"))
```

## 4. Stop / resume

Just stop the process (notebook: kill the pid; terminal: Ctrl+C). Each
agent checkpoints itself every epoch, so on the next
`python3 run_training.py` every agent continues from its own next epoch.
Nothing is lost except the tail of the epoch in flight.

## 5. Optional: pre-warm the LLM cache

The first time a calendar day is traded, the agent waits ~30-120 s for
Qwen's analysis (cached forever afterwards). To avoid that entirely:

```bash
# terminal: start the server alone, then prefill all ~600 days once
llm/llama-b11249/llama-server -m llm/qwen3-8b-q4km.gguf \
    --host 127.0.0.1 --port 8080 -t 8 -np 2 -c 8192 --jinja --no-warmup &
python3 scripts/prefill_llm_cache.py --url http://127.0.0.1:8080
```

(or just let training warm it naturally — repeated epochs on the same day
hit the cache).

## 6. Troubleshooting

- **llama-server won't start** — run `bash scripts/check_llm.sh`. Needs
  glibc >= 2.34 (Ubuntu 22.04+) plus libssl3/libgomp1 (`apt install
  libssl3 libgomp1` if they're missing). Training still runs without the
  LLM in the meantime.
- **Out of memory for the 8B model** — `run_training.py` automatically
  continues with neutral advisor features; or run `--no-llm`.
- **"no usable days in cache"** — run `python3 scripts/prepare_data.py`
  and check `data/minute/` exists.
- **Want more/fewer agents** — `python3 run_training.py --agents 4`.
