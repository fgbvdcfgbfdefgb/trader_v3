# trader_v3 — offline RL crypto-trading agent + Qwen3-8B news advisor

A fully **offline** reinforcement-learning system that learns to day-trade
**BTC, ETH and LTC** on 1-minute Binance data, helped by a local
**Qwen3-8B** LLM (llama.cpp, CPU) that reads each day's **news headlines
and market metrics** and feeds its assessment to the trading agent.

Everything the trainer needs — minute data, daily market/on-chain data,
news, the Qwen3-8B model and llama.cpp binaries — is inside this repo, so
it runs on machines with **no internet access** (e.g. a Snowflake
workspace that was synced once from GitHub).

---

## 1. What's inside

```
run_training.py            orchestrator: GPU/CPU detection, llama.cpp server,
                           parallel agents, overview PNG, clean shutdown
status.py                  quick text status of all agents
setup_offline.sh           one-time setup (assemble model, build data cache)
trader/
  config.py                all hyper-parameters and paths
  data.py                  loads the local cache (minute bars, daily, news)
  env.py                   $2000 paper-trading environment (1 day = 1 episode)
  agent.py                 Double-DQN in pure NumPy (Adam, target net, replay)
  llm.py                   Qwen3-8B advisor: prompt, JSON parsing, disk cache
  plots.py                 per-epoch PNG reports + progress/overview PNGs
  train.py                 per-agent training worker (resumable)
scripts/
  prepare_data.py          builds cache/ from data/ (gap-fills, daily metrics)
  assemble_model.py        re-joins llm/qwen3-8b-parts/* -> qwen3-8b-q4km.gguf
  prefill_llm_cache.py     pre-computes the LLM's day analyses (recommended)
  check_llm.sh             sanity-check for the bundled llama.cpp binaries
  fetch/                   reproducibility: scripts that downloaded the data
data/
  minute/<SYM>/            Binance Vision 1m klines (2025-01 .. 2026-09)
  daily/                   Fear&Greed, CoinMetrics on-chain, CoinGecko
  news/                    GDELT + Hacker News headlines per day
llm/
  llama-b11249/            llama.cpp CPU binaries (ubuntu x64, glibc >= 2.34)
  qwen3-8b-parts/          Qwen3-8B Q4_K_M GGUF in <100 MB parts + SHA256SUMS
```

## 2. Quick start (offline machine)

```bash
bash setup_offline.sh          # assemble model + build data cache (one-time)
python3 run_training.py        # train (Ctrl+C to stop - it resumes!)
```

That's it. Useful variants:

```bash
python3 run_training.py --agents 4        # force 4 parallel agents
python3 run_training.py --epochs 5000     # stop after N epochs per agent
python3 run_training.py --no-llm          # smoke-test without Qwen
python3 run_training.py --coin-mode random  # trade ONE random coin per day
python3 run_training.py --png-keep latest:300  # keep only newest 300 PNGs/agent
python3 status.py                         # text summary anytime
```

**Requirements:** Python 3.9+, `numpy`, `matplotlib` (both preinstalled on
the Snowflake ML image), ~11 GB free disk (repo + assembled model), and for
the LLM ~7 GB free RAM. On CPU-only machines without 7 GB RAM the trainer
automatically falls back to neutral advisor features and keeps training.

## 3. How it works

**One epoch = one trading day.** The agent starts each epoch with a fake
**$2,000** cash balance and 1440 one-minute steps. At every minute it sees
per-coin technical features (multi-horizon log returns, price/volume
z-scores, RSI, realised vol), its current cash/coin allocation, the time of
day, and the LLM's 9-dim assessment of the day. It picks one of 7 actions:

| action | effect |
|---|---|
| `HOLD` | do nothing |
| `BUY_BTC/ETH/LTC` | invest 25% of equity in that coin (0.1% fee) |
| `SELL_BTC/ETH/LTC` | liquidate that coin position (0.1% fee) |

Reward = fractional change in equity per minute. Learning is
**Double-DQN** (dueling-free MLP 39→256→256→7, Huber loss, Adam, target
network with Polyak averaging, 300k replay buffer) implemented in pure
NumPy — no PyTorch needed, and for a net this small NumPy is faster than
GPU anyway. Multiple agents = multiple independent processes training in
parallel (one per GPU when GPUs exist, otherwise CPU workers); each has
its own environment sampling, seed, checkpoint and reports.

**LLM advisor (Qwen3-8B on CPU via llama.cpp).** `run_training.py` starts
`llama-server` locally (no external network). At the start of each epoch
the agent asks Qwen to read **that day's** headlines (GDELT + Hacker
News, when available), the Fear & Greed index, yesterday's OHLCV, volume
vs 30-day average and on-chain activity (active addresses, transaction
count from CoinMetrics), and answer with strict JSON:

```json
{"BTC": {"sentiment": 1, "confidence": 0.7}, "ETH": {...}, "LTC": {...},
 "risk": 0.4, "volatility": 0.6, "summary": "ETF inflows continue"}
```

That becomes a 9-dim feature vector appended to every observation of the
day — the DQN learns *how much to trust the analyst* from experience.
Analyses are **cached per calendar day** (`runs/shared/llm_cache/`), so
CPU inference happens at most once per day, ever. Run
`scripts/prefill_llm_cache.py` first to warm the whole cache (~600 days)
if you want zero LLM latency during training.

**Reports.** After **every epoch**, each agent writes
`runs/agent_i/epochs/epoch_NNNNNN_<day>.png` containing: equity curve vs
buy&hold, portfolio exposure (cash/BTC/ETH/LTC), price chart with executed
trade markers, drawdown, action distribution, cumulative PnL across
epochs, fees, epsilon, buffer size and the LLM's verdict. Plus
`runs/agent_i/progress.png` (running history), `runs/agent_i/latest.png`,
`runs/overview.png` (all agents) and `runs/agent_i/metrics.jsonl` (raw
numbers).

**Resumable.** Every epoch the agent atomically saves its weights, Adam
moments, epsilon, RNG state, replay buffer (every 20 epochs) and history.
Stop any time (Ctrl+C, SIGTERM, machine reboot) — restart and it continues
from the next epoch. The last ~45 days of data are held out for greedy
evaluation, reported on the PNGs.

## 4. Data sources (all free, downloaded once — see `scripts/fetch/`)

| data | source | range |
|---|---|---|
| 1-minute BTC/ETH/LTC klines | Binance Vision public dumps | 2025-01 → 2026-09 |
| Fear & Greed index | alternative.me | 1000 days |
| on-chain (active addresses, tx count, mcap) | CoinMetrics community | full history |
| daily market charts | CoinGecko | 365 days |
| news headlines | GDELT 2.0 DOC API + HN (Algolia) | 2025-01 → 2026-09 |

News coverage is best-effort (GDELT is heavily rate-limited; some days
have no headlines) — the prompt and training handle missing news
gracefully, as "no news available".

## 5. Notes & limits

- The model is **Qwen3-8B Q4_K_M** (bartowski GGUF, 4.7 GB). GitHub's
  100 MB file cap forced us to ship it as 54 `part-*` files;
  `setup_offline.sh` stitches them back (sha256-verified).
- llama.cpp binaries are the official `ubuntu-x64` build b11249 — needs
  glibc >= 2.34 (Ubuntu 22.04+) and `libssl3`/`libgomp1`, which stock
  Ubuntu/Debian images include. `scripts/check_llm.sh` verifies.
- GPU machines: agents are pinned one-per-GPU via `CUDA_VISIBLE_DEVICES`;
  the NumPy DQN itself is CPU-based (fast for this net size), so GPUs
  mainly buy you more parallel agents.
- This is a research toy, **not financial advice** — the agent trades a
  simulated $2,000 on historical data with fixed 0.1% fees and no slippage.
