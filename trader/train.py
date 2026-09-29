"""Per-agent training worker.

One process per agent (spawned by run_training.py). Each agent:
  - samples a random historical day (BTC/ETH/LTC minute bars),
  - asks the Qwen3-8B advisor for that day's news/market analysis (cached),
  - trades the day with a $2000 paper balance (one epoch = one day),
  - learns with double-DQN (pure NumPy),
  - saves a full PNG report for EVERY epoch + checkpoint (resumable).

Run standalone:
  python -m trader.train --agent-id 0 --llm-url http://127.0.0.1:8080
"""
import argparse
import json
import os
import shutil
import signal
import sys
import time

import numpy as np

from . import config as C
from . import plots
from .agent import DQNAgent, ReplayBuffer
from .data import MarketData
from .env import DayEnv
from .llm import DayAdvisor, neutral_features

_STOP = {"flag": False}


def _handle_stop(signum, frame):
    _STOP["flag"] = True


def run_eval(agent, md, eval_days, coin_mode, advisor, n_days=3):
    """Greedy policy on held-out days; returns avg profit USD (no learning)."""
    if not eval_days:
        return None
    picks = [eval_days[int(i)] for i in
             np.linspace(0, len(eval_days) - 1, min(n_days, len(eval_days)))]
    profits = []
    for day in picks:
        feats, advice, meta = advisor.get(day, md.day_context(day))
        env = DayEnv(md.get_episode(day), feats, coin_mode)
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, eps=0.0)
            obs, r, done, info = env.step(a)
        profits.append(env.equity - C.START_BALANCE)
    return float(np.mean(profits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", type=int, default=0)
    ap.add_argument("--run-dir", default=C.RUNS_DIR)
    ap.add_argument("--cache-dir", default=C.CACHE_DIR)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = run forever (until stopped; resumable)")
    ap.add_argument("--llm-url", default="",
                    help="llama.cpp server URL; empty = stub (no LLM)")
    ap.add_argument("--llm-cache", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--coin-mode", choices=["all", "random"], default="all",
                    help="all = trade BTC+ETH+LTC together; "
                         "random = one random coin per epoch")
    ap.add_argument("--png-keep", default="all",
                    help="'all' or 'latest:N' to keep only the newest N epoch PNGs")
    ap.add_argument("--eval-every", type=int, default=C.EVAL_EVERY)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    name = f"agent_{args.agent_id}"
    run_dir = os.path.join(args.run_dir, name)
    epochs_dir = os.path.join(run_dir, "epochs")
    os.makedirs(epochs_dir, exist_ok=True)
    llm_cache = args.llm_cache or os.path.join(
        args.run_dir, "shared", C.LLM_CACHE_DIRNAME)

    md = MarketData(args.cache_dir)
    train_days, eval_days = md.split()
    rng = np.random.default_rng(args.seed)

    probe = DayEnv(md.get_episode(train_days[0]), neutral_features(),
                   coin_mode=args.coin_mode)
    agent = DQNAgent(probe.obs_dim, C.N_ACTIONS, rng)
    buffer = ReplayBuffer(C.BUFFER_CAPACITY, probe.obs_dim, rng)

    ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
    buffer_path = os.path.join(run_dir, "buffer.npz")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    start_epoch, history, evals, eps_hist = 0, [], [], []
    extra = agent.load(ckpt_path)
    if extra:
        start_epoch = int(extra.get("epoch", 0))
        history = list(extra.get("history", []))
        evals = [tuple(e) for e in extra.get("evals", [])]
        eps_hist = list(extra.get("eps_hist", []))
        if extra.get("coin_mode") not in (None, args.coin_mode):
            print(f"[{name}] WARNING: coin_mode changed "
                  f"{extra.get('coin_mode')} -> {args.coin_mode}; "
                  f"resuming weights anyway", flush=True)
        print(f"[{name}] RESUMED from epoch {start_epoch} "
              f"({len(history)} epochs of history, "
              f"{agent.total_steps:,} trained steps)", flush=True)
    buffer.load(buffer_path)

    advisor = DayAdvisor(llm_cache, args.llm_url or None)
    print(f"[{name}] start: obs_dim={agent.obs_dim} actions={agent.n_actions} "
          f"train_days={len(train_days)} eval_days={len(eval_days)} "
          f"buffer={len(buffer):,} llm={'ON ' + args.llm_url if args.llm_url else 'STUB'}",
          flush=True)

    png_keep_n = None
    if args.png_keep.startswith("latest:"):
        try:
            png_keep_n = max(1, int(args.png_keep.split(":", 1)[1]))
        except ValueError:
            png_keep_n = None

    epoch = start_epoch
    day = None
    while not _STOP["flag"]:
        if args.epochs and epoch >= args.epochs:
            break
        epoch += 1
        t0 = time.time()

        day = train_days[int(rng.integers(len(train_days)))]
        feats, advice, llm_meta = advisor.get(day, md.day_context(day))
        env = DayEnv(md.get_episode(day), feats, coin_mode=args.coin_mode)
        obs = env.reset()
        done = False
        loss_sum, n_learn = 0.0, 0
        while not done:
            if _STOP["flag"]:
                break
            a = agent.act(obs, eps=agent.epsilon())
            obs2, r, done, info = env.step(a)
            buffer.add(obs, a, r, obs2, done)
            if (len(buffer) >= C.WARMUP_STEPS
                    and (env.t % C.LEARN_EVERY) == 0):
                s, ac, rw, s2, dd = buffer.sample(C.BATCH_SIZE)
                loss, td = agent.train_batch(s, ac, rw, s2, dd)
                loss_sum += loss
                n_learn += 1
            obs = obs2

        profit = env.equity - C.START_BALANCE
        profit_pct = profit / C.START_BALANCE * 100.0
        bh = env.buyhold_equity()
        bh_profit = float(bh[-1] - C.START_BALANCE)
        counts = {a: int(np.sum(env.actions_taken == i))
                  for i, a in enumerate(C.ACTIONS)}
        eq = np.asarray(env.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(((eq - peak) / peak).min() * 100.0) if len(eq) else 0.0

        history.append(profit)
        eps_hist.append(agent.epsilon())

        eval_pnl = None
        if epoch % max(1, args.eval_every) == 0:
            eval_pnl = run_eval(agent, md, eval_days, args.coin_mode, advisor)
            if eval_pnl is not None:
                evals.append((epoch, eval_pnl))

        rec = {
            "agent": name, "epoch": epoch, "day": day,
            "coin_mode": args.coin_mode, "active_coin": env.active_coin,
            "equity": env.equity_curve, "bh_equity": bh.tolist(),
            "exposure": env.exposure.tolist(),
            "close": {c: env.close[c].tolist() for c in C.COINS},
            "actions": env.actions_taken.tolist(),
            "executed": env.executed.tolist(),
            "action_counts": counts,
            "fees": env.fees_paid, "trades": env.trades,
            "profit": profit, "profit_pct": profit_pct,
            "bh_profit": bh_profit, "alpha": profit - bh_profit,
            "max_dd_pct": max_dd,
            "history": history, "eval_history": evals,
            "epsilon": agent.epsilon(), "buffer_size": len(buffer),
            "steps_trained": agent.total_steps,
            "avg_loss": (loss_sum / n_learn) if n_learn else 0.0,
            "llm": {"source": llm_meta.get("source", "stub"),
                    "advice": advice,
                    "summary": advice.get("summary", "")},
            "wall_s": time.time() - t0,
        }
        try:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    k: rec[k] for k in
                    ("agent", "epoch", "day", "profit", "profit_pct",
                     "bh_profit", "alpha", "max_dd_pct", "trades", "fees",
                     "epsilon", "buffer_size", "steps_trained", "avg_loss",
                     "wall_s", "active_coin")}) + "\n")
        except Exception:
            pass

        agent.save(ckpt_path, extra={
            "epoch": epoch, "history": history[-4000:],
            "evals": [list(e) for e in evals[-500:]],
            "eps_hist": eps_hist[-5000:], "coin_mode": args.coin_mode,
            "seed": args.seed, "day": day,
        })
        if epoch % C.SAVE_BUFFER_EVERY == 0 or _STOP["flag"]:
            buffer.save(buffer_path)

        png_path = os.path.join(epochs_dir,
                                f"epoch_{epoch:06d}_{day}.png")
        plots.save_epoch_png(png_path, rec)
        try:
            shutil.copyfile(png_path, os.path.join(run_dir, "latest.png"))
        except Exception:
            pass
        if png_keep_n is not None and (epoch % 10) == 0:
            files = sorted(os.listdir(epochs_dir))
            while len(files) > png_keep_n:
                os.remove(os.path.join(epochs_dir, files.pop(0)))
        if (epoch % C.PROGRESS_PNG_EVERY) == 0 or _STOP["flag"]:
            plots.save_progress_png(
                os.path.join(run_dir, "progress.png"), name,
                history, evals, eps_hist)

        ev_txt = (f" eval {eval_pnl:+.2f}$" if eval_pnl is not None else "")
        print(f"[{name}] ep {epoch:5d} day {day} pnl {profit:+9.2f}$ "
              f"({profit_pct:+6.2f}%) bh {bh_profit:+8.2f}$ "
              f"alpha {profit - bh_profit:+8.2f}$ trades {env.trades:3d} "
              f"fees {env.fees_paid:6.2f}$ eps {agent.epsilon():.3f} "
              f"buf {len(buffer):7,} llm[{llm_meta.get('source', 'stub')}]"
              f"{ev_txt} {rec['wall_s']:.1f}s", flush=True)

    agent.save(ckpt_path, extra={
        "epoch": epoch, "history": history[-4000:],
        "evals": [list(e) for e in evals[-500:]],
        "eps_hist": eps_hist[-5000:], "coin_mode": args.coin_mode,
        "seed": args.seed, "day": day, "stopped": True,
    })
    buffer.save(buffer_path)
    plots.save_progress_png(os.path.join(run_dir, "progress.png"), name,
                            history, evals, eps_hist)
    print(f"[{name}] stopped at epoch {epoch} - checkpoint saved, "
          f"resumable", flush=True)


if __name__ == "__main__":
    sys.exit(main())
