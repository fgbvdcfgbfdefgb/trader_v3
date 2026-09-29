#!/usr/bin/env python3
"""Quick status report of a running/finished training session.

Run:  python3 status.py
"""
import glob
import json
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def main():
    dirs = sorted(glob.glob(os.path.join(RUNS, "agent_*")))
    if not dirs:
        print("no runs found under", RUNS)
        return
    for d in dirs:
        name = os.path.basename(d)
        p = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(p):
            print(f"{name}: no metrics yet")
            continue
        hist = []
        with open(p) as f:
            for line in f:
                try:
                    hist.append(json.loads(line))
                except Exception:
                    pass
        if not hist:
            print(f"{name}: empty")
            continue
        last = hist[-1]
        profits = np.array([h["profit"] for h in hist])
        last100 = profits[-100:]
        evals = [h for h in hist if h.get("eval_pnl") is not None]
        print(f"== {name} ==")
        print(f"  epochs: {len(hist)}  last day: {last['day']}  "
              f"last PnL: {last['profit']:+.2f}$ ({last['profit_pct']:+.2f}%)")
        print(f"  mean PnL/epoch (all): {profits.mean():+.2f}$   "
              f"(last 100): {last100.mean():+.2f}$   "
              f"win rate (last 100): {(last100 > 0).mean()*100:.0f}%")
        print(f"  cumulative training PnL: {profits.sum():+.2f}$   "
              f"avg alpha vs buy&hold: "
              f"{np.mean([h['alpha'] for h in hist]):+.2f}$")
        print(f"  epsilon: {last['epsilon']:.3f}  buffer: "
              f"{last['buffer_size']:,}  trained steps: "
              f"{last['steps_trained']:,}")
        print(f"  reports: {d}/latest.png  {d}/progress.png  "
              f"{d}/epochs/ ({len(glob.glob(os.path.join(d, 'epochs', '*.png')))} epoch PNGs)")
    ov = os.path.join(RUNS, "overview.png")
    if os.path.exists(ov):
        print("overview:", ov)


if __name__ == "__main__":
    main()
