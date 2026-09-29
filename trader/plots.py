"""PNG report rendering (pure matplotlib, Agg backend, no display needed).

- save_epoch_png():    full report of ONE epoch (one trading day) per agent
- save_progress_png(): running history of one agent
- save_overview_png(): all parallel agents side by side
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

_GREEN = "#2e7d32"
_RED = "#c62828"
_BLUE = "#1565c0"
_ORANGE = "#ef6c00"
_GREY = "#757575"


def _safe_ax(ax):
    ax.grid(True, alpha=0.3)


def save_epoch_png(path, rec):
    """rec keys: agent, epoch, day, coin_mode, active_coin, equity(list),
    bh_equity, exposure (4,1440), close {coin: arr}, actions, executed,
    action_counts, fees, trades, profit, profit_pct, bh_profit, alpha,
    history (list of past profits), eval_history, epsilon, buffer_size,
    steps_trained, llm {advice, source, summary}, wall_s, config"""
    try:
        eq = np.asarray(rec["equity"], dtype=float)
        bh = np.asarray(rec["bh_equity"], dtype=float)
        n = min(len(eq) - 1, C.MINUTES_PER_DAY)
        x = np.arange(n + 1)
        minutes = np.arange(C.MINUTES_PER_DAY)
        hh = (minutes // 60).astype(int)
        mm = minutes % 60
        tlabels = [f"{h:02d}:{m:02d}" for h, m in zip(hh, mm)]

        fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
        fig.suptitle(
            f"{rec['agent']}  |  epoch {rec['epoch']}  |  {rec['day']}"
            f"  |  PnL {rec['profit']:+,.2f} USD ({rec['profit_pct']:+.2f}%)"
            f"  |  buy&hold {rec['bh_profit']:+,.2f} USD",
            fontsize=13, fontweight="bold")

        # 1. equity vs buy&hold
        ax = axes[0, 0]
        ax.plot(x, eq, color=_BLUE, lw=1.6, label="agent equity")
        ax.plot(x, bh, color=_ORANGE, lw=1.2, ls="--", label="equal buy&hold")
        ax.axhline(C.START_BALANCE, color=_GREY, lw=1, ls=":",
                   label=f"start ${C.START_BALANCE:,.0f}")
        ax.set_title("Equity curve (1 day)")
        ax.set_xlabel("minute of day (UTC)")
        ax.set_ylabel("USD")
        step = max(1, (n + 1) // 12)
        ax.set_xticks(np.arange(0, n + 1, step))
        ax.set_xticklabels([tlabels[i] for i in range(0, n + 1, step)],
                           rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        _safe_ax(ax)

        # 2. exposure
        ax = axes[0, 1]
        expo = np.asarray(rec["exposure"], dtype=float)[:, :C.MINUTES_PER_DAY]
        labels = ["cash", "BTC", "ETH", "LTC"]
        colors = [_GREY, "#f2a900", "#627eea", "#345d9d"]
        ax.stackplot(minutes, expo, labels=labels, colors=colors, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title("Portfolio exposure")
        ax.set_xlabel("minute of day (UTC)")
        ax.legend(loc="center left", fontsize=8)
        step = max(1, C.MINUTES_PER_DAY // 12)
        ax.set_xticks(np.arange(0, C.MINUTES_PER_DAY, step))
        ax.set_xticklabels([tlabels[i] for i in range(0, C.MINUTES_PER_DAY, step)],
                           rotation=45, fontsize=7)
        _safe_ax(ax)

        # 3. price + trade markers (most traded coin, else BTC)
        ax = axes[0, 2]
        counts = rec.get("action_counts") or {}
        coin_scores = {c: counts.get(f"BUY_{c}", 0) + counts.get(f"SELL_{c}", 0)
                       for c in C.COINS}
        focus = max(coin_scores, key=lambda c: coin_scores[c]) if any(
            coin_scores.values()) else "BTC"
        close = np.asarray(rec["close"][focus], dtype=float)
        ax.plot(minutes, close, color="#37474f", lw=1.0)
        actions = np.asarray(rec["actions"], dtype=int)
        executed = np.asarray(rec["executed"], dtype=bool)
        ci = C.COINS.index(focus)
        buy_a, sell_a = 1 + 2 * ci, 2 + 2 * ci
        bm = executed & ((actions == buy_a))
        sm = executed & ((actions == sell_a))
        ax.scatter(minutes[bm], close[bm], marker="^", s=28, color=_GREEN,
                   label=f"BUY {focus}", zorder=5)
        ax.scatter(minutes[sm], close[sm], marker="v", s=28, color=_RED,
                   label=f"SELL {focus}", zorder=5)
        ax.set_title(f"{focus} price + executed trades "
                     f"({int(bm.sum())}B/{int(sm.sum())}S)")
        ax.legend(fontsize=8)
        _safe_ax(ax)

        # 4. drawdown
        ax = axes[1, 0]
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100.0
        ax.fill_between(x, dd, 0, color=_RED, alpha=0.5)
        ax.set_title(f"Drawdown (max {dd.min():.2f}%)")
        ax.set_xlabel("minute of day (UTC)")
        ax.set_ylabel("%")
        step = max(1, (n + 1) // 12)
        ax.set_xticks(np.arange(0, n + 1, step))
        ax.set_xticklabels([tlabels[i] for i in range(0, n + 1, step)],
                           rotation=45, fontsize=7)
        _safe_ax(ax)

        # 5. action distribution + stats box
        ax = axes[1, 1]
        names = C.ACTIONS
        vals = [int(counts.get(a, 0)) for a in names]
        bars = ax.bar(range(len(names)), vals,
                      color=[_GREY] + [_GREEN, _RED] * 3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, fontsize=7)
        ax.set_title("Actions taken this day")
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                        ha="center", va="bottom", fontsize=7)
        _safe_ax(ax)

        # 6. cumulative PnL across epochs
        ax = axes[1, 2]
        hist = list(rec.get("history") or []) + [rec["profit"]]
        cum = np.cumsum(hist)
        ax.plot(np.arange(1, len(cum) + 1), cum, color=_BLUE, lw=1.5)
        ax.axhline(0, color=_GREY, lw=1, ls=":")
        if rec.get("eval_history"):
            ev_x, ev_y = zip(*rec["eval_history"])
            ax.scatter(ev_x, ev_y, color=_ORANGE, s=22, zorder=5,
                       label="greedy eval (3-day avg)")
            ax.legend(fontsize=8)
        ax.set_title("Cumulative training PnL (all epochs)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("USD")
        _safe_ax(ax)

        # footer stats
        llm = rec.get("llm") or {}
        adv = llm.get("advice") or {}
        sent = " ".join(
            f"{c}:{adv.get(c, {}).get('sentiment', 0):+d}"
            f"({adv.get(c, {}).get('confidence', 0):.1f})" for c in C.COINS)
        recent = hist[-100:]
        wins = sum(1 for p in recent if p > 0)
        stats = (
            f"start ${C.START_BALANCE:,.0f} -> final ${eq[-1]:,.2f}   |   "
            f"fees ${rec['fees']:.2f}   trades {rec['trades']}   |   "
            f"alpha vs B&H {rec['alpha']:+,.2f} USD   |   "
            f"epsilon {rec['epsilon']:.3f}   buffer {rec['buffer_size']:,}   "
            f"trained steps {rec.get('steps_trained', 0):,}\n"
            f"LLM[{llm.get('source', 'n/a')}]  {sent}   "
            f"risk {adv.get('risk', 0.5):.2f}  vol {adv.get('volatility', 0.5):.2f}"
            f"  \"{adv.get('summary', '')}\"   |   "
            f"win-rate(last {len(recent)}): {wins}/{len(recent)}   |   "
            f"epoch wall {rec.get('wall_s', 0):.1f}s")
        fig.text(0.5, 0.005, stats, ha="center", fontsize=8.5, family="monospace")

        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[plots] epoch png failed: {e}", flush=True)
        return False


def save_progress_png(path, agent_name, history, evals, eps_hist):
    """history: list of per-epoch profits; evals: [(epoch, avg_pnl)];"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"{agent_name} - training progress", fontweight="bold")
        h = np.asarray(history, dtype=float)
        ep = np.arange(1, len(h) + 1)

        ax = axes[0, 0]
        ax.bar(ep, h, color=[_GREEN if v > 0 else _RED for v in h], alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("PnL per epoch (USD)")
        _safe_ax(ax)

        ax = axes[0, 1]
        ax.plot(ep, np.cumsum(h), color=_BLUE, lw=1.6, label="cumulative PnL")
        k = min(50, len(h))
        if k > 1:
            ax.plot(ep[k - 1:], np.convolve(h, np.ones(k) / k, mode="valid"),
                    color=_ORANGE, lw=1.2, label=f"rolling mean({k})")
        ax.axhline(0, color=_GREY, ls=":")
        ax.legend(fontsize=8)
        ax.set_title("Cumulative PnL (USD)")
        _safe_ax(ax)

        ax = axes[1, 0]
        if evals:
            ex, ey = zip(*evals)
            ax.plot(ex, ey, "o-", color=_ORANGE, label="greedy eval")
            ax.legend(fontsize=8)
            ax.axhline(0, color="k", lw=0.8)
        ax.set_title("Greedy evaluation on held-out days (avg USD/day)")
        _safe_ax(ax)

        ax = axes[1, 1]
        if eps_hist:
            ax.plot(np.arange(1, len(eps_hist) + 1), eps_hist, color="#6a1b9a")
            ax.set_ylim(0, 1.05)
        ax.set_title("Epsilon (exploration)")
        _safe_ax(ax)

        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[plots] progress png failed: {e}", flush=True)
        return False


def save_overview_png(path, agents):
    """agents: list of (name, history_list, evals_list)"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle("All parallel agents - training overview",
                     fontweight="bold")
        ax = axes[0]
        for name, h, _ in agents:
            if h:
                ax.plot(np.arange(1, len(h) + 1), np.cumsum(h), lw=1.5,
                        label=f"{name} ({sum(h):+,.0f}$ / {len(h)} ep)")
        ax.axhline(0, color=_GREY, ls=":")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cumulative PnL (USD)")
        ax.legend(fontsize=8)
        _safe_ax(ax)

        ax = axes[1]
        names, means, last = [], [], []
        for name, h, _ in agents:
            if h:
                names.append(name)
                means.append(float(np.mean(h[-100:])))
                last.append(float(np.sum(h)))
        w = 0.4
        r = np.arange(len(names))
        ax.bar(r - w / 2, means, w, label="mean PnL/epoch (last 100)",
               color=_BLUE)
        ax.bar(r + w / 2, [m / 10 for m in last], w,
               label="cumulative PnL / 10", color=_ORANGE)
        ax.set_xticks(r)
        ax.set_xticklabels(names, fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.legend(fontsize=8)
        _safe_ax(ax)

        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[plots] overview png failed: {e}", flush=True)
        return False
