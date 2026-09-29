#!/usr/bin/env python3
"""Warm the LLM advice cache ahead of training (optional but recommended).

Every training epoch asks Qwen3-8B to analyse that day's news once; the
answer is cached forever after. This script pre-computes those answers for
all days so training epochs never wait on the LLM.

Run AFTER the llama.cpp server is up (it starts its own copy if you pass
--start-server), e.g.:
  python3 scripts/prefill_llm_cache.py --url http://127.0.0.1:8080
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trader import config as C  # noqa: E402
from trader.data import MarketData  # noqa: E402
from trader.llm import DayAdvisor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--cache-dir", default=C.CACHE_DIR)
    ap.add_argument("--run-dir", default=C.RUNS_DIR)
    ap.add_argument("--limit", type=int, default=0, help="0 = all days")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-fetch days whose cached answer was a fallback")
    args = ap.parse_args()

    md = MarketData(args.cache_dir)
    cache_dir = os.path.join(args.run_dir, "shared", C.LLM_CACHE_DIRNAME)
    advisor = DayAdvisor(cache_dir, args.url)
    if not advisor.client or not advisor.client.healthy():
        print(f"[prefill] no healthy LLM server at {args.url} - aborting")
        return 1

    days = md.days
    if args.limit:
        days = days[:args.limit]
    n_ok, n_new, n_fail = 0, 0, 0
    t0 = time.time()
    for i, day in enumerate(days):
        p = advisor._path(day)
        if args.retry_failed and os.path.exists(p):
            try:
                with open(p) as f:
                    if json.load(f).get("source") != "llm":
                        os.remove(p)
            except Exception:
                pass
        feats, advice, meta = advisor.get(day, md.day_context(day))
        if meta.get("source") == "llm":
            n_ok += 1
        else:
            n_fail += 1
        n_new += 1
        if i % 10 == 0:
            el = time.time() - t0
            eta = el / max(1, i) * (len(days) - i) / 60
            print(f"[{i+1}/{len(days)}] {day} source={meta.get('source')} "
                  f"({el:.0f}s elapsed, ~{eta:.0f}m left)", flush=True)
    print(f"[prefill] done: {n_ok} ok, {n_fail} fallback, {n_new} processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
