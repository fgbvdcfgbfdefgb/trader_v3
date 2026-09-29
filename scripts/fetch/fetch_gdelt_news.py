#!/usr/bin/env python3
"""Fetch daily crypto news headlines from GDELT 2.0 DOC API (free, no key).

GDELT enforces ~1 request / 5 seconds -> we pace at ~6.5s. One combined
query per day covering BTC/ETH/LTC, articles classified per coin later.
Output: JSONL, one line per day: {"day": "YYYY-MM-DD", "articles": [...]}
"""
import json
import os
import sys
import time
import datetime as dt
import urllib.request
import urllib.parse

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "news", "gdelt_all.jsonl")
START = dt.date(2025, 1, 1)
END = dt.date(2026, 9, 28)
QUERY = "(bitcoin OR ethereum OR litecoin OR btc OR eth OR ltc) sourcelang:english"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) academic-dataset-builder"}


def fetch_day(d: dt.date):
    s = d.strftime("%Y%m%d000000")
    e = d.strftime("%Y%m%d235959")
    url = ("https://api.gdeltproject.org/api/v2/doc/doc?query="
           + urllib.parse.quote(QUERY)
           + "&mode=artlist&maxrecords=250&sort=datedesc&format=json"
           + f"&startdatetime={s}&enddatetime={e}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = r.read().decode("utf-8", "replace")
    j = json.loads(raw)
    return j.get("articles") or []


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["day"])
                except Exception:
                    pass
    days = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    todo = [d for d in days if d.isoformat() not in done]
    print(f"gdelt: {len(todo)} days to fetch (of {len(days)})", flush=True)
    backoff = 12.0
    with open(OUT, "a") as f:
        for i, d in enumerate(todo):
            arts = []
            err = None
            for attempt in range(4):
                try:
                    arts = fetch_day(d)
                    err = None
                    backoff = max(10.0, backoff * 0.9)
                    break
                except Exception as ex:
                    err = str(ex)
                    time.sleep(min(150.0, backoff))
                    backoff = min(150.0, backoff * 1.7)
            if err is not None:
                # do NOT persist failed days - they will be retried on rerun
                print(f"{d} SKIPPED ({err}); backoff={backoff:.0f}s", flush=True)
                time.sleep(backoff)
                continue
            rec = {"day": d.isoformat(), "articles": [
                {"t": a.get("title", ""), "d": a.get("domain", ""), "s": a.get("seendate", "")}
                for a in arts[:250]]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"[{i+1}/{len(todo)}] {d} n={len(arts)}", flush=True)
            time.sleep(backoff)
    print("GDELT_DONE", flush=True)


if __name__ == "__main__":
    main()
