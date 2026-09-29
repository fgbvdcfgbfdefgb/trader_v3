#!/usr/bin/env python3
"""Fetch daily crypto-related Hacker News stories via the Algolia HN API
(free, no key). Supplements GDELT headlines with tech-community signal.

Output: JSONL, one line per day: {"day": "...", "btc": [...], "eth": [...], "ltc": [...]}
"""
import json
import os
import sys
import time
import datetime as dt
import urllib.request
import urllib.parse

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "news", "hn_stories.jsonl")
START = dt.date(2025, 1, 1)
END = dt.date(2026, 9, 28)
COINS = {"btc": "bitcoin", "eth": "ethereum", "ltc": "litecoin"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) academic-dataset-builder"}


def fetch(coin_kw, day):
    s = int(dt.datetime(day.year, day.month, day.day, 0, 0, 0,
                        tzinfo=dt.timezone.utc).timestamp())
    e = s + 86400
    nf = f"created_at_i>{s},created_at_i<{e}"
    url = ("https://hn.algolia.com/api/v1/search_by_date?query="
           + urllib.parse.quote(coin_kw)
           + "&tags=story&numericFilters=" + urllib.parse.quote(nf)
           + "&hitsPerPage=100")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    out = []
    for h in j.get("hits", []):
        try:
            if int(h.get("points") or 0) < 2:
                continue
        except Exception:
            pass
        out.append({"t": h.get("title") or "", "p": h.get("points") or 0,
                    "c": h.get("num_comments") or 0})
    return out[:40]


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
    print(f"hn: {len(todo)} days to fetch", flush=True)
    with open(OUT, "a") as f:
        for i, d in enumerate(todo):
            rec = {"day": d.isoformat()}
            for c, kw in COINS.items():
                stories = []
                for attempt in range(4):
                    try:
                        stories = fetch(kw, d)
                        break
                    except Exception as ex:
                        time.sleep(8)
                rec[c] = stories
                time.sleep(0.35)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{i+1}/{len(todo)}] {d} btc={len(rec['btc'])} eth={len(rec['eth'])} ltc={len(rec['ltc'])}", flush=True)
            time.sleep(0.4)
    print("HN_DONE", flush=True)


if __name__ == "__main__":
    main()
