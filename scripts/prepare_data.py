#!/usr/bin/env python3
"""Build the training cache from the raw data bundled in this repo (offline).

Input (already in repo):
  data/minute/<SYM>/<SYM>-1m-YYYY-MM.zip / -YYYY-MM-DD.zip   (Binance Vision)
  data/daily/fear_greed.json                                 (alternative.me)
  data/daily/coinmetrics_{btc,eth,ltc}.csv                   (coinmetrics.io)
  data/daily/coingecko_*.json                                (coingecko.com)
  data/news/gdelt_all.jsonl                                  (GDELT 2.0 DOC API)
  data/news/hn_stories.jsonl                                 (HN Algolia API)

Output (cache/):
  <SYM>_1m.npy   (n_days*1440, 6) float64 [ts_ms, o, h, l, c, v] - gap-filled
  days.json      aligned day list
  daily.json     per-day metrics + fear&greed + on-chain deltas (LLM context)
  news.json      per-day per-coin headline lists (LLM context)

Run:  python3 scripts/prepare_data.py
"""
import csv
import datetime as dt
import glob
import io
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trader import config as C  # noqa: E402

MIN_REAL_ROWS = 1200  # a day needs >= this many real minutes to be kept


def parse_symbol_zips(sym):
    """All minute bars for one symbol -> sorted, de-duplicated (N, 6) array."""
    files = sorted(glob.glob(os.path.join(C.DATA_DIR, "minute", sym, "*.zip")))
    if not files:
        raise FileNotFoundError(f"no zip files under data/minute/{sym}")
    chunks = []
    for path in files:
        try:
            with zipfile.ZipFile(path) as z:
                name = z.namelist()[0]
                text = z.read(name).decode("utf-8", "replace")
        except Exception as e:
            print(f"  ! skipping corrupt {path}: {e}")
            continue
        rows = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",")
            head = parts[0][:2]
            if not head.isdigit():    # header line
                continue
            try:
                ts = int(float(parts[0]))
                if ts > 10 ** 14:      # microseconds -> milliseconds
                    ts //= 1000
                rows.append((ts, float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4]), float(parts[5])))
            except (ValueError, IndexError):
                continue
        if rows:
            chunks.append(np.array(rows, dtype=np.float64))
    if not chunks:
        raise RuntimeError(f"no rows parsed for {sym}")
    arr = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    del chunks
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = arr[1:, 0] != arr[:-1, 0]
    arr = arr[keep]
    return arr


def ffill(a):
    """Forward-fill NaNs (vectorized); leading NaNs back-filled."""
    n = len(a)
    if n == 0:
        return a
    valid = ~np.isnan(a)
    if not valid.any():
        return np.zeros_like(a)
    idx = np.where(valid, np.arange(n), 0)
    idx = np.maximum.accumulate(idx)
    out = a[idx]
    fv = int(np.argmax(valid))   # first valid position
    out[:fv] = a[fv]             # back-fill leading NaNs
    return out


def normalize_to_days(arr):
    """Gap-fill to a strict 1440-minute UTC grid.
    Returns (normalized (n_days*1440,6) array, kept_days, real_row_counts)."""
    day0 = dt.datetime.fromtimestamp(arr[0, 0] / 1000.0, dt.timezone.utc)
    day0 = day0.replace(hour=0, minute=0, second=0, microsecond=0)
    dayN = dt.datetime.fromtimestamp(arr[-1, 0] / 1000.0, dt.timezone.utc)
    dayN = dayN.replace(hour=0, minute=0, second=0, microsecond=0)
    ndays = int((dayN - day0).days) + 1
    kept, counts, blocks = [], [], []
    for i in range(ndays):
        s = day0 + dt.timedelta(days=i)
        s_ms = int(s.timestamp() * 1000)
        e_ms = s_ms + 86_400_000
        mask = (arr[:, 0] >= s_ms) & (arr[:, 0] < e_ms)
        sub = arr[mask]
        if sub.shape[0] < MIN_REAL_ROWS:
            continue
        out = np.empty((1440, 6), dtype=np.float64)
        out[:, 0] = s_ms + np.arange(1440) * 60_000
        idx = ((sub[:, 0] - s_ms) // 60_000).astype(np.int64)
        idx = np.clip(idx, 0, 1439)
        close = np.full(1440, np.nan)
        vol = np.zeros(1440)
        o = np.full(1440, np.nan)
        h = np.full(1440, np.nan)
        l = np.full(1440, np.nan)
        close[idx] = sub[:, 4]
        vol[idx] = sub[:, 5]
        o[idx] = sub[:, 1]
        h[idx] = sub[:, 2]
        l[idx] = sub[:, 3]
        close = ffill(close)
        o = ffill(o)
        h = ffill(h)
        l = ffill(l)
        nan_o = np.isnan(o)
        o[nan_o], h[nan_o], l[nan_o] = close[nan_o], close[nan_o], close[nan_o]
        out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5] = o, h, l, close, vol
        blocks.append(out)
        kept.append(s.date().isoformat())
        counts.append(int(sub.shape[0]))
    if not blocks:
        raise RuntimeError("no complete days found")
    return np.concatenate(blocks), kept, counts


def load_fng():
    path = os.path.join(C.DATA_DIR, "daily", "fear_greed.json")
    out = {}
    try:
        with open(path) as f:
            d = json.load(f)
        for it in d.get("data", []):
            day = dt.datetime.fromtimestamp(
                int(it["timestamp"]), dt.timezone.utc).date().isoformat()
            out[day] = (int(it["value"]), it.get("value_classification", ""))
    except Exception as e:
        print(f"  ! fear&greed unavailable: {e}")
    return out


COINMETRIC_MAP = {"BTC": "btc", "ETH": "eth", "LTC": "ltc"}


def load_onchain():
    """day -> coin -> (adr_act_cnt, tx_cnt, mcap)"""
    out = {}
    for coin, fn in COINMETRIC_MAP.items():
        path = os.path.join(C.DATA_DIR, "daily", f"coinmetrics_{fn}.csv")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    day = (row.get("time") or "")[:10]
                    if not day:
                        continue

                    def _f(k):
                        try:
                            return float(row[k]) if row.get(k) else None
                        except ValueError:
                            return None
                    rec = out.setdefault(day, {})
                    rec[coin] = (_f("AdrActCnt"), _f("TxCnt"), _f("CapMrktCurUSD"))
        except Exception as e:
            print(f"  ! coinmetrics {coin} unavailable: {e}")
    return out


RE_COIN = {
    "BTC": re.compile(r"\b(bitcoin|btc|satoshi)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|ether|eth)\b", re.I),
    "LTC": re.compile(r"\b(litecoin|ltc)\b", re.I),
}


def load_news():
    """day -> coin -> [headline, ...] (<=16 each)"""
    gdelt_path = os.path.join(C.DATA_DIR, "news", "gdelt_all.jsonl")
    hn_path = os.path.join(C.DATA_DIR, "news", "hn_stories.jsonl")
    out = {}
    seen = {}
    try:
        with open(gdelt_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                day = rec.get("day")
                if not day:
                    continue
                drec = out.setdefault(day, {c: [] for c in C.COINS})
                srec = seen.setdefault(day, {c: set() for c in C.COINS})
                for a in rec.get("articles", []):
                    t = (a.get("t") or "").strip()
                    if not t:
                        continue
                    tl = t.lower()
                    for coin in C.COINS:
                        if RE_COIN[coin].search(tl):
                            if tl not in srec[coin] and len(drec[coin]) < 16:
                                drec[coin].append(t[:160])
                                srec[coin].add(tl)
    except FileNotFoundError:
        print("  ! no GDELT news file")
    try:
        with open(hn_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                day = rec.get("day")
                if not day:
                    continue
                drec = out.setdefault(day, {c: [] for c in C.COINS})
                srec = seen.setdefault(day, {c: set() for c in C.COINS})
                for coin_key, stories in rec.items():
                    coin = str(coin_key).upper()
                    if coin not in C.COINS:
                        continue
                    for s in sorted(stories, key=lambda x: -(x.get("p") or 0)):
                        t = (s.get("t") or "").strip()
                        if not t:
                            continue
                        tl = t.lower()
                        if tl in srec[coin]:
                            continue
                        pts = s.get("p") or 0
                        if pts >= 50:
                            t = f"{t} [trending on HN, {pts} pts]"
                        if len(drec[coin]) < 16:
                            drec[coin].append(t[:170])
                            srec[coin].add(tl)
    except FileNotFoundError:
        print("  ! no HN news file")
    return out


def main():
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    norm = {}
    kept_days = None
    for coin in C.COINS:
        sym = C.SYMBOLS[coin]
        print(f"[{sym}] parsing zips ...", flush=True)
        arr = parse_symbol_zips(sym)
        print(f"[{sym}] {len(arr):,} raw rows "
              f"({arr[0,0]:.0f} -> {arr[-1,0]:.0f} ms)", flush=True)
        narr, days, counts = normalize_to_days(arr)
        print(f"[{sym}] normalized to {len(days)} days "
              f"(avg {np.mean(counts):.0f}/1440 real minutes)", flush=True)
        np.save(os.path.join(C.CACHE_DIR, f"{sym}_1m.npy"), narr)
        norm[coin] = narr
        kept_days = days if kept_days is None else kept_days
        del arr

    # align: keep only days where every coin has a normalized block
    n = min(v.shape[0] // 1440 for v in norm.values())
    kept_days = kept_days[:n]
    for coin in C.COINS:
        sym = C.SYMBOLS[coin]
        a = norm[coin][:n * 1440]
        np.save(os.path.join(C.CACHE_DIR, f"{sym}_1m.npy"), a)
        norm[coin] = a
    with open(os.path.join(C.CACHE_DIR, "days.json"), "w") as f:
        json.dump({"days": kept_days,
                   "generated": dt.datetime.now(dt.timezone.utc).isoformat()},
                  f)
    print(f"[data] aligned {len(kept_days)} days: "
          f"{kept_days[0]} .. {kept_days[-1]}", flush=True)

    # ---------------------------------------------------------- daily metrics
    fng = load_fng()
    onchain = load_onchain()
    daily = {}
    prev_close = {}
    prev_vals = {c: [] for c in C.COINS}
    for i, day in enumerate(kept_days):
        rec = {}
        f = fng.get(day)
        if f:
            rec["fng"], rec["fng_class"] = f
        for coin in C.COINS:
            A = norm[coin][i * 1440:(i + 1) * 1440]
            o, h, l, c = A[0, 1], A[:, 2].max(), A[:, 3].min(), A[-1, 4]
            v = float(A[:, 5].sum())
            crec = {"close": float(c), "open": float(o),
                    "high": float(h), "low": float(l), "volume_usd": v}
            if coin in prev_close and prev_close[coin] > 0:
                crec["ret1d"] = float(c / prev_close[coin] - 1.0)
                crec["range"] = float((h - l) / prev_close[coin])
                if len(prev_vals[coin]) > 0:
                    avg = float(np.mean(prev_vals[coin][-30:]))
                    crec["volr"] = v / avg if avg > 0 else 1.0
            oc = (onchain.get(day) or {}).get(coin)
            if oc:
                padr, ptx, _mcap = oc
                prevday = (onchain.get(prev_day_key(day)) or {}).get(coin) \
                    if day not in ("",) else None
                # d/d change computed later against previous kept day
                crec["_adr"] = padr
                crec["_tx"] = ptx
            prev_close[coin] = c
            prev_vals[coin].append(v)
            rec[coin] = crec
        daily[day] = rec
    # on-chain d/d change (uses previous calendar day's value)
    for day in kept_days:
        pday = prev_day_key(day)
        for coin in C.COINS:
            crec = daily[day].get(coin)
            if not crec:
                continue
            cur_a, cur_t = crec.pop("_adr", None), crec.pop("_tx", None)
            prev = (onchain.get(pday) or {}).get(coin)
            if prev:
                pa, pt, _ = prev
                if cur_a and pa:
                    crec["adr_chg"] = cur_a / pa - 1.0
                if cur_t and pt:
                    crec["tx_chg"] = cur_t / pt - 1.0
    with open(os.path.join(C.CACHE_DIR, "daily.json"), "w") as f:
        json.dump(daily, f)

    # ---------------------------------------------------------- news
    news = load_news()
    with open(os.path.join(C.CACHE_DIR, "news.json"), "w") as f:
        json.dump(news, f)
    cov = sum(1 for d in kept_days if news.get(d))
    n_heads = sum(len(v) for d in kept_days for v in news.get(d, {}).values())
    print(f"[data] daily metrics for {len(daily)} days; "
          f"news coverage {cov}/{len(kept_days)} days, "
          f"{n_heads:,} coin-headlines total", flush=True)
    print("[data] cache ready:", C.CACHE_DIR, flush=True)


def prev_day_key(day):
    try:
        d = dt.date.fromisoformat(day)
        return (d - dt.timedelta(days=1)).isoformat()
    except ValueError:
        return None


if __name__ == "__main__":
    main()
