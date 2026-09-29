"""Market data access for training (everything is local / offline).

Expects the cache produced by scripts/prepare_data.py:
  cache/BTCUSDT_1m.npy   (n_days*1440, 6) float64  [ts_ms, open, high, low, close, volume]
  cache/ETHUSDT_1m.npy
  cache/LTCUSDT_1m.npy
  cache/days.json        {"days": ["2025-01-01", ...]}
  cache/daily.json       per-day metrics + fear&greed + on-chain deltas
  cache/news.json        per-day per-coin headline lists
"""
import json
import os

import numpy as np

from . import config as C


class MarketData:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or C.CACHE_DIR
        self.arrays = {}
        for coin in C.COINS:
            sym = C.SYMBOLS[coin]
            path = os.path.join(self.cache_dir, sym + "_1m.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} missing - run:  python3 scripts/prepare_data.py")
            self.arrays[coin] = np.load(path)

        with open(os.path.join(self.cache_dir, "days.json")) as f:
            self.days = json.load(f)["days"]
        self.daily = self._load_json("daily.json")
        self.news = self._load_json("news.json")

        # keep only days present for every coin (they are aligned by construction)
        n = min(int(a.shape[0] // C.MINUTES_PER_DAY) for a in self.arrays.values())
        self.days = self.days[-n:] if len(self.days) > n else self.days
        self.n_days = len(self.days)
        if self.n_days == 0:
            raise RuntimeError("no usable days in cache")

    # ------------------------------------------------------------------ api
    def _load_json(self, name):
        p = os.path.join(self.cache_dir, name)
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    def split(self, eval_days=C.EVAL_DAYS):
        """(train_days, eval_days) - most recent days held out for eval."""
        n_eval = min(eval_days, max(1, self.n_days // 10))
        return self.days[:-n_eval], self.days[-n_eval:]

    def day_index(self, day):
        return self.days.index(day)

    def get_episode(self, day):
        """dict coin -> (1440, 6) float64 array for the given day."""
        i = self.day_index(day)
        s = i * C.MINUTES_PER_DAY
        e = s + C.MINUTES_PER_DAY
        out = {}
        for coin in C.COINS:
            out[coin] = np.array(self.arrays[coin][s:e], dtype=np.float64)
        return out

    def day_context(self, day):
        """Raw context used to build the LLM prompt for this day."""
        prev = None
        idx = self.days.index(day)
        if idx > 0:
            prev = self.days[idx - 1]
        return {
            "day": day,
            "prev_day": prev,
            "daily": self.daily.get(day, {}),
            "prev_daily": self.daily.get(prev, {}) if prev else {},
            "news": self.news.get(day, {}),
        }


def coin_of_action(a: int):
    """0 HOLD, else BUY/SELL_<COIN>."""
    if a <= 0:
        return None, None
    kind = "BUY" if a % 2 == 1 else "SELL"
    coin = C.COINS[(a - 1) // 2]
    return kind, coin
