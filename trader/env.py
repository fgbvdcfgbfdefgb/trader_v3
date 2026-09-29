"""One-day crypto trading environment (BTC / ETH / LTC).

- Episode = one UTC day of 1-minute bars (1440 steps).
- Start with a fake $2000 cash balance.
- 7 discrete actions: HOLD, BUY_x / SELL_x for each coin.
  * BUY  invests BUY_FRAC (25%) of current equity in that coin (0.1% fee).
  * SELL liquidates the whole position of that coin (0.1% fee).
- Reward = fractional change of total equity per minute.
- The LLM (Qwen3-8B) day-analysis features are appended to every
  observation, so the policy is conditioned on that day's news/market view.
"""
import numpy as np

from . import config as C


def _roll_mean_std(x, w):
    """Rolling mean/std over the last `w` elements (prefix windows allowed)."""
    n = x.shape[0]
    c = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(1, n + 1)
    lo = np.maximum(0, idx - w)
    cnt = (idx - lo).astype(np.float64)
    mean = (c[idx] - c[lo]) / cnt
    var = (c2[idx] - c2[lo]) / cnt - mean * mean
    var = np.maximum(var, 0.0)
    return mean, np.sqrt(var)


def _lagged_logret(logc, k):
    r = np.zeros_like(logc)
    if logc.shape[0] > k:
        r[k:] = logc[k:] - logc[:-k]
        r[:k] = logc[:k] - logc[0]
    return r


def build_coin_features(close, volume):
    """(1440, 8) per-coin technical feature matrix."""
    n = close.shape[0]
    logc = np.log(np.maximum(close, 1e-12))
    feats = []
    for k in (1, 5, 15, 60):
        feats.append(_lagged_logret(logc, k))
    ma, sd = _roll_mean_std(close, 60)
    feats.append(np.clip((close - ma) / np.maximum(sd, 1e-9), -5, 5))     # price z-score
    mav, sdv = _roll_mean_std(volume, 60)
    feats.append(np.clip((volume - mav) / np.maximum(sdv, 1e-9), -5, 5))  # volume z-score
    # RSI(14)
    d = np.diff(close, prepend=close[0])
    up = np.maximum(d, 0.0)
    dn = np.maximum(-d, 0.0)
    mu, _ = _roll_mean_std(up, 14)
    md, _ = _roll_mean_std(dn, 14)
    rsi = 100.0 - 100.0 / (1.0 + mu / np.maximum(md, 1e-12))
    feats.append((rsi - 50.0) / 50.0)
    # 60-min realised volatility of 1-min log returns (scaled)
    lr1 = _lagged_logret(logc, 1)
    _, rstd = _roll_mean_std(lr1, 60)
    feats.append(np.clip(rstd * 300.0, 0, 5))
    M = np.stack(feats, axis=1)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return M[:, :n] if M.shape[1] != n else M


class DayEnv:
    """Single-day trading environment over the three coins."""

    def __init__(self, episode_arrays, llm_features, coin_mode="all"):
        """episode_arrays: {coin: (1440,6)}; llm_features: np.ndarray(9)."""
        self.coin_mode = coin_mode
        self.arr = {c: episode_arrays[c] for c in C.COINS}
        self.n = C.MINUTES_PER_DAY
        self.close = {c: self.arr[c][:, 4] for c in C.COINS}
        self.feat = {c: build_coin_features(self.close[c], self.arr[c][:, 5])
                     for c in C.COINS}
        self.llm_features = np.asarray(llm_features, dtype=np.float64).reshape(-1)
        self.obs_dim = 3 * 8 + 4 + 2 + self.llm_features.shape[0]
        if coin_mode == "random":
            self.obs_dim += 3
        self.n_actions = C.N_ACTIONS
        self.rng = np.random.default_rng()

    # ---------------------------------------------------------------- state
    def reset(self, active_coin=None):
        self.t = 0
        self.cash = C.START_BALANCE
        self.qty = {c: 0.0 for c in C.COINS}
        self.equity = C.START_BALANCE
        self.fees_paid = 0.0
        self.trades = 0
        self.active_coin = active_coin
        if self.coin_mode == "random":
            if self.active_coin is None:
                self.active_coin = C.COINS[self.rng.integers(len(C.COINS))]
        self.equity_curve = [C.START_BALANCE]
        self.exposure = np.zeros((4, self.n))   # cash, BTC, ETH, LTC fractions
        self.actions_taken = np.zeros(self.n, dtype=np.int64)
        self.executed = np.zeros(self.n, dtype=bool)
        self._log_exposure()
        return self.observe()

    def _prices(self, t):
        return {c: float(self.close[c][t]) for c in C.COINS}

    def _equity(self, t):
        px = self._prices(t)
        return self.cash + sum(self.qty[c] * px[c] for c in C.COINS)

    def observe(self):
        t = self.t
        obs = []
        for c in C.COINS:
            obs.append(self.feat[c][t])
        px = self._prices(t)
        eq = max(self.cash + sum(self.qty[c] * px[c] for c in C.COINS), 1e-9)
        fracs = [self.cash / eq] + [self.qty[c] * px[c] / eq for c in C.COINS]
        obs.append(np.array(fracs, dtype=np.float64))
        tod = 2.0 * np.pi * t / self.n
        obs.append(np.array([np.sin(tod), np.cos(tod)]))
        obs.append(self.llm_features)
        if self.coin_mode == "random":
            oh = np.zeros(len(C.COINS))
            if self.active_coin in C.COINS:
                oh[C.COINS.index(self.active_coin)] = 1.0
            obs.append(oh)
        v = np.concatenate(obs)
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # ---------------------------------------------------------------- trade
    def _apply(self, action, px):
        if action == 0:
            return False
        # decode: 1 BUY_BTC 2 SELL_BTC 3 BUY_ETH 4 SELL_ETH 5 BUY_LTC 6 SELL_LTC
        kind = "BUY" if action % 2 == 1 else "SELL"
        coin = C.COINS[(action - 1) // 2]
        if self.coin_mode == "random" and coin != self.active_coin:
            return False
        price = px[coin]
        if kind == "BUY":
            spend = min(self.cash, C.BUY_FRAC * self.equity)
            if spend < 1.0 or price <= 0:
                return False
            self.qty[coin] += spend * (1.0 - C.FEE) / price
            fee = spend * C.FEE
            self.cash -= spend
            self.fees_paid += fee
            self.trades += 1
            return True
        else:
            value = self.qty[coin] * price
            if value < 1.0:
                return False
            self.cash += value * (1.0 - C.FEE)
            self.fees_paid += value * C.FEE
            self.qty[coin] = 0.0
            self.trades += 1
            return True

    def _log_exposure(self):
        t = min(self.t, self.n - 1)
        px = self._prices(t)
        eq = max(self.cash + sum(self.qty[c] * px[c] for c in C.COINS), 1e-9)
        self.exposure[0, t] = self.cash / eq
        for i, c in enumerate(C.COINS):
            self.exposure[1 + i, t] = self.qty[c] * px[c] / eq

    def step(self, action):
        t = self.t
        px = self._prices(t)
        self.actions_taken[t] = action
        self.executed[t] = self._apply(action, px)
        self.equity = self._equity(t)

        t2 = min(t + 1, self.n - 1)
        self.t = t2
        new_equity = self._equity(t2)
        reward = (new_equity - self.equity) / max(self.equity, 1e-9)
        self.equity = new_equity
        self.equity_curve.append(new_equity)
        self._log_exposure()

        done = t2 >= self.n - 1
        info = {
            "equity": new_equity,
            "cash": self.cash,
            "qty": dict(self.qty),
            "fees": self.fees_paid,
            "trades": self.trades,
            "active_coin": self.active_coin,
        }
        return self.observe(), reward, done, info

    # ---------------------------------------------------------------- extras
    def buyhold_equity(self):
        """Equal-weight buy&hold of the tradable coins (benchmark)."""
        coins = [self.active_coin] if (self.coin_mode == "random" and self.active_coin) else C.COINS
        eq = np.zeros(self.n)
        for c in coins:
            p = self.close[c]
            eq += (C.START_BALANCE / len(coins)) * (p / p[0])
        return eq
