"""Qwen3-8B 'market analyst' advisor (llama.cpp server on CPU, localhost).

At the start of each epoch (= one trading day) we ask the LLM to read that
day's news headlines + daily market metrics and output a compact JSON
assessment.  The parsed assessment becomes a 9-dim feature vector appended
to every observation of the episode, so the DQN policy is conditioned on
the LLM's view of the day.

All responses are cached on disk per day (runs/shared/llm_cache/), so the
expensive CPU inference happens at most once per calendar day, ever.
Everything degrades gracefully: if the server is down / times out / returns
garbage, neutral features are used and training continues.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from . import config as C

NEUTRAL_ADVICE = {
    "BTC": {"sentiment": 0, "confidence": 0.0},
    "ETH": {"sentiment": 0, "confidence": 0.0},
    "LTC": {"sentiment": 0, "confidence": 0.0},
    "risk": 0.5,
    "volatility": 0.5,
    "summary": "no data",
}
FEATURE_DIM = 9


# --------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, base_url, timeout=C.LLM_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base_url + "/health",
                                        timeout=min(10, self.timeout)) as r:
                return r.status == 200
        except Exception:
            return False

    def chat(self, messages, max_tokens=C.LLM_MAX_TOKENS,
             temperature=C.LLM_TEMPERATURE, attempts=2):
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        last_err = None
        with self._lock:  # serialize requests across threads in this process
            for attempt in range(attempts):
                try:
                    req = urllib.request.Request(
                        self.base_url + "/v1/chat/completions", data=body,
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        out = json.loads(r.read().decode("utf-8", "replace"))
                    return out["choices"][0]["message"]["content"]
                except urllib.error.HTTPError as e:
                    # 4xx (except 429) means bad request - do not retry forever
                    last_err = f"HTTP {e.code}"
                    if 400 <= e.code < 500 and e.code != 429:
                        break
                except Exception as e:
                    last_err = str(e)
                time.sleep(3)
        raise RuntimeError(f"LLM request failed: {last_err}")


# --------------------------------------------------------------------- prompt
def _fmt_coin_block(coin, daily):
    d = daily.get(coin, {}) if isinstance(daily, dict) else {}
    if not d:
        return f"{coin}: no daily metrics available."
    close = d.get("close")
    ret = d.get("ret1d")
    rng = d.get("range")
    volr = d.get("volr")
    adr = d.get("adr_chg")
    tx = d.get("tx_chg")
    parts = []
    if close is not None:
        parts.append(f"close ${close:,.0f}")
    if ret is not None:
        parts.append(f"{ret*100:+.1f}% d/d")
    if rng is not None:
        parts.append(f"range {rng*100:.1f}%")
    if volr is not None:
        parts.append(f"volume {volr:.1f}x 30d avg")
    if adr is not None:
        parts.append(f"active addresses {adr*100:+.1f}% d/d")
    if tx is not None:
        parts.append(f"tx count {tx*100:+.1f}% d/d")
    return f"{coin}: " + (", ".join(parts) if parts else "no metrics")


def build_messages(ctx):
    """ctx = MarketData.day_context(day)"""
    day = ctx["day"]
    daily = ctx["daily"]
    prev_daily = ctx["prev_daily"]
    news = ctx["news"]

    lines = [f"Date: {day} (UTC).",
             "You advise an automated crypto day-trading agent that starts the "
             "day with $2,000 cash and trades BTC, ETH and LTC."]
    fng = daily.get("fng")
    if fng is None:
        lines.append("Crypto Fear & Greed index: unavailable.")
    else:
        lines.append(f"Crypto Fear & Greed index (previous close): "
                     f"{fng} ({daily.get('fng_class', 'n/a')}).")
    lines.append("Previous-day market summary:")
    for coin in C.COINS:
        lines.append("  " + _fmt_coin_block(coin, prev_daily))
    for coin in C.COINS:
        heads = (news or {}).get(coin) or []
        if heads:
            lines.append(f"{coin} news headlines today ({len(heads)} shown):")
            for h in heads[:12]:
                h = str(h).strip().replace("\n", " ")[:160]
                lines.append(f"  - {h}")
        else:
            lines.append(f"{coin} news headlines today: none available.")
    lines.append(
        'Based on the data above, rate each coin for TODAY on a -2..2 '
        'sentiment scale (-2 = very bearish, +2 = very bullish) with your '
        'confidence, plus overall market risk and expected volatility (0..1). '
        'Use YOUR OWN estimates - never repeat the placeholder values. The '
        'summary must describe today\'s market in at most 15 words. '
        'Reply with ONLY this JSON object, no markdown, no extra text:\n'
        '{"BTC": {"sentiment": -1, "confidence": 0.6}, '
        '"ETH": {"sentiment": 0, "confidence": 0.4}, '
        '"LTC": {"sentiment": 1, "confidence": 0.3}, '
        '"risk": 0.7, "volatility": 0.8, '
        '"summary": "example text - replace with your own"}')
    system = ("You are a professional crypto market analyst. You read news "
              "and market data and output strict JSON only. /no_think")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n".join(lines)}]


# --------------------------------------------------------------------- parsing
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text):
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # unterminated think block
        text = text.split("</think>")[-1]
    return text.strip()


def extract_json_object(text):
    text = _strip_think(text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i + 1])
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found")


def _clamp(x, lo, hi, default):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return max(lo, min(hi, x))


def parse_advice(text):
    """Parse model output -> normalized advice dict (never raises)."""
    advice = json.loads(json.dumps(NEUTRAL_ADVICE))  # deep copy
    try:
        obj = extract_json_object(text)
    except Exception:
        return advice, False
    ok = False
    for key in ("BTC", "ETH", "LTC"):
        for k, v in obj.items():
            if str(k).upper() == key:
                if isinstance(v, dict):
                    advice[key]["sentiment"] = int(round(_clamp(
                        v.get("sentiment", 0), -2, 2, 0)))
                    advice[key]["confidence"] = _clamp(
                        v.get("confidence", 0), 0, 1, 0)
                else:  # bare number
                    advice[key]["sentiment"] = int(round(_clamp(v, -2, 2, 0)))
                    advice[key]["confidence"] = 0.5
                ok = True
                break
    advice["risk"] = _clamp(obj.get("risk", 0.5), 0, 1, 0.5)
    advice["volatility"] = _clamp(obj.get("volatility", 0.5), 0, 1, 0.5)
    s = obj.get("summary")
    advice["summary"] = (str(s)[:120] if s else advice["summary"])
    return advice, ok


def advice_to_features(advice, available=True):
    v = [advice[c]["sentiment"] / 2.0 for c in C.COINS] \
        + [advice[c]["confidence"] for c in C.COINS] \
        + [advice["risk"], advice["volatility"], 1.0 if available else 0.0]
    return np.asarray(v, dtype=np.float64)


def neutral_features():
    return advice_to_features(NEUTRAL_ADVICE, available=False)


# --------------------------------------------------------------------- cached
class DayAdvisor:
    """Disk-cached per-day LLM analysis shared by all parallel agents."""

    def __init__(self, cache_dir, base_url=None, timeout=C.LLM_TIMEOUT_S):
        self.cache_dir = cache_dir
        self.client = LLMClient(base_url, timeout) if base_url else None
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, day):
        return os.path.join(self.cache_dir, day + ".json")

    def get(self, day, ctx=None, force=False):
        """Returns (features np.ndarray(9), advice dict, meta dict)."""
        p = self._path(day)
        if not force and os.path.exists(p):
            try:
                with open(p) as f:
                    rec = json.load(f)
                if rec.get("source") in ("llm", "stub", "fallback"):
                    return (np.asarray(rec["features"], dtype=np.float64),
                            rec["advice"], rec)
            except Exception:
                pass
        if self.client is None or ctx is None:
            return neutral_features(), NEUTRAL_ADVICE, {"source": "stub"}
        raw, advice, source = "", NEUTRAL_ADVICE, "fallback"
        try:
            raw = self.client.chat(build_messages(ctx))
            advice, ok = parse_advice(raw)
            if ok:
                source = "llm"
        except Exception as e:
            raw = f"ERROR: {e}"
        rec = {
            "day": day,
            "source": source,
            "features": advice_to_features(
                advice, available=(source == "llm")).tolist(),
            "advice": advice,
            "raw": raw[:4000],
            "ts": time.time(),
        }
        try:
            tmp = p + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, p)
        except Exception:
            pass
        return (np.asarray(rec["features"], dtype=np.float64),
                advice, rec)

    def cached_days(self):
        try:
            return [f[:-5] for f in os.listdir(self.cache_dir)
                    if f.endswith(".json")]
        except Exception:
            return []
