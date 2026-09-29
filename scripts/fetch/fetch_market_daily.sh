#!/bin/bash
# Reproducibility script: re-downloads all market data bundled with this repo.
# (You do NOT need this - data/ is already populated.)
set -e
cd "$(dirname "$0")"/..

mkdir -p data/minute data/daily data/news

# ---- Binance Vision 1-minute klines (BTC/ETH/LTC, 2025-01 .. 2026-08 + Sep-2026 dailies)
BASE="https://data.binance.vision/data/spot"
for SYM in BTCUSDT ETHUSDT LTCUSDT; do
  mkdir -p "data/minute/$SYM"
  for Y in 2025 2026; do
    MM=12; [ "$Y" = "2026" ] && MM=8
    for M in $(seq -w 1 $MM); do
      F="data/minute/$SYM/$SYM-1m-$Y-$M.zip"
      [ -s "$F" ] || curl -sfL --retry 3 -o "$F" "$BASE/monthly/klines/$SYM/1m/$SYM-1m-$Y-$M.zip" || echo "missing $F"
    done
  done
  for D in $(seq -w 1 28); do
    F="data/minute/$SYM/$SYM-1m-2026-09-$D.zip"
    [ -s "$F" ] || curl -sfL --retry 3 -o "$F" "$BASE/daily/klines/$SYM/1m/$SYM-1m-2026-09-$D.zip" || echo "missing $F"
  done
done

# ---- Fear & Greed index (alternative.me)
curl -sfL "https://api.alternative.me/fng/?limit=1000&format=json" -o data/daily/fear_greed.json

# ---- CoinMetrics community on-chain metrics
for C in btc eth ltc; do
  curl -sfL "https://raw.githubusercontent.com/coinmetrics/data/master/csv/$C.csv" -o "data/daily/coinmetrics_$C.csv"
done

# ---- CoinGecko daily market charts (365d public limit)
for C in bitcoin ethereum litecoin; do
  curl -sfL "https://api.coingecko.com/api/v3/coins/$C/market_chart?vs_currency=usd&days=365&interval=daily" -o "data/daily/coingecko_$C.json" || echo "coingecko $C failed (optional)"
  sleep 3
done

# ---- News (GDELT is rate-limited to ~1 req / 5 s; HN via Algolia)
python3 scripts/fetch/fetch_gdelt_news.py data/news/gdelt_all.jsonl
python3 scripts/fetch/fetch_hn_news.py data/news/hn_stories.jsonl

echo "done"
