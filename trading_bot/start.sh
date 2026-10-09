#!/usr/bin/env bash
# Container entrypoint: train any missing models, then run the alert loop forever.
set -euo pipefail
SYMBOLS="${SYMBOLS:-BTC/USDT,ETH/USDT,SOL/USDT}"
TIMEFRAME="${TIMEFRAME:-1h}"
TRAIN_ARGS="${TRAIN_ARGS:---ensemble 5 --inherit elite --kill-drawdown 0.35 --kill-on-loss false --target-sharpe 99 --max-generations 20}"

IFS=',' read -ra LIST <<< "$SYMBOLS"
for sym in "${LIST[@]}"; do
  dir="runs/$(echo "$sym" | tr '/' '_')_${TIMEFRAME}"
  if [ ! -f "$dir/champion.npz" ]; then
    echo "No model for $sym $TIMEFRAME, training..."
    # shellcheck disable=SC2086
    python main.py --symbol "$sym" --timeframe "$TIMEFRAME" $TRAIN_ARGS
  fi
done

exec python signals.py --symbols "$SYMBOLS" --timeframe "$TIMEFRAME" ${SIGNAL_ARGS:-}
