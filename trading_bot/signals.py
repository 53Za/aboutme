"""Send LONG / SHORT / FLAT entry signals from the trained champion to Telegram.

Setup:
    1. In Telegram, message @BotFather -> /newbot -> copy the token.
    2. Send any message to your new bot, then open
       https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}.
    3. export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...

Run:
    python signals.py --csv prices.csv                       # one check from a CSV
    python signals.py --exchange binance --symbol BTC/USDT --timeframe 1h --every 3600
    python signals.py --csv prices.csv --dry-run             # print instead of sending

A message is sent only when the signal changes, so you get one alert per entry or exit.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import data
from agent import Agent
from env import features

NAMES = {1: "LONG 📈", -1: "SHORT 📉", 0: "FLAT ⏸ (exit / stay out)"}
MIN_BARS = 120  # features need ~90 bars of history


def send_telegram(text: str, dry_run: bool):
    if dry_run:
        print(f"[dry-run] {text}")
        return
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or use --dry-run).")
    body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", body, timeout=15) as r:
        if not json.load(r).get("ok"):
            raise RuntimeError("Telegram rejected the message")


def fetch_prices(a) -> np.ndarray:
    if a.csv:
        return data.load_csv(a.csv)
    import ccxt  # optional: pip install ccxt (public candles need no API key)
    ex = getattr(ccxt, a.exchange)()
    candles = ex.fetch_ohlcv(a.symbol, timeframe=a.timeframe, limit=max(a.bars, MIN_BARS))
    return np.array([c[4] for c in candles], dtype=float)  # close prices


def current_signal(agent: Agent, prices: np.ndarray, lookback: int, prev_pos: int) -> int:
    if len(prices) < MIN_BARS:
        raise SystemExit(f"Need at least {MIN_BARS} bars, got {len(prices)}.")
    X = features(prices, lookback)
    return agent.act(X[-1], prev_pos, greedy=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/champion.npz")
    p.add_argument("--csv")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--bars", type=int, default=300)
    p.add_argument("--every", type=int, default=0, help="seconds between checks; 0 = run once")
    p.add_argument("--state", default="runs/signal_state.json")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    saved = np.load(a.model)
    agent = Agent(0, np.random.default_rng(0), weights=saved["W"])
    lookback = int(saved["lookback"]) if "lookback" in saved else 5
    label = Path(a.csv).name if a.csv else f"{a.symbol} {a.timeframe}"
    state_path = Path(a.state)

    while True:
        state = json.loads(state_path.read_text()) if state_path.exists() else {"position": 0}
        prices = fetch_prices(a)
        pos = current_signal(agent, prices, lookback, state["position"])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if pos != state["position"] or "sent" not in state:
            send_telegram(f"{label}: {NAMES[pos]} @ {prices[-1]:.4f}\n"
                          f"was {NAMES[state['position']]}\n{now}\n"
                          "Model signal, not financial advice.", a.dry_run)
            state = {"position": pos, "sent": now}
            state_path.parent.mkdir(exist_ok=True)
            state_path.write_text(json.dumps(state))
        else:
            print(f"{now} no change ({NAMES[pos]})")
        if a.every <= 0:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
