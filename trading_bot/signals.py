"""Crypto trade alerts on Telegram: when to OPEN or CLOSE, and whether LONG or SHORT.

Setup:
    1. In Telegram, message @BotFather -> /newbot -> copy the token.
    2. Send any message to your new bot, then open
       https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}.
    3. export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
    4. Train one model per coin:  python main.py --symbol BTC/USDT --timeframe 1h

Run:
    python signals.py --symbols BTC/USDT,ETH/USDT --timeframe 1h      # runs forever
    python signals.py --symbols BTC/USDT --once --dry-run              # one check, print only

Signals are computed on closed candles only, right after each candle closes.
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

MIN_BARS = 120  # features need ~90 bars of history
SIDE = {1: "LONG", -1: "SHORT"}


def send_telegram(text: str, dry_run: bool):
    if dry_run:
        print(f"[dry-run]\n{text}\n")
        return
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or use --dry-run).")
    body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage",
                                        body, timeout=15) as r:
                if json.load(r).get("ok"):
                    return
        except OSError:
            pass
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Could not deliver Telegram message after 4 attempts")


def fmt_price(p: float) -> str:
    return f"{p:,.2f}" if p >= 1 else f"{p:.6f}"


def open_msg(symbol, side, price, tf, exchange, when):
    icon, verb = ("🟢", "BUY") if side == 1 else ("🔴", "SELL")
    return (f"{icon} OPEN {SIDE[side]} — {symbol}\n"
            f"Action: {verb} to open a {SIDE[side].lower()} position\n"
            f"Entry price: {fmt_price(price)}\n"
            f"Timeframe: {tf} · {exchange}\n{when}\n"
            "Model signal, not financial advice.")


def close_msg(symbol, side, entry, price, tf, exchange, when, opened):
    pnl = (price / entry - 1.0) * side
    verb = "SELL" if side == 1 else "BUY"
    icon = "✅" if pnl >= 0 else "❌"
    return (f"{icon} CLOSE {SIDE[side]} — {symbol}\n"
            f"Action: {verb} to close your {SIDE[side].lower()} position\n"
            f"Entry: {fmt_price(entry)} → Exit: {fmt_price(price)}\n"
            f"Result: {pnl:+.2%} (before fees)\n"
            f"Opened: {opened}\n"
            f"Timeframe: {tf} · {exchange}\n{when}")


def transition_messages(symbol, state, new_pos, price, tf, exchange, when):
    """Messages for moving from state['position'] to new_pos. A flip = close + open."""
    old, msgs = state["position"], []
    if new_pos == old:
        return msgs
    if old != 0:
        msgs.append(close_msg(symbol, old, state["entry_price"], price, tf, exchange, when,
                              state["entry_time"]))
    if new_pos != 0:
        msgs.append(open_msg(symbol, new_pos, price, tf, exchange, when))
    return msgs


def load_model(symbol, timeframe, model_path=None):
    path = Path(model_path or f"{data.model_dir(symbol, timeframe)}/champion.npz")
    if not path.exists():
        raise SystemExit(f"No model at {path}. Train it first:\n"
                         f"  python main.py --symbol {symbol} --timeframe {timeframe}")
    saved = np.load(path)
    trained_tf = str(saved["timeframe"]) if "timeframe" in saved else timeframe
    if trained_tf != timeframe:
        raise SystemExit(f"{path} was trained on {trained_tf} candles, not {timeframe}.")
    lookback = int(saved["lookback"]) if "lookback" in saved else 5
    return Agent(0, np.random.default_rng(0), weights=saved["W"]), lookback


def check_once(a, ex, models, state_path):
    states = json.loads(state_path.read_text()) if state_path.exists() else {}
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for symbol, (agent, lookback) in models.items():
        st = states.get(symbol, {"position": 0})
        prices = data.load_csv(a.csv) if a.csv else \
            data.fetch_closed_candles(ex, symbol, a.timeframe, max(a.bars, MIN_BARS))
        if len(prices) < MIN_BARS:
            print(f"{symbol}: only {len(prices)} candles, need {MIN_BARS}; skipping")
            continue
        price = float(prices[-1])
        new_pos = agent.act(features(prices, lookback)[-1], st["position"], greedy=True)

        if "started" not in st and new_pos == 0:
            send_telegram(f"🤖 Watching {symbol} ({a.timeframe} · {a.exchange})\n"
                          f"No position right now — wait for an OPEN alert.\nPrice: {fmt_price(price)}",
                          a.dry_run)
        for msg in transition_messages(symbol, st, new_pos, price, a.timeframe, a.exchange, when):
            send_telegram(msg, a.dry_run)

        if new_pos != st["position"]:
            st.update(position=new_pos, entry_price=price if new_pos else None,
                      entry_time=when if new_pos else None)
        st["started"] = st.get("started", when)
        states[symbol] = st
        side = SIDE.get(new_pos, "no position")
        print(f"{when} {symbol} {fmt_price(price)} -> {side}")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(states, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="okx")
    p.add_argument("--symbols", default="BTC/USDT", help="comma-separated, e.g. BTC/USDT,ETH/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--bars", type=int, default=300)
    p.add_argument("--model", help="override model path (single symbol only)")
    p.add_argument("--csv", help="read prices from a CSV instead of the exchange (testing)")
    p.add_argument("--state", default="runs/signal_state.json")
    p.add_argument("--once", action="store_true", help="check once and exit")
    p.add_argument("--dry-run", action="store_true", help="print messages instead of sending")
    a = p.parse_args()

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    models = {s: load_model(s, a.timeframe, a.model) for s in symbols}
    ex = None if a.csv else data.make_exchange(a.exchange)
    state_path = Path(a.state)
    tf_s = data.timeframe_seconds(a.timeframe)

    while True:
        try:
            check_once(a, ex, models, state_path)
        except (OSError, RuntimeError) as e:
            print(f"check failed, will retry next candle: {e}")
        except Exception as e:  # ccxt network/exchange errors
            if type(e).__module__.startswith("ccxt"):
                print(f"exchange error, will retry next candle: {e}")
            else:
                raise
        if a.once:
            break
        wait = tf_s - (time.time() % tf_s) + 10  # just after the next candle closes
        time.sleep(wait)


if __name__ == "__main__":
    main()
