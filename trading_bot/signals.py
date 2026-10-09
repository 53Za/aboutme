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

Optional: export AI4TRADE_TOKEN=... to also paper-trade every signal on ai4trade.ai
(see ai4trade.py).

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

import ai4trade
import data
import envfile
from agent import Agent
from env import features
from risk import RiskGuard, RiskRules

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


def open_msg(symbol, side, price, tf, exchange, when, rules=None):
    icon, verb = ("🟢", "BUY") if side == 1 else ("🔴", "SELL")
    risk = ""
    if rules:
        stop = price * (1 - side * rules.stop_loss)
        risk = (f"Stop-loss: {fmt_price(stop)} ({-rules.stop_loss:.0%})\n"
                f"Profit lock: after +{rules.protect_after:.0%}, closes if {rules.giveback:.0%} of the gain is given back\n")
    return (f"{icon} OPEN {SIDE[side]} — {symbol}\n"
            f"Action: {verb} to open a {SIDE[side].lower()} position\n"
            f"Entry price: {fmt_price(price)}\n{risk}"
            f"Timeframe: {tf} · {exchange}\n{when}\n"
            "Model signal, not financial advice.")


def close_msg(symbol, side, entry, price, tf, exchange, when, opened, reason=None):
    pnl = (price / entry - 1.0) * side
    verb = "SELL" if side == 1 else "BUY"
    icon = "✅" if pnl >= 0 else "❌"
    return (f"{icon} CLOSE {SIDE[side]} — {symbol}\n"
            f"Action: {verb} to close your {SIDE[side].lower()} position\n"
            f"Entry: {fmt_price(entry)} → Exit: {fmt_price(price)}\n"
            f"Result: {pnl:+.2%} (before fees)\n"
            f"Reason: {reason or 'model signal'}\n"
            f"Opened: {opened}\n"
            f"Timeframe: {tf} · {exchange}\n{when}")


def transition_messages(symbol, state, new_pos, price, tf, exchange, when, reason=None, rules=None):
    """Messages for moving from state['position'] to new_pos. A flip = close + open."""
    old, msgs = state["position"], []
    if new_pos == old:
        return msgs
    if old != 0:
        msgs.append(close_msg(symbol, old, state["entry_price"], price, tf, exchange, when,
                              state["entry_time"], reason))
    if new_pos != 0:
        msgs.append(open_msg(symbol, new_pos, price, tf, exchange, when, rules))
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
    return {
        "agent": Agent(0, np.random.default_rng(0), weights=saved["W"]),
        "lookback": int(saved["lookback"]) if "lookback" in saved else 5,
        "exchange": str(saved["exchange"]) if "exchange" in saved else "hyperliquid",
        "use_funding": bool(saved["use_funding"]) if "use_funding" in saved else False,
        "indicators": bool(saved["indicators"]) if "indicators" in saved else False,
        "risk": RiskRules(**json.loads(str(saved["risk"])))
                if "risk" in saved and json.loads(str(saved["risk"])) else None,
    }


def market_data(a, symbol, model, clients):
    """(prices, funding or None) from the same source and features the model was trained on."""
    if a.csv:
        return data.load_csv(a.csv), None
    bars = max(a.bars, MIN_BARS)
    if model["exchange"] == "hyperliquid":
        df = data.fetch_hyperliquid(symbol, a.timeframe, bars)
        return df["close"].to_numpy(), (df["funding"].to_numpy() if model["use_funding"] else None)
    ex = clients.setdefault(model["exchange"], data.make_exchange(model["exchange"]))
    return data.fetch_closed_candles(ex, symbol, a.timeframe, bars), None


def paper_trade(paper, symbol, st, new_pos, price, a):
    """Mirror the transition on the AI-Trader paper account. Failures are reported, not fatal."""
    coin = data.hl_coin(symbol)
    try:
        if st["position"] != 0 and st.get("paper_qty"):
            paper.close(st["position"], coin, st["paper_qty"], "trading_bot signal: close")
        st["paper_qty"] = None
        if new_pos != 0:
            qty = a.paper_usd / price
            paper.open(new_pos, coin, qty, f"trading_bot signal: open {SIDE[new_pos].lower()}")
            st["paper_qty"] = qty
    except (ai4trade.Ai4TradeError, OSError) as e:
        send_telegram(f"⚠️ AI-Trader paper trade failed for {symbol}: {e}", a.dry_run)


_model_cache: dict = {}


def current_models(symbols, a):
    """Load models, reloading any whose file changed (so a retrain takes effect without restart)."""
    out = {}
    for s in symbols:
        path = Path(a.model or f"{data.model_dir(s, a.timeframe)}/champion.npz")
        mtime = path.stat().st_mtime if path.exists() else None
        cached = _model_cache.get(s)
        if cached is None or cached[0] != mtime:
            _model_cache[s] = (mtime, load_model(s, a.timeframe, a.model))
            if cached is not None:
                print(f"{s}: reloaded retrained model")
        out[s] = _model_cache[s][1]
    return out


def check_once(a, models, clients, paper, state_path):
    states = json.loads(state_path.read_text()) if state_path.exists() else {}
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for symbol, model in models.items():
        st = states.get(symbol, {"position": 0})
        source = model["exchange"]
        prices, funding = market_data(a, symbol, model, clients)
        if len(prices) < MIN_BARS:
            print(f"{symbol}: only {len(prices)} candles, need {MIN_BARS}; skipping")
            continue
        price = float(prices[-1])
        candle = int(time.time() // data.timeframe_seconds(a.timeframe))
        if st.get("candle") == candle:
            print(f"{symbol}: already checked this candle")
            continue
        X = features(prices, model["lookback"], funding, model["indicators"])
        desired = model["agent"].act(X[-1], st["position"], greedy=True)
        reason = None
        if model["risk"]:  # same rules the model was tested with
            guard = RiskGuard(model["risk"], st.get("guard") or
                              {"pos": st["position"], "entry": st.get("entry_price")})
            new_pos, reason = guard.step(desired, price)
            st["guard"] = guard.state()
        else:
            new_pos = desired

        if "started" not in st and new_pos == 0:
            send_telegram(f"🤖 Watching {symbol} ({a.timeframe} · {source})\n"
                          f"No position right now — wait for an OPEN alert.\nPrice: {fmt_price(price)}",
                          a.dry_run)
        for msg in transition_messages(symbol, st, new_pos, price, a.timeframe, source, when, reason,
                                       model["risk"]):
            send_telegram(msg, a.dry_run)

        if new_pos != st["position"]:
            if paper and not a.dry_run:
                paper_trade(paper, symbol, st, new_pos, price, a)
            st.update(position=new_pos, entry_price=price if new_pos else None,
                      entry_time=when if new_pos else None)
        st["started"] = st.get("started", when)
        st["candle"] = candle
        states[symbol] = st
        side = SIDE.get(new_pos, "no position")
        print(f"{when} {symbol} {fmt_price(price)} -> {side}")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(states, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT", help="comma-separated, e.g. BTC/USDT,ETH/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--bars", type=int, default=300)
    p.add_argument("--model", help="override model path (single symbol only)")
    p.add_argument("--csv", help="read prices from a CSV instead of the exchange (testing)")
    p.add_argument("--state", default="runs/signal_state.json")
    p.add_argument("--paper-usd", type=float, default=10_000,
                   help="USD per paper trade on AI-Trader (used when AI4TRADE_TOKEN is set)")
    p.add_argument("--once", action="store_true", help="check once and exit")
    p.add_argument("--dry-run", action="store_true", help="print messages instead of sending")
    p.add_argument("--heartbeat-hours", type=float, default=24,
                   help="send a 'still running' status every N hours (0 = off)")
    a = p.parse_args()
    envfile.load()

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    current_models(symbols, a)  # fail fast if a model is missing
    paper = ai4trade.from_env()
    print("AI-Trader paper trading:", "ON" if paper else "off (set AI4TRADE_TOKEN to enable)")
    clients, state_path = {}, Path(a.state)
    tf_s = data.timeframe_seconds(a.timeframe)

    send_telegram(f"▶️ Bot started — watching {', '.join(symbols)} ({a.timeframe})\n"
                  f"Paper trading: {'ON' if paper else 'off'}", a.dry_run)
    last_beat, failures = time.time(), 0
    while True:
        try:
            check_once(a, current_models(symbols, a), clients, paper, state_path)
            if failures >= 3:
                send_telegram("✅ Bot recovered — checks are working again.", a.dry_run)
            failures = 0
        except Exception as e:  # network/exchange errors: keep running, retry soon
            if not isinstance(e, (OSError, RuntimeError, ValueError)) and \
                    not type(e).__module__.startswith("ccxt"):
                raise
            failures += 1
            print(f"check failed ({failures} in a row): {e}")
            if failures == 3:
                send_telegram(f"⚠️ Bot can't get market data (3 failures in a row): {e}\n"
                              "It keeps retrying every minute.", a.dry_run)
        if a.once:
            break
        if a.heartbeat_hours > 0 and time.time() - last_beat >= a.heartbeat_hours * 3600:
            send_telegram(status_text(state_path), a.dry_run)
            last_beat = time.time()
        if failures:
            time.sleep(60)
        else:
            time.sleep(tf_s - (time.time() % tf_s) + 10)  # just after the next candle closes


def status_text(state_path: Path) -> str:
    states = json.loads(state_path.read_text()) if state_path.exists() else {}
    lines = ["💓 Bot is running"]
    for sym, st in states.items():
        if st.get("position"):
            lines.append(f"{sym}: {SIDE[st['position']]} since {st['entry_time']} @ {fmt_price(st['entry_price'])}")
        else:
            lines.append(f"{sym}: no position")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
