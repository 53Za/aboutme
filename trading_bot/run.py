"""Start the bot on any OS: read .env, train missing models, then run alerts.

    python run.py
"""
import os
import subprocess
import sys
from pathlib import Path

import envfile

HERE = Path(__file__).parent
DEFAULT_TRAIN = ("--ensemble 5 --inherit elite --kill-drawdown 0.35 --kill-on-loss false "
                 "--target-sharpe 99 --max-generations 20")


def main():
    envfile.load()
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
        print("Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the .env file, save it, and start again.")
        sys.exit(3)
    symbols = os.environ.get("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
    timeframe = os.environ.get("TIMEFRAME", "1h")
    train_args = os.environ.get("TRAIN_ARGS", DEFAULT_TRAIN).split()

    ready = []
    for sym in [s.strip() for s in symbols.split(",") if s.strip()]:
        model = HERE / "runs" / f"{sym.replace('/', '_')}_{timeframe}" / "champion.npz"
        if not model.exists():
            print(f"No model for {sym} {timeframe}, training (about a minute)...")
            subprocess.run([sys.executable, "main.py", "--symbol", sym, "--timeframe", timeframe,
                            *train_args], cwd=HERE)
        if model.exists():
            ready.append(sym)
        else:
            print(f"Skipping {sym}: training produced no model (see messages above).")
    if not ready:
        print("No coins have a model, so there is nothing to watch.")
        sys.exit(3)
    symbols = ",".join(ready)

    cmd = [sys.executable, "signals.py", "--symbols", symbols, "--timeframe", timeframe,
           *os.environ.get("SIGNAL_ARGS", "").split()]
    sys.exit(subprocess.call(cmd, cwd=HERE))


if __name__ == "__main__":
    main()
