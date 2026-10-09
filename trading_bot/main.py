"""Train the evolutionary trading bot on crypto candles.

    python main.py --symbol BTC/USDT --timeframe 1h          # OKX history (default)
    python main.py --exchange kraken --symbol ETH/USDT
    python main.py --inherit elite --kill-drawdown 0.35      # keep the best survivor
    python main.py --csv prices.csv | --synthetic            # offline data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from env import features, step_returns, run, metrics
from evolve import Config, evolve, evaluate


def prep(prices, lookback):
    X = features(prices, lookback)
    return X[:-1], step_returns(prices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="okx")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--bars", type=int, default=8000, help="candles of history to train on")
    p.add_argument("--csv", help="use a CSV with a `close` column instead of the exchange")
    p.add_argument("--synthetic", action="store_true", help="offline test data")
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--out", help="default: runs/<SYMBOL>_<TIMEFRAME>")
    for name, field in Config.__dataclass_fields__.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(field.default), default=field.default)
    a = p.parse_args()
    a.periods_per_year = data.periods_per_year(a.timeframe)
    cfg = Config(**{k: getattr(a, k) for k in Config.__dataclass_fields__})

    if a.csv:
        prices = data.load_csv(a.csv)
    elif a.synthetic:
        prices = data.synthetic(seed=cfg.seed)
    else:
        print(f"downloading {a.bars} closed {a.timeframe} candles of {a.symbol} from {a.exchange}...")
        prices = data.fetch_closed_candles(data.make_exchange(a.exchange), a.symbol, a.timeframe, a.bars)
    print(f"{len(prices)} candles")
    train, val, test = data.split(prices)
    Xtr, rtr = prep(train, a.lookback)
    Xva, rva = prep(val, a.lookback)
    Xte, rte = prep(test, a.lookback)

    champion, champ_m, history, cfg_dict = evolve(Xtr, rtr, Xva, rva, cfg)

    out = Path(a.out or data.model_dir(a.symbol, a.timeframe))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "generations.csv", index=False)
    deaths = sum(not h["survived"] for h in history)
    print(f"\n{len(history)} generations, {deaths} terminated, {len(history) - deaths} survived")

    if champion is None:
        print("No generation survived. Loosen kill thresholds, add data, or richer features.")
        return

    # Out-of-sample check on data no generation trained on or was selected against.
    test_m = evaluate(champion, Xte, rte, cfg.cost, cfg.periods_per_year)
    bh_m = metrics(*run(lambda f, pos: 1, Xte, rte, 0.0), cfg.periods_per_year)
    print(f"champion  validation: {json.dumps({k: round(v, 4) for k, v in champ_m.items()})}")
    print(f"champion  TEST      : {json.dumps({k: round(v, 4) for k, v in test_m.items()})}")
    print(f"buy&hold  TEST      : {json.dumps({k: round(v, 4) for k, v in bh_m.items()})}")
    if len(history) > 1:
        print(f"note: champion was picked from {len(history)} generations on validation data; "
              "trust the TEST row, not the validation row.")

    np.savez(out / "champion.npz", W=champion.W, lookback=a.lookback,
             exchange=a.exchange, symbol=a.symbol, timeframe=a.timeframe)
    (out / "result.json").write_text(json.dumps(
        {"config": cfg_dict, "validation": champ_m, "test": test_m, "buy_and_hold_test": bh_m}, indent=2))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
