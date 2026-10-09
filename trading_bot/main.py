"""Run the evolutionary trading bot.

    python main.py                         # synthetic data
    python main.py --csv prices.csv        # your data (needs a `close` column)
    python main.py --inherit elite         # keep the best survivor instead of wiping
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
    p.add_argument("--csv")
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--out", default="runs")
    for name, field in Config.__dataclass_fields__.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(field.default), default=field.default)
    a = p.parse_args()
    cfg = Config(**{k: getattr(a, k) for k in Config.__dataclass_fields__})

    prices = data.load_csv(a.csv) if a.csv else data.synthetic(seed=cfg.seed)
    train, val, test = data.split(prices)
    Xtr, rtr = prep(train, a.lookback)
    Xva, rva = prep(val, a.lookback)
    Xte, rte = prep(test, a.lookback)

    champion, champ_m, history, cfg_dict = evolve(Xtr, rtr, Xva, rva, cfg)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(out / "generations.csv", index=False)
    deaths = sum(not h["survived"] for h in history)
    print(f"\n{len(history)} generations, {deaths} terminated, {len(history) - deaths} survived")

    if champion is None:
        print("No generation survived. Loosen kill thresholds, add data, or richer features.")
        return

    # Out-of-sample check on data no generation trained on or was selected against.
    test_m = evaluate(champion, Xte, rte, cfg.cost)
    bh_m = metrics(*run(lambda f, pos: 1, Xte, rte, 0.0))
    print(f"champion  validation: {json.dumps({k: round(v, 4) for k, v in champ_m.items()})}")
    print(f"champion  TEST      : {json.dumps({k: round(v, 4) for k, v in test_m.items()})}")
    print(f"buy&hold  TEST      : {json.dumps({k: round(v, 4) for k, v in bh_m.items()})}")
    if len(history) > 1:
        print(f"note: champion was picked from {len(history)} generations on validation data; "
              "trust the TEST row, not the validation row.")

    np.savez(out / "champion.npz", W=champion.W)
    (out / "result.json").write_text(json.dumps(
        {"config": cfg_dict, "validation": champ_m, "test": test_m, "buy_and_hold_test": bh_m}, indent=2))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
