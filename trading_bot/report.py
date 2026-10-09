"""Summarize live signal performance from runs/trades.csv.

    python report.py                 # print the summary
    python report.py --telegram      # also send it to Telegram
"""
import argparse
from pathlib import Path

import pandas as pd

import envfile
from signals import send_telegram


def summary(path: Path, fee_pct: float) -> str:
    if not path.exists():
        return "No closed trades yet."
    t = pd.read_csv(path)
    if t.empty:
        return "No closed trades yet."
    t["net"] = t["pnl_pct"] - 2 * fee_pct
    lines = [f"📊 Live signal report — {len(t)} closed trades "
             f"({t['exit_time'].min()[:10]} → {t['exit_time'].max()[:10]})",
             f"Win rate: {(t['net'] > 0).mean():.0%} · Sum of trade %: {t['net'].sum():+.2f}% "
             f"(after {fee_pct}% fee per side)",
             f"Avg win {t.loc[t.net > 0, 'net'].mean():+.2f}% · avg loss {t.loc[t.net <= 0, 'net'].mean():+.2f}%"]
    for sym, g in t.groupby("symbol"):
        lines.append(f"{sym}: {len(g)} trades, {(g.net > 0).mean():.0%} wins, {g.net.sum():+.2f}%")
    for reason, g in t.groupby("reason"):
        lines.append(f"  closed by {reason.split(' (')[0]}: {len(g)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="runs/trades.csv")
    p.add_argument("--fee", type=float, default=0.05, help="fee+slippage %% per side")
    p.add_argument("--telegram", action="store_true")
    a = p.parse_args()
    envfile.load()
    text = summary(Path(a.trades), a.fee)
    print(text)
    if a.telegram:
        send_telegram(text, dry_run=False)


if __name__ == "__main__":
    main()
