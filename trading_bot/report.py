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
    size = t["size"].fillna(1.0) if "size" in t else 1.0
    t["net"] = (t["pnl_pct"] - 2 * fee_pct) * size  # % of your normal trade amount
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


def news_accuracy(path: Path, horizons=(4, 24)) -> str:
    """Were the AI's bullish/bearish calls on medium/high-impact news right N hours later?"""
    if not path.exists():
        return "No news analysed yet."
    log = pd.read_csv(path)
    coins = [c[:-4] for c in log.columns if c.endswith("_dir")]
    log = log[log["impact"].isin(["medium", "high"])]
    if log.empty:
        return "No medium/high-impact news yet."
    import data
    lines = [f"📰 News AI scorecard — {len(log)} medium/high headlines"]
    for c in coins:
        calls = log[log[f"{c}_dir"].isin(["bullish", "bearish"]) & log[f"{c}_price"].notna()]
        if calls.empty:
            continue
        candles = data.fetch_hyperliquid(f"{c}/USDT", "1h", 5000)
        ts = pd.to_datetime(candles["ts"], unit="ms", utc=True) + pd.Timedelta(hours=1)  # close time
        close = pd.Series(candles["close"].to_numpy(), index=ts)
        for h in horizons:
            hits = n = 0
            for _, r in calls.iterrows():
                t = pd.Timestamp(r["logged_at"].replace(" UTC", ""), tz="UTC") + pd.Timedelta(hours=h)
                later = close[close.index >= t]
                if later.empty:
                    continue  # not enough time has passed
                move = later.iloc[0] / float(r[f"{c}_price"]) - 1
                n += 1
                hits += (move > 0) == (r[f"{c}_dir"] == "bullish")
            if n:
                lines.append(f"{c} after {h}h: {hits}/{n} calls right ({hits / n:.0%})")
    if len(lines) == 1:
        lines.append("Not enough time has passed to score any calls yet.")
    lines.append("50% is a coin flip; trust it only after ~50+ scored calls.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="runs/trades.csv")
    p.add_argument("--fee", type=float, default=0.05, help="fee+slippage %% per side")
    p.add_argument("--telegram", action="store_true")
    a = p.parse_args()
    envfile.load()
    text = summary(Path(a.trades), a.fee) + "\n\n" + news_accuracy(Path(a.trades).with_name("news_log.csv"))
    print(text)
    if a.telegram:
        send_telegram(text, dry_run=False)


if __name__ == "__main__":
    main()
