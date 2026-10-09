"""Pump.fun scanner — TEST MODE: picks new meme coins with early traction, logs them,
and checks what happened 1h and 24h later. It never buys anything.

    python pumpscan.py                 # run forever (every PUMP_INTERVAL_MIN minutes)
    python pumpscan.py --once          # one scan + outcome check
    python pumpscan.py --report        # print the scorecard

Picks go to runs/pump_picks.csv. By default nothing is sent per pick (PUMP_ALERTS=off);
only a daily scorecard goes to Telegram. Turn alerts on only if the scorecard shows the
picks were profitable after fees over a meaningful sample (100+ picks scored at 24h).

Data: pump.fun's public frontend API (new coins) + DexScreener (live price, volume, buys/sells).
"""
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import envfile

PUMP_API = "https://frontend-api-v3.pump.fun/coins"
DEX_API = "https://api.dexscreener.com/tokens/v1/solana/"
GT_API = "https://api.geckoterminal.com/api/v2/networks/solana/dexes/pump-fun/pools"
FIELDS = ["picked_at", "mint", "symbol", "name", "age_min", "mcap_usd", "price_usd", "vol_h1",
          "buys_h1", "sells_h1", "dex", "url", "price_1h", "ret_1h", "price_24h", "ret_24h",
          "source", "buyers_h1"]

# Filters: "early traction, not obviously dead or fake". Tuned by reasoning, not by results.
RULES = {
    "min_age_min": 10, "max_age_min": 180,        # past the first-seconds sniper war, still early
    "min_mcap": 15_000, "max_mcap": 500_000,      # someone is buying; not already huge
    "min_of_ath": 0.70,                           # not already dumped from its peak
    "min_vol_h1": 10_000,                         # real trading in the last hour (USD)
    "min_buys_h1": 50,                            # many buys, not a handful of wallets
    "min_buy_sell_ratio": 1.0,                    # more buying than selling
    "max_picks_per_scan": 3,
    "max_creator_coins": 2,                       # skip serial launchers
    "min_buyers_h1": 30,                          # unique buyers (GeckoTerminal only): no wash trading
}
ROUND_TRIP_FEE = 0.03  # ~3% for DEX fees + slippage on thin meme pools, buy + sell


class Blocked(RuntimeError):
    """The source refused us (Cloudflare page, 403, or persistent 429)."""


def _get(url: str, tries: int = 4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (trading-bot pumpscan)",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) or (e.code == 429 and i == tries - 1):
                raise Blocked(f"HTTP {e.code} from {url.split('/')[2]}") from None
            if e.code != 429:
                raise
            time.sleep(10 * (i + 1))  # rate limited: back off
        except json.JSONDecodeError:
            raise Blocked(f"non-JSON reply (bot protection?) from {url.split('/')[2]}") from None
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(5)


def dex_info(mints: list[str]) -> dict[str, dict]:
    """Best (most liquid) DexScreener pair per mint: price, 1h volume, buys/sells."""
    out = {}
    for i in range(0, len(mints), 30):
        for p in _get(DEX_API + ",".join(mints[i:i + 30])) or []:
            mint = p.get("baseToken", {}).get("address")
            if not mint or not p.get("priceUsd"):
                continue
            liq = (p.get("liquidity") or {}).get("usd") or 0
            if mint in out and liq <= out[mint]["liq"]:
                continue
            h1 = (p.get("txns") or {}).get("h1") or {}
            out[mint] = {"price": float(p["priceUsd"]), "liq": liq, "dex": p.get("dexId", ""),
                         "vol_h1": float((p.get("volume") or {}).get("h1") or 0),
                         "buys_h1": int(h1.get("buys") or 0), "sells_h1": int(h1.get("sells") or 0),
                         "url": p.get("url", "")}
    return out


def pumpfun_candidates() -> list[dict]:
    """Recently traded coins from pump.fun's own API (has ATH, socials, creator)."""
    now_ms = time.time() * 1000
    out = []
    for c in _get(f"{PUMP_API}?offset=0&limit=50&sort=last_trade_timestamp&order=DESC"
                  "&includeNsfw=false") or []:
        if c.get("nsfw") or c.get("is_banned"):
            continue
        mcap = c.get("usd_market_cap") or 0
        out.append({"mint": c["mint"], "symbol": c.get("symbol", ""), "name": c.get("name", ""),
                    "age_min": (now_ms - c.get("created_timestamp", now_ms)) / 60_000, "mcap": mcap,
                    "ath": c.get("ath_market_cap") or mcap, "creator": c.get("creator", ""),
                    "socials": bool(c.get("twitter") or c.get("telegram") or c.get("website")),
                    "buyers_h1": None, "source": "pump.fun"})
    return out


def geckoterminal_candidates(pages: int = 3) -> list[dict]:
    """Backup source: busiest pump.fun bonding-curve pools on GeckoTerminal (has unique buyers;
    no ATH, socials or creator, so those filters are skipped)."""
    out, now = [], datetime.now(timezone.utc)
    for page in range(1, pages + 1):
        for p in (_get(f"{GT_API}?sort=h24_volume_usd_desc&page={page}") or {}).get("data", []):
            a = p["attributes"]
            created = datetime.fromisoformat(a["pool_created_at"].replace("Z", "+00:00"))
            h1 = a.get("transactions", {}).get("h1", {})
            sym = a.get("name", "").split(" / ")[0]
            out.append({"mint": p["relationships"]["base_token"]["data"]["id"].split("_", 1)[1],
                        "symbol": sym, "name": sym,
                        "age_min": (now - created).total_seconds() / 60,
                        "mcap": float(a.get("market_cap_usd") or a.get("fdv_usd") or 0),
                        "ath": None, "creator": "", "socials": None,
                        "buyers_h1": int(h1.get("buyers") or 0), "source": "geckoterminal"})
        time.sleep(2.5)  # GeckoTerminal allows ~30 requests/minute
    return out


class Scanner:
    def __init__(self, runs: Path):
        self.picks_path = runs / "pump_picks.csv"
        self.state_path = runs / "pump_state.json"
        runs.mkdir(parents=True, exist_ok=True)
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.creators: dict[str, int] = st.get("creators", {})
        self.last_summary = st.get("last_summary", 0)
        self.source = os.environ.get("PUMP_SOURCE", "auto")  # auto | pumpfun | geckoterminal
        self.blocked_note = ""

    def save_state(self):
        self.state_path.write_text(json.dumps({"creators": self.creators,
                                               "last_summary": self.last_summary}))

    def rows(self) -> list[dict]:
        if not self.picks_path.exists():
            return []
        with self.picks_path.open() as f:
            return list(csv.DictReader(f))

    def write_rows(self, rows: list[dict]):
        tmp = self.picks_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(self.picks_path)

    # ---- picking -----------------------------------------------------------------------
    def candidates(self) -> list[dict]:
        if self.source == "geckoterminal":
            return geckoterminal_candidates()
        try:
            return pumpfun_candidates()
        except Blocked as e:
            if self.source == "pumpfun":
                raise
            if not self.blocked_note:
                print(f"pump.fun refused ({e}); using GeckoTerminal instead")
            self.blocked_note = str(e)
            return geckoterminal_candidates()

    def scan(self) -> list[dict]:
        rows = self.rows()
        have = {r["mint"] for r in rows}
        names = {r["name"].strip().lower() for r in rows} | {r["symbol"].strip().lower() for r in rows}
        r = RULES

        pre = []
        for c in self.candidates():
            if (c["mint"] in have
                    or not (r["min_age_min"] <= c["age_min"] <= r["max_age_min"])
                    or not (r["min_mcap"] <= c["mcap"] <= r["max_mcap"])
                    or (c["ath"] and c["mcap"] < r["min_of_ath"] * c["ath"])
                    or c["socials"] is False                              # None = unknown (backup source)
                    or (c["buyers_h1"] is not None and c["buyers_h1"] < r["min_buyers_h1"])
                    or c["name"].strip().lower() in names                 # copycat of an earlier pick
                    or c["symbol"].strip().lower() in names
                    or (c["creator"] and self.creators.get(c["creator"], 0) >= r["max_creator_coins"])):
                continue
            pre.append((c, c["age_min"], c["mcap"]))
        if not pre:
            return []

        info = dex_info([c["mint"] for c, _, _ in pre])
        picks = []
        for c, age, mcap in pre:
            d = info.get(c["mint"])
            if (not d or d["vol_h1"] < r["min_vol_h1"] or d["buys_h1"] < r["min_buys_h1"]
                    or d["buys_h1"] < r["min_buy_sell_ratio"] * max(1, d["sells_h1"])):
                continue
            picks.append((d["vol_h1"], c, age, mcap, d))
        picks.sort(key=lambda x: -x[0])

        new = []
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for _, c, age, mcap, d in picks[: r["max_picks_per_scan"]]:
            new.append({"picked_at": when, "mint": c["mint"], "symbol": c.get("symbol", ""),
                        "name": c.get("name", "")[:60], "age_min": round(age), "mcap_usd": round(mcap),
                        "price_usd": d["price"], "vol_h1": round(d["vol_h1"]), "buys_h1": d["buys_h1"],
                        "sells_h1": d["sells_h1"], "dex": d["dex"],
                        "url": d["url"] or f"https://pump.fun/coin/{c['mint']}",
                        "source": c["source"], "buyers_h1": c["buyers_h1"] if c["buyers_h1"] is not None else ""})
            if c["creator"]:
                self.creators[c["creator"]] = self.creators.get(c["creator"], 0) + 1
        if new:
            self.write_rows(rows + new)
            self.save_state()
        return new

    # ---- outcome tracking -------------------------------------------------------------
    def check_outcomes(self) -> int:
        rows, now, due = self.rows(), datetime.now(timezone.utc), []
        for row in rows:
            age_h = (now - datetime.strptime(row["picked_at"], "%Y-%m-%d %H:%M UTC")
                     .replace(tzinfo=timezone.utc)).total_seconds() / 3600
            for h in (1, 24):
                # check once the horizon has passed; give up waiting after 2x the horizon
                if not row.get(f"ret_{h}h") and h <= age_h:
                    due.append((row, h, age_h))
        if not due:
            return 0
        info = dex_info(sorted({row["mint"] for row, _, _ in due}))
        for row, h, age_h in due:
            d = info.get(row["mint"])
            if d:
                row[f"price_{h}h"] = d["price"]
                row[f"ret_{h}h"] = round(d["price"] / float(row["price_usd"]) - 1, 4)
            elif age_h > 2 * h:
                row[f"price_{h}h"] = 0
                row[f"ret_{h}h"] = -1.0  # no market left: count as a total loss
        self.write_rows(rows)
        return len(due)


def scorecard(rows: list[dict]) -> str:
    lines = [f"🧪 Pump scanner TEST — {len(rows)} picks logged (no trades, no alerts)"]
    for h in (1, 24):
        rets = [float(r[f"ret_{h}h"]) for r in rows if r.get(f"ret_{h}h")]
        if not rets:
            lines.append(f"After {h}h: nothing scored yet")
            continue
        net = [x - ROUND_TRIP_FEE for x in rets]
        rets.sort()
        avg = sum(net) / len(net)
        lines.append(
            f"After {h}h ({len(rets)} scored): avg {avg:+.0%} after fees · median {rets[len(rets) // 2]:+.0%}"
            f" · up {sum(x > 0 for x in net) / len(net):.0%} · doubled {sum(x >= 1 for x in rets)}"
            f" · lost >50% {sum(x <= -0.5 for x in rets)}")
    scored = [r for r in rows if r.get("ret_24h")]
    if len(scored) >= 100:
        avg24 = sum(float(r["ret_24h"]) - ROUND_TRIP_FEE for r in scored) / len(scored)
        lines.append("Verdict: " + ("profitable on average after fees ✅" if avg24 > 0
                                    else "loses money on average ❌ — keep alerts off"))
    else:
        lines.append(f"Verdict: wait for 100+ picks scored at 24h ({len(scored)} so far)")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--once", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print Telegram messages instead of sending")
    a = p.parse_args()
    envfile.load()
    from signals import send_telegram  # same Telegram settings as the main bot

    sc = Scanner(Path(a.runs))
    if a.report:
        print(scorecard(sc.rows()))
        return
    every = float(os.environ.get("PUMP_INTERVAL_MIN", "5")) * 60
    alerts = os.environ.get("PUMP_ALERTS", "off").lower() == "on"
    print(f"pump scanner: test mode, every {every / 60:g} min, source {sc.source}, "
          f"per-pick alerts {'ON' if alerts else 'off'}")
    while True:
        try:
            for r in sc.scan():
                print(f"pick [{r['source']}]: {r['symbol']} age {r['age_min']}m mcap ${int(r['mcap_usd']):,} "
                      f"vol1h ${int(r['vol_h1']):,} buys/sells {r['buys_h1']}/{r['sells_h1']}")
                if alerts:
                    send_telegram(f"🎰 HIGH-RISK meme coin (scanner pick, not advice)\n"
                                  f"{r['name']} (${r['symbol']}) · age {r['age_min']} min\n"
                                  f"Market cap ${int(r['mcap_usd']):,} · 1h volume ${int(r['vol_h1']):,}\n"
                                  f"{r['url']}", a.dry_run)
            n = sc.check_outcomes()
            print(f"{datetime.now(timezone.utc):%H:%M} scan done ({len(sc.rows())} picks so far"
                  + (f", scored {n}" if n else "") + ")")
            if time.time() - sc.last_summary >= 24 * 3600 and sc.rows():
                send_telegram(scorecard(sc.rows()), a.dry_run)
                sc.last_summary = time.time()
                sc.save_state()
        except Exception as e:  # keep scanning through API hiccups
            print(f"scan failed: {type(e).__name__}: {e}")
        if a.once:
            break
        time.sleep(every)


if __name__ == "__main__":
    main()
