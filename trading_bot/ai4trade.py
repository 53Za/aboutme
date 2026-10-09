"""Paper-trade the bot's signals on AI-Trader (https://ai4trade.ai, github.com/HKUDS/AI-Trader).

Every OPEN/CLOSE alert becomes a simulated trade at the platform's live price, giving an
independent, timestamped track record. Note: trades on the platform are public and other
agents can follow/copy them.

One-time registration (run it yourself; use a fresh password you don't use elsewhere):
    python ai4trade.py register --name MyCryptoBot --email you@example.com
    export AI4TRADE_TOKEN=...          # printed by the command above; keep it secret

Check the paper account:
    python ai4trade.py status
"""
import argparse
import getpass
import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("AI4TRADE_URL", "https://ai4trade.ai")
OPEN_ACTION = {1: "buy", -1: "short"}
CLOSE_ACTION = {1: "sell", -1: "cover"}


class Ai4TradeError(RuntimeError):
    pass


def _request(method: str, path: str, body: dict | None = None, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise Ai4TradeError(f"{e.code} {detail}") from None


class PaperAccount:
    def __init__(self, token: str):
        self.token = token

    def trade(self, action: str, coin: str, quantity: float, note: str = "") -> dict:
        """Simulated trade at the platform's current price (Method 2: price 0, executed_at now)."""
        return _request("POST", "/api/signals/realtime", {
            "market": "crypto", "action": action, "symbol": coin, "price": 0,
            "quantity": round(quantity, 6), "content": note, "executed_at": "now",
        }, self.token)

    def open(self, side: int, coin: str, quantity: float, note: str = "") -> dict:
        return self.trade(OPEN_ACTION[side], coin, quantity, note)

    def close(self, side: int, coin: str, quantity: float, note: str = "") -> dict:
        return self.trade(CLOSE_ACTION[side], coin, quantity, note)

    def positions(self) -> dict:
        return _request("GET", "/api/positions", token=self.token)


def from_env() -> PaperAccount | None:
    token = os.environ.get("AI4TRADE_TOKEN")
    return PaperAccount(token) if token else None


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--email", required=True)
    sub.add_parser("status")
    a = p.parse_args()

    if a.cmd == "register":
        pw = getpass.getpass("New AI-Trader password (use a fresh one): ")
        res = _request("POST", "/api/claw/agents/selfRegister",
                       {"name": a.name, "email": a.email, "password": pw})
        print(f"registered agent {res.get('name')} (id {res.get('agent_id')})")
        print(f"export AI4TRADE_TOKEN={res['token']}")
    else:
        acct = from_env()
        if not acct:
            raise SystemExit("Set AI4TRADE_TOKEN first (see `python ai4trade.py register`).")
        print(json.dumps(acct.positions(), indent=2))


if __name__ == "__main__":
    main()
