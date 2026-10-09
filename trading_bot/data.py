"""Price data: load a CSV with a `close` column, or generate synthetic prices."""
import numpy as np
import pandas as pd


def load_csv(path: str, column: str = "close") -> np.ndarray:
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.lower() == column), None)
    if col is None:
        raise ValueError(f"{path} has no '{column}' column (found {list(df.columns)})")
    return df[col].astype(float).to_numpy()


def synthetic(n: int = 6000, seed: int = 0) -> np.ndarray:
    """Regime-switching random walk: trending and mean-reverting stretches."""
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    drift, regime_left = 0.0, 0
    for t in range(n):
        if regime_left == 0:
            drift = rng.choice([-0.0006, 0.0, 0.0006])
            regime_left = rng.integers(200, 800)
        regime_left -= 1
        rets[t] = drift + rng.normal(0, 0.01)
    return 100 * np.exp(np.cumsum(rets))


def split(prices: np.ndarray, train: float = 0.6, val: float = 0.2):
    """Chronological train / validation / test split. Test is never seen during evolution."""
    a, b = int(len(prices) * train), int(len(prices) * (train + val))
    return prices[:a], prices[a:b], prices[b:]


# ---- crypto exchange data (ccxt) -------------------------------------------------

def make_exchange(name: str):
    """Public-data exchange client. Honours HTTPS_PROXY / REQUESTS_CA_BUNDLE if set."""
    import os
    import ccxt  # pip install ccxt
    opts = {"enableRateLimit": True}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        opts["httpsProxy"] = proxy
    ex = getattr(ccxt, name)(opts)
    if os.environ.get("REQUESTS_CA_BUNDLE"):
        ex.session.verify = os.environ["REQUESTS_CA_BUNDLE"]
    return ex


def fetch_closed_candles(ex, symbol: str, timeframe: str, bars: int) -> np.ndarray:
    """Close prices of the last `bars` *closed* candles (the still-forming one is dropped)."""
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = now - (bars + 1) * tf_ms
    rows = {}
    while since < now:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
        if not batch:
            break
        for ts, *_ohlc, close, _vol in batch:
            rows[ts] = close
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    closed = [rows[t] for t in sorted(rows) if t + tf_ms <= now]
    return np.array(closed[-bars:], dtype=float)


def timeframe_seconds(timeframe: str) -> int:
    unit, n = timeframe[-1], int(timeframe[:-1])
    return {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit] * n


def periods_per_year(timeframe: str) -> int:
    """Crypto trades 24/7, so annualise by candles per calendar year."""
    return int(365 * 86400 / timeframe_seconds(timeframe))


def model_dir(symbol: str, timeframe: str) -> str:
    return f"runs/{symbol.replace('/', '_')}_{timeframe}"
