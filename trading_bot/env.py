"""Minimal trading environment: one asset, positions short / flat / long."""
import numpy as np

ACTIONS = np.array([-1, 0, 1])  # short, flat, long


def funding_features(funding: np.ndarray) -> np.ndarray:
    """Perp funding rate: positive = longs pay shorts (crowded long), negative = crowded short."""
    bps = funding * 1e4
    avg24 = np.array([bps[max(0, t - 23):t + 1].mean() for t in range(len(bps))])
    scale = 0.5  # fixed (no look-ahead): typical hourly funding is ~0.1 bps, extremes several bps
    return np.column_stack([bps / scale, avg24 / scale, (bps - avg24) / scale])


def indicator_features(prices: np.ndarray) -> np.ndarray:
    """Classic indicators (as in NoFxAiOS/nofx market/data_indicators.go), scaled to roughly [-1, 1]."""
    import pandas as pd
    p = pd.Series(prices)
    d = p.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-12))
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    vol = p.pct_change().rolling(30, min_periods=2).std() * p + 1e-12
    mid, sd = p.rolling(20, min_periods=2).mean(), p.rolling(20, min_periods=2).std() + 1e-12
    hi, lo = p.rolling(20, min_periods=1).max(), p.rolling(20, min_periods=1).min()
    return np.column_stack([
        (rsi - 50) / 25,                       # RSI(14)
        hist / vol,                            # MACD histogram in volatility units
        (p - mid) / (2 * sd),                  # Bollinger %b, centred
        2 * (p - lo) / (hi - lo + 1e-12) - 1,  # Donchian(20) position
    ]).astype(float)


def features(prices: np.ndarray, lookback: int, funding: np.ndarray | None = None,
             indicators: bool = False) -> np.ndarray:
    """Per-step features built only from information available at time t."""
    logp = np.log(prices)
    r = np.diff(logp, prepend=logp[0])
    feats = []
    for k in range(1, lookback + 1):
        feats.append(np.roll(r, k - 1))                       # recent returns
    for w in (10, 30, 90):
        mom = logp - np.roll(logp, w)
        feats.append(mom)                                     # momentum
    vol = np.array([r[max(0, t - 30):t + 1].std() + 1e-8 for t in range(len(r))])
    X = np.column_stack(feats) / vol[:, None]                 # volatility-normalised
    if funding is not None:
        X = np.column_stack([X, funding_features(np.asarray(funding, dtype=float))])
    if indicators:
        X = np.column_stack([X, indicator_features(prices)])
    X[:90] = 0.0                                              # warm-up rows have no history
    return np.clip(X, -5, 5)


def step_returns(prices: np.ndarray) -> np.ndarray:
    """Simple return earned from t to t+1."""
    return prices[1:] / prices[:-1] - 1.0


VOL_SHORT, VOL_LONG, MIN_SIZE = 24, 720, 0.25


def vol_size(rets_before: np.ndarray) -> float:
    """Position size in [0.25, 1]: smaller when the last day was wilder than the last month.

    Uses only returns that already happened (causal). 1.0 = normal size.
    """
    if len(rets_before) < VOL_SHORT * 2:
        return 1.0
    short = rets_before[-VOL_SHORT:].std()
    normal = np.median([rets_before[max(0, i - VOL_SHORT):i].std()
                        for i in range(max(VOL_SHORT, len(rets_before) - VOL_LONG),
                                       len(rets_before) + 1, VOL_SHORT)])
    if short <= 0 or not np.isfinite(normal):
        return 1.0
    return float(np.clip(normal / short, MIN_SIZE, 1.0))


def run(policy_fn, X, rets, cost: float, guard=None, sizing=False, history=None):
    """Roll a policy over a window. Returns per-step net PnL, equity curve and positions.

    `guard` (risk.RiskGuard) can override the policy, exactly as it does in live alerts.
    `sizing` scales each new trade by vol_size(); `history` = returns before the window,
    so the volatility estimate is warm from the first step.
    """
    pos, size = 0, 1.0
    pnl, positions = np.empty(len(rets)), np.empty(len(rets), dtype=int)
    px = np.concatenate([[1.0], np.cumprod(1.0 + rets)])  # relative price at each decision
    past = np.concatenate([history if history is not None else [], rets])
    off = len(past) - len(rets)
    for t in range(len(rets)):
        new_pos = policy_fn(X[t], pos)
        if guard is not None:
            new_pos, _ = guard.step(new_pos, px[t])
        new_size = size
        if new_pos != pos:
            new_size = vol_size(past[:off + t]) if (sizing and new_pos != 0) else 1.0
        pnl[t] = new_pos * new_size * rets[t] - cost * abs(new_pos * new_size - pos * size)
        pos, size = new_pos, new_size
        positions[t] = pos
    equity = np.cumprod(1.0 + pnl)
    return pnl, equity, positions


def metrics(pnl: np.ndarray, equity: np.ndarray, positions: np.ndarray,
            periods_per_year: int = 252) -> dict:
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    dd = 1.0 - equity / peak
    sd = pnl.std()
    return {
        "return": float(equity[-1] - 1.0),
        "max_drawdown": float(dd.max()),
        "sharpe": float(pnl.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "trades": int(np.count_nonzero(np.diff(positions, prepend=0))),
    }
