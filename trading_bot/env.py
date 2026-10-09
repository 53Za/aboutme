"""Minimal trading environment: one asset, positions short / flat / long."""
import numpy as np

ACTIONS = np.array([-1, 0, 1])  # short, flat, long


def features(prices: np.ndarray, lookback: int) -> np.ndarray:
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
    X[:90] = 0.0                                              # warm-up rows have no history
    return np.clip(X, -5, 5)


def step_returns(prices: np.ndarray) -> np.ndarray:
    """Simple return earned from t to t+1."""
    return prices[1:] / prices[:-1] - 1.0


def run(policy_fn, X, rets, cost: float):
    """Roll a policy over a window. Returns per-step net PnL, equity curve and positions."""
    pos, pnl, positions = 0, np.empty(len(rets)), np.empty(len(rets), dtype=int)
    for t in range(len(rets)):
        new_pos = policy_fn(X[t], pos)
        pnl[t] = new_pos * rets[t] - cost * abs(new_pos - pos)
        pos = positions[t] = new_pos
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
