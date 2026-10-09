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
