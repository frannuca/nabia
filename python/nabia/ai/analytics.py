"""Statistics of the underlying assets, computed from prices up to a date.

Pure pandas/numpy functions on a wide price frame (dates x symbols), used by
the stats MCP server and by the rule-based baseline policy. Every function
only looks at the rows it is given, so callers control the as-of date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def aligned_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns between consecutive dates where every asset has a price."""
    return prices.dropna(how="any").pct_change().iloc[1:]


def correlation_matrix(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Correlation of the last `window` aligned returns."""
    r = aligned_returns(prices).tail(window)
    if len(r) < 3:
        raise ValueError(f"need at least 3 aligned returns, have {len(r)}")
    return r.corr()


def average_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal correlations."""
    values = corr.to_numpy()
    n = values.shape[0]
    return float((values.sum() - np.trace(values)) / (n * n - n)) if n > 1 else float("nan")


def rolling_volatility(prices: pd.DataFrame, windows: tuple[int, ...] = (20, 60, 120)) -> pd.DataFrame:
    """Per symbol: annualized volatility over each window (its own returns, not
    aligned with the other assets), plus where the 20-day volatility sits in
    its last year of history (percentile, 0..1)."""
    out = {}
    for symbol in prices.columns:
        r = prices[symbol].dropna().pct_change().dropna()
        row = {f"vol_{w}d": float(r.tail(w).std() * np.sqrt(PERIODS_PER_YEAR)) if len(r) >= w else float("nan")
               for w in windows}
        rolling = r.rolling(20).std().dropna().tail(PERIODS_PER_YEAR)
        row["vol_20d_percentile_1y"] = float((rolling <= rolling.iloc[-1]).mean()) if len(rolling) else float("nan")
        out[symbol] = row
    return pd.DataFrame(out).T


def trend_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: past returns (1m, 3m, 6m, 12m), distance from the 50- and
    200-day moving averages, a volatility-scaled 12-1 momentum score, and the
    current and 1-year maximum drawdowns."""
    out = {}
    for symbol in prices.columns:
        p = prices[symbol].dropna()
        last = p.iloc[-1]
        row: dict[str, float] = {}
        for label, n in (("ret_1m", 21), ("ret_3m", 63), ("ret_6m", 126), ("ret_12m", 252)):
            row[label] = float(last / p.iloc[-n - 1] - 1) if len(p) > n else float("nan")
        for n in (50, 200):
            row[f"vs_ma{n}"] = float(last / p.tail(n).mean() - 1) if len(p) >= n else float("nan")
        r = p.pct_change().dropna()
        vol = r.tail(252).std() * np.sqrt(PERIODS_PER_YEAR)
        if len(p) > 252 and vol > 0:
            row["momentum_12_1"] = float((p.iloc[-22] / p.iloc[-253] - 1) / vol)  # skip the last month
        else:
            row["momentum_12_1"] = float("nan")
        year = p.tail(253)
        drawdown = 1 - year / year.cummax()
        row["drawdown"] = float(drawdown.iloc[-1])
        row["max_drawdown_1y"] = float(drawdown.max())
        out[symbol] = row
    return pd.DataFrame(out).T


def market_regime(prices: pd.DataFrame) -> dict[str, float]:
    """Cross-asset summary: average correlation now (60d) vs over the last year,
    median volatility percentile, and the share of assets in an uptrend."""
    corr_now = average_correlation(correlation_matrix(prices, 60))
    corr_year = average_correlation(correlation_matrix(prices, 252)) if len(aligned_returns(prices)) >= 60 else float("nan")
    vols = rolling_volatility(prices)
    trends = trend_signals(prices)
    return {
        "avg_correlation_60d": corr_now,
        "avg_correlation_1y": corr_year,
        "median_vol_percentile_1y": float(vols["vol_20d_percentile_1y"].median()),
        "share_above_ma200": float((trends["vs_ma200"] > 0).mean()),
        "share_positive_12m": float((trends["ret_12m"] > 0).mean()),
    }


def trend_tilted_budgets(prices: pd.DataFrame, strength: float = 0.5, min_budget: float = 0.02,
                         max_budget: float = 0.40) -> dict[str, float]:
    """A transparent rule-based policy: equal risk budgets tilted towards assets
    with positive volatility-scaled momentum, bounded and normalized.

    budget_i ~ 1 + strength * clip(momentum_12_1_i, -1, 1). Used as a baseline
    for the agents and as a deterministic decision rule in tests.
    """
    momentum = trend_signals(prices)["momentum_12_1"].fillna(0.0).clip(-1.0, 1.0)
    raw = (1.0 + strength * momentum).clip(lower=1e-6)
    return project_budgets(raw.to_dict(), min_budget, max_budget)


def project_budgets(budgets: dict[str, float], min_budget: float, max_budget: float) -> dict[str, float]:
    """Normalizes positive budgets to sum to 1 with every value in [min_budget, max_budget].

    Values beyond a bound are fixed at that bound and the rest rescaled to fill
    the remainder, repeating until nothing is out of bounds.
    """
    if not budgets:
        raise ValueError("no budgets")
    if any(not np.isfinite(v) or v <= 0 for v in budgets.values()):
        raise ValueError("budgets must be positive numbers")
    n = len(budgets)
    if not (min_budget * n <= 1.0 + 1e-12 and max_budget * n >= 1.0 - 1e-12):
        raise ValueError("bounds are infeasible for this number of assets")
    fixed: dict[str, float] = {}
    free = dict(budgets)
    while True:
        remaining = 1.0 - sum(fixed.values())
        total = sum(free.values())
        scaled = {k: v * remaining / total for k, v in free.items()}
        low = {k for k, v in scaled.items() if v < min_budget - 1e-12}
        high = {k for k, v in scaled.items() if v > max_budget + 1e-12}
        if not low and not high:
            result = {**fixed, **scaled}
            return {k: result[k] for k in budgets}
        for k in low:
            fixed[k] = min_budget
            free.pop(k)
        for k in high:
            fixed[k] = max_budget
            free.pop(k)
        if not free:
            result = {k: fixed[k] / sum(fixed.values()) for k in budgets}
            return result
