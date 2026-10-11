"""pandas interface to the C++ risk budgeting portfolio engine."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import _nabia
except ImportError as error:  # pragma: no cover - depends on the build
    raise ImportError(
        "the nabia C++ extension (nabia._nabia) is not built. From the repository root run\n"
        "  cmake --preset python && cmake --build --preset python\n"
        "or install the package with  pip install ./python"
    ) from error

__all__ = ["BacktestResult", "TargetWeights", "backtest", "latest_weights"]


@dataclass(frozen=True)
class BacktestResult:
    """Output of :func:`backtest`. All frames are indexed by date."""

    weights: pd.DataFrame
    """Weights held over each period (row t earns the returns of date t); 1 - sum is cash."""
    target_weights: pd.DataFrame
    """Weights set at each rebalance date (after volatility targeting)."""
    risk_contributions: pd.DataFrame
    """Share of portfolio risk of each asset at each rebalance (rows sum to 1, match the budgets)."""
    leverage: pd.Series
    """Sum of the target weights at each rebalance."""
    performance: pd.DataFrame
    """Per date: gross_return, transaction_cost, net_return, turnover, pnl, equity, drawdown."""
    metrics: dict[str, float]
    """Summary statistics from the first rebalance on (Sharpe, max drawdown, turnover, ...)."""
    all_converged: bool
    threads_used: int
    dropped_dates: int
    """Price dates dropped because some asset had no price."""

    @property
    def pnl(self) -> pd.Series:
        """Change in equity per date, in currency (the unit of ``initial_capital``)."""
        return self.performance["pnl"]

    @property
    def equity(self) -> pd.Series:
        return self.performance["equity"]

    @property
    def net_returns(self) -> pd.Series:
        return self.performance["net_return"]


@dataclass(frozen=True)
class TargetWeights:
    """Output of :func:`latest_weights`: the position to hold from ``date`` on."""

    date: pd.Timestamp
    weights: pd.Series
    risk_contributions: pd.Series
    leverage: float
    converged: bool
    dropped_dates: int


def _to_days(index: pd.Index) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.DatetimeIndex(index)
        except (TypeError, ValueError) as error:
            raise TypeError("prices must be indexed by dates (a DatetimeIndex or convertible to one)") from error
    if index.tz is not None:
        index = index.tz_localize(None)
    return np.ascontiguousarray(index.values.astype("datetime64[D]").astype(np.int64))


def _from_days(days: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.asarray(days, dtype=np.int64).astype("datetime64[D]"), name="date")


def _config_json(params: Mapping[str, Any] | None) -> str:
    """Portfolio parameters to JSON; dates may be given as date/datetime/Timestamp."""
    clean: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if isinstance(value, (dt.date, pd.Timestamp)):
            value = pd.Timestamp(value).strftime("%Y-%m-%d")
        elif isinstance(value, pd.Series):
            value = {str(k): float(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        clean[key] = value
    return json.dumps(clean)


def _inputs(prices: pd.DataFrame, params: Mapping[str, Any] | None) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame (dates x assets)")
    if prices.shape[1] == 0:
        raise ValueError("prices has no columns")
    values = np.ascontiguousarray(prices.to_numpy(dtype=np.float64, na_value=np.nan))
    return _to_days(prices.index), [str(c) for c in prices.columns], values, _config_json(params)


def backtest(
    prices: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
    budget_schedule: pd.DataFrame | None = None,
) -> BacktestResult:
    """Walk-forward risk budgeting backtest.

    Parameters
    ----------
    prices
        Asset prices, one column per asset, indexed by date in increasing order.
        Missing values (NaN) are allowed: dates where any asset has no price are
        dropped, and returns are computed between the remaining dates.
    params
        Portfolio parameters, with the same keys as the ``nabia_rb_backtest``
        JSON configuration (see ``apps/rb_backtest/README.md``), e.g.::

            {"covariance_window": 120, "rebalance_frequency": 21,
             "budgets": {"SPX": 0.5, "ZN": 0.5}, "target_volatility": 0.10,
             "transaction_cost": 0.0005, "initial_capital": 1_000_000}

        ``start`` / ``end`` may be strings or dates. Unknown keys raise ValueError.
    budget_schedule
        Optional time-varying risk budgets: one row per decision date, one
        column per asset of ``prices`` (any order). Budgets decided on date d
        are used by every rebalance at or after d, until the next row; earlier
        rebalances use ``params["budgets"]`` (equal risk if absent).

    Raises
    ------
    ValueError
        On invalid parameters or data (the message names the problem).
    """
    days, assets, values, config = _inputs(prices, params)
    schedule_days = schedule_values = None
    if budget_schedule is not None:
        missing = set(assets) - {str(c) for c in budget_schedule.columns}
        extra = {str(c) for c in budget_schedule.columns} - set(assets)
        if missing or extra:
            raise ValueError(f"budget_schedule columns must be the price columns (missing {sorted(missing)}, "
                             f"unknown {sorted(extra)})")
        ordered = budget_schedule.rename(columns=str)[assets].sort_index()
        schedule_days = _to_days(ordered.index)
        schedule_values = np.ascontiguousarray(ordered.to_numpy(dtype=np.float64))
    r = _nabia.backtest(days, assets, values, config, schedule_days, schedule_values)
    dates = _from_days(r["dates"])
    rebalance_dates = _from_days(r["rebalance_dates"])
    columns = pd.Index(r["assets"], name="asset")
    performance = pd.DataFrame(
        {
            "gross_return": r["gross_returns"],
            "transaction_cost": r["costs"],
            "net_return": r["net_returns"],
            "turnover": r["turnover"],
            "pnl": r["pnl"],
            "equity": r["equity"],
            "drawdown": r["drawdown"],
        },
        index=dates,
    )
    return BacktestResult(
        weights=pd.DataFrame(r["weights"], index=dates, columns=columns),
        target_weights=pd.DataFrame(r["target_weights"], index=rebalance_dates, columns=columns),
        risk_contributions=pd.DataFrame(r["risk_contributions"], index=rebalance_dates, columns=columns),
        leverage=pd.Series(r["leverage"], index=rebalance_dates, name="leverage"),
        performance=performance,
        metrics=dict(r["metrics"]),
        all_converged=bool(r["all_converged"]),
        threads_used=int(r["threads_used"]),
        dropped_dates=int(r["dropped_dates"]),
    )


def latest_weights(prices: pd.DataFrame, params: Mapping[str, Any] | None = None) -> TargetWeights:
    """Risk budgeting weights to hold from the last date of ``prices`` on.

    Uses the last ``covariance_window`` returns, exactly as a backtest would at
    a rebalance on that date (same estimator, budgets, risk measure and
    volatility target). ``rebalance_frequency`` and costs play no role.
    """
    days, assets, values, config = _inputs(prices, params)
    r = _nabia.latest_weights(days, assets, values, config)
    columns = pd.Index(r["assets"], name="asset")
    return TargetWeights(
        date=_from_days(r["date"])[0],
        weights=pd.Series(r["weights"], index=columns, name="weight"),
        risk_contributions=pd.Series(r["risk_contributions"], index=columns, name="risk_contribution"),
        leverage=float(r["leverage"]),
        converged=bool(r["converged"]),
        dropped_dates=int(r["dropped_dates"]),
    )
