"""pandas interface to the time-indexed engines: portfolio backtests and factor regressions.

The C++ engines work on row positions; these wrappers keep the pandas index
(dates or anything ordered) and column names on the way in and out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import regression as _regression
from . import strategy as _strategy


def _make_params(cls: type, params: Any, overrides: Mapping[str, Any]) -> Any:
    """`params` (or a default `cls`) with `overrides` applied. Enum fields also
    accept their member name as a string, e.g. risk_measure="ValueAtRisk"."""
    # A copy, so the caller's parameters are never modified.
    out = cls() if params is None else cls(**{name: getattr(params, name) for name in _fields(params)})
    for name, value in overrides.items():
        current = getattr(out, name)  # AttributeError on unknown fields
        if isinstance(value, str) and hasattr(type(current), "__members__"):
            value = type(current).__members__[value]
        setattr(out, name, value)
    return out


def _fields(obj: Any) -> list[str]:
    return [n for n, v in vars(type(obj)).items() if isinstance(v, property) and not n.startswith("_")]


# --- Portfolio backtests ---------------------------------------------------------


@dataclass
class PortfolioFrames:
    weights: pd.DataFrame             # held over each period
    target_weights: pd.DataFrame      # at each rebalance
    risk_contributions: pd.DataFrame  # relative contributions at each rebalance
    leverage: pd.Series               # sum of target weights at each rebalance
    performance: pd.DataFrame         # gross_returns, costs, net_returns, turnover, equity, pnl, drawdown
    metrics: dict[str, float]
    all_converged: bool
    computation: Any                  # the raw strategy.PortfolioComputation


def metrics_dict(metrics: Any) -> dict[str, float]:
    """A PerformanceMetrics as a plain dict."""
    return {name: getattr(metrics, name) for name in _fields(metrics)}


def backtest(returns: pd.DataFrame, params: Any = None, *, budget_schedule: pd.DataFrame | None = None,
             **overrides: Any) -> PortfolioFrames:
    """Walk-forward risk budgeting backtest of `returns` (simple returns, rows in time order).

    Parameters are a ``strategy.RiskBudgetingPortfolioParams`` and/or its fields as
    keyword arguments; ``budgets`` may be a mapping {asset: budget}. Row d of
    ``budget_schedule`` (columns = assets) holds budgets used by every rebalance
    at or after time d.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    assets = list(returns.columns)
    budgets = overrides.pop("budgets", None)
    p = _make_params(_strategy.RiskBudgetingPortfolioParams, params, overrides)
    if budgets is not None:
        if isinstance(budgets, Mapping):
            missing = set(assets) - set(budgets)
            if missing:
                raise ValueError(f"budgets missing for {sorted(missing)}")
            budgets = [budgets[a] for a in assets]
        p.budgets = np.asarray(budgets, dtype=float)

    rows = values = None
    if budget_schedule is not None:
        schedule = budget_schedule[assets].sort_index()
        rows = returns.index.searchsorted(schedule.index, side="left").tolist()
        values = schedule.to_numpy(dtype=float)

    c = _strategy.RiskBudgetingPortfolio(p).run(returns.to_numpy(dtype=float), rows, values)
    index = returns.index
    rebalances = index[np.asarray(c.rebalance_rows, dtype=int)]
    performance = pd.DataFrame(
        {name: getattr(c, name) for name in ("gross_returns", "costs", "net_returns", "turnover", "equity", "pnl",
                                             "drawdown")},
        index=index)
    return PortfolioFrames(
        weights=pd.DataFrame(c.weights, index=index, columns=assets),
        target_weights=pd.DataFrame(c.target_weights, index=rebalances, columns=assets),
        risk_contributions=pd.DataFrame(c.risk_contributions, index=rebalances, columns=assets),
        leverage=pd.Series(c.leverage, index=rebalances, name="leverage"),
        performance=performance,
        metrics=metrics_dict(c.metrics),
        all_converged=c.all_converged,
        computation=c,
    )


def performance(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Performance metrics of a series of simple returns."""
    return metrics_dict(_strategy.compute_performance(np.asarray(returns, dtype=float), periods_per_year))


# --- Factor regression -----------------------------------------------------------


@dataclass
class RegressionFrames:
    coefficients: pd.Series           # selected factors (final model)
    intercept: float
    alpha: float
    lambda_: float
    predictions: pd.Series            # model fit on the training rows; out of sample from test_start
    residuals: pd.Series
    test_start: Any                   # first out-of-sample index value, or None
    best_by_size: pd.DataFrame        # best subset of each size
    bootstrap: pd.DataFrame | None    # per selected factor: mean, std_error, lower, upper, selection_frequency
    computation: Any                  # the raw regression.RegressionComputation


def regress(target: pd.Series, factors: pd.DataFrame, params: Any = None, **overrides: Any) -> RegressionFrames:
    """Elastic net regression of `target` on `factors` with factor, lambda and alpha selection.

    The two are aligned on their common index and rows with a missing value
    are dropped. Parameters are a ``regression.RegressionEngineParams`` and/or
    its fields as keyword arguments, e.g. ``min_factors=1, max_factors=3,
    train_fraction=0.7, bootstrap_samples=500``.
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    data = factors.join(pd.Series(target, name="__target__"), how="inner").dropna()
    names = [str(c) for c in factors.columns]
    p = _make_params(_regression.RegressionEngineParams, params, overrides)
    y = data.pop("__target__").to_numpy(dtype=float)
    c = _regression.RegressionEngine(p).run(y, data.to_numpy(dtype=float), names)

    index = data.index
    sizes = pd.DataFrame([
        {
            "size": len(e.factors),
            "factors": [names[j] for j in e.factors],
            "alpha": e.alpha,
            "lambda": e.lambda_,
            "nonzero": e.nonzero,
            "validation_mse": e.validation_mse,
            "train_mse": e.train_mse,
            "train_r2": e.train_r2,
            "test_mse": e.test_mse,
            "test_r2": e.test_r2,
            "criterion": e.criterion,
            "score": e.score,
        }
        for e in c.best_by_size
    ]).set_index("size")
    boot = None
    if c.bootstrap is not None:
        b = c.bootstrap
        boot = pd.DataFrame(
            {"mean": b.mean, "std_error": b.std_error, "lower": b.lower, "upper": b.upper,
             "selection_frequency": b.selection_frequency},
            index=pd.Index(c.factor_names, name="factor"))
    return RegressionFrames(
        coefficients=pd.Series(c.coefficients, index=pd.Index(c.factor_names, name="factor"), name="coefficient"),
        intercept=c.intercept,
        alpha=c.alpha,
        lambda_=c.lambda_,
        predictions=pd.Series(c.predictions, index=index, name="predictions"),
        residuals=pd.Series(c.residuals, index=index, name="residuals"),
        test_start=index[c.train_rows] if c.test_rows > 0 else None,
        best_by_size=sizes,
        bootstrap=boot,
        computation=c,
    )


__all__ = ["PortfolioFrames", "RegressionFrames", "backtest", "metrics_dict", "performance", "regress"]
