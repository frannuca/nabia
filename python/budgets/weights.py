"""Portfolio weights from the risk budgets with the nabia risk budgeting optimizer (C++, nabia_python_bridge): at
every rebalance date, the long-only weights (sum 1) whose risk contributions match the asset budgets, for a
covariance of the assets' weekly returns over the trailing window (weekly, so that markets closing at different
hours do not bias the correlations), shrunk toward its diagonal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nabia_python_bridge import risk

from datalayer import MarketDB
from exposures.proxies import asset_returns

MEASURES = {"Volatility": lambda c, W: risk.Volatility(c),
            "ValueAtRisk": lambda c, W: risk.ValueAtRisk(c, W["confidence_level"]),
            "ExpectedShortfall": lambda c, W: risk.ExpectedShortfall(c, W["confidence_level"])}


def covariance(R: pd.DataFrame, shrinkage: float, periods: int = 52) -> np.ndarray:
    """Annualized sample covariance, shrunk toward its diagonal: (1 - s) C + s diag(C)."""
    c = R.cov().values * periods
    return (1 - shrinkage) * c + shrinkage * np.diag(np.diag(c))


def optimize(cov: np.ndarray, budgets: np.ndarray, W: dict, x0=None):
    """(weights, result) of the risk budgeting problem for one date."""
    rb = risk.RiskBudgeting(risk.RiskBudgetingParams(budgets=budgets, tolerance=W["tolerance"]),
                            MEASURES[W["risk_measure"]](cov, W))
    return rb.optimize(x0), rb.result


def compute_weights(cfg: dict, start: str = "", end: str = "9999") -> pd.DataFrame:
    """Weights at every date of the risk_budgets table in [start, end], stored in portfolio_weights and returned."""
    W, assets = cfg["weights"], cfg["exposures"]["assets"]
    with MarketDB(cfg["db"]) as db:
        budgets = pd.DataFrame(db.budget_history(start, end))
        if budgets.empty:
            return budgets
        budgets = budgets.pivot(index="date", columns="asset", values="budget").reindex(columns=assets)
        R = np.expm1(asset_returns(db, assets))  # weekly simple returns
        rows, x0 = [], None
        for date, b in budgets.iterrows():
            win = R[R.index <= date].tail(W["cov_window_weeks"]).dropna()  # returns up to the rebalance date
            if len(win) < 0.8 * W["cov_window_weeks"]:
                continue
            w, res = optimize(covariance(win, W["shrinkage"]), b.values, W, x0)
            x0 = w  # warm start of the next date
            rows += [{"date": date, "asset": a, "budget": float(b[a]), "weight": float(w[k]),
                      "risk_contribution": float(res.relative_contributions[k]), "portfolio_risk": float(res.risk),
                      "converged": int(res.converged)} for k, a in enumerate(assets)]
        db.save_weights(rows)
    return pd.DataFrame(rows)
