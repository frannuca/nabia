"""Backtests of plain ERC and the sentiment overlay, long-only or long-short, and the risk-contribution check.

Long-only runs on the nabia backtest engine. Long-short needs a direction per asset and rebalance, which the engine
does not have: `simulate` repeats the engine's accounting (same covariance estimator, rebalance rows, drift and
costs) and, at each rebalance, solves nabia RiskBudgeting on the covariance of the directed returns D r,
Sigma' = D Sigma D, giving exposures w' > 0 and weights w = D w'. Each asset keeps its risk budget whatever its
side. Run with all-long directions, `simulate` reproduces the engine (checked on every run).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from nabia_python_bridge import frames, risk, strategy

from .budgets import directions, modulate


def window_cov(cfg, window):
    """The engine's estimator: (exponentially) weighted, mean-centred, unbiased covariance of a returns window."""
    P, n = cfg["portfolio"], len(window)
    w = (np.full(n, 1.0 / n) if P["covariance_estimator"] == "Sample"
         else 2.0 ** (-np.arange(n)[::-1] / P["ewma_halflife"]))
    w /= w.sum()
    x = window - w @ window
    return (x.T * w) @ x / (1 - w @ w)


def solve(cfg, cov, b, d):
    """nabia risk budgeting on D Sigma D: weights D w' and relative risk contributions (= b)."""
    vol = risk.Volatility(d[:, None] * cov * d[None, :])
    P = cfg["portfolio"]
    x = np.asarray(risk.RiskBudgeting(risk.RiskBudgetingParams(budgets=b, tolerance=P.get("solver_tolerance", 1e-8),
                                                                 max_iter=P.get("solver_max_iter", 5000)), vol).optimize())
    return d * x, np.asarray(vol.compute_risk_contributions(x)) / vol.compute_risk(x)


def simulate(cfg, returns, sched, dirs):
    """The engine's walk-forward accounting with directions: budgets and directions in force at each rebalance
    (the last schedule row at or before it; b0 / long before the first), drifting weights, costs on turnover."""
    P, R, dates = cfg["portfolio"], returns.values, returns.index
    W, (T, N) = P["covariance_window"], R.shape
    b0 = np.array([cfg["assets"][a][2] for a in returns.columns])
    rows = set(range(W - 1, T, P["rebalance_frequency"]))
    w, equity, rec, targets, rcs = np.zeros(N), 1.0, [], {}, {}
    for t in range(T):
        gross = w @ R[t]
        w = w * (1 + R[t]) / (1 + gross)
        cost = traded = 0.0
        if t in rows:
            k = sched.index.searchsorted(dates[t], side="right") - 1
            b, d = (sched.iloc[k].values, dirs.iloc[k].values) if k >= 0 else (b0, np.ones(N))
            target, rcs[dates[t]] = solve(cfg, window_cov(cfg, R[t - W + 1:t + 1]), b, d)
            traded = np.abs(target - w).sum()
            cost, w, targets[dates[t]] = P["transaction_cost"] * traded, target, target
        net = (1 + gross) * (1 - cost) - 1
        equity *= 1 + net
        rec.append((gross, cost, net, traded, equity))
    perf = pd.DataFrame(rec, dates, ["gross_returns", "costs", "net_returns", "turnover", "equity"])
    perf["drawdown"] = strategy.drawdown_series(perf.equity.values, 1.0)
    metrics = frames.performance(perf.net_returns.iloc[W - 1:], P.get("periods_per_year", 252))
    metrics |= {"rebalances": len(targets), "average_turnover": perf.turnover.sum() / len(targets),
                "annualized_turnover": perf.turnover.sum() * 252 / (T - W + 1), "total_costs": perf.costs.sum()}
    return SimpleNamespace(performance=perf, metrics=metrics, target_weights=pd.DataFrame(targets, returns.columns).T,
                           risk_contributions=pd.DataFrame(rcs, returns.columns).T)


def backtests(cfg, returns, s, b0):
    """Plain ERC with b0 (nabia engine) and the sentiment overlay in the configured mode, with the same covariance,
    rebalancing and costs. Returns (bench, overlay, budget schedule, direction schedule)."""
    assets, P, B = list(returns.columns), cfg["portfolio"], cfg["budgets"]
    returns = returns.loc[returns.index[returns.index.get_loc(s.index[0]) - P["covariance_window"]]:]
    bench = frames.backtest(returns, budgets=dict(zip(assets, b0)), **P)
    sched, dirs, prev_b, prev_d = {}, {}, b0, np.ones(len(assets))
    for day in bench.target_weights.index:
        if day not in s.index:
            continue
        sd = s.loc[day].values
        prev_d = dirs[day] = directions(cfg, sd, prev_d) if B["mode"] == "long_short" else np.ones(len(assets))
        conviction = np.where(prev_d > 0, sd, 1 - sd)  # long_only: = s
        prev_b = sched[day] = (b0 if B["mode"] == "long_short" and not B["modulate_long_short"]
                               else modulate(cfg, conviction, b0, prev_b))
    sched, dirs = pd.DataFrame(sched, index=assets).T, pd.DataFrame(dirs, index=assets).T
    if B["mode"] == "long_only":
        overlay = frames.backtest(returns, budgets=dict(zip(assets, b0)), budget_schedule=sched, **P)
        replay = simulate(cfg, returns, sched, dirs)  # the long-short simulator must reproduce the engine
        assert np.allclose(replay.performance.equity, overlay.performance.equity, rtol=0, atol=1e-6)
    else:
        overlay = simulate(cfg, returns, sched, dirs)
    return bench, overlay, sched, dirs


def check_risk_contributions(cfg, returns, sched, dirs):
    """nabia RiskBudgeting on D Sigma D at every rebalance: risk contributions must match the budgets (1e-4)."""
    W, err = cfg["portfolio"]["covariance_window"], 0.0
    for day, b in sched.iterrows():
        _, rc = solve(cfg, window_cov(cfg, returns.loc[:day].values[-W:]), b.values, dirs.loc[day].values)
        err = max(err, np.abs(rc - b.values).max())
    assert err < 1e-4, err
    return err
