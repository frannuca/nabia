"""Positions from a sentiment signal: budgets, directions, weights and risk contributions at every rebalance of a
lookback window and on the as-of date, and the window's performance against plain ERC, written as CSV and JSON."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nabia_python_bridge import frames

from .backtest import rebalance_step, solve, window_cov


def position(cfg, returns, s, b0, prev_b, prev_d, day):
    """Budgets, directions, weights and risk contributions on `day` (covariance of the last window up to it)."""
    b, d = rebalance_step(cfg, s.loc[day].values, b0, prev_b, prev_d)
    W = cfg["portfolio"]["covariance_window"]
    w, rc = solve(cfg, window_cov(cfg, returns.loc[:day].values[-W:]), b, d)
    return pd.DataFrame({"s": s.loc[day].values, "direction": d, "budget": b, "weight": w, "risk_contribution": rc},
                        index=returns.columns)


def window_metrics(result, start):
    m = frames.performance(result.performance.net_returns.loc[start:], 252)
    return {k: float(m[k]) for k in ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]}


def write_positions(cfg, returns, s, b0, bench, overlay, sched, dirs, out, lookback=252, as_of=None):
    """signal.csv, rebalances.csv, current.csv and summary.json in `out`, for the `lookback` business days up to
    as_of (default: the last date). Returns (current position, summary)."""
    day = s.index[s.index <= pd.Timestamp(as_of)][-1] if as_of else s.index[-1]
    start = s.index[max(0, s.index.get_loc(day) - lookback + 1)]
    ones = np.ones(len(b0))
    rows = [position(cfg, returns, s, b0, sched.iloc[k - 1].values if k else b0, dirs.iloc[k - 1].values if k else ones, t)
            .rename_axis("asset").reset_index().assign(date=t) for k, t in enumerate(sched.index) if start <= t <= day]
    last = sched.index[sched.index <= day][-1]
    current = position(cfg, returns, s, b0, sched.loc[last].values, dirs.loc[last].values, day)
    out.mkdir(parents=True, exist_ok=True)
    s.loc[start:day].to_csv(out / "signal.csv")
    columns = ["date", "asset", "s", "direction", "budget", "weight", "risk_contribution"]
    (pd.concat(rows)[columns] if rows else pd.DataFrame(columns=columns)).to_csv(out / "rebalances.csv", index=False)
    current.to_csv(out / "current.csv")
    summary = {"as_of": str(day.date()), "window": [str(start.date()), str(day.date())], "rebalances": len(rows),
               "mode": cfg["budgets"]["mode"], "position": current.round(6).to_dict("index"),
               "window_performance": {"sentiment overlay": window_metrics(overlay, start), "ERC (b0)": window_metrics(bench, start)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n== position as of {day.date()} ({cfg['budgets']['mode']}) ==\n{current.round(4)}")
    print(f"\n== {start.date()} .. {day.date()}: {len(rows)} rebalances ==\n{pd.DataFrame(summary['window_performance']).round(4)}")
    return current, summary
