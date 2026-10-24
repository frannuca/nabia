"""Walk-forward risk parity and risk budgeting backtests on real data.

Every month the engine estimates the covariance from the trailing window, solves
for the risk budgeting weights, optionally scales them to a volatility target and
trades, paying transaction costs. We compare risk parity variants with an equally
weighted benchmark, then change the budgets over time (a tactical tilt).

Covers: strategy.RiskBudgetingPortfolio (run with and without a budget schedule,
parameters), RiskBudgetingPortfolioParams, PortfolioRiskMeasure,
CovarianceEstimator, PortfolioComputation, PerformanceMetrics,
compute_performance, equity_curve, drawdown_series; frames.backtest,
frames.performance, frames.metrics_dict.

Run from the python/ folder (or in PyCharm, with python/.venv as interpreter):
    .venv/bin/python examples/03_portfolio_backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _plots
from _data import header, load_returns
from nabia_python_bridge import frames, strategy

ASSETS = ["^GSPC", "^STOXX50E", "^N225", "ZN=F", "ZB=F", "GC=F", "HG=F"]


def summary(metrics: strategy.PerformanceMetrics) -> dict[str, float]:
    return {"ann. return": metrics.annualized_return, "ann. vol": metrics.annualized_volatility,
            "Sharpe": metrics.sharpe_ratio, "Sortino": metrics.sortino_ratio, "max DD": metrics.max_drawdown,
            "Calmar": metrics.calmar_ratio, "turnover/yr": metrics.annualized_turnover,
            "costs": metrics.total_costs}


def plot_variants(dates: pd.DatetimeIndex, results: dict[str, strategy.PortfolioComputation],
                  benchmark_returns: np.ndarray, first: int) -> None:
    """Equity curves and drawdowns of the risk parity variants against equal weights."""
    fig, (ax_e, ax_d) = _plots.figure(2, 1, "Risk parity variants vs equal weights", size=(11, 7),
                                      sharex=True, height_ratios=[2, 1])
    curves = {name: c.net_returns[first:] for name, c in results.items()}
    curves["equal weights"] = benchmark_returns
    for k, (name, r) in enumerate(curves.items()):
        style = dict(color="k", lw=1.6, ls="--") if name == "equal weights" else dict(color=f"C{k}", lw=1.1)
        equity = strategy.equity_curve(r, start=1.0)
        ax_e.plot(dates[first:], equity, label=f"{name} (Sharpe {strategy.compute_performance(r, 252).sharpe_ratio:.2f})",
                  **style)
        ax_d.plot(dates[first:], -strategy.drawdown_series(equity, start=1.0), **style)
    ax_e.set_yscale("log")
    ax_e.set(ylabel="equity (log scale)", title="net of 5 bp transaction costs")
    ax_e.legend(fontsize=8)
    _plots.percent(ax_d)
    ax_d.set(ylabel="drawdown")
    _plots.save(fig, "equity_drawdown")


def plot_allocation(out: frames.PortfolioFrames, vol_target: strategy.PortfolioComputation,
                    dates: pd.DatetimeIndex, tilt: pd.Timestamp) -> None:
    """Weights and risk shares over time (with the 2022 tilt), and the volatility target's leverage."""
    fig, axes = _plots.figure(3, 1, "Allocation through time (frames.backtest, budgets tilted in 2022)",
                              size=(11, 9.5), sharex=True)
    for ax, frame, title in ((axes[0], out.target_weights, "target weights"),
                             (axes[1], out.risk_contributions, "share of volatility")):
        ax.stackplot(frame.index, frame.T.to_numpy(), labels=frame.columns, step="post", alpha=0.9)
        ax.axvline(tilt, color="k", ls="--", lw=1)
        ax.set(title=title, ylim=(0, 1))
        _plots.percent(ax)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[1].annotate("budget tilt", (tilt, 0.97), xytext=(5, 0), textcoords="offset points", va="top")
    ax = axes[2]
    rebalance_dates = dates[vol_target.rebalance_rows]
    ax.step(rebalance_dates, vol_target.leverage, where="post", label="leverage (8% vol target)")
    ax.step(out.leverage.index, out.leverage, where="post", label="leverage (no target)", color="0.5")
    ax.set(title="gross exposure at each rebalance")
    ax.legend()
    _plots.save(fig, "allocation")


def main() -> None:
    returns = load_returns(ASSETS, start="2012-01-01")
    R = returns.to_numpy()
    print(f"{len(returns)} daily returns of {len(ASSETS)} assets, {returns.index[0].date()} to "
          f"{returns.index[-1].date()}")

    # ------------------------------------------------------------------------------
    header("1. Risk parity variants vs equal weights (index-free engine)")
    base = dict(covariance_window=126, rebalance_frequency=21, transaction_cost=0.0005, periods_per_year=252,
                initial_capital=1.0, solver_tolerance=1e-8, solver_max_iter=5000)
    variants = {
        "ERC, sample cov": strategy.RiskBudgetingPortfolioParams(**base),
        "ERC, EWMA cov": strategy.RiskBudgetingPortfolioParams(
            **base, covariance_estimator=strategy.CovarianceEstimator.Exponential, ewma_halflife=30.0),
        # Without expected returns ES is a multiple of volatility (same weights): use the window mean.
        "ERC on ES 97.5% with mu": strategy.RiskBudgetingPortfolioParams(
            **base, risk_measure=strategy.PortfolioRiskMeasure.ExpectedShortfall, confidence_level=0.975,
            use_expected_returns=True),
        "ERC, 8% vol target": strategy.RiskBudgetingPortfolioParams(
            **base, target_volatility=0.08, max_leverage=3.0),
        "ERC, weekly, no drift": strategy.RiskBudgetingPortfolioParams(
            **{**base, "rebalance_frequency": 5}, drift_weights=False),
    }
    rows = {}
    results: dict[str, strategy.PortfolioComputation] = {}
    for name, params in variants.items():
        engine = strategy.RiskBudgetingPortfolio(params)
        c = engine.run(R)
        results[name] = c
        rows[name] = summary(c.metrics)
    # Benchmark: equal weights rebalanced daily, from the same first date as the strategies.
    first = results["ERC, sample cov"].rebalance_rows[0] + 1
    benchmark = strategy.compute_performance(R[first:].mean(axis=1), 252)
    rows["equal weights"] = summary(benchmark)
    print(pd.DataFrame(rows).T.round(3).to_string())
    print(f"(risk budgeting problems solved on {results['ERC, sample cov'].threads_used} threads; num_threads=0 = all cores)")

    c = results["ERC, 8% vol target"]
    p = variants["ERC, 8% vol target"]
    print(f"\nvol-targeted run: {len(c.rebalance_rows)} rebalances, first at "
          f"{returns.index[c.rebalance_rows[0]].date()}, all converged {c.all_converged}")
    print(f"  leverage: min {c.leverage.min():.2f}, median {np.median(c.leverage):.2f}, "
          f"max {c.leverage.max():.2f} (cap {p.max_leverage})")
    print(f"  final equity {c.equity[-1]:.3f}, total PnL {c.pnl.sum():.3f}, worst drawdown {c.drawdown.max():.1%}, "
          f"costs {c.costs.sum():.2%} of equity")
    print(f"  latest target weights: {dict(zip(ASSETS, c.target_weights[-1].round(3).tolist()))}")
    print(f"  latest risk shares:    {dict(zip(ASSETS, c.risk_contributions[-1].round(3).tolist()))}")
    print(f"  arrays: weights {c.weights.shape}, gross {c.gross_returns.shape}, net {c.net_returns.shape}, "
          f"turnover {c.turnover.shape}")

    # Metrics, equity and drawdown helpers on any return series.
    m = strategy.compute_performance(c.net_returns[first:], periods_per_year=252)
    equity = strategy.equity_curve(c.net_returns[first:], start=1.0)
    dd = strategy.drawdown_series(equity, start=1.0)
    print(f"  recomputed: {m.periods} periods, total return {m.total_return:.1%} "
          f"(equity_curve: {equity[-1] - 1:.1%}), max drawdown {m.max_drawdown:.1%} (drawdown_series: {dd.max():.1%}),")
    print(f"  longest drawdown {m.max_drawdown_duration} days, hit rate {m.hit_rate:.1%}, "
          f"best day {m.best_period:.2%}, worst day {m.worst_period:.2%}, "
          f"avg turnover per rebalance {c.metrics.average_turnover:.3f}, rebalances {c.metrics.rebalances}")
    print("  (trading statistics are filled by the engine; compute_performance leaves them at 0: "
          f"{m.average_turnover})")

    # ------------------------------------------------------------------------------
    header("2. Budgets that change over time (tactical tilt from 2022 on)")
    # Rows of `returns` from which each row of budgets applies: index-free engine.
    tilt_row = int(returns.index.searchsorted(pd.Timestamp("2022-01-03")))
    schedule = np.array([[0.15, 0.10, 0.10, 0.10, 0.05, 0.30, 0.20]])   # less duration, more real assets
    engine = strategy.RiskBudgetingPortfolio(variants["ERC, sample cov"])
    tilted = engine.run(R, schedule_rows=[tilt_row], schedule_budgets=schedule)
    after = [k for k, r in enumerate(tilted.rebalance_rows) if r >= tilt_row][0]
    print(f"risk shares before the tilt: {tilted.risk_contributions[after - 1].round(3)}")
    print(f"risk shares after the tilt:  {tilted.risk_contributions[after].round(3)}")
    print(f"engine parameters: window {engine.parameters.covariance_window}, every "
          f"{engine.parameters.rebalance_frequency} days, measure {engine.parameters.risk_measure.name}")
    since = slice(tilt_row, None)
    for name, run in [("static ERC", results["ERC, sample cov"]), ("tilted", tilted)]:
        mm = strategy.compute_performance(run.net_returns[since], 252)
        print(f"  {name:<10} since 2022: return {mm.annualized_return:6.2%}, vol {mm.annualized_volatility:.2%}, "
              f"Sharpe {mm.sharpe_ratio:.2f}")

    # ------------------------------------------------------------------------------
    header("3. The same through pandas: frames.backtest")
    budget_schedule = pd.DataFrame(schedule, index=[pd.Timestamp("2022-01-03")], columns=ASSETS)
    out: frames.PortfolioFrames = frames.backtest(returns, covariance_window=126, rebalance_frequency=21, transaction_cost=0.0005,
                          risk_measure="Volatility", covariance_estimator="Sample",
                          budgets={a: 1.0 for a in ASSETS},            # mapping by asset name (normalized)
                          budget_schedule=budget_schedule)
    print("target weights, last 3 rebalances:\n" + out.target_weights.tail(3).round(3).to_string())
    print("\nrisk contributions around the tilt:\n" +
          out.risk_contributions.loc["2021-11-01":"2022-03-01"].round(3).to_string())
    print("\nperformance (last 3 days):\n" + out.performance.tail(3).round(5).to_string())
    yearly = out.performance["net_returns"].groupby(out.performance.index.year).apply(lambda r: (1 + r).prod() - 1)
    print("\ncalendar-year returns: " + ", ".join(f"{y}: {v:.1%}" for y, v in yearly.items()))
    print(f"\nmetrics: Sharpe {out.metrics['sharpe_ratio']:.2f}, max drawdown {out.metrics['max_drawdown']:.1%}; "
          f"leverage series {len(out.leverage)} points; converged {out.all_converged}")
    eq_weights = returns.iloc[first:].mean(axis=1)
    print(f"frames.performance(equal weights): Sharpe {frames.performance(eq_weights)['sharpe_ratio']:.2f}; "
          f"metrics_dict keys: {len(frames.metrics_dict(out.computation.metrics))}")

    plot_variants(returns.index, results, R[first:].mean(axis=1), first)
    plot_allocation(out, results["ERC, 8% vol target"], returns.index, pd.Timestamp("2022-01-03"))
    _plots.show()


if __name__ == "__main__":
    main()
