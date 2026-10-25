import numpy as np
import pandas as pd
import pytest

from nabia_python_bridge import frames, risk, strategy

COV = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.01], [0.0, 0.01, 0.01]])


def test_measures():
    w = np.array([0.3, 0.3, 0.4])
    vol = risk.Volatility(COV)
    assert vol.compute_risk(w) == pytest.approx(np.sqrt(w @ COV @ w))
    assert vol.compute_risk_contributions(w).sum() == pytest.approx(vol.compute_risk(w))
    var = risk.ValueAtRisk(COV, 0.99)
    assert var.compute_risk(w) == pytest.approx(2.326348 * vol.compute_risk(w), rel=1e-5)
    es = risk.ExpectedShortfall(COV, 0.975, np.full(3, 0.001))
    assert es.confidence_level == 0.975 and es.dim == 3
    with pytest.raises(ValueError):
        risk.Volatility(np.array([[1.0, 2.0], [0.0, 1.0]]))  # not symmetric


@pytest.mark.parametrize("measure", [risk.Volatility(COV), risk.ValueAtRisk(COV, 0.99),
                                     risk.ExpectedShortfall(COV, 0.975)])
def test_risk_budgeting_matches_budgets(measure):
    budgets = np.array([0.5, 0.3, 0.2])
    rb = risk.RiskBudgeting(risk.RiskBudgetingParams(budgets=budgets), measure)
    w = rb.optimize()
    assert w.sum() == pytest.approx(1.0)
    assert rb.result.converged
    assert rb.result.relative_contributions == pytest.approx(budgets, abs=1e-6)
    assert isinstance(rb.measure, type(measure))


def test_performance_metrics():
    r = np.array([0.01, -0.02, 0.015, 0.0])
    m = strategy.compute_performance(r, 252)
    assert m.total_return == pytest.approx(np.prod(1 + r) - 1)
    assert strategy.equity_curve(r, 100.0)[-1] == pytest.approx(100 * np.prod(1 + r))
    assert strategy.drawdown_series(np.array([1.0, 0.9, 1.1]), 1.0) == pytest.approx([0.0, 0.1, 0.0])


def make_returns(rng, n=300):
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0003, [0.01, 0.02, 0.005], size=(n, 3)), index=index, columns=["eq", "cmd", "bd"])


def test_portfolio_engine(rng):
    returns = make_returns(rng)
    engine = strategy.RiskBudgetingPortfolio(covariance_window=60, rebalance_frequency=20, transaction_cost=0.0005)
    c = engine.run(returns.to_numpy())
    assert c.weights.shape == (300, 3)
    assert c.rebalance_rows[0] == 59
    assert c.target_weights.shape[0] == len(c.rebalance_rows)
    assert c.all_converged
    assert c.metrics.rebalances == len(c.rebalance_rows)
    with pytest.raises(ValueError):
        strategy.RiskBudgetingPortfolio(covariance_window=1)


def test_frames_backtest(rng):
    returns = make_returns(rng)
    out = frames.backtest(returns, covariance_window=60, rebalance_frequency=20, risk_measure="ExpectedShortfall",
                          budgets={"eq": 0.5, "cmd": 0.25, "bd": 0.25}, target_volatility=0.1)
    assert out.weights.index.equals(returns.index)
    assert list(out.target_weights.columns) == ["eq", "cmd", "bd"]
    assert out.risk_contributions.iloc[0].to_numpy() == pytest.approx([0.5, 0.25, 0.25], abs=1e-6)
    assert out.target_weights.index[0] == returns.index[59]
    assert set(out.performance.columns) >= {"net_returns", "equity", "drawdown"}
    assert "sharpe_ratio" in out.metrics
    assert out.computation.metrics.periods > 0


def test_frames_budget_schedule(rng):
    returns = make_returns(rng)
    change = returns.index[150]
    schedule = pd.DataFrame([[0.6, 0.2, 0.2]], index=[change], columns=["eq", "cmd", "bd"])
    out = frames.backtest(returns, covariance_window=60, rebalance_frequency=10, budget_schedule=schedule)
    rc = out.risk_contributions
    assert rc.loc[rc.index < change].iloc[-1].to_numpy() == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert rc.loc[rc.index >= change].iloc[0].to_numpy() == pytest.approx([0.6, 0.2, 0.2], abs=1e-6)
