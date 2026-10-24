import json

import numpy as np
import pandas as pd
import pytest

import nabia


def make_prices(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vols = np.array([0.010, 0.006, 0.015, 0.012])
    common = rng.normal(0, 1, (rows, 1))
    shocks = 0.5 * common + rng.normal(0, 1, (rows, 4))
    returns = 0.0002 + vols * shocks
    dates = pd.bdate_range("2021-01-04", periods=rows)
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=["SPX", "ZN", "CL", "GC"])


def test_backtest_returns_pandas_objects_indexed_by_date():
    prices = make_prices()
    res = nabia.backtest(prices, {"covariance_window": 60, "rebalance_frequency": 20, "transaction_cost": 0.0005,
                                  "initial_capital": 1_000_000})
    assert isinstance(res.weights.index, pd.DatetimeIndex)
    assert list(res.weights.columns) == ["SPX", "ZN", "CL", "GC"]
    assert res.weights.index.equals(prices.index[1:])  # one row per return date
    assert len(res.target_weights) == (len(prices) - 1 - 60) // 20 + 1
    assert res.target_weights.index[0] == prices.index[60]  # first rebalance at the close of return row 59
    assert set(res.performance.columns) == {"gross_return", "transaction_cost", "net_return", "turnover", "pnl",
                                            "equity", "drawdown"}
    # Equal risk contributions by default, and accounting identities.
    np.testing.assert_allclose(res.risk_contributions.to_numpy(), 0.25, atol=1e-6)
    np.testing.assert_allclose(res.equity.iloc[-1], 1_000_000 * np.prod(1 + res.net_returns), rtol=1e-12)
    np.testing.assert_allclose(res.pnl.sum(), res.equity.iloc[-1] - 1_000_000, rtol=1e-9)
    assert res.metrics["rebalances"] == len(res.target_weights)
    assert res.metrics["total_costs"] > 0
    assert res.all_converged
    assert res.dropped_dates == 0


def test_budgets_by_name_dates_and_volatility_target():
    prices = make_prices()
    params = {
        "covariance_window": 60,
        "rebalance_frequency": 10,
        "budgets": pd.Series({"GC": 0.1, "CL": 0.2, "ZN": 0.3, "SPX": 0.4}),  # any order
        "start": pd.Timestamp("2021-03-01"),
        "target_volatility": 0.08,
        "max_leverage": 3.0,
    }
    res = nabia.backtest(prices, params)
    assert res.weights.index[0] > pd.Timestamp("2021-03-01")
    expected = pd.Series({"SPX": 0.4, "ZN": 0.3, "CL": 0.2, "GC": 0.1})
    np.testing.assert_allclose(res.risk_contributions.to_numpy(), np.tile(expected.to_numpy(), (len(res.target_weights), 1)), atol=1e-6)
    np.testing.assert_allclose(res.target_weights.sum(axis=1), res.leverage, rtol=1e-12)


def test_missing_prices_drop_their_dates():
    prices = make_prices()
    prices.iloc[[50, 51, 120], 1] = np.nan  # ZN missing on three dates
    res = nabia.backtest(prices, {"covariance_window": 60, "rebalance_frequency": 20})
    assert res.dropped_dates == 3
    assert prices.index[50] not in res.weights.index


def test_latest_weights_match_a_rebalance_on_the_last_date():
    prices = make_prices()
    params = {"covariance_window": 80, "rebalance_frequency": 1, "budgets": [0.4, 0.3, 0.2, 0.1]}
    latest = nabia.latest_weights(prices, params)
    full = nabia.backtest(prices, params)
    assert latest.date == prices.index[-1]
    pd.testing.assert_series_equal(latest.weights, full.target_weights.iloc[-1].rename("weight"), atol=1e-7)
    np.testing.assert_allclose(latest.risk_contributions.to_numpy(), [0.4, 0.3, 0.2, 0.1], atol=1e-6)
    assert latest.converged
    assert latest.weights.sum() == pytest.approx(1.0)


def test_invalid_input_raises():
    prices = make_prices(100)
    with pytest.raises(ValueError, match="unknown key"):
        nabia.backtest(prices, {"rebalance_frequncy": 5})
    with pytest.raises(ValueError, match="covariance_window"):
        nabia.backtest(prices, {"covariance_window": 1})
    with pytest.raises(ValueError, match="budget"):
        nabia.backtest(prices, {"covariance_window": 20, "budgets": {"SPX": 1.0}})
    with pytest.raises(ValueError, match="at least covariance_window"):
        nabia.latest_weights(prices, {"covariance_window": 200})
    with pytest.raises(TypeError):
        nabia.backtest(prices.to_numpy(), {})  # not a DataFrame
    with pytest.raises(ValueError, match="strictly increasing"):
        nabia.backtest(prices.iloc[::-1], {"covariance_window": 20})


def test_same_parameters_as_the_cli_example():
    """The CLI's JSON configuration is accepted unchanged."""
    from pathlib import Path

    # A copy of cpp/apps/rb_backtest/example, the input of the C++ command-line application.
    example = Path(__file__).parent / "data" / "rb_backtest_example"
    params = json.loads((example / "portfolio.json").read_text())
    prices = pd.read_csv(example / "prices.csv", index_col="date", parse_dates=True)
    res = nabia.backtest(prices, params)
    assert res.metrics["rebalances"] == 52  # as printed by nabia_rb_backtest
    assert res.equity.iloc[-1] == pytest.approx(1396739.39, abs=0.01)
