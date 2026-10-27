"""The sentiment risk-budgeting pipeline: budgets, directions, the long-short simulator and the tree."""

import numpy as np
import pandas as pd
import pytest

from nabia_python_bridge import frames
from nabia.sentiment_rb.config import load_config
from nabia.sentiment_rb.features import ancestors, asset_matrix
from nabia.sentiment_rb.features.tree import base_budgets, rollup
from nabia.sentiment_rb.portfolio import directions, modulate, simulate
from nabia.sentiment_rb.portfolio.backtest import solve, window_cov


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def returns(cfg):
    rng = np.random.default_rng(3)
    vol = np.array([v["vol"] for v in cfg["assets"].values()]) / np.sqrt(252)
    return pd.DataFrame(rng.normal(0.0002, 1, (700, len(vol))) * vol, pd.bdate_range("2021-01-01", periods=700),
                        list(cfg["assets"]))


def test_budgets_positive_normalized_bounded_and_capped(cfg):
    b0 = base_budgets(cfg)
    s_neutral = modulate(cfg, np.full(len(b0), 0.5), b0, None)
    np.testing.assert_allclose(s_neutral, b0)  # s = 0.5 everywhere: the base budgets
    prev = b0
    for s in np.random.default_rng(0).uniform(0, 1, (20, len(b0))):
        b = modulate(cfg, s, b0, prev)
        assert (b > 0).all() and b.sum() == pytest.approx(1)
        assert (np.abs(b / prev - 1) <= cfg["budgets"]["max_change"] + 1e-3).all()
        prev = b


def test_directions_hysteresis(cfg):
    lo, hi = cfg["budgets"]["short_below"], cfg["budgets"]["long_above"]
    d = directions(cfg, np.array([lo - 0.01, 0.9]), np.ones(2))
    np.testing.assert_array_equal(d, [-1, 1])
    d = directions(cfg, np.array([(lo + hi) / 2, (lo + hi) / 2]), np.array([-1.0, 1.0]))
    np.testing.assert_array_equal(d, [-1, 1])  # inside the neutral band the side is kept
    np.testing.assert_array_equal(directions(cfg, np.array([hi + 0.01]), np.array([-1.0])), [1])


def test_short_positions_keep_their_risk_budget(cfg, returns):
    b = base_budgets(cfg)
    d = np.array([1, -1, 1, -1, 1, 1.0])
    w, rc = solve(cfg, window_cov(cfg, returns.values[-252:]), b, d)
    np.testing.assert_allclose(rc, b, atol=1e-6)
    assert (np.sign(w) == d).all()
    # risk contributions of the actual (signed) weights on the original covariance equal the budgets too
    cov = window_cov(cfg, returns.values[-252:])
    np.testing.assert_allclose(w * (cov @ w) / (w @ cov @ w), b, atol=1e-6)


def test_simulator_reproduces_the_nabia_engine_when_all_long(cfg, returns):
    assets, P = list(returns.columns), cfg["portfolio"]
    b0 = base_budgets(cfg, assets)
    engine = frames.backtest(returns, budgets=dict(zip(assets, b0)), **P)
    sched = pd.DataFrame([b0[::-1] / b0[::-1].sum()] * len(engine.target_weights), engine.target_weights.index, assets)
    engine = frames.backtest(returns, budgets=dict(zip(assets, b0)), budget_schedule=sched, **P)
    sim = simulate(cfg, returns, sched, pd.DataFrame(1.0, sched.index, assets))
    np.testing.assert_allclose(sim.performance.equity, engine.performance.equity, atol=1e-8)
    np.testing.assert_allclose(sim.target_weights.values, engine.target_weights.values, atol=1e-7)
    assert sim.metrics["sharpe_ratio"] == pytest.approx(engine.metrics["sharpe_ratio"], abs=1e-9)


def test_assets_take_the_features_of_their_configured_factors_once(cfg):
    dates = pd.bdate_range("2020-01-01", periods=5)
    feats = {fid: (pd.Series(1.0, dates), node, sign) for fid, (node, _, _, sign, _) in cfg["features"].items()}
    for asset, spec in cfg["assets"].items():
        expected = {f for f, (_, n, sign) in feats.items()
                    if n in spec["factors"] and (sign.get(spec["class"], 0) if isinstance(sign, dict) else sign)}
        assert set(asset_matrix(cfg, feats, asset).columns) == expected
    assert asset_matrix(cfg, feats, "GOLD")["USD"].iloc[0] == -1  # a strong dollar is bearish for gold
    assert "USD" not in asset_matrix(cfg, feats, "SPX") and ancestors(cfg, "US.EQ.TECH") == ["US.EQ.TECH", "US.EQ", "US", "GLOBAL"]
    assert not any(v[3] == "asset" for v in cfg["taxonomy"].values())  # the assets are not in the tree
    assert rollup(cfg, "US.EQ.TECH") == [("US.EQ.TECH", 0), ("US.EQ", 1)]  # equity news stops below the US region
    assert rollup(cfg, "APAC.MACRO.CN") == [("APAC.MACRO.CN", 0), ("APAC.MACRO", 1), ("APAC", 2), ("GLOBAL", 3)]


def test_a_config_file_replaces_the_portfolio(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"assets": {"NKY": {"name": "Nikkei 225", "class": "EQ", "budget": 1.0, "vol": 0.2, "ticker": "^N225", '
                    '"factors": {"GLOBAL": 1, "APAC": 1, "APAC.EQ.JP": 1, "APAC.FX.JPY": -1}}}}')
    assert list(load_config(path)["assets"]) == ["NKY"]
    path.write_text('{"assets": {"X": {"name": "x", "class": "EQ", "budget": 1, "vol": 0.2, "factors": {"NOPE": 1}}}}')
    with pytest.raises(ValueError, match="NOPE"):
        load_config(path)
