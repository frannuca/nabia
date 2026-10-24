import numpy as np
import pandas as pd
import pytest

from nabia_python_bridge import frames, regression


def synthetic(rng, n=300, k=6):
    X = rng.normal(size=(n, k))
    y = 1.0 + 2.0 * X[:, 0] - 1.5 * X[:, 3] + 0.5 * rng.normal(size=n)
    return X, y


def test_elastic_net_lambda_zero_is_ols(rng):
    X, y = synthetic(rng)
    en = regression.ElasticNet(lambda_=0.0, alpha=1.0, tolerance=1e-14)
    en.fit(X, y)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    assert en.result.intercept == pytest.approx(beta[0], abs=1e-6)
    assert en.result.coefficients == pytest.approx(beta[1:], abs=1e-6)
    assert en.predict(X) == pytest.approx(en.result.predict(X))


def test_lambda_path(rng):
    X, y = synthetic(rng)
    lmax = regression.lambda_max(X, y, 1.0)
    grid = regression.lambda_grid(lmax, 10, 1e-3)
    path = regression.elastic_net_path(X, y, regression.ElasticNetParams(alpha=1.0), grid)
    assert len(path) == 10
    assert path[0].nonzero == 0
    assert path[-1].nonzero >= 2  # the true factors enter as lambda decreases
    assert path[-1].coefficients[[0, 3]] == pytest.approx([2.0, -1.5], abs=0.1)


def test_engine_selects_true_factors(rng):
    X, y = synthetic(rng)
    params = regression.RegressionEngineParams(min_factors=1, max_factors=2, alphas=[0.5, 1.0], n_lambdas=20,
                                               bootstrap_samples=100, bootstrap_block_size=5)
    r = regression.RegressionEngine(params).run(y, X, [f"f{j}" for j in range(6)])
    assert r.factor_names == ["f0", "f3"]
    assert r.search_used == regression.SubsetSearch.Exhaustive
    assert r.best.test_r2 > 0.8
    assert r.bootstrap is not None and r.bootstrap.coefficients.shape == (100, 2)
    assert r.validation_curve.mse.shape == (2, 20)
    assert len(r.best_by_size) == 2


def test_engine_in_sample_and_enums(rng):
    X, y = synthetic(rng)
    r = regression.RegressionEngine(max_factors=3, train_fraction=1.0, alphas=[1.0], n_lambdas=10,
                                    in_sample_criterion=regression.InSampleCriterion.BIC,
                                    search=regression.SubsetSearch.Forward).run(y, X)
    assert r.test_rows == 0 and np.isnan(r.best.test_mse)
    assert r.search_used == regression.SubsetSearch.Forward
    with pytest.raises(ValueError):
        regression.RegressionEngine(train_fraction=0.0)


def test_frames_regress_aligns_and_names(rng):
    X, y = synthetic(rng)
    index = pd.date_range("2021-01-01", periods=len(y))
    factors = pd.DataFrame(X, index=index, columns=list("abcdef"))
    target = pd.Series(y, index=index)
    target.iloc[5] = np.nan
    out = frames.regress(target.iloc[:-10], factors, max_factors=2, alphas=[1.0], n_lambdas=15,
                         validation="WalkForward", bootstrap_samples=50)
    assert list(out.coefficients.index) == ["a", "d"]
    assert len(out.predictions) == len(y) - 11
    assert out.test_start in index
    assert list(out.best_by_size.index) == [1, 2]
    assert list(out.bootstrap.index) == ["a", "d"]
    assert out.computation.subsets_evaluated == 6 + 15
