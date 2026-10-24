"""Factor replication: recover a known combination of real market series.

We build a "fund" whose daily return is a hidden mix of three real series
(Nasdaq 100, gold, 10-year Treasury futures) plus a small alpha and tracking
noise, then ask nabia which of 14 candidate factors it holds and in what weights.

    fund = 0.0001 + 0.50 NDX + 0.30 gold + 0.20 10y notes + noise (2% a year)

The candidates include close substitutes (S&P 500 vs Nasdaq, silver vs gold,
5y/30y vs 10y bonds), which is where the elastic net and out-of-sample factor
selection matter.

Covers: regression.lambda_max, lambda_grid, elastic_net_path, ElasticNet,
ElasticNetParams, ElasticNetFit, RegressionEngine, RegressionEngineParams,
SubsetSearch, ValidationScheme, InSampleCriterion, SubsetEvaluation,
ValidationCurve, BootstrapSummary, RegressionComputation, frames.regress.

Run from the python/ folder (or in PyCharm, with python/.venv as interpreter):
    .venv/bin/python examples/01_factor_replication.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _plots
from _data import UNIVERSE, header, load_returns
from _plots import plt
from nabia_python_bridge import frames, regression

TRUE_WEIGHTS = {"^NDX": 0.50, "GC=F": 0.30, "ZN=F": 0.20}
TRUE_ALPHA = 0.0001                     # 1 bp a day
TRACKING_NOISE = 0.02 / np.sqrt(252)    # 2% annualized tracking error


def make_fund(returns: pd.DataFrame, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    mix = sum(w * returns[s] for s, w in TRUE_WEIGHTS.items())
    return (TRUE_ALPHA + mix + rng.normal(0.0, TRACKING_NOISE, len(returns))).rename("fund")


def compare(coefficients: pd.Series) -> pd.DataFrame:
    """True vs estimated weights over the union of both factor sets."""
    names = list(dict.fromkeys([*TRUE_WEIGHTS, *coefficients.index]))
    return pd.DataFrame({"true": [TRUE_WEIGHTS.get(n, 0.0) for n in names],
                         "estimated": [coefficients.get(n, 0.0) for n in names]}, index=names).round(4)


def plot_path(path, names: list[str]) -> None:
    """Coefficient paths of the lasso: true factors in colour, the others in grey."""
    lambdas = np.array([f.lambda_ for f in path])
    coefs = np.array([f.coefficients for f in path])
    fig, (ax, ax2) = _plots.figure(1, 2, "Lasso path: which factors enter as the penalty is relaxed")
    for j, name in enumerate(names):
        true = name in TRUE_WEIGHTS
        ax.plot(lambdas, coefs[:, j], lw=1.8 if true else 0.8, color=None if true else "0.65",
                label=f"{name} (true {TRUE_WEIGHTS[name]:.2f})" if true else None, zorder=3 if true else 1)
    for name, w in TRUE_WEIGHTS.items():
        ax.axhline(w, ls=":", lw=0.8, color="0.3")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="lambda (log scale, decreasing)", ylabel="coefficient", title="coefficients (grey: other factors)")
    ax.legend()
    ax2.plot(lambdas, [f.nonzero for f in path], drawstyle="steps-post", label="non-zero coefficients")
    ax2.plot(lambdas, [f.degrees_of_freedom for f in path], label="effective degrees of freedom")
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set(xlabel="lambda (log scale, decreasing)", title="model size")
    ax2.legend()
    _plots.save(fig, "lasso_path")


def plot_selection(r, params, names: list[str], fund: pd.Series) -> None:
    """Subset size, validation curve, bootstrap intervals and the out-of-sample replication."""
    fig, axes = _plots.figure(2, 2, "Out-of-sample factor selection")
    ax = axes[0, 0]
    sizes = [len(e.factors) for e in r.best_by_size]
    ax.plot(sizes, [e.test_r2 for e in r.best_by_size], "o-", label="test R²")
    ax.plot(sizes, [e.train_r2 for e in r.best_by_size], "s--", label="train R²")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}\n" + "\n".join(names[j] for j in e.factors) for n, e in zip(sizes, r.best_by_size)],
                       fontsize=7)
    ax.set(ylabel="R²", title="best subset of each size")
    ax.legend()

    ax = axes[0, 1]
    curve = r.validation_curve
    for a, alpha in enumerate(curve.alphas):
        ax.errorbar(curve.lambdas[a], curve.mse[a], yerr=curve.se[a], capsize=2, lw=1, label=f"alpha = {alpha}")
    ax.plot(r.lambda_, r.best.validation_mse, "k*", ms=12, label="selected")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="lambda", ylabel="validation MSE (log)", title="validation curve of the selected subset")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    boot = r.bootstrap
    x = np.arange(len(r.factor_names))
    ax.bar(x, boot.mean, color="C0", alpha=0.35, label="bootstrap mean")
    ax.errorbar(x, r.coefficients, yerr=[r.coefficients - boot.lower, boot.upper - r.coefficients], fmt="o",
                color="C0", capsize=4, label=f"estimate, {params.confidence_level:.0%} interval")
    ax.plot(x, [TRUE_WEIGHTS.get(n, 0.0) for n in r.factor_names], "rx", ms=10, mew=2, label="true weight")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x, r.factor_names)
    ax.set(ylabel="coefficient", title=f"block bootstrap ({boot.samples} samples)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    dates = fund.index
    test = dates[r.train_rows:]
    ax.plot(test, np.cumsum(fund.to_numpy()[r.train_rows:]), label="fund")
    ax.plot(test, np.cumsum(r.predictions[r.train_rows:]), label="replication", alpha=0.8)
    ax.plot(test, np.cumsum(r.residuals[r.train_rows:]), label="cumulative residual", color="0.4", lw=0.8)
    _plots.percent(ax)
    ax.set(ylabel="cumulative return (sum)", title="out of sample: fund vs replication")
    ax.legend()
    _plots.save(fig, "selection")


def plot_variants(fund: pd.Series, out, ins_bic) -> None:
    """frames.regress predictions, residuals and in-sample BIC selection."""
    fig, axes = _plots.figure(1, 2, "Replication through frames.regress")
    ax = axes[0]
    both = pd.DataFrame({"fund": fund, "replication": out.predictions}).dropna()
    rolling = (both["fund"] - both["replication"]).rolling(63).std() * np.sqrt(252)
    ax.plot(rolling, label="63-day tracking error")
    ax.axhline(TRACKING_NOISE * np.sqrt(252), color="r", ls="--", lw=1, label="true noise (2%)")
    ax.axvline(out.test_start, color="0.3", ls=":", label="test start")
    _plots.percent(ax, decimals=1)
    ax.set(title="rolling tracking error, annualized")
    ax.legend()
    ax = axes[1]
    resid = out.residuals.dropna()
    ax.hist(resid * 1e4, bins=80, density=True, alpha=0.6, label="residuals")
    grid = np.linspace(resid.min(), resid.max(), 200)
    sd = resid.std()
    ax.plot(grid * 1e4, np.exp(-0.5 * (grid / sd) ** 2) / (sd * 1e4 * np.sqrt(2 * np.pi)), "k", lw=1,
            label="normal fit")
    ax.set(xlabel="daily residual (bp)", title=f"residuals; in-sample BIC picks {', '.join(ins_bic)}")
    ax.legend()
    _plots.save(fig, "frames_regress")


def main() -> None:
    returns = load_returns(UNIVERSE, start="2012-01-01")
    fund = make_fund(returns)
    X = returns.to_numpy()
    y = fund.to_numpy()
    names = list(returns.columns)
    print(f"{len(returns)} daily returns, {returns.index[0].date()} to {returns.index[-1].date()}, "
          f"{len(names)} candidate factors")
    print("correlations with the true factors:\n" +
          returns.corr().loc[list(TRUE_WEIGHTS), ["^GSPC", "^NDX", "SI=F", "GC=F", "ZF=F", "ZN=F", "ZB=F"]]
          .round(2).to_string())

    # ------------------------------------------------------------------------------
    header("1. The lasso path: which factors enter first as the penalty is relaxed")
    lmax = regression.lambda_max(X, y, alpha=1.0)
    grid = regression.lambda_grid(lmax, 60, 1e-4)
    path = regression.elastic_net_path(X, y, regression.ElasticNetParams(alpha=1.0), grid)
    entered: list[str] = []
    for fit in path:
        for j in np.flatnonzero(fit.coefficients):
            if names[j] not in entered:
                entered.append(names[j])
                print(f"  lambda = {fit.lambda_:.2e}: {names[j]:<10} enters ({fit.nonzero} active, "
                      f"df = {fit.degrees_of_freedom:.1f})")
    print(f"lambda_max = {lmax:.3e}: above it every coefficient is 0 "
          f"(first fit: {path[0].nonzero} non-zero, intercept = mean(y) = {path[0].intercept:.2e})")
    plot_path(path, names)

    # A single elastic net at a chosen penalty, through the class interface.
    en = regression.ElasticNet(regression.ElasticNetParams(lambda_=grid[30], alpha=0.5, fit_intercept=True,
                                                           standardize=True, tolerance=1e-7, max_iter=100000))
    en.fit(X, y)
    fit: regression.ElasticNetFit = en.result
    print(f"\nElasticNet(lambda={fit.lambda_:.2e}, alpha=0.5): {fit.nonzero} factors, "
          f"{fit.iterations} sweeps, converged={fit.converged}")
    print(f"  in-sample RMSE {np.sqrt(np.mean((y - en.predict(X)) ** 2)):.2e} "
          f"(same through the fit: {np.allclose(en.predict(X), fit.predict(X))})")

    # ------------------------------------------------------------------------------
    header("2. Out-of-sample factor selection (train on the first 70%, score on the last 30%)")
    params = regression.RegressionEngineParams(
        min_factors=1, max_factors=4,                     # subsets of 1..4 factors
        search=regression.SubsetSearch.Exhaustive,        # all 14 + 91 + 364 + 1001 subsets
        train_fraction=0.7,
        alphas=[0.0, 0.5, 1.0], n_lambdas=30, lambda_min_ratio=1e-4,
        validation=regression.ValidationScheme.BlockedKFold, validation_folds=5,
        bootstrap_samples=500, bootstrap_block_size=20,   # 20-day blocks keep the volatility clustering
        confidence_level=0.95, bootstrap_seed=42,
        fit_intercept=True, standardize=True,             # penalize factors on a common scale
    )
    engine = regression.RegressionEngine(params)
    r: regression.RegressionComputation = engine.run(y, X, names)
    print(f"{r.subsets_evaluated} subsets evaluated ({r.search_used.name}) on {r.threads_used} threads; "
          f"train {r.train_rows} rows, test {r.test_rows} rows; all converged: {r.all_converged}")
    print("\nbest subset of each size (lower test MSE is better):")
    for e in r.best_by_size:  # SubsetEvaluation
        print(f"  {len(e.factors)} factors {str([names[j] for j in e.factors]):<42} test R2 {e.test_r2:7.4f}  "
              f"test RMSE {np.sqrt(e.test_mse):.2e}  alpha {e.alpha:.2f}  lambda {e.lambda_:.1e}")
    b: regression.SubsetEvaluation = r.best
    print(f"\nselected: {r.factor_names} (columns {r.factor_positions}), alpha = {r.alpha}, lambda = {r.lambda_:.2e}")
    print(f"  validation MSE {b.validation_mse:.3e} +- {b.validation_se:.1e}, train RMSE {np.sqrt(b.train_mse):.2e}, "
          f"train R2 {b.train_r2:.4f}, "
          f"test R2 {b.test_r2:.4f}, df {b.degrees_of_freedom:.1f}")
    print(f"  intercept {r.intercept * 1e4:.2f} bp/day (true {TRUE_ALPHA * 1e4:.2f})")
    print(compare(pd.Series(r.coefficients, index=r.factor_names)).to_string())

    curve: regression.ValidationCurve = r.validation_curve
    a, l = np.unravel_index(np.argmin(curve.mse), curve.mse.shape)
    print(f"\nvalidation curve: {curve.mse.shape[0]} alphas x {curve.mse.shape[1]} lambdas, "
          f"minimum at alpha = {curve.alphas[a]}, lambda = {curve.lambdas[a, l]:.2e} "
          f"(MSE {curve.mse[a, l]:.3e} +- {curve.se[a, l]:.1e})")

    boot: regression.BootstrapSummary = r.bootstrap
    print(f"\nblock bootstrap ({boot.samples} samples), {params.confidence_level:.0%} intervals:")
    for j, name in enumerate(r.factor_names):
        print(f"  {name:<6} mean {boot.mean[j]:.4f}  se {boot.std_error[j]:.4f}  "
              f"[{boot.lower[j]:.4f}, {boot.upper[j]:.4f}]  selected in {boot.selection_frequency[j]:.0%}")
    print(f"  intercept {boot.intercept_mean * 1e4:.2f} +- {boot.intercept_std_error * 1e4:.2f} bp/day; "
          f"test RMSE {np.sqrt(boot.test_mse_mean):.2e} [{np.sqrt(boot.test_mse_lower):.2e}, "
          f"{np.sqrt(boot.test_mse_upper):.2e}]; raw draws {boot.coefficients.shape}")
    significant = [n for j, n in enumerate(r.factor_names) if boot.lower[j] > 0 or boot.upper[j] < 0]
    print(f"  factors whose interval excludes 0: {significant}"
          + ("" if len(significant) == len(r.factor_names) else
             "  <- the others won on the test set by noise; selecting on the test set is optimistic"))
    if r.alpha == 0.0:
        print("  (alpha = 0 is ridge: coefficients are never exactly 0, so selection frequencies are all 100%)")
    oos = r.residuals[r.train_rows:]
    print(f"  realized tracking error out of sample: {oos.std() * np.sqrt(252):.2%} a year "
          f"(true noise 2.00%)")
    plot_selection(r, params, names, fund)

    # ------------------------------------------------------------------------------
    header("3. Variants: in-sample criteria, forward search, walk-forward validation")
    in_sample = {}
    for criterion in (regression.InSampleCriterion.BIC, regression.InSampleCriterion.AIC,
                      regression.InSampleCriterion.MSE):
        ins = regression.RegressionEngine(
            min_factors=1, max_factors=5, train_fraction=1.0,           # in sample: no test rows
            in_sample_criterion=criterion, alphas=[1.0], n_lambdas=20,
            search=regression.SubsetSearch.Forward).run(y, X, names)
        in_sample[criterion.name] = ins.factor_names
        print(f"  in sample, {criterion.name:<3}: {str(ins.factor_names):<48} "
              f"criterion {ins.best.criterion:12.6g}  ({ins.subsets_evaluated} subsets)")
    print("  (MSE alone always prefers the largest subset; BIC/AIC charge for each extra factor)")

    wf = regression.RegressionEngine(
        max_factors=3, alphas=[0.5, 1.0], n_lambdas=20,
        validation=regression.ValidationScheme.WalkForward, validation_folds=4,
        validation_gap=5,                    # 5 days left out between training and validation blocks
        one_standard_error_rule=True,        # prefer the sparsest model within 1 SE of the best
        refit_on_full_sample=True,           # final coefficients use every row
        search=regression.SubsetSearch.Auto, max_exhaustive_subsets=200,
        num_threads=0).run(y, X, names)
    print(f"  walk-forward + 1 SE + refit ({wf.search_used.name} search): {wf.factor_names} "
          f"-> {np.round(wf.coefficients, 4)}, lambda {wf.lambda_:.2e}")
    print(f"  engine parameters are kept: max_factors = {engine.parameters.max_factors}")

    # ------------------------------------------------------------------------------
    header("4. The same through pandas: frames.regress")
    fund_with_gap = fund.copy()
    fund_with_gap.iloc[100:105] = np.nan            # missing fund NAVs: those dates are dropped
    out: frames.RegressionFrames = frames.regress(fund_with_gap, returns, min_factors=2, max_factors=4, train_fraction=0.7,
                         alphas=[0.5, 1.0], validation="WalkForward", bootstrap_samples=200,
                         bootstrap_block_size=20)
    print(f"selected {list(out.coefficients.index)}, out of sample from {out.test_start.date()}")
    print(compare(out.coefficients).to_string())
    print("\nbootstrap table:\n" + out.bootstrap.round(4).to_string())
    print("\nbest subset by size:\n" + out.best_by_size[["factors", "test_r2", "alpha", "lambda"]].to_string())
    print(f"\npredictions: {len(out.predictions)} dates; last 3:\n"
          + pd.DataFrame({"fund": fund, "replication": out.predictions}).dropna().tail(3).to_string())
    plot_variants(fund, out, in_sample["BIC"])
    _plots.show()


if __name__ == "__main__":
    main()
