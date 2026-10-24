"""Risk measures, risk parity and risk budgeting on a real multi-asset universe.

An equally weighted portfolio of equities, bonds and commodities looks diversified
by capital but not by risk: equities dominate its volatility. Risk parity (equal
risk contribution) and risk budgeting fix that by choosing weights whose risk
contributions match chosen budgets, for volatility, Gaussian VaR or Gaussian
expected shortfall.

Covers: risk.Volatility, ValueAtRisk, ExpectedShortfall (compute_risk,
compute_risk_contributions, compute_marginal_risk_contributions, dim,
covariance_matrix, confidence_level), RiskBudgeting (optimize, result,
parameters, measure), RiskBudgetingParams, RiskBudgetingResult.

Run from the python/ folder (or in PyCharm, with python/.venv as interpreter):
    .venv/bin/python examples/02_risk_parity_and_budgeting.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _plots
from _data import header, load_returns
from nabia_python_bridge import risk

ASSETS = ["^GSPC", "^STOXX50E", "^N225", "ZN=F", "ZB=F", "GC=F", "HG=F"]
CLASSES = {"^GSPC": "equity", "^STOXX50E": "equity", "^N225": "equity", "ZN=F": "bond", "ZB=F": "bond",
           "GC=F": "commodity", "HG=F": "commodity"}


def table(columns: dict[str, np.ndarray]) -> str:
    return (pd.DataFrame(columns, index=ASSETS) * 100).round(2).to_string() + "\n(in %)"


def by_class(x: np.ndarray) -> dict[str, float]:
    return pd.Series(x, index=ASSETS).groupby(CLASSES).sum().round(3).to_dict()


def plot_allocations(vol: risk.Volatility, portfolios: dict[str, np.ndarray], budgets: np.ndarray) -> None:
    """Capital weights next to risk shares (stacked by asset) for each portfolio."""
    fig, (ax_w, ax_r) = _plots.figure(1, 2, "Capital allocation vs risk allocation (volatility)", size=(12, 4.6))
    labels = list(portfolios)
    x = np.arange(len(labels))
    for ax, risk_view in ((ax_w, False), (ax_r, True)):
        bottom = np.zeros(len(labels))
        for i, asset in enumerate(ASSETS):
            values = []
            for w in portfolios.values():
                rc = vol.compute_risk_contributions(w)
                values.append((rc / rc.sum())[i] if risk_view else w[i])
            ax.bar(x, values, bottom=bottom, label=asset, color=f"C{i}", edgecolor="white", lw=0.5)
            bottom += values
        ax.set_xticks(x, labels, rotation=15)
        _plots.percent(ax)
        ax.set(title="share of volatility" if risk_view else "capital weights")
    # Budget targets as markers on the budgeting bar.
    k = labels.index("budget 40/40/20")
    ax_r.scatter(np.full(len(ASSETS), k + 0.45), np.cumsum(budgets), marker="<", color="k", s=25, zorder=5,
                 label="budget (cumulative)")
    ax_r.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    _plots.save(fig, "allocations")


def plot_measures(returns: pd.DataFrame, vol, var, es, weights: dict[str, np.ndarray]) -> None:
    """Correlations, and the risk of each portfolio under the three measures."""
    fig, (ax_c, ax_m) = _plots.figure(1, 2, "Correlations and portfolio risk by measure", size=(12, 4.6))
    corr = returns.corr().to_numpy()
    im = ax_c.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_c.set_xticks(range(len(ASSETS)), ASSETS, rotation=45)
    ax_c.set_yticks(range(len(ASSETS)), ASSETS)
    ax_c.grid(False)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            ax_c.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if abs(corr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax_c, shrink=0.8)
    ax_c.set_title("daily return correlations")
    names = list(weights)
    x = np.arange(len(names))
    for k, (label, measure) in enumerate([("volatility", vol), ("VaR 99%", var), ("ES 97.5% (with mu)", es)]):
        ax_m.bar(x + (k - 1) * 0.27, [measure.compute_risk(w) for w in weights.values()], 0.27, label=label)
    ax_m.set_xticks(x, names, rotation=15)
    _plots.percent(ax_m)
    ax_m.set(title="annual portfolio risk")
    ax_m.legend()
    _plots.save(fig, "measures")


def main() -> None:
    returns = load_returns(ASSETS, start="2022-01-01")
    cov = returns.cov().to_numpy() * 252                 # annualized
    mu = returns.mean().to_numpy() * 252
    print(f"covariance from {len(returns)} daily returns ({returns.index[0].date()} to {returns.index[-1].date()})")
    print("annualized volatility: " + ", ".join(f"{a} {v:.1%}" for a, v in zip(ASSETS, np.sqrt(np.diag(cov)))))

    vol = risk.Volatility(cov)
    var = risk.ValueAtRisk(cov, 0.99)                    # mu = 0
    es = risk.ExpectedShortfall(cov, 0.975, mu)          # with expected returns
    print(f"\nmeasures over {vol.dim} assets; VaR at {var.confidence_level}, ES at {es.confidence_level}; "
          f"covariance stored: {np.allclose(vol.covariance_matrix, cov)}")

    # ------------------------------------------------------------------------------
    header("1. Equal weights: diversified in capital, not in risk")
    w = np.full(len(ASSETS), 1 / len(ASSETS))
    rc = vol.compute_risk_contributions(w)
    print(f"volatility {vol.compute_risk(w):.2%}, 99% VaR {var.compute_risk(w):.2%}, "
          f"97.5% ES {es.compute_risk(w):.2%} (annual)")
    print(table({"weight": w, "marginal": vol.compute_marginal_risk_contributions(w),
                 "share of risk": rc / rc.sum()}))
    print(f"risk by asset class: {by_class(rc / rc.sum())}")
    print(f"Euler: contributions add up to the risk: {rc.sum():.6f} = {vol.compute_risk(w):.6f}")

    # ------------------------------------------------------------------------------
    header("2. Risk parity (equal risk contribution) for each measure")
    weights = {}
    measures: list[tuple[str, risk.IRiskMeasure]] = [("volatility", vol), ("VaR 99%", var), ("ES 97.5%", es)]
    for name, measure in measures:
        rb = risk.RiskBudgeting(risk.RiskBudgetingParams(), measure)   # empty budgets = equal risk
        weights[name] = rb.optimize()
        res: risk.RiskBudgetingResult = rb.result
        print(f"{name:<11} risk {res.risk:.2%}, budget error {res.budget_error:.1e}, "
              f"{res.iterations} iterations, converged {res.converged}, measure {type(rb.measure).__name__}")
    inverse_vol = 1 / np.sqrt(np.diag(cov))
    weights["inverse vol"] = inverse_vol / inverse_vol.sum()
    print(table(weights))
    erc_rc = vol.compute_risk_contributions(weights["volatility"])
    naive_rc = vol.compute_risk_contributions(weights["inverse vol"])
    print(f"share of volatility, max - min: ERC {np.ptp(erc_rc / erc_rc.sum()):.1e}, "
          f"inverse vol {np.ptp(naive_rc / naive_rc.sum()):.3f} (it ignores correlations)")
    print("(Without expected returns, VaR and ES are multiples of volatility, so their risk parity\n"
          " weights match; with mu in the ES, assets with higher expected returns get more weight.)")

    # ------------------------------------------------------------------------------
    header("3. Risk budgeting: 40% equity / 40% bonds / 20% commodities")
    budgets = np.array([0.20, 0.10, 0.10, 0.20, 0.20, 0.10, 0.10])
    params = risk.RiskBudgetingParams(budgets=budgets, tolerance=1e-10, max_iter=5000, memory=20)
    rb = risk.RiskBudgeting(params, vol)
    start = time.perf_counter()
    x = rb.optimize()
    cold = rb.result
    elapsed = time.perf_counter() - start
    print(table({"budget": budgets, "weight": x, "share of risk": cold.relative_contributions}))
    print(f"risk by class: {by_class(cold.relative_contributions)}; weight by class: {by_class(x)}")
    print(f"portfolio volatility {cold.risk:.2%}; contributions sum {cold.risk_contributions.sum():.4%}; "
          f"{cold.iterations} iterations in {elapsed * 1e3:.2f} ms")
    print(f"parameters kept: budgets {rb.parameters.budgets.round(2)}, tolerance {rb.parameters.tolerance}")

    # Warm start: a small budget change re-solved from the previous weights.
    tilted = risk.RiskBudgeting(risk.RiskBudgetingParams(budgets=budgets + np.array([0.02, 0, 0, -0.02, 0, 0, 0])),
                                vol)
    tilted.optimize(initial_guess=x)
    print(f"tilted budgets from the previous solution: {tilted.result.iterations} iterations "
          f"(cold start: {cold.iterations})")
    plot_allocations(vol, {"equal weights": w, "inverse vol": weights["inverse vol"],
                           "risk parity": weights["volatility"], "ERC on ES": weights["ES 97.5%"],
                           "budget 40/40/20": x}, budgets)
    plot_measures(returns, vol, var, es, {"equal weights": w, "inverse vol": weights["inverse vol"],
                                         **{f"ERC {k}": v for k, v in weights.items() if k != "inverse vol"},
                                         "budget 40/40/20": x})

    # ------------------------------------------------------------------------------
    header("4. Errors are Python exceptions")
    for label, call in [
        ("negative budget", lambda: risk.RiskBudgeting(risk.RiskBudgetingParams(budgets=-budgets), vol).optimize()),
        ("wrong size", lambda: vol.compute_risk(np.ones(3))),
        ("non-PSD covariance", lambda: risk.Volatility(np.array([[1.0, 2.0], [2.0, 1.0]]))),
    ]:
        try:
            call()
        except ValueError as error:
            print(f"  {label}: ValueError: {error}")
    _plots.show()


if __name__ == "__main__":
    main()
