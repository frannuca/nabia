"""Optimizers with Python objectives, on real portfolio and calibration problems.

1. L-BFGS: the maximum Sharpe ratio portfolio (smooth, unconstrained), checked
   against its closed form Sigma^-1 mu.
2. Augmented Lagrangian and SQP: long-only mean-variance allocation with a
   budget constraint and position limits, as linear or box constraints.
3. Nelder-Mead: maximum likelihood GARCH(1,1) volatility model of S&P 500
   returns, in a box (derivative-free, a few parameters).
4. Differential evolution: calibrate the spans of an EWMA crossover rule on
   gold (noisy, non-smooth objective: a global search).

Covers: optimization.LBFGS (objective + gradient, value_and_gradient),
LBFGSParams, LBFGSResult, LBFGSStatus, AugmentedLagrangian(+Options, +Result),
SQP(+Options, +Result), ConvexOptimizationParams, LinearConstraint (num_eq,
num_ineq, is_consistent, is_satisfied), BoxConstraints, to_linear_constraint,
NelderMead(+Params, +Result), NelderMeadStatus, DifferentialEvolution(+Params,
+Result), DEStrategy.

Run from the python/ folder (or in PyCharm, with python/.venv as interpreter):
    .venv/bin/python examples/06_optimization.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _plots
from _data import header, load_prices, load_returns
from nabia_python_bridge import optimization as opt
from nabia_python_bridge import signal, strategy

ASSETS = ["^GSPC", "^STOXX50E", "^N225", "ZN=F", "ZB=F", "GC=F", "HG=F"]


def plot_portfolios(mu, cov, cparams_for, mean_variance_for, max_sharpe: np.ndarray,
                    portfolios: dict[str, np.ndarray]) -> None:
    """Long-only efficient frontier (augmented Lagrangian over risk aversions), the solutions, and their weights."""
    fig, (ax_f, ax_w) = _plots.figure(1, 2, "Mean-variance portfolios", size=(13, 4.8))
    frontier = []
    for gamma in np.geomspace(0.5, 200, 40):
        f, g = mean_variance_for(gamma)
        x = opt.AugmentedLagrangian(cparams_for(), f, g).optimize(np.full(len(mu), 1 / len(mu)))
        frontier.append((np.sqrt(x @ cov @ x), mu @ x))
    frontier = np.array(frontier)
    ax_f.plot(frontier[:, 0], frontier[:, 1], "k-", lw=1.2, label="long-only frontier, w <= 30%")
    ax_f.scatter(np.sqrt(np.diag(cov)), mu, c="0.5", s=20)
    for a, x, y in zip(ASSETS, np.sqrt(np.diag(cov)), mu):
        ax_f.annotate(a, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    markers = {"max Sharpe (L-BFGS, unconstrained)": "*", "mean-variance (AL)": "o", "mean-variance (SQP)": "x",
               "box only (SQP)": "s"}
    for (name, w), m in zip(portfolios.items(), markers.values()):
        ax_f.plot(np.sqrt(w @ cov @ w), mu @ w, m, ms=11 if m == "*" else 8, label=name,
                  mfc="none" if m in "os" else None)
    vols = np.linspace(0, frontier[:, 0].max() * 1.1, 50)
    sr = (mu @ max_sharpe) / np.sqrt(max_sharpe @ cov @ max_sharpe)
    ax_f.plot(vols, sr * vols, "C0:", lw=1, label=f"max Sharpe line ({sr:.2f})")
    _plots.percent(ax_f)
    _plots.percent(ax_f, "x")
    ax_f.set(xlabel="volatility", ylabel="expected return", title="risk and return (in-sample means)")
    ax_f.legend(fontsize=7)
    x = np.arange(len(ASSETS))
    width = 0.8 / len(portfolios)
    for k, (name, w) in enumerate(portfolios.items()):
        ax_w.bar(x + (k - (len(portfolios) - 1) / 2) * width, w, width, label=name)
    ax_w.axhline(0.30, color="r", ls="--", lw=0.8, label="30% cap")
    ax_w.axhline(0, color="k", lw=0.6)
    ax_w.set_xticks(x, ASSETS)
    _plots.percent(ax_w)
    ax_w.set(title="weights (max Sharpe scaled to 100% gross)")
    ax_w.legend(fontsize=7)
    _plots.save(fig, "portfolios")


def plot_garch(dates, r_pct: np.ndarray, theta: np.ndarray) -> None:
    """GARCH(1,1) conditional volatility against 21-day realized volatility."""
    omega, alpha, beta = theta
    s2 = np.empty_like(r_pct)
    s2[0] = r_pct.var()
    for t in range(1, len(r_pct)):
        s2[t] = omega + alpha * r_pct[t - 1] ** 2 + beta * s2[t - 1]
    fig, ax = _plots.figure(1, 1, "GARCH(1,1) fitted by Nelder-Mead: S&P 500", size=(11, 4))
    realized = pd.Series(r_pct, index=dates).rolling(21).std() * np.sqrt(252) / 100
    ax.plot(dates, np.abs(r_pct) * np.sqrt(252) / 100, color="0.8", lw=0.5, label="|daily return|, annualized")
    ax.plot(realized, color="C0", lw=1, label="21-day realized volatility")
    ax.plot(dates, np.sqrt(s2 * 252) / 100, color="C3", lw=1, label="GARCH conditional volatility")
    ax.axhline(np.sqrt(omega / (1 - alpha - beta) * 252) / 100, color="k", ls="--", lw=0.8, label="long-run volatility")
    ax.set_ylim(0, 1.0)
    _plots.percent(ax)
    ax.set(title=f"omega {omega:.4f}, alpha {alpha:.3f}, beta {beta:.3f}")
    ax.legend(fontsize=8)
    _plots.save(fig, "garch")


def plot_de(objective, bounds: opt.BoxConstraints, solutions: dict[str, np.ndarray]) -> None:
    """The in-sample Sharpe surface of the crossover rule, and where each DE strategy ended."""
    fast = np.linspace(bounds.lower_bounds[0], bounds.upper_bounds[0], 40)
    slow = np.linspace(bounds.lower_bounds[1], bounds.upper_bounds[1], 40)
    sharpe = np.array([[-objective(np.array([f, s])) if f < s else np.nan for f in fast] for s in slow])
    fig, ax = _plots.figure(1, 1, "Differential evolution on a noisy objective: EWMA crossover on gold",
                            size=(8, 5.5))
    mesh = ax.pcolormesh(fast, slow, sharpe, cmap="RdYlGn", shading="auto")
    fig.colorbar(mesh, ax=ax, label="in-sample Sharpe (long/short, 2 bp costs)")
    for k, (name, x) in enumerate(solutions.items()):
        ax.plot(*x, "o", ms=9, mfc="none", mew=2, color=f"C{k}", label=name)
    ax.grid(False)
    ax.set(xlabel="fast span (days)", ylabel="slow span (days)", title="grid evaluation (40 x 40) and DE optima")
    ax.legend(fontsize=8, loc="lower right")
    _plots.save(fig, "differential_evolution")


def main() -> None:
    returns = load_returns(ASSETS, start="2015-01-01")
    mu = returns.mean().to_numpy() * 252
    cov = returns.cov().to_numpy() * 252
    n = len(ASSETS)
    print(f"{len(returns)} daily returns of {n} assets; annualized means {dict(zip(ASSETS, mu.round(3).tolist()))}")

    # ------------------------------------------------------------------------------
    header("1. L-BFGS: maximum Sharpe ratio portfolio")

    def neg_sharpe(w):
        return -(mu @ w) / np.sqrt(w @ cov @ w)

    def neg_sharpe_grad(w):
        s = np.sqrt(w @ cov @ w)
        return -(mu / s - (mu @ w) * (cov @ w) / s ** 3)

    params = opt.LBFGSParams(dim=n, tolerance=1e-10, max_iter=500, memory=10, armijo_c1=1e-4, backtrack=0.5,
                             max_line_search=60, flat_tolerance=1e-10, wolfe_c2=0.9)
    solver = opt.LBFGS(params, neg_sharpe, neg_sharpe_grad)
    w = solver.optimize(np.full(n, 1 / n))
    r: opt.LBFGSResult = solver.result
    # Sharpe is scale invariant: scale to 100% gross exposure (the net sum can be near 0).
    w = w / np.abs(w).sum()
    closed = np.linalg.solve(cov, mu)
    closed /= np.abs(closed).sum()
    print(f"status {r.status.name} after {r.iterations} iterations / {r.evaluations} evaluations, "
          f"|grad| {r.gradient_norm:.1e}, Sharpe {-r.objective:.3f}")
    print(f"matches Sigma^-1 mu: {np.allclose(w, closed, atol=1e-6)}; weights {dict(zip(ASSETS, (w.round(3) + 0.0).tolist()))}")
    print(f"(unconstrained: largest long {ASSETS[int(np.argmax(w))]}, largest short {ASSETS[int(np.argmin(w))]}, "
          "from in-sample means; section 2 adds constraints)")

    short = opt.LBFGS(opt.LBFGSParams(dim=n, max_iter=3), neg_sharpe, neg_sharpe_grad)
    short.optimize(np.full(n, 1 / n))
    print(f"with max_iter=3: {short.result.status.name}; did not converge: "
          f"{short.result.status in (opt.LBFGSStatus.MaxIterations, opt.LBFGSStatus.LineSearchFailed)}")
    both = opt.LBFGS(params, lambda x: (neg_sharpe(x), neg_sharpe_grad(x)))   # value_and_gradient style
    both.optimize(np.full(n, 1 / n))
    print(f"value_and_gradient callback: same optimum {np.isclose(both.result.objective, r.objective)}, "
          f"converged {both.result.status == opt.LBFGSStatus.Converged}")

    # ------------------------------------------------------------------------------
    header("2. Constrained mean-variance: sum w = 1, 0 <= w <= 30%")
    risk_aversion = 4.0

    def mean_variance(w):
        return 0.5 * risk_aversion * w @ cov @ w - mu @ w

    def mean_variance_grad(w):
        return risk_aversion * cov @ w - mu

    cap = 0.30
    linear = opt.LinearConstraint(A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                                  A_ineq=np.vstack([np.eye(n), -np.eye(n)]),
                                  b_ineq=np.concatenate([np.full(n, cap), np.zeros(n)]))
    print(f"{linear.num_eq} equality and {linear.num_ineq} inequality rows; consistent for dim {n}: "
          f"{linear.is_consistent(n)}")
    cparams = opt.ConvexOptimizationParams(dim=n, tolerance=1e-9, max_iter=200, constraints=linear)

    al = opt.AugmentedLagrangian(cparams, mean_variance, mean_variance_grad,
                                 opt.AugmentedLagrangianOptions(initial_penalty=10.0, penalty_growth=10.0,
                                                                max_penalty=1e10, required_decrease=0.25,
                                                                initial_inner_tolerance=1e-1, max_inner_iter=1000,
                                                                lbfgs_memory=10))
    t0 = time.perf_counter()
    x_al = al.optimize(np.full(n, 1 / n))
    t_al = time.perf_counter() - t0
    a: opt.AugmentedLagrangianResult = al.result
    sqp = opt.SQP(cparams, mean_variance, mean_variance_grad, opt.SQPOptions(
        qp_tolerance=1e-12, qp_max_iter=200, merit_c1=1e-4, backtrack=0.5, max_line_search=40,
        initial_merit_penalty=1.0, merit_penalty_margin=1e-3))
    t0 = time.perf_counter()
    x_sqp = sqp.optimize(np.full(n, 1 / n))
    t_sqp = time.perf_counter() - t0
    s: opt.SQPResult = sqp.result
    print(f"augmented Lagrangian: converged {a.converged}, {a.outer_iterations} outer / {a.inner_iterations} inner "
          f"iterations, violation {a.constraint_violation:.1e}, stationarity {a.stationarity:.1e}, "
          f"penalty {a.penalty:.0e}, {t_al * 1e3:.1f} ms")
    print(f"SQP:                  converged {s.converged}, {s.iterations} iterations / {s.evaluations} evaluations, "
          f"violation {s.constraint_violation:.1e}, complementarity {s.complementarity:.1e}, {t_sqp * 1e3:.1f} ms")
    print(f"same solution: {np.allclose(x_al, x_sqp, atol=1e-5)}; feasible: {linear.is_satisfied(x_sqp, 1e-7)}")
    print(f"weights: {dict(zip(ASSETS, (x_sqp.round(3) + 0.0).tolist()))}")
    print(f"budget multiplier {s.lambda_eq[0]:+.4f}; binding limits (mu > 0): "
          f"{[ASSETS[i % n] + (' <= 30%' if i < n else ' >= 0') for i in np.flatnonzero(s.mu_ineq > 1e-8)]}")
    print(f"expected return {mu @ x_sqp:.2%}, volatility {np.sqrt(x_sqp @ cov @ x_sqp):.2%}, "
          f"objective {s.objective:.5f} (AL {a.objective:.5f})")

    # Bounds alone as BoxConstraints (no budget constraint: the solver may invest more or less than 100%).
    box = opt.BoxConstraints([0.0] * n, [cap] * n)
    as_linear = opt.to_linear_constraint(box, n)
    print(f"\nbox {box.lower_bounds[0]:.0%}..{box.upper_bounds[0]:.0%} per asset; BoxConstraints -> LinearConstraint with {as_linear.num_ineq} inequality rows; "
          f"to_linear_constraint(None) has {opt.to_linear_constraint(None, n).num_ineq}")
    boxed = opt.SQP(opt.ConvexOptimizationParams(dim=n, tolerance=1e-9, max_iter=200, constraints=box),
                    mean_variance, mean_variance_grad)
    xb = boxed.optimize()
    print(f"box only (no budget): invested {xb.sum():.1%}, weights {(xb.round(3) + 0.0).tolist()}")

    def mean_variance_for(gamma):
        return (lambda w: 0.5 * gamma * w @ cov @ w - mu @ w), (lambda w: gamma * cov @ w - mu)

    plot_portfolios(mu, cov, lambda: opt.ConvexOptimizationParams(dim=n, tolerance=1e-9, max_iter=200,
                                                                   constraints=linear),
                    mean_variance_for, w, {"max Sharpe (L-BFGS, unconstrained)": w, "mean-variance (AL)": x_al,
                                           "mean-variance (SQP)": x_sqp, "box only (SQP)": xb})

    # ------------------------------------------------------------------------------
    header("3. Nelder-Mead: GARCH(1,1) maximum likelihood on S&P 500 returns")
    spx_returns = load_returns(["^GSPC"], start="2015-01-01")["^GSPC"]
    r_spx = spx_returns.to_numpy() * 100                  # in %
    sample_var = r_spx.var()

    def garch_nll(theta):
        omega, alpha, beta = theta
        if alpha + beta >= 0.999:
            return 1e10                                    # stationarity: outside the feasible region
        s2 = np.empty_like(r_spx)
        s2[0] = sample_var
        for t in range(1, len(r_spx)):
            s2[t] = omega + alpha * r_spx[t - 1] ** 2 + beta * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + r_spx ** 2 / s2)

    nm_params = opt.NelderMeadParams(dim=3, bounds=opt.BoxConstraints([1e-6, 0.0, 0.0], [1.0, 0.5, 0.999]),
                                     tolerance_f=1e-10, tolerance_x=1e-8, max_iter=2000, max_evaluations=4000,
                                     initial_step=0.05, adaptive=True)
    nm = opt.NelderMead(nm_params, garch_nll)
    t0 = time.perf_counter()
    omega, alpha, beta = nm.optimize(np.array([0.05, 0.1, 0.85]))
    res: opt.NelderMeadResult = nm.result
    print(f"status {res.status.name} ({res.status == opt.NelderMeadStatus.Converged}), {res.iterations} iterations, {res.evaluations} evaluations, "
          f"{time.perf_counter() - t0:.1f} s; -log L = {res.objective:.1f}")
    long_run = np.sqrt(omega / (1 - alpha - beta) * 252)
    print(f"omega {omega:.4f}, alpha {alpha:.3f}, beta {beta:.3f}: persistence {alpha + beta:.3f}, "
          f"volatility half-life {np.log(0.5) / np.log(alpha + beta):.0f} days, long-run vol {long_run:.1f}%")
    plot_garch(spx_returns.index, r_spx, np.array([omega, alpha, beta]))
    classic = opt.NelderMead(opt.NelderMeadParams(dim=3, bounds=nm_params.bounds, adaptive=False, reflection=1.0,
                                                  expansion=2.0, contraction=0.5, shrink=0.5, max_evaluations=300),
                             garch_nll)
    classic.optimize(np.array([0.05, 0.1, 0.85]))
    c = classic.result
    print(f"classic coefficients, at most 300 evaluations: {c.status.name} after {c.evaluations} "
          f"(budget hit: {c.status in (opt.NelderMeadStatus.MaxIterations, opt.NelderMeadStatus.MaxEvaluations)}), -log L {c.objective:.1f}")

    # ------------------------------------------------------------------------------
    header("4. Differential evolution: EWMA crossover spans for gold (in-sample Sharpe)")
    gold = load_prices(["GC=F"], start="2010-01-01")["GC=F"].to_numpy()
    daily = np.diff(gold) / gold[:-1]
    evaluations = 0

    def neg_sharpe_of_rule(x):
        nonlocal evaluations
        evaluations += 1
        fast_span, slow_span = x
        if fast_span >= slow_span:
            return 10.0
        fast = signal.EWMA(alpha=signal.EWMA.alpha_from_span(fast_span)).apply(gold)
        slow = signal.EWMA(alpha=signal.EWMA.alpha_from_span(slow_span)).apply(gold)
        pos = np.where(fast > slow, 1.0, -1.0)[:-1]       # long/short, decided at the close
        pnl = pos[250:] * daily[250:] - 0.0002 * np.abs(np.diff(pos))[249:]   # 2 bp per unit traded
        return -strategy.compute_performance(pnl, 252).sharpe_ratio

    bounds = opt.BoxConstraints([2.0, 10.0], [100.0, 400.0])
    solutions = {}
    for strat in (opt.DEStrategy.Rand1, opt.DEStrategy.Best1, opt.DEStrategy.CurrentToBest1, opt.DEStrategy.Rand2):
        evaluations = 0
        de_params = opt.DifferentialEvolutionParams(dim=2, bounds=bounds, population_size=20, max_generations=40,
                                                    strategy=strat, mutation_min=0.5, mutation_max=1.0,
                                                    crossover=0.9, tolerance=1e-6, absolute_tolerance=1e-6,
                                                    seed=11, num_threads=4)
        de = opt.DifferentialEvolution(de_params, neg_sharpe_of_rule)
        x = de.optimize()
        d: opt.DifferentialEvolutionResult = de.result
        solutions[f"{strat.name} (Sharpe {-d.objective:.2f})"] = x
        print(f"  {strat.name:<15} spans {x[0]:5.1f}/{x[1]:5.1f}: Sharpe {-d.objective:.2f}, "
              f"{d.generations} generations, {d.evaluations} evaluations on {d.threads_used} threads, "
              f"converged {d.converged}")
    print("  (in-sample optimum of a noisy objective: expect a lower Sharpe out of sample;\n"
          "   threads are safe with a Python objective but share the GIL)")
    again = opt.DifferentialEvolution(opt.DifferentialEvolutionParams(dim=2, bounds=bounds, population_size=20,
                                                                      max_generations=40, seed=11, num_threads=1),
                                      neg_sharpe_of_rule).optimize()
    first = opt.DifferentialEvolution(opt.DifferentialEvolutionParams(dim=2, bounds=bounds, population_size=20,
                                                                      max_generations=40, seed=11, num_threads=4),
                                      neg_sharpe_of_rule).optimize()
    print(f"  same seed, 1 vs 4 threads: identical result {np.array_equal(again, first)}")
    plot_de(neg_sharpe_of_rule, bounds, solutions)
    _plots.show()


if __name__ == "__main__":
    main()
