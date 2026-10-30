# nabia_python_bridge

Python bindings of the whole nabia C++ library (pybind11). Each Python submodule
corresponds to one area of the library:

| Module | Contents |
|---|---|
| `stats` | `mean`, `variance`, `stddev`, `autocorrelation`; `normal_pdf/cdf/quantile` (vectorized) |
| `signal` | `EWMA`, `ButterworthBandPass` (streaming `process` / batch `apply`), `moving_average`, `exponential_moving_average`, `diff` |
| `forecast` | `KalmanFilter` (local level / linear trend or custom), `GaussianProcess` with SE, Matérn 5/2, periodic, sum and product kernels (`k1 + k2`, `k1 * k2`), `SimpleExponentialSmoothing`, `Forecast` |
| `optimization` | `LBFGS`, `AugmentedLagrangian`, `SQP`, `NelderMead`, `DifferentialEvolution` with Python objectives; `BoxConstraints`, `LinearConstraint` |
| `risk` | `Volatility`, `ValueAtRisk`, `ExpectedShortfall`, `RiskBudgeting` |
| `strategy` | `RiskBudgetingPortfolio` backtests, `compute_performance`, `equity_curve`, `drawdown_series` |
| `regression` | `ElasticNet`, `elastic_net_path`, `lambda_max`, `lambda_grid`, `RegressionEngine` (factor subset, lambda and alpha selection, validation, bootstrap) |
| `frames` | pandas interface: `backtest(returns_df, ...)`, `regress(target, factors_df, ...)`, `performance(returns)` |

The `DataFrame`/`TimeSeries` containers are not bound, because pandas does the same job in Python.
Where the C++ engines take time-indexed data, `frames` maps the pandas index to row positions and back.

## Build

The bindings are part of the C++ project (`cpp/`). Their sources are in `cpp/bindings`: the pybind11
C++ in `src/` and the package's Python files (re-exports, the pandas interface `frames`) in
`nabia_python_bridge/`. The package has two extension modules: `_core` (the whole library) and
`_rb_backtest` (the JSON-configured backtest).

```sh
# Wheels for CPython 3.10-3.14 on this machine, each tested installed (needs uv) -> dist/
cpp/scripts/build_wheels.sh                              # or: build_wheels.sh 3.12 3.13
NABIA_WHEEL_BUILD_TYPE=RelWithDebInfo cpp/scripts/build_wheels.sh 3.14   # with debug symbols

# Current interpreter only
python3 -m pip wheel ./cpp --no-deps -w dist

# Windows (PowerShell; Visual Studio 2022 Build Tools, Git, Python): CPython 3.10-3.14, each tested
cpp\scripts\build_wheels.ps1                              # or: -Versions 3.12,3.13

# Linux / macOS / Windows with cibuildwheel (configured in cpp/pyproject.toml; Linux needs Docker)
pipx run cibuildwheel==4.3.4 --platform linux cpp

# Every platform in GitHub Actions; all-wheels = every wheel + the sdist; a v* tag makes a Release
gh workflow run wheels.yml
gh run download <run-id> -n all-wheels -D dist

# Development build for debugging in CLion: modules written next to the .py files
cd cpp && cmake --preset python-bridge && cmake --build --preset python-bridge
PYTHONPATH=cpp/bindings cpp/.venv/bin/python my_script.py
```

Install the wheel from `dist/` with `pip install --no-index --find-links dist nabia_python_bridge`
(`cpp/scripts/fetch_wheels.sh` downloads the wheels built by GitHub Actions into `dist/`).
The tests of the bindings are in `cpp/bindings/tests`; the wheel scripts run them
on every wheel. Every command is explained in the chapter "Wheels, cibuildwheel and GitHub Actions" of
[`doc/nabia.pdf`](../../doc/nabia.pdf).

The wheel needs only the system C++ runtime: nabia, iir1 and Eigen are linked statically, and numpy is
installed by pip. A wheel works only with its CPython version, and on macOS the default oldest supported
version is 13.0 (`MACOSX_DEPLOYMENT_TARGET`).

## Conventions

- Arrays are numpy `float64`: vectors are 1-d, and matrices have one row per observation (time).
- Parameter structs take their fields as keyword arguments, and unknown names raise `AttributeError`:
  `signal.EWMAParams(alpha=0.2)`. Models also accept the fields directly: `signal.EWMA(alpha=0.2)`.
  `lambda` is spelled `lambda_`.
- Enums are Python enums (`regression.SubsetSearch.Forward`). In `frames`, a member name also works (`risk_measure="ValueAtRisk"`).
- C++ `std::invalid_argument` becomes `ValueError`, and "not fitted" errors become `RuntimeError`.
  An exception raised in a Python objective propagates out of `optimize()`.
- Fitting, backtests, regressions and optimizations release the GIL.
  Python objectives take the GIL back for each call, so `DifferentialEvolution(num_threads > 1)` is safe.
  It is only faster if the objective itself releases the GIL.
- Results (`.result`, the returned structs) are copies, so they stay valid after the next run.

## Examples

Usage from Python, the theory behind each module and how the wheel is built are in
[`doc/nabia.pdf`](../../doc/nabia.pdf) (Parts III–V). The snippets below are a quick tour.

```python
import numpy as np
import nabia_python_bridge as nb
from nabia_python_bridge import frames

# Optimization with a Python objective
p = nb.optimization.LBFGSParams(dim=2, tolerance=1e-10)
x = nb.optimization.LBFGS(p, lambda x: ((x - 1) ** 2).sum(), lambda x: 2 * (x - 1)).optimize()

# Gaussian process forecast with a trend + seasonality kernel
k = nb.forecast.SquaredExponentialKernel(1.0, 20.0) + nb.forecast.PeriodicKernel(1.0, 1.0, 7.0)
gp = nb.forecast.GaussianProcess(nb.forecast.GaussianProcessParams(kernel=k))
gp.fit(y)
fc = gp.forecast(10); fc.mean, fc.lower(0.95), fc.upper(0.95)

# Risk budgeting
rb = nb.risk.RiskBudgeting(nb.risk.RiskBudgetingParams(budgets=[0.5, 0.3, 0.2]), nb.risk.Volatility(cov))
weights = rb.optimize()

# Backtest on pandas returns
bt = frames.backtest(returns_df, covariance_window=120, rebalance_frequency=21,
                     budgets={"^GSPC": 0.4, "ZN=F": 0.4, "GC=F": 0.2}, target_volatility=0.10)
bt.weights, bt.performance, bt.metrics

# Factor regression: 1 to 3 factors chosen out of sample (last 30%), with bootstrap intervals
reg = frames.regress(fund_returns, factor_returns, min_factors=1, max_factors=3, train_fraction=0.7,
                     alphas=[0.0, 0.5, 1.0], bootstrap_samples=500, bootstrap_block_size=5)
reg.coefficients, reg.bootstrap, reg.best_by_size, reg.test_start
```
