# nabia — C++ project

The C++20 numerical library of nabia, for researching and running **systematic trend strategies**,
with its tests, a command-line application and the Python bindings that build the
`nabia_python_bridge` wheel. Open this folder (`cpp/`) in CLion: it is a self-contained CMake project.

A trend strategy has four stages: it extracts a trend from noisy prices, forecasts
where that trend is heading and how uncertain the forecast is, turns those views
into positions sized by risk, and calibrates its own parameters. nabia provides
the numerical building blocks for each stage, built on Eigen.

| Folder | Contents |
|---|---|
| `include/nabia`, `src` | the library `nabia::nabia` (headers and sources, same layout) |
| `tests` | Catch2 unit tests |
| `apps/rb_backtest` | `nabia_rb_backtest`, the command-line backtest |
| `examples` | C++ usage examples |
| `bindings` | pybind11 bindings and the Python files of the `nabia_python_bridge` package ([README](bindings/README.md)) |
| `pyproject.toml`, `scripts/build_wheels.sh` | the wheel build (scikit-build-core, cibuildwheel) |
| `tools/lldb` | LLDB formatters for Eigen |

## Contents

### Signal extraction — `signal/`
Causal filters (output at time t uses only data up to t, so they can run live)
behind a common `IFilter<T>` interface. Each supports streaming (`process`) and batch (`apply`) use.
- `EWMA` — exponentially weighted moving average, the classic trend filter.
- `ButterworthBandPass` — isolates trend cycles within a chosen frequency band
  (removes the level and the high-frequency noise).
- `moving_average`, `exponential_moving_average`, `diff`.

### Trend forecasting — `forecast/`
Models behind a common `IForecaster<T>` interface: `fit(series)`,
`forecast(horizon)` returning a mean and variance for each step ahead, and prediction intervals.
- `KalmanFilter` — linear Gaussian state-space models. Includes local-level and
  local-linear-trend models for the trend and its slope, missing data, a smoother
  and the likelihood.
- `GaussianProcess` — non-parametric trend and seasonality, with
  squared-exponential, Matérn 5/2, periodic, sum and product kernels and
  maximum-likelihood hyperparameters.
- `SimpleExponentialSmoothing` — level-only baseline.

### Risk and allocation — `risk/`
Gaussian risk measures with per-asset risk contributions, used to size and combine positions.
- `Volatility`, `ValueAtRisk`, `ExpectedShortfall`.
- `RiskBudgeting<Measure>` — long-only weights whose risk contributions match
  given budgets (equal risk contribution by default), for any of the measures above.

### Portfolio construction and backtesting — `strategy/`
- `RiskBudgetingPortfolio` — walk-forward backtest of a long-only risk-budgeting
  portfolio built from a `DataFrame` of asset returns. Settings: the trailing
  covariance window (sample or exponentially weighted), the rebalance frequency
  (every k points), the risk measure and budgets, optional volatility targeting
  with a leverage cap, and proportional transaction costs. Weights drift
  between trades. The risk budgeting problems of all rebalance dates are
  solved in parallel (`num_threads`), with results identical for every thread count.
- Results keep the input time index: weights held each period, target weights
  and risk contributions at each rebalance, gross and net returns, costs,
  turnover, equity, PnL and drawdown.
- `PerformanceMetrics` — total and annualized return, volatility, Sharpe,
  Sortino, maximum drawdown and its duration, Calmar, hit rate, turnover and costs.

### Time series containers — `dataframe/`
`TimeSeries<Index>` and `DataFrame<Index>`: Eigen vectors and matrices with a
strictly increasing time index. The index can be dates (`Date`), timestamps
(`Timestamp`), integers (`Step`) or any ordered type. Supports column access by
name, lookup and slicing by time, column selection, dropping rows with missing
values (to align series), and simple and log returns.

### Optimization — `optimization/`
Solvers for portfolio construction and for calibrating strategy parameters,
behind a common `IOptimizer<T, R>` interface.
- `convex/` — `LBFGS` (unconstrained), `AugmentedLagrangian` and `SQP`
  (linear equality, inequality and box constraints).
- `nongradopt/` — derivative-free `NelderMead` (box-constrained) and
  `DifferentialEvolution` (global search, with the population evaluated in parallel).
  Use them on noisy or non-smooth objectives such as backtest performance.

### Factor regression — `regression/`
- `ElasticNet` — elastic net linear regression (glmnet parametrization: `lambda`
  sets the penalty, `alpha` mixes lasso and ridge) by coordinate descent, with
  standardization, warm-started lambda paths and effective degrees of freedom.
- `RegressionEngine` — regresses a target series on a set of factors and chooses
  the factors, lambda and alpha. It searches subsets of `min_factors..max_factors`
  factors (exhaustive or forward stepwise). For each subset it tunes (alpha, lambda)
  by time-series cross-validation on the training rows (blocked K-fold or walk-forward,
  with an optional gap and the one-standard-error rule). It keeps the subset that
  extrapolates best: lowest MSE on the rows after the first `train_fraction`.
  With `train_fraction = 1` the comparison is in-sample, by BIC, AIC or MSE.
  A moving block bootstrap of the selected model gives coefficient standard errors,
  percentile intervals, selection frequencies and the spread of the out-of-sample
  error. Subsets and bootstrap samples run in parallel, and the results are identical
  for every thread count.
- Derivations (coordinate update, `lambda_max`, degrees of freedom, cross-validation,
  BIC/AIC, block bootstrap) and usage from C++ and Python: chapter "Elastic net factor regression:
  derivations" of [`../doc/nabia.pdf`](../doc/nabia.pdf).

### Statistics — `stats/`
Descriptive statistics, autocorrelation, and normal density, CDF and quantile.

## Python bindings — `bindings/`
`nabia_python_bridge` exposes every area above to Python (numpy arrays, Python
objectives for the optimizers, pandas helpers for the backtest and the regression
engine). It is distributed as a wheel, built into `../dist` for each Python version:

```sh
scripts/build_wheels.sh                 # CPython 3.10-3.14, each wheel tested (needs uv)
scripts/build_wheels.sh 3.14            # one version
```

Python environments install it from there; the bindings' tests are in `bindings/tests`. See [`bindings/README.md`](bindings/README.md).

## Applications

- `nabia_rb_backtest` (`apps/rb_backtest/`) — runs a risk budgeting backtest
  from a CSV of asset prices and a JSON file of portfolio parameters. It writes a
  CSV with the weights, risk contributions, transaction costs, returns, PnL,
  equity and drawdown at every date, plus a metrics summary. See
  `apps/rb_backtest/README.md`.

## Building

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
```

Presets: `debug`, `release`, `releaseWithDebugInfo`, `asan`, and `python-bridge` (a development build of
the bindings, with `cpp/.venv` as the interpreter: numpy, pandas, pytest). Dependencies: Eigen and iir1 (fetched
automatically if not installed), nlohmann/json for the applications, Catch2 for the tests, pybind11 for
the bindings. Turn the applications off with `-DNABIA_BUILD_APPS=OFF`.

## Debugging Eigen values

LLDB (the debugger CLion uses on macOS) shows Eigen objects as raw storage
pointers. `tools/lldb/eigen_lldb.py` makes it show their values instead:

```
(Eigen::MatrixXd) m = 2x3 [[1, 2, 3], [4, 5, 6]]
(Eigen::VectorXd) v = 4 [0.1, 0.2, 0.3, 0.4]
```

It covers `Matrix`, `Array` (dynamic or fixed size, row or column major),
contiguous `Map` views and references to them. Expanding a variable lists its
elements as `[i]` or `[i,j]`, and large matrices are shortened in the one-line
summary. Enable it once by adding the import to `~/.lldbinit`, which both
command-line `lldb` and CLion read:

```sh
echo "command script import $PWD/tools/lldb/eigen_lldb.py" >> ~/.lldbinit
```

## Documentation

The whole project, this library included, is documented in one PDF, [`../doc/nabia.pdf`](../doc/nabia.pdf):
the numerical core with its mathematics (Part III), the full derivations of the forecasters, the optimizers and
the elastic net (Part IV), the application, bindings and wheel packaging (Part V), with code excerpts and
`file:line` pointers extracted from this source tree when it is built (`../doc/build.sh`).
