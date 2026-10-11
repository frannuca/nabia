# nabia

A C++20 numerical library for researching and running **systematic trend strategies**.

A trend strategy has four stages: it extracts a trend from noisy prices, forecasts
where that trend is heading and how uncertain the forecast is, turns those views
into positions sized by risk, and calibrates its own parameters. nabia provides
the numerical building blocks for each stage, built on Eigen.

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

### Statistics — `stats/`
Descriptive statistics, autocorrelation, and normal density, CDF and quantile.

## Python and AI agents — `python/`
pandas interface to the C++ engine (`nabia.backtest`, `nabia.latest_weights`).
A market data downloader for equity indices, bond futures, commodities and crypto,
stored in SQLite (PostgreSQL-ready). Three Claude agents (news analyzer, stats
analyzer, execution) that set the risk budgets through MCP tools, with a backtest
mode for their decisions. See `python/README.md`.

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

Presets: `debug`, `release`, `asan`. Dependencies: Eigen and iir1 (fetched
automatically if not installed), nlohmann/json for the applications, Catch2
for the tests. Turn the applications off with `-DNABIA_BUILD_APPS=OFF`.

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

`doc/` holds LaTeX notes on the theory, proofs and implementation of the
algorithms. Its subfolders mirror the library (`doc/forecast/`,
`doc/optimization/convex/`, ...) and share a style file in `doc/common/`.

[`docs/nabia_methodology.pdf`](docs/nabia_methodology.pdf) is the project-wide
document. It covers the architecture and the methodology of every component:
the C++ engine, market data, the AI agents with news sentiment and relevance,
and the web app. Its source is `doc/project/nabia_methodology.tex`. The code
excerpts in it are extracted from the source at build time by
`doc/project/extract_snippets.py`, so they always match the current code
(`--target nabia_doc_nabia_methodology`).

The PDFs are built with the library whenever `pdflatex` is installed (a TeX
distribution such as TeX Live or MacTeX); turn them off with
`-DNABIA_BUILD_DOCS=OFF`. To build only the documentation:

```sh
cmake --preset debug                              # once, to configure
cmake --build --preset debug --target nabia_docs  # all PDFs
cmake --build --preset debug --target nabia_doc_kalman_filter   # a single one
```

Each PDF is compiled in the matching folder of the build tree (for example
`build/debug/doc/forecast/kalman_filter.pdf`) and copied next to its source
(`doc/forecast/kalman_filter.pdf`), so the current PDFs can be read directly
from the repository. If a document fails to compile, its LaTeX log is in the
build tree (`build/debug/doc/forecast/kalman_filter.log`). To compile one by hand
without CMake:

```sh
cd doc/forecast
TEXINPUTS=../common: pdflatex kalman_filter.tex   # run 2-3 times for references
```
