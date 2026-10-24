# Examples

The numbered scripts use `nabia_python_bridge`, the C++ library installed from its wheel. Each one
runs on real daily market data (equity indices, Treasury futures, metals and natural gas) and prints
its results. Together they use every function, class, field and enum exported by the bridge;
`tests/bridge/test_examples_coverage.py` checks that. `risk_budgeting_from_market_data.py` uses the
`nabia` package: it downloads data into SQLite, backtests a risk budgeting portfolio and prints the
weights to hold now.

| Script | What it shows |
|---|---|
| `01_factor_replication.py` | A fund built as a hidden mix of real series (0.5 Nasdaq 100 + 0.3 gold + 0.2 10y notes, plus alpha and noise). The elastic net path, out-of-sample selection of 1 to 4 of 14 candidate factors, the (alpha, lambda) validation curve, block bootstrap intervals, in-sample BIC/AIC/MSE, forward search, walk-forward validation, and `frames.regress` |
| `02_risk_parity_and_budgeting.py` | Volatility / VaR / ES and their risk contributions. Equal weights vs risk parity for each measure vs inverse volatility. A 40/40/20 risk budget by asset class, warm starts, and error handling |
| `03_portfolio_backtest.py` | Walk-forward risk parity backtests: sample vs EWMA covariance, ES with expected returns, a volatility target, weekly rebalancing without drift, against equal weights. Time-varying budgets, performance helpers, and `frames.backtest` |
| `04_signals_and_stats.py` | Fat tails and volatility clustering (stats). SMA/EMA, EWMA streaming vs batch, a Butterworth band-pass cycle, and an EWMA crossover trend rule |
| `05_forecasting.py` | Realized volatility with a local level, a custom AR(1)+noise Kalman filter and exponential smoothing. The gold trend with a local linear trend. Kernel algebra, and Gaussian process forecasts of natural gas, all scored on a holdout |
| `06_optimization.py` | L-BFGS max-Sharpe portfolio, constrained mean-variance with the augmented Lagrangian and SQP, GARCH(1,1) maximum likelihood with Nelder-Mead, and EWMA rule calibration with differential evolution |

Run them from `python/` with the project's environment (`scripts/setup_env.sh` installs the wheel,
pandas, matplotlib and yfinance), or in PyCharm with `python/.venv` as the interpreter:

```sh
.venv/bin/python examples/01_factor_replication.py
NABIA_NO_SHOW=1 .venv/bin/python examples/01_factor_replication.py   # save the figures only
.venv/bin/python examples/risk_budgeting_from_market_data.py --db sqlite:///../data/market.db
```

## Plots

Each script saves its figures as PNG in `examples/figures/` (gitignored), named `<script>_<figure>.png`.
When the matplotlib backend is interactive, it also opens them at the end. Set `NABIA_NO_SHOW=1`
(or `MPLBACKEND=Agg`) to only save them. `_plots.py` holds the shared style and helpers.

| Script | Figures |
|---|---|
| `01` | `lasso_path`: coefficient paths against lambda (true factors highlighted) and model size / effective df. `selection`: best test R² by subset size, the validation curve per alpha with its standard errors, bootstrap intervals against the true weights, and cumulative fund vs replication out of sample. `frames_regress`: rolling tracking error against the 2% noise, and the residual distribution |
| `02` | `allocations`: capital weights next to risk shares for equal weights, inverse vol, risk parity, ERC on ES and the 40/40/20 budget (with its targets). `measures`: the correlation matrix, and each portfolio's volatility, VaR and ES |
| `03` | `equity_drawdown`: net equity curves and drawdowns of every variant against equal weights. `allocation`: target weights and risk shares through time around the 2022 budget tilt, and the volatility target's leverage |
| `04` | `stylized_facts`: the return density against the normal, tail frequencies relative to the normal, and autocorrelations of returns vs squared returns. `trend`: SMA/EMA on the price, the band-pass cycle, and the crossover rules against buy & hold |
| `05` | `volatility`: realized volatility, the AR(1) smoothed state, holdout forecasts and standardized innovations. `gold_trend`: smoothed level, forecasts with intervals, and the filtered vs smoothed slope. `gas_gp`: GP holdout forecasts per kernel, the fit and forecast, kernel shapes, and the fitted Gram matrix |
| `06` | `portfolios`: the long-only efficient frontier with the L-BFGS / AL / SQP solutions and their weights. `garch`: GARCH conditional volatility against realized volatility. `differential_evolution`: the Sharpe surface of the crossover rule with each DE strategy's optimum |

Data: `_data.py` downloads closes from Yahoo Finance (`yfinance`) and caches them as CSV in `examples/.data/`,
so later runs are offline. Without a connection or a cache, it reads the shared database
`../data/market.db` (filled by `python -m nabia.marketdata --db sqlite:///../data/market.db`). Delete `.data/` to refresh. The futures series are front-month
continuous contracts, so they jump at rolls.

Numbers change with the data window. The scripts compute their comments from the results
(selected factors, significant coefficients, largest positions) rather than hard-coding them.
