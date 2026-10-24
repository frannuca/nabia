# nabia — Python project

Research on the nabia C++ library from Python: the `nabia` package, examples on real market data, and
the tests of the library's Python bindings. Open this folder (`python/`) in PyCharm. It contains no C++:
the library comes from the `nabia_python_bridge` wheel built by the C++ project (`../cpp`).

| Module | What it does |
|---|---|
| `nabia_python_bridge` (wheel) | The whole C++ library: filters, forecasters, optimizers, risk measures, risk budgeting, portfolio backtests, elastic net regression, and pandas helpers (`frames`) |
| `nabia.backtest`, `nabia.latest_weights` | Risk budgeting backtests and today's target weights on pandas prices, with the JSON parameters of the command-line tool |
| `nabia.analytics` | Returns, correlations, rolling volatilities, trend signals, drawdowns, market regime and budget projections on price frames |
| `nabia.marketdata` | Downloads equity indices, bond futures, commodities (Yahoo Finance) and crypto (exchanges through ccxt) into SQLite / PostgreSQL |

The agents, their MCP servers and the web app are a separate project, `../webui`, built on this package.

## Setup

```sh
../cpp/scripts/build_wheels.sh 3.14     # once, and after C++ changes: the wheel, into ../dist (needs uv)
scripts/setup_env.sh                    # .venv: the wheel + nabia (editable) + market data, examples and test extras
.venv/bin/python -m pytest              # bridge tests (tests/bridge) and package tests
```

`setup_env.sh` creates `.venv` with `python3` (or `PYTHON=python3.12 scripts/setup_env.sh`) and
reinstalls the wheel every time, so run it again after rebuilding the wheel. In PyCharm, select
`python/.venv` as the project interpreter; run configurations work from this folder.

To step into the C++ code from a Python script, build a wheel with debug symbols
(`NABIA_WHEEL_BUILD_TYPE=RelWithDebInfo ../cpp/scripts/build_wheels.sh 3.14`), run `setup_env.sh`,
and attach CLion's debugger to the Python process.

## The C++ library

```python
import nabia_python_bridge as nb
from nabia_python_bridge import frames

rb = nb.risk.RiskBudgeting(nb.risk.RiskBudgetingParams(budgets=[0.5, 0.3, 0.2]), nb.risk.Volatility(cov))
weights = rb.optimize()
reg = frames.regress(fund_returns, factor_returns, min_factors=1, max_factors=3, train_fraction=0.7)
```

Module list and conventions: [`../cpp/bindings/README.md`](../cpp/bindings/README.md).

## Backtests from pandas

```python
import nabia

result = nabia.backtest(prices, {           # prices: DataFrame, DatetimeIndex x assets; NaN dates are dropped
    "covariance_window": 120, "rebalance_frequency": 21,
    "budgets": {"^GSPC": 0.4, "ZN=F": 0.4, "GC=F": 0.2},
    "target_volatility": 0.10, "transaction_cost": 0.0005, "initial_capital": 1_000_000,
})
result.weights          # weights held each day
result.performance      # gross/net return, costs, turnover, pnl, equity, drawdown
result.metrics          # Sharpe, max drawdown, turnover, ...

nabia.latest_weights(prices, params).weights    # the position to hold from the last date on
nabia.backtest(prices, params, budget_schedule=budgets_by_date)   # time-varying risk budgets
```

Parameters use the same keys as the `nabia_rb_backtest` command line tool of the C++ project
(`cpp/apps/rb_backtest/README.md`): both run the same C++ parser, shipped in the wheel as
`nabia_python_bridge._rb_backtest`.

## Market data

```sh
python -m nabia.marketdata --list                                  # the default universe
python -m nabia.marketdata --db sqlite:///../data/market.db --start 2010-01-01
python -m nabia.marketdata --db sqlite:///../data/market.db --asset-class crypto --crypto-exchange coinbase
```

Downloads are incremental: re-running fetches only recent days (with a 5-day
overlap that picks up revisions). Each asset records its source (`yahoo` or
`crypto`). Moving to PostgreSQL later only changes the URL
(`postgresql+psycopg://user:pass@host/db`, after `.venv/bin/pip install -e ".[postgres]"`).

Caveats:
- Yahoo futures tickers are front-month continuous series. They jump at
  contract rolls, and WTI (`CL=F`) went negative in April 2020, where returns are undefined.
- Crypto trades every day. Aligned with exchange-traded assets, its weekend
  prices are dropped, and returns span the gap.

## Examples

[`examples/`](examples/README.md): six scripts on `nabia_python_bridge` (factor replication, risk parity
and budgeting, backtests, signals, forecasting, optimization), each plotting its results, and an end to
end example on the `nabia` package. Together they use every function the bridge exports
(`tests/bridge/test_examples_coverage.py` checks it).

```sh
.venv/bin/python examples/01_factor_replication.py
NABIA_NO_SHOW=1 .venv/bin/python examples/01_factor_replication.py   # only save the figures
```

## Documentation

[`doc/nabia_python_reference.pdf`](doc/nabia_python_reference.pdf): this project's reference — the
bridge from Python, backtests, analytics with their formulas, market data, and the examples. Its code
excerpts are extracted from the current sources. Rebuild it with `doc/build.sh` (needs `pdflatex`).
The methods of the C++ library are derived in `../cpp/doc/nabia_cpp_reference.pdf`.
