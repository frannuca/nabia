# nabia — Python project

The Python side of nabia. Open this folder in PyCharm: its `.venv` is the interpreter and `pyproject.toml` the
only Python project file. It contains no C++: the library comes from the `nabia_python_bridge` wheel built by the
C++ project (`../cpp`) into `../dist`. The methods are described in [`../doc/nabia.pdf`](../doc/nabia.pdf)
(Parts VI and VII for this project).

| Module | What it does |
|---|---|
| `nabia.sentiment_rb` | the sentiment risk-budgeting pipeline: LLM news sentiment and indicators → elastic net per asset → sentiment score → modulated budgets (long-only) or directions (long-short) → backtest against ERC |
| `nabia.backtest`, `nabia.latest_weights` | risk-budgeting backtests and today's target weights on pandas prices, with the JSON parameters of the command-line tool |
| `nabia.analytics` | returns, correlations, rolling volatilities, trend signals, drawdowns, market regime |
| `nabia.marketdata` | downloads equity indices, bond futures, commodities (Yahoo Finance) and crypto (ccxt) into SQLite / PostgreSQL |
| `nabia_python_bridge` (wheel) | the whole C++ library: filters, forecasters, optimizers, risk measures, risk budgeting, backtests, elastic net, pandas helpers (`frames`) |

## Setup

```sh
../cpp/scripts/build_wheels.sh 3.14   # the wheel, into ../dist (or: scripts/fetch_wheels.sh to download the CI's)
scripts/setup_env.sh                  # .venv: the wheel + nabia (editable) + market data and test extras
.venv/bin/python -m pytest            # bridge tests (tests/bridge), package tests, pipeline tests
```

`setup_env.sh` reinstalls the wheel every time, so run it again after the C++ code changes. `PYTHON=python3.12`
chooses the interpreter of a new `.venv`. On Windows: `scripts\setup_env.ps1`, `scripts\fetch_wheels.ps1`.

## The sentiment risk-budgeting pipeline

```sh
.venv/bin/python -m nabia.sentiment_rb                       # long-only, offline (synthetic data, mocked LLM), ~20 s
.venv/bin/python -m nabia.sentiment_rb --show                # also open the figures
.venv/bin/python -m nabia.sentiment_rb --config my.json      # override any CONFIG key, e.g. {"normalize": {"method": "phi"}}
.venv/bin/python -m nabia.sentiment_rb --config <(echo '{"budgets": {"mode": "long_short"}}')   # long-short
.venv/bin/python -m nabia.sentiment_rb --ic-sweep 0 0.1 0.2 0.3 0.5   # alpha against news quality
.venv/bin/pip install -e ".[llm]" && .venv/bin/python -m nabia.sentiment_rb --live   # Claude (ANTHROPIC_API_KEY)
```

Results go to `output/sentiment_rb` (`--out` to change): CSV tables (benchmark comparison, IC, snapshot, budgets,
signal, selection frequencies, in-sample vs out-of-sample IC), the LLM audit log, and the figures `portfolio.png`,
`budgets.png`, `model.png` (`sweep.png` for a sweep). `nabia-sentiment` is the same command, installed as a script.

- **Long-only**: the normalized sentiment s ∈ [0,1] scales each risk budget, b = b0·(0.5 + s), renormalized and
  capped at ±25% change per rebalance; nabia's backtest engine computes the weights.
- **Long-short**: an asset is shorted when s < `short_below` (long again above `long_above`); it keeps its risk
  budget, its returns change sign, and the optimizer works on D Σ D (D = diag(±1)), weights D w'.
- **Synthetic news with a known skill**: each node's tone correlates with its assets' forward returns
  (`synthetic.sentiment_ic`); `--ic-sweep` measures how much of that edge survives the pipeline.

| Folder (`nabia/sentiment_rb/`) | Files | Area |
|---|---|---|
| — | `config.py`, `pipeline.py`, `__main__.py` | settings, `run()`, command line and sweep |
| `data/` | `synthetic.py` | synthetic market data, indicators and news |
| `features/` | `tree.py`, `transforms.py` | indicator tree; point-in-time transforms, candidate features, target |
| `llm/` | `schemas.py`, `client.py`, `news.py` | prompt output schemas; mock or audited Anthropic client; dedup, scoring, node features |
| `model/` | `elastic_net.py` | elastic net per asset (purged walk-forward CV, positivity, stability), score S and s |
| `portfolio/` | `budgets.py`, `backtest.py` | budgets and directions; backtests, long-short simulator, risk-contribution check |
| `report/` | `tables.py`, `figures.py` | tables and CSV; figures |
| `prompts/` | `triage_v1.md`, `scoring_v1.md`, `digest_v1.md` | versioned prompts |

The specification is [`../doc/sentiment_risk_budgeting_spec.md`](../doc/sentiment_risk_budgeting_spec.md).

## Backtests and market data

```python
import nabia

result = nabia.backtest(prices, {"covariance_window": 120, "rebalance_frequency": 21,
                                 "budgets": {"^GSPC": 0.4, "ZN=F": 0.4, "GC=F": 0.2}, "target_volatility": 0.10})
result.weights, result.performance, result.metrics
nabia.latest_weights(prices, params).weights          # the position to hold from the last date on
```

```sh
.venv/bin/python -m nabia.marketdata --list                                   # the default universe
.venv/bin/python -m nabia.marketdata --db sqlite:///market.db --start 2010-01-01
```
