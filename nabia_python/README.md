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
| `nabia.news` | downloads news (RSS, GDELT), stores it as JSON, scores it with an LLM (local Ollama or Claude) |
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
- **Sentiment factor tree** (`TREE` in `config.py`, 85 nodes, sentiment only): a global factor at the root; the
  regions US, EU and APAC, each with macro data, central banks, equity sectors, government bonds, credit and
  currencies; and the commodity complex (energy, precious, industrial metals, agriculture). Each news score also
  feeds the same-class ancestors of its node, x `llm.rollup` (0.5) per level, so detail news (e.g. `US.EQ.TECH`)
  reaches the broader nodes (`US.EQ`).
- **Assets** are not in the tree: the run's JSON lists them (`assets`, replacing the default table), each with its
  name, class, base budget, volatility, Yahoo ticker and the factor nodes it may load on, with a sign, e.g. gold:
  `{"GLOBAL": 1, "CMD": 1, "CMD.PRECIOUS": 1, "CMD.PRECIOUS.GOLD": 1, "US.FX.USD": -1}`. Those nodes' features are
  the asset's candidates, and the elastic net learns the mapping.
- **Synthetic news with a known skill**: each node's tone correlates with its assets' forward returns
  (`synthetic.sentiment_ic`); `--ic-sweep` measures how much of that edge survives the pipeline.

### Real data

```sh
.venv/bin/python -m nabia.sentiment_rb --data real --out output/real                 # real prices + example news, offline
.venv/bin/python -m nabia.sentiment_rb --data real --news my_news.json --out output/real_news
.venv/bin/python -m nabia.sentiment_rb --data real --news none.json                  # indicators only (no news file)
.venv/bin/python -m nabia.sentiment_rb.data.example_news --ic 0.3                    # regenerate the example news
.venv/bin/python -m nabia.sentiment_rb --data real --live --config <(echo '{"llm": {"live_days": 0}}') --batch
```

- **Prices and indicators**: Yahoo Finance (assets: ^GSPC, ^STOXX50E, Brent, gold, 10y note futures, HYG; VIX, MOVE,
  dollar index, a 3-2-1 crack spread) and FRED (high-yield OAS, NFCI, 10y term premium), mapped in `CONFIG["real"]`,
  cached as CSV in `data/cache` (`"data": {"refresh": true}` downloads again). Features without a free source are
  dropped in real mode.
- **News**: a JSON file of articles (`publish_ts`, `received_ts`, `source`, `headline`, `body`), `data.news_file`.
  `data/example_news.json` is a stand-in made from the real assets' forward returns (IC 0.2, it looks ahead on purpose)
  until a news feed exists. Offline, articles are read by a lexicon (nodes by name or alias, tone by word polarity).
- **Live scoring** (`--live`, `pip install -e ".[llm]"`, `ANTHROPIC_API_KEY`): `"llm": {"live_days": 0}` scores every
  day; every response is cached in `data/llm_cache.jsonl`, so an article is paid for once; `--batch` first submits the
  uncached requests as Message Batches (triage, then scoring, then digests).

### Run configurations and the weights script

```sh
.venv/bin/python -m nabia.sentiment_rb --config configs/synthetic.json --out output/syn   # full report, synthetic
.venv/bin/python -m nabia.sentiment_rb --config configs/real.json --out output/real        # full report, real
.venv/bin/python scripts/sentiment_weights.py configs/real.json                            # weights only
```

- `config.py` has three parts: `CONFIG`, the methodology shared by every run (factor tree, assets, features,
  model, budgets, portfolio, LLM); and `SYNTHETIC` and `REAL`, the data settings of each kind of run. A JSON file
  chooses one with `data.source` and overrides any key (`configs/synthetic.json`, `configs/real.json`).
- `scripts/sentiment_weights.py CONFIG.json` computes the signal, budgets, directions and weights over the lookback
  window `backtest.lookback_days`, ending at `backtest.as_of` (null = the last date; a date also cuts the data there).
  The news comes from `data.scores_file` (no model calls), or else from `data.news_file` scored with `llm.provider`
  (`mock` offline, or `ollama` / `anthropic`). It writes `signal.csv`, `rebalances.csv` (date, asset, s, direction,
  budget, weight, risk contribution), `current.csv` (the position to hold) and `summary.json` (the position and the
  window's performance against ERC) to `backtest.out`.

### The staged real-data pipeline

```sh
.venv/bin/python scripts/pipeline/collect_news_1.py configs/real.json      # articles -> data.news_file
.venv/bin/python scripts/pipeline/digest_news_2.py configs/real.json       # LLM daily digests -> data.digests_file
.venv/bin/python scripts/pipeline/score_news_3.py configs/real.json        # LLM article scores -> data.scores_file
.venv/bin/python scripts/pipeline/calibrate_betas_4.py configs/real.json   # elastic-net betas -> data.betas_file
.venv/bin/python scripts/pipeline/backtest_5.py configs/real.json          # portfolio with the stored betas -> backtest.out
```

Each step reads the files of the steps before it, so a step runs once and the later ones reuse its result
(`nabia/sentiment_rb/stages.py`):

1. **Collect** (`collect`: rss, feeds, gkg, since, until, per_day, weekends, workers): new articles, deduplicated,
   appended to `data.news_file`. Run it again, e.g. daily, to add the latest items.
2. **Digest**: the LLM (`llm.provider`: ollama, anthropic, or mock offline) writes one macro digest per day (risk
   appetite, growth, inflation, geopolitics, credit, central banks, commodity supply).
3. **Score**: the LLM routes every article to the factor-tree nodes it concerns and scores its sentiment per node.
   Steps 2 and 3 save their file every 25 items, and cache every response in `llm.cache`, so a run resumes.
4. **Calibrate**: the market data and both sentiment files give each asset its candidate features (from its
   factors). The purged walk-forward elastic net is fitted at every refit date, and the betas are stored per refit
   date and asset (non-zero coefficients, the candidate features, the model settings).
5. **Backtest**: the stored betas map the aggregated sentiment to each asset's S and s, from the day after each
   refit to the next. Budgets and directions follow (`budgets`), and the portfolio is backtested against ERC.
   Writes the report tables and figures and the positions (`backtest`: lookback_days, as_of, out). A calibration
   whose features no longer match the configuration is refused: calibrate again.

### News: download, score, use

```sh
.venv/bin/python -m nabia.news fetch                                         # today's RSS items -> data/news.json (run daily)
.venv/bin/python -m nabia.news fetch --no-rss --gkg --since 2024-01-01      # history from GDELT's raw files, 10/day
.venv/bin/python -m nabia.news stats
.venv/bin/python -m nabia.news score                                       # local Ollama (gemma4:26b) -> data/news_scores.json
.venv/bin/python -m nabia.sentiment_rb --data real --scores data/news_scores.json --out output/real_scored
```

- **Sources**: 9 RSS feeds (MarketWatch, CNBC, FT, Investing.com, OilPrice, Federal Reserve, ECB, Bank of England)
  with `received_ts` = download time (no history: run daily, e.g. from cron); GDELT's raw GKG files for history
  (two 15-minute slices per business day, finance titles only, financial outlets first, `received_ts` = the file's
  timestamp); the GDELT search API (`--gdelt`) rate-limits hard.
- **Store**: `data/news.json`, `{"description", "articles": [{id, publish_ts, received_ts, source, feed, headline, body,
  url}]}` (UTC ISO 8601), deduplicated by URL and by headline on the same day, appended in place.
- **Scores**: `nabia.news score` runs triage, scoring and the daily digest on every article with the LLM (`--provider
  ollama|anthropic`, `--model`), writing `data/news_scores.json`: per article its triage and node scores (sentiment,
  confidence, relevance, novelty, weight), per day its digest, plus the model and prompt versions. Responses are
  cached, so an interrupted run resumes. `--scores` makes the pipeline read this file instead of calling a model.
- **Ollama**: `--llm ollama` (pipeline) or `--provider ollama` (scoring) use a local model with structured output;
  `gemma4:26b` was the fastest good model on this machine (about 2 s per call, one request at a time). Parallel
  requests (`OLLAMA_NUM_PARALLEL` > 1 with `--workers`) made every request several times slower here, so keep 1 worker.

| Folder (`nabia/sentiment_rb/`) | Files | Area |
|---|---|---|
| — | `config.py`, `pipeline.py`, `__main__.py` | settings, `run()`, command line and sweep |
| `data/` | `synthetic.py`, `real.py`, `example_news.py` | synthetic data; real prices, indicators and news files; example news |
| `features/` | `tree.py`, `transforms.py` | factor tree (ancestors, roll-up, orientation) and the assets' factors; point-in-time transforms, candidate features, target |
| `llm/` | `schemas.py`, `client.py`, `news.py`, `lexicon.py`, `batch.py` | schemas; mock or cached Anthropic client; dedup, scoring, node features; offline text scorer; Message Batches |
| `model/` | `elastic_net.py` | elastic net per asset (purged walk-forward CV, positivity, stability), score S and s |
| `portfolio/` | `budgets.py`, `backtest.py` | budgets and directions; backtests, long-short simulator, risk-contribution check |
| `report/` | `tables.py`, `figures.py` | tables and CSV; figures |
| `prompts/` | `triage.md`, `scoring.md`, `digest.md` | prompts (version = hash of the file) |

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
