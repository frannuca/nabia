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
| `features/` | `tree.py`, `transforms.py` | indicator tree; point-in-time transforms, candidate features, target |
| `llm/` | `schemas.py`, `client.py`, `news.py`, `lexicon.py`, `batch.py` | schemas; mock or cached Anthropic client; dedup, scoring, node features; offline text scorer; Message Batches |
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
