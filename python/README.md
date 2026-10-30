# News sentiment on economic factors

Collects financial news into a SQLite database, scores every article with an LLM on 27 economic factors, and maps
the portfolio assets to those factors from data. The mapping gives each asset a signed exposure per factor, so news
sentiment can modulate risk budgets.

```
collector/   RSS feeds (today's items) and GDELT GKG files (history since 2020) -> news table
scoring/     one LLM call per article -> news_scoring and news_scores tables (prompt.md is the prompt)
exposures/   FRED series and prices -> factor proxies -> rolling regressions -> asset_exposures table
budgets/     factor news signals x exposures -> asset scores -> risk budgets -> risk budgeting weights
doc/         news_risk_budgeting.tex / .pdf: the documentation, with its figures from the database
datalayer/   the SQLite tables and their reads and writes (NewsDB, MarketDB), and the configuration loader
config/      config.json (database, collection, LLM, exposures) and factors.json (the 27 factors and their proxies)
```

Documentation, with the mathematics of every step: [`doc/news_risk_budgeting.pdf`](doc/news_risk_budgeting.pdf)
(rebuilt with `doc/build.sh`).

## Setup

```sh
cd python
python3 -m venv .venv
.venv/bin/pip install --find-links ../dist -e .   # numpy, pandas and the nabia_python_bridge wheel of ../dist
```

The wheel comes from `cpp/scripts/build_wheels.sh` or `cpp/scripts/fetch_wheels.sh`. `run.sh` does this setup on its
first run. `collector`, `scoring` and `datalayer` use only the standard library. `exposures` needs numpy and pandas,
and `budgets` also needs the wheel. The document needs matplotlib (`.venv/bin/pip install matplotlib`) and LaTeX.

## Running each step

All commands run from `python/` and take an optional configuration file as first argument (default
`config/config.json`). `py` stands for `.venv/bin/python`. Each step reads what the previous one wrote in
`data/market.db`, so the steps can be run one at a time and re-run independently.

| step | command | reads | writes |
|---|---|---|---|
| all | `./run.sh [config.json]` | | every table below |
| 1. collect | `py -m collector` | RSS feeds, GDELT GKG | `news` |
| 2. score | `py -m scoring` | `news` | `news_scoring`, `news_scores` |
| 3. exposures | `py -m exposures` | FRED, `prices`, `news_scores` | `series`, `asset_exposures` |
| 4–6. budgets and weights | `py -m budgets` | `news_scores`, `asset_exposures`, `prices` | `risk_budgets`, `portfolio_weights` |
| document | `doc/build.sh` | the tables above | `doc/news_risk_budgeting.pdf` |

**1. Collect the news.** RSS gives today's items. GKG gives the history and resumes after the last GKG day stored.

```sh
py -m collector                           # RSS + GKG up to today
py -m collector --since 2024-01-01        # GKG from a given day (articles already stored are skipped)
py -m collector --until 2024-06-01        # GKG end day (excluded)
py -m collector --import old/news.json    # only import a news.json file of the former nabia_python project
```

**2. Score the articles.** One LLM call per unscored article (see LLM providers below). A run can be stopped at
any time; the next one resumes.

```sh
py -m scoring --dry-run                   # how many articles are left and their prompt tokens
py -m scoring --limit 100                 # a slice
py -m scoring --since 2026-01-01 --until 2026-02-01
py -m scoring --provider anthropic --workers 8   # through an API (key in ANTHROPIC_API_KEY)
```

**3. Estimate the exposures.** This downloads the FRED proxies (a few seconds), estimates every quarter end since
`exposures.start` and prints the latest map.

```sh
py -m exposures                           # download FRED, estimate, print
py -m exposures --no-download             # reuse the FRED series already stored
```

**4–6. Factor signals, risk budgets and weights.** These are computed every Friday since `budgets.start`. The
command prints the latest budgets, each factor's contribution to the asset scores, and the risk budgeting weights.

```sh
py -m budgets                             # budgets.start to today
py -m budgets --start 2025-01-03 --end 2025-12-26
```

**Check the results** in the database, or from Python with `MarketDB(path).exposures(date)`, `.budgets(date)` and
`.weights(date)`:

```sh
sqlite3 ../data/market.db "SELECT * FROM portfolio_weights WHERE date = (SELECT max(date) FROM portfolio_weights)"
sqlite3 ../data/market.db "SELECT model, prompt, count(*), sum(relevant) FROM news_scoring GROUP BY 1, 2"
```

**Tests and document**

```sh
py -m unittest discover -s tests          # the LLM and the APIs are stubbed
doc/build.sh                              # figures and tables from the database, then the PDF
```

## LLM providers

`llm.provider` in `config/config.json` picks the model; `--provider` overrides it for one run.
- **`ollama`** (the default): a local model (`llm.ollama.model`, `gemma4:26b`) served at `llm.ollama.host`.
- **`openai`**: the Chat Completions API with structured outputs. Set `OPENAI_API_KEY`. `base_url` can point to any
  OpenAI-compatible server.
- **`anthropic`**: the Messages API with forced tool use and a cached system prompt. Set `ANTHROPIC_API_KEY`.

```sh
OPENAI_API_KEY=sk-... python3 -m scoring --provider openai --limit 50
```

For reasoning models that accept only their default temperature, set `llm.temperature` to `null`.

`llm.workers` (or `--workers`) > 1 sends that many requests in parallel. This is useful with the APIs; with Ollama,
set `OLLAMA_NUM_PARALLEL` as well. Rate limits (HTTP 429) are retried after the server's `Retry-After`, and
`llm.requests_per_minute` caps the request rate across workers. To score the backlog through an API:

```sh
.venv/bin/python -m scoring --provider anthropic --dry-run      # articles to score and their input tokens
ANTHROPIC_API_KEY=... .venv/bin/python -m scoring --provider anthropic --workers 8 --limit 200   # a first slice
ANTHROPIC_API_KEY=... .venv/bin/python -m scoring --provider anthropic --workers 8               # the rest
```

Each model's scores are stored separately, so an API run doesn't overwrite the Ollama scores. `exposures` averages
an article's scores across models and prompt versions.

## Factors

`config/factors.json` lists 27 factors in groups: growth, corporate, policy, fiscal, commodity, inflation,
financial, fx and event. For every factor, sentiment > 0 means the factor itself goes up (more growth, more
inflation, tighter policy, more stress, more conflict), whether or not that is good news for markets; `positive`
states it for each factor. Each factor also has:
- `group` and `region` tags, which replace a tree;
- where one exists, `proxy`: its weekly market move. The proxy is a sum of terms `[series, transform, weight]`, for
  example `POLICY_FED` = change of `fred:DGS2` (the 2-year yield) and `TECH_AI` = log return of `^NDX` minus that of
  `^GSPC`.

Editing the factors or `scoring/prompt.md` changes the prompt version, so every article is scored again under the
new version. The old scores stay in the database.

## Asset exposures

`python -m exposures` maps each asset of `exposures.assets` (symbols of the `prices` table) to the factors. At
every quarter end, over the trailing `window_weeks` of weekly returns:
1. **Orthogonalize the proxies** in the order of `factors.json` (QR decomposition). Each proxy becomes its shock net
   of the factors listed before it: the 2-year yield net of equities is a monetary-policy shock, Brent net of
   growth is an energy-supply shock, breakevens net of growth and energy are inflation. The order is the
   identification assumption; change it in the file.
2. **Regress each asset** on the orthogonalized proxies. The exposure is the standardized beta (a partial
   correlation, −1..1), with its t-statistic.
3. **News fit.** Every factor with at least `news_min_weeks` weeks of news in the window is also fitted on its
   weekly news sentiment: the correlation of the asset's returns with it. This fit fills the factors without a
   proxy (EARNINGS, the non-Fed central banks, GEOPOLITICS, TRADE_POLICY…). With `news_override`, it also replaces
   the proxy fit when |t| ≥ `t_min`, which captures what a proxy can't, such as a Fed shock on equities.
4. **Keep, shrink or borrow.** An exposure is kept when |t| ≥ `t_min` and its sign held in a share ≥ `stability`
   of the last `stability_windows` estimates; otherwise it is 0. A kept exposure is shrunk by `group_shrink`
   toward the mean of the asset's exposures in the same group. A factor with neither proxy nor enough news borrows
   `group_fallback` × that group mean (method `group`), or gets 0 (method `none`).

`MarketDB(path).exposures(as_of)` returns the map in force at a date (point in time). The growth proxies are equity
indices, so from the proxies alone an equity index gets no POLICY_FED exposure of its own. The news fit restores it
once enough Fed news is scored.

## Risk budgets

`python -m budgets` computes, at every rebalance date t (`budgets.frequency`, Fridays), using only what was known
before t:
1. **Factor signal.** The confidence-weighted mean sentiment of the articles received before t, each decayed with a
   half-life of `half_life_days`. It is then shrunk by its evidence: signal = level × n / (n + `evidence`), where n
   is the decayed sum of confidences, so a factor with a handful of articles moves little. Range −1..1; > 0 means
   the factor is going up.
2. **Asset score.** S = Σ_f e_f × signal_f / Σ_f |e_f|, with the latest exposure map estimated before t. Range
   −1..1; > 0 means the news is bullish for the asset (e.g. hawkish Fed news × a −0.9 bond exposure gives a
   negative bond score).
3. **Budget.** base × (1 + `strength` × S), clipped to [`min_multiplier`, `max_multiplier`], then normalized to sum
   to 1. `base` is {asset: weight}; null means equal budgets.

The CLI prints the latest budgets and each factor's contribution to every asset's score.
`MarketDB(path).budgets(date)` returns the budgets in force at a date.

## Weights

The weights come from the nabia risk budgeting optimizer (`nabia_python_bridge.risk.RiskBudgeting`, C++, L-BFGS on
the convex problem min_y R(y) − Σ b_a ln y_a). At each rebalance they are the long-only weights, summing to 1, whose
risk contributions equal the budgets. The covariance is that of weekly returns over `cov_window_weeks` (104),
annualized and shrunk toward its diagonal by `shrinkage` (0.1). `risk_measure` is `Volatility`, `ValueAtRisk` or
`ExpectedShortfall`; without expected returns all three give the same weights. Each rebalance starts from the
previous weights.

## Database

The tables are added to `data/market.db`, next to its other tables (`prices` is read for the assets and proxies):

| table | one row per | columns |
|---|---|---|
| `news` | article | url, headline, body, source, published_at, received_at (UTC, point in time) |
| `news_scoring` | article × model × prompt version | relevant, scored_at |
| `news_scores` | article × model × prompt version × factor | sentiment (−1..1), confidence (0..1) |
| `series` | downloaded series × date | value (the FRED proxies, key `fred:<id>`) |
| `asset_exposures` | quarter end × asset × factor | exposure, raw, t_stat, stability, method |
| `risk_budgets` | rebalance date × asset | base, score, multiplier, budget, exposures (as_of of the map used) |
| `portfolio_weights` | rebalance date × asset | budget, weight, risk_contribution, portfolio_risk, converged |

`NewsDB(path).scores(model, prompt, since, until)` returns the factor scores with their article's times: the input
of a later aggregation per factor and day.
