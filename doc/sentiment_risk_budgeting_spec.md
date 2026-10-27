# Sentiment-Modulated Risk Budgeting — Implementation Spec

> Purpose: build a pipeline that maps each portfolio underlying (equity, commodity, fixed income) to a set of sentiment indicators, aggregates them via elastic-net betas into one sentiment score per asset, normalizes it to [0,1], and uses it to modulate that asset's risk budget in a risk-budgeting (ERC-style) portfolio optimization. News sentiment is produced by an LLM (Anthropic API).
>
> Audience: Claude Code. Implement in Python 3.11+. Follow the "Implementation plan" and "Acceptance criteria" sections.

---

## 0. Overview

```
raw data ──► point-in-time alignment ──► transforms (stationary, oriented, z-scored)
                                              │
LLM news pipeline ──► node-level daily features ┘
                                              ▼
                     per-asset candidate features (inherited down the indicator tree)
                                              ▼
                     Elastic net per asset (target = forward vol-scaled return)
                                              ▼
                     S_i,t = Σ β_ik z_k,t   (prediction ex-intercept)
                                              ▼
                     s_i,t ∈ [0,1]  (rolling percentile / Φ, optional cross-sectional blend)
                                              ▼
                     b_i = b_i⁰ · (m_min + (m_max − m_min)·s_i), renormalized
                                              ▼
                     Risk-budgeting optimizer: min ½wᵀΣw − Σ b_i ln w_i
```

---

## 1. Sentiment factor tree (taxonomy)

The taxonomy is a tree of sentiment factors only; the portfolio's assets are not part of it. Each asset is configured in the run's JSON file with the factor nodes it may load on (each with a sign): the indicators and LLM features of those nodes are its **candidate features**, and the elastic net selects and weights them per asset.

Node ID convention: the path below the root. `GLOBAL`; the regions `US`, `EU`, `APAC` and the commodity complex `CMD`; per region the branches `<REGION>.MACRO`, `.CB` (central banks), `.EQ`, `.RATES`, `.CREDIT`, `.FX`; then detail nodes such as `US.EQ.TECH`, `APAC.FX.JPY`, `CMD.ENERGY.CRUDE`.

### Global factor (`GLOBAL`)
- VIX, MOVE, FX implied vol (CVIX)
- Credit spreads: US HY/IG OAS, iTraxx Main / Crossover, CDX IG/HY
- Financial conditions: Chicago Fed NFCI, GS FCI (if available)
- Economic surprise indices (e.g. Citi ESI)
- Surveys: AAII bull–bear spread, BofA Fund Manager Survey (monthly)
- USD index, gold/copper ratio (risk-on/off proxies)
- **LLM**: risk_appetite, growth_tone, inflation_tone, geopolitical_stress, credit_tone, central-bank hawk/dove (`cb_tone`) — from Prompt C

### Regions and the commodity complex

| Equity (`<REGION>.EQ`) | Commodity (`CMD`) | Rates and credit (`<REGION>.RATES`, `<REGION>.CREDIT`) |
|---|---|---|
| VIX term structure (VIX9D/VIX/VIX3M) | CFTC COT managed-money net (z-score) | MOVE, swaption vol |
| SKEW index, 25Δ risk reversals | Curve shape / roll yield (backwardation vs contango) | Term premium (ACM) |
| Equity put/call ratio | OVX, GVZ | CFTC Treasury-futures positioning |
| Breadth (% above 200-day MA) | ETF holdings (GLD, USO, …) | JPM Treasury client survey |
| Equity fund flows (EPFR) | China PMI, inventory surprises (EIA, LME) | IG/HY fund flows, new-issue concessions |
| **LLM**: equity market tone | **LLM**: commodity_supply_tone | **LLM**: rates tone, credit tone |

### Sectors
- Equity: sectors per region (GICS or ICB). Sector ETF flows, sector news tone (LLM).
- Commodity: energy / base metals / precious metals / agriculture. Crack spreads, specific inventories, complex news tone (LLM).
- Rates and credit: govt curve segment (short / long dated), IG & HY, EM sovereign. Curve slopes, breakevens, swap spreads, segment news tone (LLM).

### Single markets and names (detail nodes)
- Single stocks: LLM name-level news tone + event flags, analyst revision breadth, earnings-call tone, short interest, insider buying, single-name option skew.
- Issuer bonds: issuer CDS, CDS–bond basis, equity-implied credit signal, rating-outlook changes, LLM issuer tone.
- Commodities: contract-specific news tone (LLM), positioning.

### LLM features per node (all levels)
For each node produce 5 daily features:
- `llm_level` — decay-weighted mean sentiment
- `llm_volume` — log(1 + article count) (attention)
- `llm_dispersion` — std of article scores (disagreement)
- `llm_mom5` — 5-day change of `llm_level`
- `llm_surprise` — `llm_level` − 60-day mean of `llm_level`

---

## 2. Quantitative pipeline

### 2.1 Point-in-time alignment
- Lag every series by its publication delay (COT: released Friday for Tuesday data; surveys monthly; macro data on release date; LLM features by article **received** timestamp).
- Store `as_of` and `available_at` for every observation. Never use a value before `available_at`.

### 2.2 Transforms
- Make stationary: levels → changes or rolling z-scores (window configurable, default 756 trading days ≈ 3y; min 252). Clip z at ±3.
- **Orient** every feature so higher = more bullish / risk-on for the asset's price (store a `sign` per feature-asset mapping in config). Note: risk-off is bullish for govt bonds.

### 2.3 Elastic net per asset
- Target: forward h-day vol-scaled return `y_t = r_{t→t+h} / σ_t` (h configurable, default 21). Alternative target (config): forward realized vol.
- `ElasticNetCV` with `l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`, standardized features.
- CV: `TimeSeriesSplit(n_splits=5, gap=h)` — purge/embargo overlapping targets.
- `positive=True` (features are pre-oriented → prevents nonsensical sign flips). Configurable.
- Refit on rolling or expanding window, monthly or quarterly (config).
- Optional stability selection: bootstrap B fits (block bootstrap), keep features selected in > 60% of runs.
- If an asset's data is too sparse: pooled regression over the assets of a region or branch with shared betas (config flag).

### 2.4 Aggregation
`S_i,t = Σ_k β_ik · z_k,t` (prediction minus intercept). All-zero betas ⇒ S = 0 ⇒ neutral.

### 2.5 Normalization to [0,1]
- Default: rolling empirical percentile of S over lookback (default 756).
- Alternative: `Φ(S / σ_S)` with rolling σ.
- Optional cross-sectional rank across assets at each date; final `s = α·s_ts + (1−α)·s_xs` (α config, default 1.0).
- **Do not** use full-sample min-max (lookahead + outlier-sensitive).
- Smooth with EWMA (half-life config, default 5 days).

### 2.6 Risk-budget modulation
- `b_i = b_i⁰ · (m_min + (m_max − m_min) · s_i)`, default `m_min=0.5`, `m_max=1.5`; renormalize Σb = 1.
- Never allow b_i = 0 (log-barrier requires b > 0).
- Cap per-rebalance change of b_i (default ±25% relative).

### 2.7 Optimizer (Spinu / Roncalli risk budgeting)
`min_w ½ wᵀΣw − Σ b_i ln w_i`, `w > 0`, then `w ← w / Σw`.
- Σ: EWMA or Ledoit-Wolf shrinkage covariance (config).
- Verify risk contributions `RC_i = w_i (Σw)_i / wᵀΣw ≈ b_i` (tolerance 1e-4).

### 2.8 Reference sketch

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

def fit_asset(X, y, h):
    cv = TimeSeriesSplit(n_splits=5, gap=h)
    sc = StandardScaler().fit(X)
    m = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9], cv=cv, positive=True).fit(sc.transform(X), y)
    return sc, m

def sentiment_score(sc, m, X_hist, x_now, lookback=756):
    S_hist = m.predict(sc.transform(X_hist[-lookback:])) - m.intercept_
    S_now = m.predict(sc.transform(x_now[None]))[0] - m.intercept_
    return (S_hist < S_now).mean()          # rolling percentile in [0,1]

def budgets(s, b0, lo=0.5, hi=1.5):
    b = b0 * (lo + (hi - lo) * np.asarray(s))
    return b / b.sum()

def risk_budget_weights(Sigma, b):
    f = lambda w: 0.5 * w @ Sigma @ w - b @ np.log(w)
    g = lambda w: Sigma @ w - b / w
    w = minimize(f, np.ones(len(b)) / len(b), jac=g,
                 bounds=[(1e-10, None)] * len(b), method="L-BFGS-B").x
    return w / w.sum()
```

---

## 3. LLM news-sentiment pipeline

### 3.1 Steps
1. **Ingest** news (newswire / RSS / GDELT / vendor). Store `publish_ts`, `received_ts`, `source`, `headline`, `body`, `url`.
2. **Deduplicate**: embeddings, cosine > 0.9 within 48h = same story cluster. `novelty = 1` for first article in cluster, decaying (e.g. `1/(1+n_prior)`) for later ones.
3. **Triage** (Prompt A, small/cheap model): relevance + mapping to taxonomy nodes.
4. **Score** (Prompt B, stronger model): sentiment per (article, node) — only for relevant articles.
5. **Daily macro digest** (Prompt C): global / central-bank / commodity-supply tone from the day's deduplicated headlines.
6. **Aggregate** per node and day:
   `s_node,t = Σ_a w_a s_a e^{−λ(t − t_a)} / Σ_a w_a e^{−λ(t − t_a)}`,
   `w_a = relevance × confidence × novelty × source_weight`. λ from half-life config (default 3 days).
7. Emit the 5 LLM features per node (§1) as standard time series, z-scored like all other indicators.

### 3.2 Engineering requirements
- Anthropic Python SDK; model IDs in config (`triage_model`, `scoring_model`, `digest_model`) — do not hardcode.
- Temperature 0; structured JSON output (tool use / JSON schema); validate with pydantic; retry on schema failure (max 2) then log + skip.
- Prompt caching for system prompts; Message Batches API for backfills.
- Persist raw request/response + model ID + prompt version hash for every call (audit & re-scoring).
- Rate limiting + exponential backoff.
- Prompts stored as versioned files under `prompts/` (not inline strings).

### 3.3 Lookahead bias (critical)
LLMs know what happened after historical articles. Mitigations (implement as config options):
- Restrict backtests to news after the model's training cutoff, **or**
- Anonymize historical articles (entity names → "Company A", dates → relative) before scoring, keeping a mapping to node IDs outside the LLM.
- Treat forward out-of-sample performance as the true validation.

---

## 4. Prompts

### Prompt A — Triage & mapping (small model)

**System**
```
You are a financial news router for a multi-asset portfolio. Your job is to decide
whether a news item carries information relevant to any node of the asset taxonomy
provided, and to map it to those nodes. You do not score sentiment.

Rules:
- Use only the article text. Do not use knowledge of events after the article's timestamp.
- Map to the most specific node(s) justified by the text. Add a parent node only if the
  article has a broader macro/sector implication of its own.
- channel = "direct" if the node's asset is the subject of the article; "spillover" if it is
  affected via sector, supply chain, competitor, or macro linkage.
- Ignore pure market-recap articles that only describe price moves without new information
  (set relevant=false, reason="price_recap").
- Respond with JSON only, matching the schema.
```

**User**
```
<timestamp>{publish_ts_utc}</timestamp>
<source>{source_name}</source>
<headline>{headline}</headline>
<body>{body_truncated_to_N_tokens}</body>

<taxonomy>
{node_id | name | asset_class | aliases/tickers/issuers  — one per line}
</taxonomy>

Schema:
{
  "relevant": bool,
  "reason": "new_information" | "price_recap" | "opinion_only" | "irrelevant",
  "event_type": "earnings" | "guidance" | "m&a" | "rating_action" | "default_restructuring" |
                "litigation_regulatory" | "management" | "central_bank" | "macro_data" |
                "fiscal_policy" | "geopolitics" | "supply_disruption" | "inventory" |
                "opec_policy" | "weather" | "flows_positioning" | "other",
  "nodes": [ { "node_id": str, "relevance": float 0-1, "channel": "direct" | "spillover" } ]
}
```

### Prompt B — Sentiment scoring per node (stronger model)

**System**
```
You are a buy-side analyst scoring news for a quantitative sentiment model. For each
taxonomy node given, judge the directional implication of THIS article for the PRICE of
that node's asset over the next 1–20 trading days.

Scale (sentiment):
 +1.0  major unambiguously positive new information (e.g. large guidance raise, supply shock
       removing significant output for a commodity, surprise dovish pivot for govt bonds)
 +0.5  clearly positive, moderate magnitude
 +0.2  mildly positive / incremental
  0.0  neutral, mixed, or no directional content
 -0.2 / -0.5 / -1.0  symmetric negatives

Asset-class conventions (direction is always w.r.t. the asset's PRICE):
- Equity: earnings/guidance vs stated expectations, margins, demand, M&A (target positive;
  acquirer depends on price/financing), litigation, regulation, management credibility.
- Commodity: supply down / demand up = positive. Inventory draws positive, builds negative.
  Producer-cartel cuts positive. Demand-side macro weakness negative.
- Government bonds: hawkish policy, upside inflation, strong growth, heavy issuance =
  NEGATIVE (yields up). Dovish, disinflation, risk-off / flight-to-quality = POSITIVE.
- Credit (IG/HY, issuer bonds): issuer credit quality and default risk. Deleveraging,
  upgrades, refinancing success positive; leverage-increasing M&A, downgrades, covenant
  stress, liquidity problems negative. Equity-friendly buybacks may be credit-NEGATIVE.

Rules:
- Use only the article. Do not use knowledge of what happened after the timestamp.
- Score the NEW information relative to expectations stated in the article (consensus,
  guidance, prior). An in-line result is ~0 even if the absolute number is good.
- Do not treat reported price moves as sentiment; they are reactions, not information.
- Rumours, unnamed sources, opinion pieces: keep the direction but lower confidence.
- "spillover" nodes usually deserve smaller magnitudes than "direct" nodes.
- Respond with JSON only, matching the schema.
```

**User**
```
<timestamp>{publish_ts_utc}</timestamp>
<source>{source_name}</source>
<headline>{headline}</headline>
<body>{body}</body>

<nodes>
{node_id | name | asset_class | channel from Prompt A}
</nodes>

Schema:
{
  "scores": [
    {
      "node_id": str,
      "sentiment": float -1..1,
      "confidence": float 0-1
    }
  ]
}
```

### Prompt C — Daily macro & central-bank digest (global and regions)

**System**
```
You are a macro strategist. From the batch of headlines/summaries for one day, produce
scores of the prevailing news tone. Scores reflect the content of this day's news only,
not your view of markets. Use no knowledge beyond the latest timestamp in the batch.

Definitions (-1..+1):
- risk_appetite: +1 strongly risk-on news flow, -1 strongly risk-off.
- growth_tone: +1 upside growth news, -1 recession/contraction news.
- inflation_tone: +1 upside inflation pressure, -1 disinflation.
- geopolitical_stress: 0 calm, +1 severe escalation (one-sided scale 0..1).
- cb_tone per central bank (Fed, ECB, BoE, SNB, BoJ, PBoC): +1 hawkish, -1 dovish,
  null if no relevant news.
- credit_tone: +1 easing credit conditions / strong issuance, -1 stress/defaults.
- commodity_supply_tone (energy, metals, ags): +1 supply tightening, -1 loosening.
Give key_drivers as up to 5 short phrases. Respond with JSON only.
```

**User**
```
<date>{yyyy-mm-dd}</date>
<items>
{ts | source | headline | 1–2 sentence summary}   (deduplicated, max ~300 items)
</items>

Schema:
{
  "risk_appetite": float, "growth_tone": float, "inflation_tone": float,
  "geopolitical_stress": float, "credit_tone": float,
  "cb_tone": {"FED": float|null, "ECB": float|null, "BOE": float|null,
              "SNB": float|null, "BOJ": float|null, "PBOC": float|null},
  "commodity_supply_tone": {"energy": float|null, "metals": float|null, "ags": float|null},
  "item_count_used": int,
  "key_drivers": [str]
}
```

Orientation note: when mapping Prompt C outputs to assets, apply per-asset signs in config (e.g. `cb_tone.FED` hawkish → negative for US Treasuries, `inflation_tone` → positive for commodities/breakevens).

---

## 5. Implementation plan

### 5.1 Repository layout
```
sentiment_rb/
  pyproject.toml
  config/
    settings.yaml          # windows, horizons, model IDs, modulation bounds, flags
    taxonomy.yaml          # nodes: id, parent, name, asset_class, aliases
    assets.yaml            # asset_id, node path, base budget b0, return series ref
    features.yaml          # feature_id, source, node_id, publication_lag, transform, sign per asset_class
  prompts/
    triage.md
    scoring.md
    digest.md
  src/sentiment_rb/
    data/        loaders.py, pit.py            # point-in-time store (as_of, available_at)
    features/    transforms.py, tree.py        # z-score, orient, inherit features down the tree
    llm/         client.py, schemas.py, triage.py, scoring.py, digest.py, dedup.py, aggregate.py
    model/       elastic_net.py, stability.py  # per-asset & pooled fits, purged CV
    signal/      aggregate.py, normalize.py    # S_i,t, percentile/Φ, cross-sectional blend, EWMA
    portfolio/   covariance.py, budgets.py, optimizer.py
    backtest/    engine.py, metrics.py         # walk-forward, costs, turnover, IC
    cli.py                                     # typer CLI
  tests/
  notebooks/     (optional exploration)
```

### 5.2 Build order
1. Config loading + taxonomy tree + feature inheritance (`tree.py`).
2. Point-in-time data store and transforms (with synthetic data for tests).
3. Elastic net per asset with purged CV; aggregation; normalization.
4. Budget modulation + risk-budgeting optimizer.
5. Walk-forward backtest engine (refit schedule, rebalance schedule, costs).
6. LLM pipeline: schemas → client (retry, caching, logging) → triage → scoring → digest → dedup → node aggregation → features.
7. CLI: `ingest`, `score-news`, `build-features`, `fit`, `signal`, `optimize`, `backtest`.

### 5.3 Tests (pytest)
- Feature mapping: an asset receives the features of its configured factor nodes exactly once.
- PIT: no feature value used before `available_at` (property test with random lags).
- Purged CV: no train index within `h` of any test index.
- Normalization outputs ∈ [0,1]; no full-sample statistics used.
- Budgets: positive, sum to 1, respect bounds and change cap.
- Optimizer: risk contributions match budgets within 1e-4 on random PD matrices.
- LLM: schema validation, retry path, sign conventions (mock responses: hawkish Fed → negative for UST node).
- Aggregation: decay weighting and novelty weighting match hand-computed examples.

---

## 6. Validation & monitoring

- **Benchmark**: backtest vs plain ERC with b⁰ (same Σ, same rebalance, after costs). Overlay must add value.
- **IC**: correlation of S_i,t (and each LLM feature) with next-period returns, per asset and pooled.
- **Beta stability**: track selection frequency and sign of β over refits; group near-duplicate features (e.g. VIX vs VIX term structure) within a node if unstable.
- **LLM quality**: hand-label ~300 articles; target Spearman ≥ 0.6 and directional agreement ≥ 85%. Perturbation test (paraphrase, shuffled order) → scores stable. Sign sanity tests.
- **Redundancy**: correlation of LLM features vs vendor sentiment (e.g. RavenPack) if available.
- **Cost control**: triage everything, score only relevant (typically 10–30%); batch + cache.
- **Reporting**: per rebalance — s_i, b_i, w_i, RC_i, top contributing features per asset, LLM key_drivers.

---

## 7. Acceptance criteria
- `backtest` CLI runs end-to-end on synthetic data with no network access (LLM mocked).
- All tests pass; no lookahead (PIT + purged CV tests green).
- Every LLM call is reproducible from stored prompt version + model ID + raw response.
- Config-only changes (no code edits) to: taxonomy, features, horizons, modulation bounds, models, normalization method.
- Report output includes the benchmark comparison and IC tables.
