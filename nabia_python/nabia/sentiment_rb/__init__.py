"""Sentiment-modulated risk budgeting, end to end (doc/sentiment_risk_budgeting_spec.md).

Point-in-time indicators and LLM news sentiment, inherited down an indicator tree, are combined per asset
by an elastic net into a sentiment score S, normalized to s in [0,1], which scales the asset's risk budget
in a risk-budgeting portfolio, backtested against plain ERC. The numerical work runs in the nabia wheel:
elastic net with purged walk-forward CV and block-bootstrap stability selection (regression engine),
normal CDF and EWMA (stats, signal), risk budgeting and risk contributions (risk), walk-forward backtest
with a budget schedule and performance metrics (strategy / frames). Data and news are synthetic so it
runs offline; the LLM is mocked unless --live (Anthropic API, ANTHROPIC_API_KEY, last `live_days` only).

    .venv/bin/python -m nabia.sentiment_rb                 # offline
    .venv/bin/python -m nabia.sentiment_rb --live # score the most recent news with Claude
    .venv/bin/python -m nabia.sentiment_rb --config c.json # override any CONFIG key (deep merge)
    .venv/bin/python -m nabia.sentiment_rb --ic-sweep 0 0.1 0.2 0.3 0.5   # alpha against news quality

Synthetic news is an "oracle" by default: each node's tone correlates with the forward returns of its assets
(synthetic.sentiment_ic), over the days after the news becomes usable, so the run measures how much of a
known news edge survives point-in-time features, the purged elastic net and the budget overlay.

Layout, one folder per functional group:
  config.py, pipeline.py, __main__.py   settings, run(), command line
  data/       synthetic.py               synthetic market data, indicators and news
  features/   tree.py, transforms.py     indicator tree; point-in-time transforms, candidate features, target
  llm/        schemas.py, client.py, news.py   output schemas; mock/Anthropic client; dedup, scoring, node features
  model/      elastic_net.py             elastic net per asset, walk-forward S and s
  portfolio/  budgets.py, backtest.py    budget modulation; ERC and overlay backtests, risk-contribution check
  report/     tables.py, figures.py      tables and CSV outputs; figures
  prompts/    versioned prompt files (results go to output/sentiment_rb in the working directory)
"""

from .pipeline import run

__all__ = ["run"]
