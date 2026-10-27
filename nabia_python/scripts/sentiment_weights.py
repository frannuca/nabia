"""Sentiment risk-budgeting weights from a run configuration: the signal, budgets, directions and weights of every
rebalance in a lookback window, and the position to hold from the as-of date on.

    .venv/bin/python scripts/sentiment_weights.py configs/real.json

The configuration (JSON, any CONFIG key; see nabia/sentiment_rb/config.py) chooses the data and the news:
  data.source       "real" (Yahoo Finance / FRED prices and indicators) or "synthetic"
  data.scores_file  LLM scores of the news (python -m nabia.news score): no model calls; or else
  data.news_file    news articles, scored with llm.provider: "mock" (offline lexicon), "ollama" or "anthropic"
  backtest          lookback_days (business days), as_of (a date; null = the last date), out (output folder)
  budgets           mode long_only / long_short, neutral band short_below / long_above, modulate_long_short

Writes to backtest.out: signal.csv (daily s in the window), rebalances.csv (date, asset, s, direction, budget, weight,
risk contribution), current.csv (the position at as_of), summary.json (the position and the window's performance
against plain ERC).
"""

from __future__ import annotations

import sys
from pathlib import Path

from nabia.sentiment_rb.config import load_config, merge
from nabia.sentiment_rb.features.tree import base_budgets
from nabia.sentiment_rb.pipeline import signal
from nabia.sentiment_rb.portfolio import backtests
from nabia.sentiment_rb.portfolio.positions import write_positions


def main(config_path):
    cfg = load_config(Path(config_path))
    B = cfg.get("backtest", {})
    lookback, as_of, out = B.get("lookback_days", 252), B.get("as_of"), Path(B.get("out", "output/backtest"))
    if as_of and cfg["data"]["source"] == "real":
        cfg = merge(cfg, {"data": {"end": as_of}})  # nothing after the as-of date is loaded
    live = cfg["llm"]["provider"] != "mock" and not cfg["data"].get("scores_file")
    if live:
        cfg = merge(cfg, {"llm": {"live_days": 0}})  # every article through the model (cached responses are reused)
    sig = signal(cfg, out, live)
    cfg, returns, s = sig.cfg, sig.returns, sig.s
    b0 = base_budgets(cfg)
    bench, overlay, sched, dirs = backtests(cfg, returns, s, b0)
    write_positions(cfg, returns, s, b0, bench, overlay, sched, dirs, out, lookback, as_of)
    print(f"\nsignal.csv, rebalances.csv, current.csv, summary.json in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: sentiment_weights.py CONFIG.json")
    main(sys.argv[1])
