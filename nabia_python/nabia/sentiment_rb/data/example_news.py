"""Example news in the JSON format of real news, made from the real assets' returns (a stand-in until a feed exists).

Each taxonomy node's daily tone correlates, with coefficient --ic, with the forward vol-scaled return of its assets
on the real prices: the example LOOKS AHEAD on purpose, so that a real-data run has news with a known skill. The
articles carry only what real news has (timestamps, source, headline, body): headlines name the node and use a
vocabulary that the offline scorer (llm/lexicon.py) reads back.

    python -m nabia.sentiment_rb.data.example_news                       # data/example_news.json, IC 0.2
    python -m nabia.sentiment_rb.data.example_news --ic 0.3 --out data/my_news.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import SYNTHETIC, load_config, merge
from .real import NEWS_COLUMNS, real_market
from .synthetic import news_truth, synthetic_news


def make_example_news(cfg, returns, rng, ic):
    """Articles (real-news columns only) whose tone has correlation `ic` with each node's forward returns."""
    oracle = merge(json.loads(json.dumps(cfg)), {"synthetic": {**SYNTHETIC["synthetic"], "sentiment": "oracle", "sentiment_ic": ic}})
    truth = news_truth(oracle, None, returns, rng)
    return synthetic_news(oracle, truth, rng)[NEWS_COLUMNS]


def write_news(news, path, meta):
    records = news.assign(publish_ts=news.publish_ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          received_ts=news.received_ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict("records")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "articles": records}, indent=None, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(prog="python -m nabia.sentiment_rb.data.example_news", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", type=float, default=0.2, help="correlation of news tone with forward returns (default 0.2)")
    ap.add_argument("--out", type=Path, default=Path("data/example_news.json"))
    ap.add_argument("--config", type=Path, help="JSON overrides (assets, dates, data.cache_dir, ...)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(args.config, "real")
    returns, _ = real_market(cfg)
    news = make_example_news(cfg, returns, np.random.default_rng(args.seed), args.ic)
    meta = {"description": "Example news generated from real asset returns: each node's tone correlates with the "
                           "forward vol-scaled return of its assets (it looks ahead on purpose). Replace with real news.",
            "ic": args.ic, "seed": args.seed, "horizon_days": cfg["horizon"], "assets": {a: v["ticker"] for a, v in cfg["assets"].items()},
            "start": str(returns.index[0].date()), "end": str(returns.index[-1].date())}
    write_news(news, args.out, meta)
    print(f"{len(news)} articles, {returns.index[0].date()} .. {returns.index[-1].date()}, IC {args.ic} -> {args.out}")


if __name__ == "__main__":
    main()
