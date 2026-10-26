"""Scores the stored news with the sentiment pipeline's LLM steps (triage, scoring, daily digest), by default with a
local Ollama model, and writes a scores file the pipeline reads without calling a model again. Live responses are
cached (llm.cache), so an interrupted run resumes where it stopped."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..sentiment_rb.data import load_news
from ..sentiment_rb.data.scores import write_scores
from ..sentiment_rb.llm import LLM, run_news


def score_news(cfg, news_path, scores_path, audit_dir=Path("output/news_scoring")):
    news = load_news(news_path)
    if news.empty:
        raise SystemExit(f"no articles in {news_path}")
    first, last = news.received_ts.min().normalize(), news.received_ts.max().normalize()
    dates = pd.bdate_range(first, last + pd.offsets.BDay(1))  # business days; the pipeline maps to trading dates
    audit_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(cfg, audit_dir, True, np.random.default_rng(cfg["seed"]))
    print(f"scoring {len(news)} articles ({first.date()} .. {last.date()}) with {cfg['llm']['provider']}: "
          f"{llm.model('scoring')}, {cfg['llm']['workers']} workers")
    details, partial_digests = {}, {}
    meta = {"description": "LLM scores of the articles of " + str(news_path), "news_file": str(news_path),
            "provider": llm.provider, "model": llm.model("scoring"),
            "prompts": {kind: llm.prompts[kind][2] for kind in llm.prompts}, "articles_total": len(news)}
    save = lambda: write_scores(scores_path, details, partial_digests, meta, complete=False)  # intermediate results
    records, digests = run_news(cfg, news, dates, llm, dates[0], None, details, partial_digests, save)
    write_scores(scores_path, details, digests, meta)
    relevant = sum(a["relevant"] for a in details.values())
    print(f"{relevant} relevant articles, {len(records)} node scores, {len(digests)} digests -> {scores_path}")
    return records, digests
