"""The staged real-data pipeline, one function per step, each reading the files of the steps before it (paths in
data.*) so that a step runs once and the later ones reuse its result:

    1 collect     articles (RSS, GDELT GKG history)                -> data.news_file
    2 digest      LLM daily macro digests of the articles          -> data.digests_file
    3 score       LLM triage and node scores of every article       -> data.scores_file
    4 calibrate   elastic net per asset, walk forward: the betas   -> data.betas_file
    5 backtest    stored betas -> sentiment S, s -> budgets and directions -> portfolio, against ERC
                  -> backtest.out (report tables and figures, positions over the lookback window)

scripts/pipeline/*_1.py .. *_5.py run them from a configuration file (configs/real.json).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import report
from .config import merge
from .data import load_news, real_market
from .data.scores import load_betas, load_digests, load_scores, write_betas, write_scores
from .features import asset_matrix, build_features, vol_scaled_target
from .features.tree import base_budgets
from .llm import LLM, node_features, run_news
from .model import apply_betas, fit_betas
from .portfolio import backtests, check_risk_contributions
from .portfolio.positions import write_positions


def _path(cfg, key):
    if not cfg["data"].get(key):
        raise SystemExit(f"set data.{key} in the configuration")
    return Path(cfg["data"][key])


def _real(cfg):
    if cfg["data"]["source"] != "real":
        raise SystemExit("the staged pipeline runs on real data: set data.source to \"real\"")
    return cfg


def collect(cfg):
    """Step 1: new articles into data.news_file (RSS feeds: today's items; GDELT GKG: history from collect.since)."""
    from ..news import NewsStore
    from ..news.collect import collect_gkg, collect_rss
    C, path = cfg["collect"], _path(_real(cfg), "news_file")
    store, added, failed = NewsStore(path), 0, []
    if C["rss"]:
        added, failed = collect_rss(store, C["feeds"])
    if C["gkg"]:
        since = dt.date.fromisoformat(C["since"])
        until = dt.date.fromisoformat(C["until"]) if C["until"] else None
        n, bad = collect_gkg(store, since, until, C["per_day"], C["weekends"], C["workers"])
        added, failed = added + n, failed + bad
    store.save()
    print(f"{added} new articles, {len(store.articles)} in {path}" + (f"; failed: {len(failed)}" if failed else ""))
    return store.stats()


def _llm_stage(cfg, stage, key):
    """Steps 2 and 3: one LLM stage over every article of data.news_file into data.<key>, saved every 25 items
    (complete = false until the end); live responses are cached (llm.cache), so a run resumes where it stopped."""
    news_path, out = _path(_real(cfg), "news_file"), _path(cfg, key)
    news = load_news(news_path)
    if news.empty:
        raise SystemExit(f"no articles in {news_path}: run collect_news_1 first")
    first, last = news.received_ts.min().normalize(), news.received_ts.max().normalize()
    dates = pd.bdate_range(first, last + pd.offsets.BDay(1))  # business days; later steps map to trading dates
    live = cfg["llm"]["provider"] != "mock"
    audit = Path(cfg["llm"].get("audit_dir", "output/llm_audit")) / stage
    audit.mkdir(parents=True, exist_ok=True)
    llm = LLM(cfg, audit, live, np.random.default_rng(cfg["seed"]))
    print(f"{stage}: {len(news)} articles ({first.date()} .. {last.date()}), {llm.provider} {llm.model('scoring') if live else ''}")
    details, digests = {}, {}
    prompts = ["digest"] if stage == "digests" else ["triage", "scoring"]
    meta = {"description": f"LLM {stage} of the articles of {news_path}", "news_file": str(news_path), "provider": llm.provider,
            "model": llm.model("scoring") if live else "mock", "prompts": {k: llm.prompts[k][2] for k in prompts},
            "articles_total": len(news)}
    save = lambda done=False: write_scores(out, details, digests, meta, complete=done)
    records, found = run_news(cfg, news, dates, llm, dates[0] if live else dates[-1] + pd.Timedelta(days=1), None,
                              details, digests, save, stages=(stage,))
    digests.update(found)
    save(True)
    print(f"{len(records)} node scores, {len(digests)} digests -> {out}")
    return out


def digest(cfg):
    """Step 2: the LLM's daily macro digest of each day's articles -> data.digests_file."""
    return _llm_stage(cfg, "digests", "digests_file")


def score(cfg):
    """Step 3: the LLM's triage and node scores of every article -> data.scores_file."""
    return _llm_stage(cfg, "articles", "scores_file")


def inputs(cfg):
    """Market data and the stored sentiment (scores and digests files) as the model's inputs: per asset its candidate
    features X and target y."""
    returns, raw = real_market(_real(cfg))
    cfg = {**cfg, "features": {k: v for k, v in cfg["features"].items() if k in raw}}  # with a real source only
    dates = returns.index
    scores, digests = load_scores(_path(cfg, "scores_file"), dates)
    if cfg["data"].get("digests_file"):
        digests = load_digests(cfg["data"]["digests_file"], dates)
    print(f"{len(returns)} days {dates[0].date()} .. {dates[-1].date()}, indicators {list(raw.columns)}, "
          f"{len(scores)} node scores, {len(digests)} digests")
    feats = build_features(cfg, raw, node_features(cfg, scores, dates), digests, dates)
    X = {a: asset_matrix(cfg, feats, a) for a in cfg["assets"]}
    y = {a: vol_scaled_target(cfg, returns[a]) for a in cfg["assets"]}
    return SimpleNamespace(cfg=cfg, returns=returns, dates=dates, X=X, y=y)


def calibrate(cfg):
    """Step 4: the walk-forward elastic net of every asset; its betas per refit -> data.betas_file."""
    inp = inputs(cfg)
    betas, in_sample = fit_betas(inp.cfg, inp.X, inp.y, inp.dates)
    M = inp.cfg["model"]
    meta = {"description": "elastic-net betas per refit date and asset (non-zero coefficients; the others are 0)",
            "data": {k: inp.cfg["data"].get(k) for k in ["start", "end", "scores_file", "digests_file"]},
            "horizon": inp.cfg["horizon"], "model": {k: M[k] for k in ["l1_ratios", "cv_folds", "positive", "refit", "window", "min_train", "stability"]},
            "assets": inp.cfg["assets"]}
    write_betas(_path(cfg, "betas_file"), betas, in_sample, {a: list(x.columns) for a, x in inp.X.items()}, meta)
    used = pd.DataFrame({a: [int((b != 0).sum()) for (d, x), b in sorted(betas.items()) if x == a] for a in inp.X},
                        index=sorted({d for d, _ in betas}))
    print(f"{len(used)} refits {used.index[0].date()} .. {used.index[-1].date()}; non-zero betas per asset (last 4 refits):\n{used.tail(4)}"
          if len(used) else "no refit: not enough history (model.min_train)")
    print(f"-> {cfg['data']['betas_file']}")
    return betas


def backtest(cfg, show=False):
    """Step 5: the stored betas map the sentiment features to S and s per asset; budgets and directions follow, and
    the portfolio is backtested against plain ERC. Writes the report (tables, figures) and the positions."""
    B = cfg["backtest"]
    if B.get("as_of"):
        cfg = merge(cfg, {"data": {"end": B["as_of"]}})  # nothing after the as-of date is loaded
    out = Path(B["out"])
    out.mkdir(parents=True, exist_ok=True)
    inp = inputs(cfg)
    cfg, returns, X, y = inp.cfg, inp.returns, inp.X, inp.y
    betas, in_sample = load_betas(_path(cfg, "betas_file"), X)
    betas = {k: v for k, v in betas.items() if k[0] <= inp.dates[-1]}
    S, s, ics = apply_betas(cfg, X, y, inp.dates, betas, in_sample)
    assets, b0 = list(cfg["assets"]), base_budgets(cfg)
    bench, overlay, sched, dirs = backtests(cfg, returns, s, b0)
    check_risk_contributions(cfg, returns, sched, dirs)
    t = report.tables(cfg, X, y, S, s, betas, bench, overlay, sched, b0)
    report.show_tables(t, sched.index[-1])
    report.save(out, t, sched, s, None, bench, overlay, ics)
    report.plots(cfg, out, bench, overlay, sched, s, b0, assets, ics, t["selection_frequency"], S.index[0], show)
    write_positions(cfg, returns, s, b0, bench, overlay, sched, dirs, out, B["lookback_days"], B.get("as_of"))
    print(f"\nreport, figures and positions in {out}")
    return t
