"""The whole pipeline for one configuration (spec section 0)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import report
from .data import load_news, news_truth, real_market, synthetic_market, synthetic_news
from .data.scores import load_digests, load_scores
from .features import asset_matrix, build_features, vol_scaled_target
from .features.tree import base_budgets
from .llm import LLM, node_features, run_news
from .model import walk_forward_scores
from .portfolio import backtests, check_risk_contributions


def market_and_news(cfg, out, live, rng):
    """(cfg, returns, raw indicators, news, synthetic truth or None): synthetic data, or real prices, indicators and
    news (in real runs cfg keeps only the features that have a real source)."""
    if cfg["data"]["source"] == "real":
        returns, raw = real_market(cfg)  # Yahoo Finance and FRED, cached
        cfg = {**cfg, "features": {k: v for k, v in cfg["features"].items() if k in raw}}  # with a real source only
        news = load_news(cfg["data"]["news_file"]) if not cfg["data"].get("scores_file") else None
        print(f"real data: {len(returns)} days {returns.index[0].date()} .. {returns.index[-1].date()}, "
              f"indicators {list(raw.columns)}, news {cfg['data'].get('scores_file') or cfg['data']['news_file']}")
        return cfg, returns, raw, news, None
    f, returns, raw = synthetic_market(cfg, rng)
    truth = news_truth(cfg, f, returns, rng)
    return cfg, returns, raw, synthetic_news(cfg, truth, rng), truth


def news_scores(cfg, out, news, dates, live, rng, truth=None):
    """(node scores, digests): read from data.scores_file, or computed by the LLM (live calls with `live`, from
    llm.live_days before the end; the offline mock otherwise)."""
    if scores_file := cfg["data"].get("scores_file"):  # LLM scores computed beforehand: no model calls
        scores, digests = load_scores(scores_file, dates)
        if cfg["data"].get("digests_file"):
            digests = load_digests(cfg["data"]["digests_file"], dates)
        print(f"news scores from {scores_file}: {len(scores)} node scores, {len(digests)} digests")
        return scores, digests
    live_days = cfg["llm"]["live_days"]
    live_from = (dates[0] if live_days == 0 else dates[-live_days]) if live else dates[-1] + pd.Timedelta(days=1)
    llm = LLM(cfg, out, live, rng)
    scores, digests = run_news(cfg, news, dates, llm, live_from, truth)
    print(f"news: {len(news)} articles, {len(scores)} node scores, {len(digests)} digests "
          f"({'live from ' + str(live_from.date()) + ', ' + llm.provider + ' ' + llm.model('scoring') if live else 'offline mock'})")
    return scores, digests


def signal(cfg, out, live=False):
    """Data -> news scores -> features -> walk-forward elastic net: everything up to the normalized sentiment s.
    Returns a namespace with cfg, returns, news, digests, X, y, S, s, betas, ics, rng."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    cfg, returns, raw, news, truth = market_and_news(cfg, out, live, rng)
    dates = returns.index
    scores, digests = news_scores(cfg, out, news, dates, live, rng, truth)
    feats = build_features(cfg, raw, node_features(cfg, scores, dates), digests, dates)
    X = {a: asset_matrix(cfg, feats, a) for a in cfg["assets"]}
    y = {a: vol_scaled_target(cfg, returns[a]) for a in cfg["assets"]}
    S, s, betas, ics = walk_forward_scores(cfg, X, y, dates)
    print(f"elastic net: {len(betas)} fits, signal from {S.index[0].date()}")
    return SimpleNamespace(cfg=cfg, returns=returns, news=news, digests=digests, X=X, y=y, S=S, s=s, betas=betas, ics=ics)


def run(cfg, out, live=False, show=False, verbose=True):
    """Data -> news -> features -> elastic net -> s -> budgets -> backtests -> report; returns headline numbers."""
    sig = signal(cfg, out, live)
    cfg, returns, news, digests, X, y, S, s, betas, ics = (sig.cfg, sig.returns, sig.news, sig.digests, sig.X, sig.y,
                                                          sig.S, sig.s, sig.betas, sig.ics)
    assets = list(cfg["assets"])
    b0 = base_budgets(cfg, assets)

    bench, overlay, sched, dirs = backtests(cfg, returns, s, b0)
    rc_err = check_risk_contributions(cfg, returns, sched, dirs)

    t = report.tables(cfg, X, y, S, s, betas, bench, overlay, sched, b0)
    if verbose:
        report.show_tables(t, sched.index[-1])
    print("\nkey drivers:", digests[max(digests)].key_drivers if digests else [], f"| max |RC - b| (nabia RiskBudgeting) = {rc_err:.1e}",
          f"| max budget change {(sched.pct_change().abs().max().max()):.1%} "
          f"| mode {cfg['budgets']['mode']}, short positions {(dirs < 0).to_numpy().mean():.0%}")
    report.save(out, t, sched, s, news, bench, overlay, ics)
    report.plots(cfg, out, bench, overlay, sched, s, b0, assets, ics, t["selection_frequency"], S.index[0], show)
    if verbose:
        print(f"\noutputs (tables, figures, budgets, news, LLM audit log) in {out}")
    return {"sentiment_ic": cfg.get("synthetic", {}).get("sentiment_ic"), "mode": cfg["budgets"]["mode"], "erc_sharpe": bench.metrics["sharpe_ratio"],
            "overlay_sharpe": overlay.metrics["sharpe_ratio"],
            "excess_return": overlay.metrics["annualized_return"] - bench.metrics["annualized_return"],
            "ic_S": t["ic"].loc["pooled", "IC(S, y)"], "ic_llm_level": t["llm_ic"]["level"]}
