"""The whole pipeline for one configuration (spec section 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import report
from .data import load_news, news_truth, real_market, synthetic_market, synthetic_news
from .data.scores import load_scores
from .features import asset_matrix, build_features, vol_scaled_target
from .llm import LLM, node_features, run_news
from .model import walk_forward_scores
from .portfolio import backtests, check_risk_contributions


def run(cfg, out, live=False, show=False, verbose=True):
    """Data -> news -> features -> elastic net -> s -> budgets -> backtests -> report; returns headline numbers."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    assets = list(cfg["assets"])
    b0 = np.array([cfg["assets"][a][2] for a in assets])

    if cfg["data"]["source"] == "real":
        returns, raw = real_market(cfg)  # Yahoo Finance and FRED, cached
        cfg = {**cfg, "features": {k: v for k, v in cfg["features"].items() if k in raw}}  # with a real source only
        news, truth = load_news(cfg["data"]["news_file"]), None
        print(f"real data: {len(returns)} days {returns.index[0].date()} .. {returns.index[-1].date()}, "
              f"indicators {list(raw.columns)}, news file {cfg['data']['news_file']}")
    else:
        f, returns, raw = synthetic_market(cfg, rng)
        truth = news_truth(cfg, f, returns, rng)
        news = synthetic_news(cfg, truth, rng)
    dates = returns.index
    live_days = cfg["llm"]["live_days"]
    live_from = (dates[0] if live_days == 0 else dates[-live_days]) if live else dates[-1] + pd.Timedelta(days=1)
    scores_file = cfg["data"].get("scores_file")
    if scores_file:  # LLM scores computed beforehand (python -m nabia.news score): no model calls
        scores, digests = load_scores(scores_file, dates)
        print(f"news scores from {scores_file}: {len(scores)} node scores, {len(digests)} digests")
    else:
        llm = LLM(cfg, out, live, rng)
        scores, digests = run_news(cfg, news, dates, llm, live_from, truth)
        print(f"news: {len(news)} articles, {len(scores)} node scores, {len(digests)} digests "
              f"({'live from ' + str(live_from.date()) + ', ' + llm.provider + ' ' + llm.model('scoring') if live else 'offline mock'})")

    feats = build_features(cfg, raw, node_features(cfg, scores, dates), digests, dates)
    X = {a: asset_matrix(cfg, feats, a) for a in assets}
    y = {a: vol_scaled_target(cfg, returns[a]) for a in assets}
    S, s, betas, ics = walk_forward_scores(cfg, X, y, dates)
    print(f"elastic net: {len(betas)} fits, signal from {S.index[0].date()}")

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
    return {"sentiment_ic": cfg["synthetic"]["sentiment_ic"], "mode": cfg["budgets"]["mode"], "erc_sharpe": bench.metrics["sharpe_ratio"],
            "overlay_sharpe": overlay.metrics["sharpe_ratio"],
            "excess_return": overlay.metrics["annualized_return"] - bench.metrics["annualized_return"],
            "ic_S": t["ic"].loc["pooled", "IC(S, y)"], "ic_llm_level": t["llm_ic"]["level"]}
