"""The whole pipeline for one configuration (spec section 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import report
from .data import news_truth, synthetic_market, synthetic_news
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

    f, returns, raw = synthetic_market(cfg, rng)
    dates = returns.index
    truth = news_truth(cfg, f, returns, rng)
    news = synthetic_news(cfg, truth, rng)
    live_from = dates[-cfg["llm"]["live_days"]] if live else dates[-1] + pd.Timedelta(days=1)
    scores, digests = run_news(cfg, news, truth, LLM(cfg, out, live, rng), live_from)
    print(f"news: {len(news)} articles, {len(scores)} node scores, {len(digests)} digests "
          f"({'live from ' + str(live_from.date()) if live else 'mocked LLM'})")

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
    print("\nkey drivers:", digests[max(digests)].key_drivers, f"| max |RC - b| (nabia RiskBudgeting) = {rc_err:.1e}",
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
