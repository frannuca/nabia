"""Risk budgets from the news: at every rebalance date t, using only what was known before t,
  1. factor signal: each factor's decayed, confidence-weighted mean sentiment of the articles received before t
     (half-life half_life_days), shrunk by its evidence: signal = level * n / (n + evidence), n the decayed sum of
     confidences, so a factor with a few articles moves little; -1..1, > 0 = the factor is going up;
  2. asset score: the exposure-weighted mean of the factor signals, S = sum_f e_f * signal_f / sum_f |e_f|, with the
     latest exposure map estimated at or before t; -1..1, > 0 = the news is bullish for the asset;
  3. budget: base budget * (1 + strength * S), within [min_multiplier, max_multiplier], normalized to sum 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datalayer import MarketDB, NewsDB
from exposures.proxies import article_scores


def factor_signals(scores: pd.DataFrame, dates: pd.DatetimeIndex, factors: list, half_life: float, evidence: float):
    """Signal of every factor (columns) at every date (rows) from the article scores received strictly before it."""
    days = pd.date_range(min(dates.min(), scores.received_at.min().normalize() if len(scores) else dates.min()), dates.max())
    day = scores.assign(day=scores.received_at.dt.normalize(), ws=scores.sentiment * scores.confidence)
    g = day.groupby(["day", "factor"])[["ws", "confidence"]].sum()
    ws, w = (g[c].unstack("factor").reindex(index=days, columns=factors).fillna(0.0) for c in ("ws", "confidence"))
    decay = 0.5 ** (1 / half_life)
    # decayed sums of the previous days: x_t = decay * x_{t-1} + (day t-1 total), i.e. articles of day t not yet used
    num, den = (pd.DataFrame(_decayed(x.shift(1).fillna(0.0).values, decay), days, factors) for x in (ws, w))
    level = (num / den.where(den > 1e-12)).fillna(0.0)
    return (level * den / (den + evidence)).reindex(dates)


def _decayed(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x)
    for k in range(len(x)):
        out[k] = x[k] + (decay * out[k - 1] if k else 0.0)
    return out


def asset_budgets(e: pd.DataFrame, signal: pd.Series, base: pd.Series, B: dict) -> pd.DataFrame:
    """Scores, multipliers and budgets of the assets (rows of the exposure matrix e: asset x factor) for one date."""
    weight = e.abs().sum(axis=1)
    score = (e @ signal.reindex(e.columns).fillna(0.0)) / weight.where(weight > 0)
    score = score.fillna(0.0)
    mult = (1 + B["strength"] * score).clip(B["min_multiplier"], B["max_multiplier"])
    budget = base * mult
    return pd.DataFrame({"base": base, "score": score, "multiplier": mult, "budget": budget / budget.sum()})


def compute(cfg: dict, start=None, end=None) -> pd.DataFrame:
    """Risk budgets at every rebalance date in [start, end] (default: the configuration's budgets section), stored in
    the risk_budgets table and returned."""
    B, assets, factors = cfg["budgets"], cfg["exposures"]["assets"], list(cfg["factors"])
    base = pd.Series({a: (B["base"] or {}).get(a, 1.0) for a in assets})
    base /= base.sum()
    dates = pd.date_range(start or B["start"], end or B["end"] or pd.Timestamp.now().normalize(), freq=B["frequency"])
    with MarketDB(cfg["db"]) as db, NewsDB(cfg["db"]) as news_db:
        signals = factor_signals(article_scores(news_db, cfg), dates, factors, B["half_life_days"], B["evidence"])
        rows = []
        for t in dates:
            emap = pd.DataFrame(db.exposures(str((t - pd.Timedelta(days=1)).date())))  # estimated before t
            if emap.empty:
                continue
            e = emap.pivot(index="asset", columns="factor", values="exposure").reindex(index=assets, columns=factors).fillna(0.0)
            out = asset_budgets(e, signals.loc[t], base, B)
            rows += [{"date": str(t.date()), "asset": a, **{k: float(v) for k, v in r.items()}, "exposures": emap.as_of[0]}
                     for a, r in out.iterrows()]
        db.save_budgets(rows)
    return pd.DataFrame(rows)


def contributions(cfg: dict, date: str) -> pd.DataFrame:
    """Why the budgets are what they are at a date: e_f * signal_f / sum |e| per asset (rows) and factor (columns),
    whose row sums are the asset scores."""
    B, assets, factors = cfg["budgets"], cfg["exposures"]["assets"], list(cfg["factors"])
    t = pd.Timestamp(date)
    with MarketDB(cfg["db"]) as db, NewsDB(cfg["db"]) as news_db:
        signal = factor_signals(article_scores(news_db, cfg), pd.DatetimeIndex([t]), factors,
                                B["half_life_days"], B["evidence"]).loc[t]
        emap = pd.DataFrame(db.exposures(str((t - pd.Timedelta(days=1)).date())))
    e = emap.pivot(index="asset", columns="factor", values="exposure").reindex(index=assets, columns=factors).fillna(0.0)
    return e.mul(signal, axis=1).div(e.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
