"""Mapping the portfolio assets to the factors from data: at every quarter end, over the trailing window of weekly
returns,
  1. the factor proxies are orthogonalized in the order of factors.json (QR decomposition: each proxy net of those
     before it), so every column is a distinct shock and the regression is not confused by collinearity;
  2. each asset's returns are regressed on them: the exposure is the standardized beta (a partial correlation,
     -1..1) with its t-statistic;
  3. every factor with enough weeks of news is also fitted on its weekly news sentiment (correlation with the
     asset's returns): this fit fills the factors without a proxy and, with news_override, replaces the proxy fit
     when it is significant (the news captures what a proxy cannot, e.g. a Fed shock on equities);
  4. an exposure is kept when |t| >= t_min and its sign held in a share >= stability of the recent windows, then
     shrunk toward the mean of the asset's exposures in the same group; a factor with neither proxy nor enough
     news borrows group_fallback times that group mean (method "group").
The result is the asset_exposures table: point-in-time maps asset -> factor -> signed exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datalayer import MarketDB, NewsDB

from .proxies import asset_returns, download_fred, news_sentiment, proxy_returns


def fit_window(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardized betas and t-statistics of y on the columns of P orthogonalized in order. With X = QR (X the
    centred proxies), column k of Q·R_kk is proxy k net of proxies 0..k-1, so its standardized beta is
    sign(R_kk)·(Q_k·y)/|y| and its t-statistic sign(R_kk)·(Q_k·y)/sigma."""
    X, y = P - P.mean(0), y - y.mean()
    Q, R = np.linalg.qr(X)
    qy, sign = Q.T @ y, np.sign(np.diag(R))
    resid = y - Q @ qy
    sigma = np.sqrt(resid @ resid / max(len(y) - P.shape[1] - 1, 1))
    return sign * qy / np.linalg.norm(y), sign * qy / sigma


def fit_news(s: pd.Series, y: pd.Series, min_weeks: int):
    """Correlation of the asset's weekly returns with a factor's weekly sentiment and its t-statistic, or None
    when the factor has fewer than min_weeks weeks of news in the window."""
    ok = s.notna() & y.notna()
    if ok.sum() < min_weeks:
        return None
    r = float(np.corrcoef(s[ok], y[ok])[0, 1])
    return r, r * np.sqrt((ok.sum() - 2) / max(1 - r * r, 1e-12))


def estimate(cfg: dict, download: bool = True) -> pd.DataFrame:
    """Estimates the exposures at every quarter end with a full window, stores them, returns them."""
    E, factors = cfg["exposures"], cfg["factors"]
    with MarketDB(cfg["db"]) as db, NewsDB(cfg["db"]) as news_db:
        if download:
            download_fred(db, factors)
        P, A, S = proxy_returns(db, factors), asset_returns(db, E["assets"]), news_sentiment(news_db, cfg)
        P = P[P.index >= E["start"]]
        rows, history = [], {}  # history: (asset, factor, method) -> raw exposures of the previous windows
        for as_of in pd.date_range(P.index[0], P.index[-1], freq="QE"):
            win = P[(P.index <= as_of)].tail(E["window_weeks"])
            win = win.loc[:, win.notna().mean() >= 0.9]  # proxies with (almost) complete data in the window
            if len(win) < E["window_weeks"] or win.empty:
                continue
            for asset in E["assets"]:
                y = A[asset].reindex(win.index)
                ok = win.notna().all(axis=1) & y.notna()
                fits = {name: (raw, t, "proxy") for name, raw, t in zip(win.columns, *fit_window(win[ok].values, y[ok].values))} \
                    if ok.sum() > 2 * win.shape[1] else {}
                for name in factors:
                    if name in S and (f := fit_news(S[name].reindex(win.index), y, E["news_min_weeks"])):
                        if name not in fits or (E["news_override"] and abs(f[1]) >= E["t_min"]):
                            fits[name] = (*f, "news")
                rows += decide(cfg, as_of, asset, fits, history)
        db.save_exposures(rows)
    return pd.DataFrame(rows)


def decide(cfg, as_of, asset, fits, history) -> list[dict]:
    """The exposures of one asset at one date from its fits {factor: (raw, t, method)}: significance and
    sign-stability tests (over the previous fits of the same method), shrinkage toward the group mean, and the group
    fallback for factors without a fit."""
    E, factors = cfg["exposures"], cfg["factors"]
    out = {}
    for name, (raw, t, method) in fits.items():
        past = history.setdefault((asset, name, method), [])
        past.append(raw)
        recent = past[-E["stability_windows"]:]
        stability = float(np.mean(np.sign(recent) == np.sign(raw)))
        keep = abs(t) >= E["t_min"] and stability >= E["stability"]
        out[name] = dict(raw=float(raw), t_stat=float(t), stability=stability, exposure=float(raw) if keep else 0.0,
                         method=method)
    groups = {}
    for name, o in out.items():
        groups.setdefault(factors[name]["group"], []).append(o["exposure"])
    mean = {g: float(np.mean(v)) for g, v in groups.items()}
    rows, w = [], E["group_shrink"]
    for name, f in factors.items():
        o = out.get(name) or dict(raw=None, t_stat=None, stability=None, method="group" if f["group"] in mean else "none",
                                  exposure=E["group_fallback"] * mean.get(f["group"], 0.0))
        if o["method"] in ("proxy", "news") and o["exposure"]:
            o["exposure"] = (1 - w) * o["exposure"] + w * mean[f["group"]]
        rows.append({"as_of": str(as_of.date()), "asset": asset, "factor": name, **o})
    return rows
