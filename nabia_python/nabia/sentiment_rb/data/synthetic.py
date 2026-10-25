"""Synthetic point-in-time market data (latent sentiment factors, returns, indicators) and news articles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.transforms import vol_scaled_target
from ..features.tree import ancestors, node_sign


def ar1(rng, n, k, phi):
    x = np.zeros((n, k))
    e = rng.standard_normal((n, k)) * np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def synthetic_market(cfg, rng):
    syn, nodes = cfg["synthetic"], list(cfg["taxonomy"])
    dates = pd.bdate_range(cfg["start"], periods=cfg["days"])
    f = pd.DataFrame(ar1(rng, len(dates), len(nodes), syn["factor_persistence"]), dates, nodes)  # latent sentiment
    common = rng.standard_normal(len(dates))
    returns = {}
    for a, (node, cls, _, vol) in cfg["assets"].items():
        path = ancestors(cfg, node)
        drift = sum(node_sign(cfg, n, cls) * f[n] for n in path) / np.sqrt(len(path))
        load = syn["market_loading"][cls]
        shock = load * common + np.sqrt(1 - load**2) * rng.standard_normal(len(dates))
        returns[a] = (syn["signal_strength"] * drift + shock) * vol / np.sqrt(252) + syn["sharpe"][cls] * vol / 252
    raw = {}  # indicator levels by as_of date (before publication lag)
    for fid, (node, _, transform, _, truth) in cfg["features"].items():
        noise = rng.standard_normal(len(dates))
        if transform == "diff":   # a stock (e.g. ETF holdings) whose changes carry the signal
            raw[fid] = pd.Series(np.cumsum(truth * f[node] + 2 * noise), dates)
        else:                     # a level with slow nuisance drift
            raw[fid] = truth * f[node] + noise + np.cumsum(0.02 * rng.standard_normal(len(dates)))
    return f, pd.DataFrame(returns), pd.DataFrame(raw)


def news_truth(cfg, f, returns, rng):
    """Daily 'true tone' per node, ~N(0,1). Oracle: corr(tone, node's forward vol-scaled return) = sentiment_ic,
    over the h days that start when the news becomes usable (after the LLM publication lag)."""
    syn = cfg["synthetic"]
    if syn["sentiment"] == "latent":
        return f
    ic, lag = syn["sentiment_ic"], cfg["llm"]["lag"]
    fwd = pd.DataFrame({a: vol_scaled_target(cfg, returns[a]).shift(-lag) for a in returns})
    truth = {}
    for node in cfg["taxonomy"]:
        members = [a for a, v in cfg["assets"].items() if node in ancestors(cfg, v[0])]
        x = sum(node_sign(cfg, node, cfg["assets"][a][1]) * fwd[a] for a in members) / len(members)
        z = ((x - x.mean()) / x.std()).fillna(0.0)  # the last h days have no future: pure noise
        truth[node] = ic * z + np.sqrt(1 - ic**2) * rng.standard_normal(len(z))
    return pd.DataFrame(truth)


PHRASES = [(-0.6, "slumps on severe downturn fears warning"), (-0.25, "weakens as outlook is cut"),
           (0.25, "steady with mixed signals"), (0.6, "gains on improving demand outlook"),
           (9, "surges after strong upside surprise")]


def synthetic_news(cfg, truth, rng):
    tax, nodes, arts = cfg["taxonomy"], list(cfg["taxonomy"]), []
    p = np.array([4 if n.startswith("L0") else 2 if n.startswith("L1") else 1 for n in nodes], float)
    for t in truth.index:
        for _ in range(rng.poisson(cfg["llm"]["news_per_day"])):
            node = rng.choice(nodes, p=p / p.sum())
            tone = float(np.tanh(truth.at[t, node] + cfg["synthetic"]["article_noise"] * rng.standard_normal()))
            name, kind = tax[node][1], rng.choice(["new", "recap", "dup"], p=[0.78, 0.12, 0.10])
            if kind == "dup" and arts:  # re-publication of a recent story
                arts.append({**arts[-1], "id": len(arts), "source": "blog", "headline": arts[-1]["headline"] + " (update)"})
                continue
            words = next(w for lim, w in PHRASES if tone < lim)
            headline = (f"{name} moved {rng.normal(0, 1):+.1f}% in today's session" if kind == "recap"
                        else f"{name} {words} {rng.choice(['analysts', 'officials', 'traders', 'data'])} say")
            pub = t + pd.Timedelta(hours=float(rng.uniform(0, 23)))
            arts.append({"id": len(arts), "publish_ts": pub, "received_ts": pub + pd.Timedelta(minutes=float(rng.uniform(1, 90))),
                         "source": rng.choice(["wire", "paper", "blog"]), "headline": headline, "body": headline + ".",
                         "_node": node, "_tone": tone, "_recap": kind == "recap"})
    return pd.DataFrame(arts)
