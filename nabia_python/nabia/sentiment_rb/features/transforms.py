"""Point-in-time transforms, oriented candidate features per asset (inherited down the tree), and the target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nabia_python_bridge import signal

from ..config import RISK_ON
from .tree import ancestors, sign_for


def transform(cfg, series, lag, kind):
    """Point in time (value usable `lag` days after as_of), stationary, rolling z-score clipped at +-z_clip."""
    x = series.shift(lag)
    x = x.diff(5) if kind == "diff" else x
    r = x.rolling(cfg["z_window"], min_periods=cfg["z_min"])
    return ((x - r.mean()) / r.std()).clip(-cfg["z_clip"], cfg["z_clip"])


def build_features(cfg, raw, llm_nodes, digests, dates):
    """{feature: (z-scored series, node, sign)} for indicators, node LLM features and digest outputs."""
    feats = {fid: (transform(cfg, raw[fid], lag, kind), node, sign)
             for fid, (node, lag, kind, sign, _) in cfg["features"].items()}
    for col in llm_nodes:
        kind, node = col[4:].split("@")
        s = cfg["llm_feature_signs"][kind]
        feats[col] = (transform(cfg, llm_nodes[col], cfg["llm"]["lag"], "z"), node,
                      {c: s * RISK_ON[c] for c in RISK_ON} if node == "L0.GLOBAL" else s)
    for name, (node, sign, path) in cfg["digest_features"].items():
        values = {d: (lambda v: v if v is not None else 0.0)(
            getattr(dg, path[0]) if len(path) == 1 else getattr(dg, path[0]).get(path[1])) for d, dg in digests.items()}
        feats[f"digest_{name}"] = (transform(cfg, pd.Series(values, dtype=float).reindex(dates).fillna(0.0), cfg["llm"]["lag"], "z"), node, sign)
    return feats


def asset_matrix(cfg, feats, asset):
    """Candidate features of an asset: those of every ancestor node, oriented (+ = bullish), each exactly once."""
    node, cls = cfg["assets"][asset][:2]
    path = set(ancestors(cfg, node))
    cols = {fid: sign_for(sign, cls) * z for fid, (z, n, sign) in feats.items()
            if n in path and sign_for(sign, cls) and z.notna().any()}  # features with no data at all are left out
    return pd.DataFrame(cols)


def vol_scaled_target(cfg, r):
    """y_t = r_{t->t+h} / sigma_t, sigma from a nabia EWMA of squared returns, scaled to the horizon."""
    h = cfg["horizon"]
    sigma = np.sqrt(signal.exponential_moving_average(r.values**2, 1 - 0.5 ** (1 / cfg["vol_halflife"])) * h)
    price = (1 + r).cumprod()
    return (price.shift(-h) / price - 1) / pd.Series(sigma, r.index)
