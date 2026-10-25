"""Risk budgets and directions: b = b0 (m_min + (m_max - m_min) s), renormalized, with a cap on the change;
in long-short mode, +1 / -1 per asset from the sentiment."""

from __future__ import annotations

import numpy as np


def modulate(cfg, s, b0, prev):
    B = cfg["budgets"]
    b = b0 * (B["m_min"] + (B["m_max"] - B["m_min"]) * s)
    b /= b.sum()
    for _ in range(50 if prev is not None else 0):  # cap the relative change per rebalance
        b = np.clip(b, prev * (1 - B["max_change"]), prev * (1 + B["max_change"]))
        b /= b.sum()
    assert (b > 0).all() and abs(b.sum() - 1) < 1e-12
    return b


def directions(cfg, s, prev):
    """+1 long, -1 short. Short below short_below, long again above long_above; in between the side is kept."""
    B, d = cfg["budgets"], prev.copy()
    d[s < B["short_below"]], d[s > B["long_above"]] = -1.0, 1.0
    return d
