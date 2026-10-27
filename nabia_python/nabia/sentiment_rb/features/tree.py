"""The sentiment factor tree (ancestors, roll-up, orientation per asset class) and the assets mapped onto it."""

from __future__ import annotations

import numpy as np

from ..config import RISK_ON


def ancestors(cfg, node):
    out = []
    while node:
        out.append(node)
        node = cfg["taxonomy"][node][0]
    return out


def is_mixed(cfg, node):
    """A node of no single asset class (global, region, macro, central bank): its sentiment is risk appetite."""
    return cfg["taxonomy"][node][2] is None


def factors(cfg, asset):
    """{node: exposure}: the factor nodes the asset may load on (its configuration), each with its sign."""
    return cfg["assets"][asset]["factors"]


def asset_class(cfg, asset):
    return cfg["assets"][asset]["class"]


def base_budgets(cfg, assets=None):
    return np.array([cfg["assets"][a]["budget"] for a in (assets if assets is not None else cfg["assets"])])


def own_node(cfg, asset):
    """The asset's most specific factor (the deepest with a positive exposure), e.g. for the own-node IC."""
    return max((n for n, e in factors(cfg, asset).items() if e > 0), key=lambda n: len(ancestors(cfg, n)))


def rollup(cfg, node):
    """[(node, level)]: the node (level 0) and its ancestors of the same asset class (sentiment is about that class's
    prices, so it is not mixed into a parent of another class)."""
    tax, out, cls = cfg["taxonomy"], [], cfg["taxonomy"][node][2]
    while node and tax[node][2] == cls:
        out.append((node, len(out)))
        node = tax[node][0]
    return out


def sign_for(sign, asset_class):
    return sign.get(asset_class, 0) if isinstance(sign, dict) else sign


def node_sign(cfg, node, asset_class):  # a class node's factor is bullish for that class; a mixed one is risk appetite
    return RISK_ON[asset_class] if is_mixed(cfg, node) else 1
