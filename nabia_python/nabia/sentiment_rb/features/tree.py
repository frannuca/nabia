"""The indicator tree: ancestors of a node, and feature orientation per asset class."""

from __future__ import annotations

from ..config import RISK_ON


def ancestors(cfg, node):
    out = []
    while node:
        out.append(node)
        node = cfg["taxonomy"][node][0]
    return out


def sign_for(sign, asset_class):
    return sign.get(asset_class, 0) if isinstance(sign, dict) else sign


def node_sign(cfg, node, asset_class):  # latent node factors are bullish for their subtree, L0 is risk appetite
    return RISK_ON[asset_class] if node == "L0.GLOBAL" else 1
