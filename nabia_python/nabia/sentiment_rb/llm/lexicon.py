"""Offline reading of real-format articles: taxonomy nodes by name, sentiment by a word lexicon.

Used by the mock LLM when an article carries no synthetic truth (real or example news, offline). It is a crude
stand-in for the model: it finds the most specific node named in the headline and scores finance vocabulary.
"""

from __future__ import annotations

import math
import re

#: Word polarity for the asset's price (tanh of the sum is the article's tone).
WORDS = {
    "surges": 0.5, "surge": 0.5, "soars": 0.5, "jumps": 0.35, "rallies": 0.4, "rally": 0.4, "gains": 0.3, "rises": 0.2,
    "strong": 0.3, "upside": 0.3, "improving": 0.2, "beats": 0.4, "upgrade": 0.4, "record": 0.2, "rebound": 0.3,
    "dovish": 0.3, "optimism": 0.3, "eases": 0.2,
    "slumps": -0.5, "slump": -0.5, "plunges": -0.6, "tumbles": -0.5, "falls": -0.2, "drops": -0.3, "slides": -0.3,
    "weakens": -0.3, "weak": -0.3, "downturn": -0.3, "severe": -0.2, "fears": -0.2, "warning": -0.2, "cut": -0.2,
    "misses": -0.4, "downgrade": -0.4, "selloff": -0.4, "recession": -0.5, "default": -0.5, "hawkish": -0.3,
    "losses": -0.3, "stress": -0.3,
}
RECAP = re.compile(r"moved [+-]?\d+(\.\d+)?% in today's session|closed (up|down) \d", re.I)


def tone(text: str) -> float:
    """Sentiment in (-1, 1): tanh of the summed word polarities."""
    return math.tanh(sum(WORDS.get(w, 0.0) for w in re.findall(r"[a-z]+", text.lower())))


def node_names(cfg) -> dict[str, list[str]]:
    """Every name of every node: its display name and its aliases, lower case."""
    return {n: [v[1].lower(), *cfg.get("aliases", {}).get(n, [])] for n, v in cfg["taxonomy"].items()}


def match_node(cfg, text: str) -> str | None:
    """The node with the longest name found in the text (the most specific), or None."""
    text, best = text.lower(), None
    for node, names in node_names(cfg).items():
        for name in names:
            if re.search(r"(?<![\w&])" + re.escape(name) + r"(?![\w])", text) and (best is None or len(name) > best[1]):
                best = (node, len(name))
    return best[0] if best else None


def is_recap(text: str) -> bool:
    """A market recap: price moves without new information."""
    return bool(RECAP.search(text))
