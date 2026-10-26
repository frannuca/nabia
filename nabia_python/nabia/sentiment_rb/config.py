"""Settings: every window, horizon, model, bound and the taxonomy, features and assets (override with JSON)."""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SIGNS = lambda eq, cmd, govt, credit: {"EQ": eq, "CMD": cmd, "GOVT": govt, "CREDIT": credit}


RISK_ON = SIGNS(1, 1, -1, 1)  # risk-on is bullish for everything but government bonds

CONFIG = {
    "seed": 7, "start": "2015-01-01", "days": 2600, "horizon": 21,
    # "synthetic": generated data (offline, a known truth); "real": prices and indicators from Yahoo Finance and
    # FRED (cached as CSV in cache_dir; refresh re-downloads) and news articles from news_file (JSON).
    "data": {"source": "synthetic", "start": "2014-01-01", "end": None, "cache_dir": "data/cache", "refresh": False,
             "news_file": "data/example_news.json", "scores_file": None},  # scores_file: LLM scores, no model calls
    "z_window": 756, "z_min": 252, "z_clip": 3.0, "vol_halflife": 20,
    "model": {"l1_ratios": [.1, .3, .5, .7, .9], "cv_folds": 5, "positive": True, "refit": "QE",
              "window": "expanding", "rolling_days": 1260, "min_train": 504, "threads": 0,
              "stability": {"samples": 50, "threshold": 0.6}},
    "normalize": {"method": "percentile", "lookback": 756, "alpha_ts": 1.0, "ewma_halflife": 5},
    # mode "long_only": s scales the budgets. "long_short": an asset is shorted when s < short_below (back to long
    # when s > long_above: a neutral band with hysteresis); its budget stays b0, unless modulate_long_short, which
    # scales it by the conviction (s for longs, 1 - s for shorts).
    "budgets": {"m_min": 0.5, "m_max": 1.5, "max_change": 0.25, "mode": "long_only",
                "short_below": 0.35, "long_above": 0.5, "modulate_long_short": False},
    "portfolio": {"covariance_estimator": "Exponential", "covariance_window": 252, "ewma_halflife": 60,
                  "rebalance_frequency": 21, "transaction_cost": 0.0005},
    "llm": {"triage_model": "claude-haiku-5-5", "scoring_model": "claude-sonnet-5-5",
            "digest_model": "claude-sonnet-5-5", "half_life": 3, "lag": 1, "dedup_cosine": 0.9,
            "dedup_hours": 48, "max_retries": 2, "live_days": 5, "anonymize": True, "news_per_day": 4,
            "source_weight": {"wire": 1.0, "paper": 0.8, "blog": 0.4},
            # live calls are cached here (key: prompt version, model, request), so each article is paid for once;
            # batch: submit a run's uncached live requests as Message Batches first; live_days 0 = every day live
            "cache": "data/llm_cache.jsonl", "batch": False, "batch_poll_seconds": 30,
            # live provider: "anthropic" (models above) or "ollama" (a local model, structured output, temperature 0);
            # workers: articles scored concurrently by live calls (Anthropic; a local Ollama model is fastest with 1:
            # with OLLAMA_NUM_PARALLEL > 1 every request slowed down several times on the development machine)
            "provider": "anthropic", "ollama_model": "gemma4:26b", "ollama_host": "http://localhost:11434",
            "ollama_options": {"temperature": 0, "num_ctx": 8192}, "ollama_think": False, "ollama_timeout": 600,
            "workers": 1},
    # node: parent, display name (aliases), asset class of the subtree
    "taxonomy": {
        "L0.GLOBAL": [None, "Global markets", None],
        "L1.EQ": ["L0.GLOBAL", "Equities", "EQ"], "L1.CMD": ["L0.GLOBAL", "Commodities", "CMD"],
        "L1.FI": ["L0.GLOBAL", "Fixed income", "FI"],
        "L2.EQ.US.BROAD": ["L1.EQ", "US equities", "EQ"], "L2.EQ.EU.BROAD": ["L1.EQ", "European equities", "EQ"],
        "L2.CMD.ENERGY": ["L1.CMD", "Energy", "CMD"], "L2.CMD.PRECIOUS": ["L1.CMD", "Precious metals", "CMD"],
        "L2.FI.GOVT10Y": ["L1.FI", "Treasuries 10y", "GOVT"], "L2.FI.HY": ["L1.FI", "High yield credit", "CREDIT"],
        "L3.SPX": ["L2.EQ.US.BROAD", "S&P 500", "EQ"], "L3.SX5E": ["L2.EQ.EU.BROAD", "Euro Stoxx 50", "EQ"],
        "L3.WTI": ["L2.CMD.ENERGY", "WTI crude", "CMD"], "L3.GOLD": ["L2.CMD.PRECIOUS", "Gold", "CMD"],
        "L3.UST10": ["L2.FI.GOVT10Y", "US 10y Treasury", "GOVT"], "L3.HY": ["L2.FI.HY", "US high yield", "CREDIT"],
    },
    # Other names under which a node appears in headlines (the offline scorer maps articles to nodes by name).
    "aliases": {"L0.GLOBAL": ["global markets", "world markets", "risk sentiment"], "L1.EQ": ["equities", "stocks"],
                "L1.CMD": ["commodities"], "L1.FI": ["fixed income", "bonds"], "L2.EQ.US.BROAD": ["us equities", "wall street"],
                "L2.EQ.EU.BROAD": ["european equities", "european stocks"], "L2.CMD.ENERGY": ["energy", "oil"],
                "L2.CMD.PRECIOUS": ["precious metals"], "L2.FI.GOVT10Y": ["treasuries", "treasury yields"],
                "L2.FI.HY": ["high yield", "junk bonds"], "L3.SPX": ["s&p 500", "s&p"], "L3.SX5E": ["euro stoxx", "stoxx 50"],
                "L3.WTI": ["wti", "crude", "brent"], "L3.GOLD": ["gold", "bullion"], "L3.UST10": ["10-year treasury", "10y treasury"],
                "L3.HY": ["us high yield", "high-yield bonds"]},
    "assets": {"SPX": ["L3.SPX", "EQ", 0.20, 0.16], "SX5E": ["L3.SX5E", "EQ", 0.15, 0.18],
               "WTI": ["L3.WTI", "CMD", 0.10, 0.35], "GOLD": ["L3.GOLD", "CMD", 0.15, 0.15],
               "UST10": ["L3.UST10", "GOVT", 0.25, 0.07], "HY": ["L3.HY", "CREDIT", 0.15, 0.06]},
    # feature: node, publication lag (days), transform ("z" | "diff"), sign per asset class (oriented: + = bullish);
    # "truth" is the synthetic loading on the node's latent factor (0 = pure noise).
    "features": {
        "VIX": ["L0.GLOBAL", 0, "z", SIGNS(-1, -1, 1, -1), -1], "HY_OAS": ["L0.GLOBAL", 0, "z", SIGNS(-1, -1, 1, -1), -1],
        "NFCI": ["L0.GLOBAL", 5, "z", SIGNS(-1, -1, 1, -1), -1], "AAII": ["L0.GLOBAL", 5, "z", RISK_ON, 1],
        "USD": ["L0.GLOBAL", 0, "z", SIGNS(-1, -1, 1, -1), 0],
        "PUT_CALL": ["L1.EQ", 0, "z", -1, -1], "BREADTH": ["L1.EQ", 0, "z", 1, 1],
        "COT_NET": ["L1.CMD", 3, "z", 1, 1], "ROLL_YIELD": ["L1.CMD", 0, "z", 1, 1],
        "TERM_PREMIUM": ["L1.FI", 0, "z", -1, -1], "MOVE": ["L1.FI", 0, "z", -1, -1],
        "CRACK": ["L2.CMD.ENERGY", 0, "z", 1, 1], "GLD_HOLDINGS": ["L2.CMD.PRECIOUS", 0, "diff", 1, 1],
        "HY_FLOWS": ["L2.FI.HY", 2, "z", 1, 1], "US_SECTOR_FLOWS": ["L2.EQ.US.BROAD", 0, "z", 1, 0],
    },
    # Prompt C outputs: node, sign per asset class, path in the digest JSON
    "digest_features": {
        "risk_appetite": ["L0.GLOBAL", RISK_ON, ["risk_appetite"]], "growth_tone": ["L0.GLOBAL", RISK_ON, ["growth_tone"]],
        "inflation_tone": ["L0.GLOBAL", SIGNS(-1, 1, -1, -1), ["inflation_tone"]],
        "geopolitical_stress": ["L0.GLOBAL", SIGNS(-1, 1, 1, -1), ["geopolitical_stress"]],
        "credit_tone": ["L1.FI", SIGNS(0, 0, 0, 1), ["credit_tone"]],
        "cb_fed": ["L0.GLOBAL", -1, ["cb_tone", "FED"]], "cb_ecb": ["L0.GLOBAL", -1, ["cb_tone", "ECB"]],
        "supply_energy": ["L2.CMD.ENERGY", 1, ["commodity_supply_tone", "energy"]],
        "supply_metals": ["L2.CMD.PRECIOUS", 1, ["commodity_supply_tone", "metals"]],
    },
    # Real data (data.source = "real"): a Yahoo ticker per asset (the energy asset uses Brent: WTI settled negative on
    # 2020-04-20), and the indicators with a free source; the other features are dropped in real mode.
    "real": {
        "assets": {"SPX": "^GSPC", "SX5E": "^STOXX50E", "WTI": "BZ=F", "GOLD": "GC=F", "UST10": "ZN=F", "HY": "HYG"},
        "indicators": {"VIX": ["yahoo", "^VIX"], "MOVE": ["yahoo", "^MOVE"], "USD": ["yahoo", "DX-Y.NYB"],
                       "HY_OAS": ["fred", "BAMLH0A0HYM2"], "NFCI": ["fred", "NFCI"], "TERM_PREMIUM": ["fred", "THREEFYTP10"],
                       "CRACK": ["crack", "CL=F RB=F HO=F"]},
    },
    "llm_feature_signs": {"level": 1, "volume": -1, "dispersion": -1, "mom5": 1, "surprise": 1},
    "synthetic": {"factor_persistence": 0.97, "signal_strength": 0.15,  # latent half-life ~3 weeks
                  "market_loading": {"EQ": 0.7, "CMD": 0.35, "GOVT": -0.3, "CREDIT": 0.5},
                  "sharpe": {"EQ": 0.4, "CMD": 0.2, "GOVT": 0.3, "CREDIT": 0.4},  # risk premium
                  # news tone: "oracle" = driven by each node's forward returns with correlation sentiment_ic
                  # (0 = noise, 1 = perfect foresight); "latent" = by the latent factors behind the returns
                  "sentiment": "oracle", "sentiment_ic": 0.2, "article_noise": 0.3},
}


def merge(base, override):
    for k, v in override.items():
        base[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path: Path | None = None) -> dict:
    """CONFIG, deep-merged with a JSON file of overrides."""
    return merge(json.loads(json.dumps(CONFIG)), json.loads(path.read_text()) if path else {})
