"""Settings. CONFIG: the methodology shared by every run (factor tree, assets, features, windows, model, budgets,
portfolio, LLM). SYNTHETIC and REAL: the data of each kind of run. A run's JSON file picks one with data.source and
overrides any key: load_config(path) = CONFIG + SYNTHETIC or REAL + the file (configs/synthetic.json, configs/real.json)."""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SIGNS = lambda eq, cmd, govt, credit: {"EQ": eq, "CMD": cmd, "GOVT": govt, "CREDIT": credit}


RISK_ON = SIGNS(1, 1, -1, 1)  # risk-on is bullish for everything but government bonds

CONFIG = {
    "seed": 7, "horizon": 21,
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
            "dedup_hours": 48, "max_retries": 2, "live_days": 5, "anonymize": False, "news_per_day": 4,
            "source_weight": {"wire": 1.0, "paper": 0.8, "blog": 0.4},
            # live calls are cached here (key: prompt version, model, request), so each article is paid for once;
            # batch: submit a run's uncached live requests as Message Batches first; live_days 0 = every day live
            "cache": "data/llm_cache.jsonl", "batch": False, "batch_poll_seconds": 30,
            # live provider: "anthropic" (models above) or "ollama" (a local model, structured output, temperature 0);
            # workers: articles scored concurrently by live calls (Anthropic; a local Ollama model is fastest with 1:
            # with OLLAMA_NUM_PARALLEL > 1 every request slowed down several times on the development machine)
            "provider": "anthropic", "ollama_model": "gemma4:26b", "ollama_host": "http://localhost:11434",
            "ollama_options": {"temperature": 0, "num_ctx": 8192}, "ollama_think": False, "ollama_timeout": 600,
            "workers": 1, "audit_dir": "output/llm_audit",
            # a node's scores also feed its ancestors of the same asset class (x rollup per level; 0 = off), so the
            # detail nodes without an asset (sectors, central banks, macro data) reach the assets above them
            "rollup": 0.5},
    "taxonomy": {},  # flattened from TREE below: node -> [parent, name, asset class (None = mixed), kind]
    "aliases": {},   # node -> other names under which it appears in headlines (the offline scorer matches names)
    # The portfolio: not part of the factor tree. Per asset its name, class (EQ, CMD, GOVT, CREDIT), base risk budget,
    # annual volatility (synthetic data), Yahoo ticker (real data) and the sentiment factors it may load on, with the
    # sign of the exposure: the features of these nodes are its candidates, and the elastic net learns the mapping.
    # A run's JSON file that sets "assets" replaces this whole table.
    "assets": {
        "SPX": {"name": "S&P 500", "class": "EQ", "budget": 0.20, "vol": 0.16, "ticker": "^GSPC",
                "factors": {"GLOBAL": 1, "US": 1, "US.EQ": 1}},
        "SX5E": {"name": "Euro Stoxx 50", "class": "EQ", "budget": 0.15, "vol": 0.18, "ticker": "^STOXX50E",
                 "factors": {"GLOBAL": 1, "EU": 1, "EU.EQ": 1, "EU.FX.EUR": -1}},  # a strong euro hurts exporters
        "WTI": {"name": "Crude oil", "class": "CMD", "budget": 0.10, "vol": 0.35, "ticker": "BZ=F",  # Brent: WTI went negative
                "factors": {"GLOBAL": 1, "CMD": 1, "CMD.ENERGY": 1, "CMD.ENERGY.CRUDE": 1, "US.FX.USD": -1, "APAC.MACRO.CN": 1}},
        "GOLD": {"name": "Gold", "class": "CMD", "budget": 0.15, "vol": 0.15, "ticker": "GC=F",
                 "factors": {"GLOBAL": 1, "CMD": 1, "CMD.PRECIOUS": 1, "CMD.PRECIOUS.GOLD": 1, "US.FX.USD": -1}},
        "UST10": {"name": "US 10y Treasury note", "class": "GOVT", "budget": 0.25, "vol": 0.07, "ticker": "ZN=F",
                  "factors": {"GLOBAL": 1, "US": 1, "US.RATES": 1, "US.RATES.LONG": 1}},
        "HY": {"name": "US high yield bonds", "class": "CREDIT", "budget": 0.15, "vol": 0.06, "ticker": "HYG",
               "factors": {"GLOBAL": 1, "US": 1, "US.CREDIT": 1, "US.CREDIT.HY": 1}},
    },
    # feature: node, publication lag (days), transform ("z" | "diff"), sign per asset class (oriented: + = bullish);
    # "truth" is the synthetic loading on the node's latent factor (0 = pure noise).
    "features": {
        "VIX": ["GLOBAL", 0, "z", SIGNS(-1, -1, 1, -1), -1], "HY_OAS": ["GLOBAL", 0, "z", SIGNS(-1, -1, 1, -1), -1],
        "NFCI": ["GLOBAL", 5, "z", SIGNS(-1, -1, 1, -1), -1], "AAII": ["GLOBAL", 5, "z", RISK_ON, 1],
        "USD": ["US.FX.USD", 0, "z", 1, 0],
        "PUT_CALL": ["US.EQ", 0, "z", -1, -1], "BREADTH": ["US.EQ", 0, "z", 1, 1],
        "COT_NET": ["CMD", 3, "z", 1, 1], "ROLL_YIELD": ["CMD", 0, "z", 1, 1],
        "TERM_PREMIUM": ["US.RATES", 0, "z", -1, -1], "MOVE": ["US.RATES", 0, "z", -1, -1],
        "CRACK": ["CMD.ENERGY", 0, "z", 1, 1], "GLD_HOLDINGS": ["CMD.PRECIOUS", 0, "diff", 1, 1],
        "HY_FLOWS": ["US.CREDIT", 2, "z", 1, 1], "US_SECTOR_FLOWS": ["US.EQ", 0, "z", 1, 0],
    },
    # Prompt C outputs: node, sign per asset class, path in the digest JSON
    "digest_features": {
        "risk_appetite": ["GLOBAL", RISK_ON, ["risk_appetite"]], "growth_tone": ["GLOBAL", RISK_ON, ["growth_tone"]],
        "inflation_tone": ["GLOBAL", SIGNS(-1, 1, -1, -1), ["inflation_tone"]],
        "geopolitical_stress": ["GLOBAL", SIGNS(-1, 1, 1, -1), ["geopolitical_stress"]],
        "credit_tone": ["US.CREDIT", 1, ["credit_tone"]],
        "cb_fed": ["US", -1, ["cb_tone", "FED"]], "cb_ecb": ["EU", -1, ["cb_tone", "ECB"]],
        "supply_energy": ["CMD.ENERGY", 1, ["commodity_supply_tone", "energy"]],
        "supply_metals": ["CMD.PRECIOUS", 1, ["commodity_supply_tone", "metals"]],
    },
    "llm_feature_signs": {"level": 1, "volume": -1, "dispersion": -1, "mom5": 1, "surprise": 1},
}

# Synthetic runs: generated market data and news with a known truth (offline).
SYNTHETIC = {
    "data": {"source": "synthetic"}, "start": "2015-01-01", "days": 2600,
    "synthetic": {"factor_persistence": 0.97, "signal_strength": 0.15,  # latent half-life ~3 weeks
                  "market_loading": {"EQ": 0.7, "CMD": 0.35, "GOVT": -0.3, "CREDIT": 0.5},
                  "sharpe": {"EQ": 0.4, "CMD": 0.2, "GOVT": 0.3, "CREDIT": 0.4},  # risk premium
                  # news tone: "oracle" = driven by each node's forward returns with correlation sentiment_ic
                  # (0 = noise, 1 = perfect foresight); "latent" = by the latent factors behind the returns
                  "sentiment": "oracle", "sentiment_ic": 0.2, "article_noise": 0.3},
}

# Real runs: prices and indicators from Yahoo Finance and FRED (cached as CSV in cache_dir; refresh re-downloads),
# news articles from news_file (JSON), their LLM scores (scores_file) and daily digests (digests_file), and the
# calibrated elastic-net betas (betas_file): the files of the staged pipeline (scripts/pipeline/*_1.py .. *_5.py).
# collect: the news download of step 1; backtest: the as-of date, lookback window and output folder of the
# positions (step 5 and scripts/sentiment_weights.py).
REAL = {
    "data": {"source": "real", "start": "2014-01-01", "end": None, "cache_dir": "data/cache", "refresh": False,
             "news_file": "data/example_news.json", "scores_file": None, "digests_file": None, "betas_file": None},
    # the indicators with a free source (the asset tickers are in "assets"); the other features are dropped
    "real": {
        "indicators": {"VIX": ["yahoo", "^VIX"], "MOVE": ["yahoo", "^MOVE"], "USD": ["yahoo", "DX-Y.NYB"],
                       "HY_OAS": ["fred", "BAMLH0A0HYM2"], "NFCI": ["fred", "NFCI"], "TERM_PREMIUM": ["fred", "THREEFYTP10"],
                       "CRACK": ["crack", "CL=F RB=F HO=F"]},
    },
    "collect": {"rss": True, "feeds": None, "gkg": True, "since": "2024-01-01", "until": None, "per_day": 10,
                "weekends": False, "workers": 4},
    "backtest": {"lookback_days": 252, "as_of": None, "out": "output/backtest"},
}


def N(key, name, cls, kind, aliases=(), *children):
    """A factor node of TREE: its key (the ID is the parent's ID + "." + key), name, asset class, kind, aliases."""
    return key, name, cls, kind, list(aliases), children


# The sentiment factor tree: a generic global factor at the root; regions (US, EU, APAC) and the commodity complex;
# per region its macro data, central banks, equity sectors, government bonds, credit and currencies; single markets
# (crude oil, gold, high yield, ...) at the leaves. Kinds: global, region, market, macro, policy, sector, currency.
# The portfolio's assets are not in the tree: "assets" lists the factors each one may load on.
REGION = lambda key, name, aliases, macro, cb, eq, rates, credit, fx: N(key, name, None, "region", aliases,
    N("MACRO", f"{name} macro data", None, "macro", (), *macro), N("CB", f"{name} central banks", None, "policy", (), *cb),
    N("EQ", f"{name} equities", "EQ", "sector", eq[0], *eq[1:]), N("RATES", f"{name} government bonds", "GOVT", "sector", rates[0], *rates[1:]),
    N("CREDIT", f"{name} corporate credit", "CREDIT", "sector", credit[0], *credit[1:]), N("FX", f"{name} currencies", "FX", "currency", fx[0], *fx[1:]))
TREE = N("GLOBAL", "Global markets", None, "global", ["global markets", "world markets", "risk sentiment", "world economy"],
    REGION("US", "United States", ["us economy", "u.s. economy", "american economy"],
           [N("GROWTH", "US growth", None, "macro", ["us gdp", "us growth", "us pmi"]),
            N("LABOR", "US labor market", None, "macro", ["payrolls", "jobless claims", "us unemployment"]),
            N("INFLATION", "US inflation", None, "macro", ["us cpi", "us inflation", "pce"])],
           [N("FED", "Federal Reserve", None, "policy", ["the fed", "fomc", "powell"])],
           [["us equities", "us stocks", "wall street", "stocks", "equities", "s&p 500", "s&p"],
            N("TECH", "US technology stocks", "EQ", "sector", ["nasdaq", "big tech", "tech stocks", "chipmakers"]),
            N("FIN", "US banks", "EQ", "sector", ["bank stocks", "wall street banks"]), N("ENERGY", "US energy stocks", "EQ", "sector", ["oil majors"]),
            N("HEALTH", "US healthcare stocks", "EQ", "sector", ["pharma stocks", "biotech"]),
            N("CONSUMER", "US consumer stocks", "EQ", "sector", ["retailers"]), N("INDUSTRIALS", "US industrials", "EQ", "sector")],
           [["us treasuries", "treasuries", "treasury yields", "bonds"], N("LONG", "US long-dated Treasuries", "GOVT", "sector", ["10-year treasury", "10y treasury", "10-year yield", "30-year treasury"]),
            N("FRONT", "US short-dated Treasuries", "GOVT", "sector", ["2-year treasury", "t-bills"])],
           [["credit markets", "corporate bonds"], N("IG", "US investment grade", "CREDIT", "sector", ["investment grade", "investment-grade bonds"]),
            N("HY", "US high yield", "CREDIT", "sector", ["high yield credit", "high yield", "junk bonds", "high-yield bonds"])],
           [[], N("USD", "US dollar", "FX", "currency", ["dollar", "greenback", "dollar index"])]),
    REGION("EU", "Europe", ["eurozone", "euro zone", "europe", "european economy"],
           [N("GROWTH", "Euro area growth", None, "macro", ["euro area gdp", "eurozone pmi"]),
            N("INFLATION", "Euro area inflation", None, "macro", ["eurozone inflation", "euro area inflation"]),
            N("UK", "UK economy", None, "macro", ["uk economy", "british economy"])],
           [N("ECB", "European Central Bank", None, "policy", ["ecb", "lagarde"]), N("BOE", "Bank of England", None, "policy", ["boe"]),
            N("SNB", "Swiss National Bank", None, "policy", ["snb"])],
           [["european stocks", "european equities", "euro stoxx 50", "euro stoxx", "stoxx 50"],
            N("BANKS", "European banks", "EQ", "sector"), N("INDUSTRIALS", "European industrials", "EQ", "sector"),
            N("LUXURY", "European luxury stocks", "EQ", "sector", ["luxury stocks"]), N("ENERGY", "European energy stocks", "EQ", "sector"),
            N("UK", "UK equities", "EQ", "sector", ["ftse 100", "ftse"]), N("DE", "German equities", "EQ", "sector", ["dax"])],
           [[], N("BUNDS", "German Bunds", "GOVT", "sector", ["bunds", "bund yields"]),
            N("PERIPHERY", "Euro periphery bonds", "GOVT", "sector", ["btps", "italian bonds"]), N("GILTS", "UK gilts", "GOVT", "sector", ["gilts", "gilt yields"])],
           [["european credit"]],
           [[], N("EUR", "Euro", "FX", "currency", ["euro"]), N("GBP", "British pound", "FX", "currency", ["sterling", "pound"]),
            N("CHF", "Swiss franc", "FX", "currency")]),
    REGION("APAC", "Asia-Pacific", ["asia-pacific", "asia pacific", "asia"],
           [N("CN", "China economy", None, "macro", ["chinese economy", "china gdp", "china pmi"]),
            N("JP", "Japan economy", None, "macro", ["japanese economy", "japan gdp"]), N("IN", "India economy", None, "macro", ["indian economy"])],
           [N("BOJ", "Bank of Japan", None, "policy", ["boj"]), N("PBOC", "People's Bank of China", None, "policy", ["pboc"]),
            N("RBA", "Reserve Bank of Australia", None, "policy", ["rba"])],
           [["asian stocks", "asian equities"], N("JP", "Japanese equities", "EQ", "sector", ["nikkei", "topix", "japanese stocks"]),
            N("CN", "Chinese equities", "EQ", "sector", ["chinese stocks", "hang seng", "csi 300", "shanghai composite"]),
            N("TECH", "Asian technology stocks", "EQ", "sector", ["asian chipmakers"]), N("IN", "Indian equities", "EQ", "sector", ["sensex", "nifty"])],
           [[], N("JGB", "Japanese government bonds", "GOVT", "sector", ["jgbs", "jgb yields"]),
            N("CGB", "Chinese government bonds", "GOVT", "sector")],
           [["asian credit"]],
           [[], N("JPY", "Japanese yen", "FX", "currency", ["yen"]), N("CNY", "Chinese yuan", "FX", "currency", ["yuan", "renminbi"]),
            N("AUD", "Australian dollar", "FX", "currency", ["aussie dollar"])]),
    N("CMD", "Commodities", "CMD", "market", ["commodities"],
      N("ENERGY", "Energy", "CMD", "sector", ["energy", "oil", "opec"],
        N("CRUDE", "Crude oil", "CMD", "sector", ["wti", "crude", "brent"]), N("NATGAS", "Natural gas", "CMD", "sector", ["lng"]),
        N("REFINED", "Refined products", "CMD", "sector", ["gasoline", "diesel"])),
      N("PRECIOUS", "Precious metals", "CMD", "sector", [],
        N("GOLD", "Gold", "CMD", "sector", ["bullion"]), N("SILVER", "Silver", "CMD", "sector")),
      N("BASE", "Industrial metals", "CMD", "sector", ["base metals"],
        N("COPPER", "Copper", "CMD", "sector"), N("ALUMINIUM", "Aluminium", "CMD", "sector", ["aluminum"]),
        N("IRON", "Iron ore", "CMD", "sector")),
      N("AGS", "Agriculture", "CMD", "sector", ["grains", "crops"],
        N("GRAINS", "Grain crops", "CMD", "sector", ["wheat", "corn", "soybeans"]), N("SOFTS", "Softs", "CMD", "sector", ["coffee", "sugar", "cocoa"]))))


def flatten(tree, parent=None):
    """{node ID: [parent, name, asset class, kind]} and {node ID: aliases} of a TREE, parents before children."""
    key, name, cls, kind, aliases, children = tree
    node = key if parent in (None, "GLOBAL") else f"{parent}.{key}"
    taxonomy, names = {node: [parent, name, cls, kind]}, {node: aliases} if aliases else {}
    for child in children:
        t, a = flatten(child, node)
        taxonomy, names = taxonomy | t, names | a
    return taxonomy, names


CONFIG["taxonomy"], CONFIG["aliases"] = flatten(TREE)


def merge(base, override):
    for k, v in override.items():
        base[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path: Path | None = None, source: str | None = None) -> dict:
    """CONFIG + the data defaults of the run's kind (SYNTHETIC or REAL: `source`, else data.source in the file, else
    synthetic) + the JSON file's overrides."""
    file = json.loads(Path(path).read_text()) if path else {}
    source = source or file.get("data", {}).get("source", "synthetic")
    if source not in ("synthetic", "real"):
        raise ValueError(f"data.source must be synthetic or real, not {source!r}")
    defaults = merge(json.loads(json.dumps(CONFIG)), json.loads(json.dumps(SYNTHETIC if source == "synthetic" else REAL)))
    if "assets" in file:  # the portfolio is replaced, not merged asset by asset
        defaults["assets"] = {}
    cfg = merge(merge(defaults, file), {"data": {"source": source}})
    for a, spec in cfg["assets"].items():
        unknown = [n for n in spec["factors"] if n not in cfg["taxonomy"]]
        if unknown:
            raise ValueError(f"asset {a}: factors not in the taxonomy: {unknown}")
    return cfg
