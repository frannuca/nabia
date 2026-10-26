"""Real data: daily prices from Yahoo Finance and series from FRED (cached as CSV), news articles from a JSON file.

    returns, raw = real_market(cfg)      # returns: business days x assets; raw: indicator levels by as-of date
    news = load_news(path)               # one row per article: id, publish_ts, received_ts, source, headline, body

Every downloaded series is cached in data.cache_dir (one CSV per symbol) and read from there on later runs, so a
run is reproducible and works offline once the cache exists; data.refresh = true downloads again.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

NEWS_COLUMNS = ["id", "publish_ts", "received_ts", "source", "headline", "body"]


def _cached(cfg, name, fetch):
    """A daily series from the CSV cache, or from `fetch()` (then cached)."""
    path = Path(cfg["data"]["cache_dir"]) / (re.sub(r"[^\w.-]", "_", name) + ".csv")
    if path.exists() and not cfg["data"]["refresh"]:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(name)
    series = fetch().dropna().rename(name)
    if series.empty:
        raise RuntimeError(f"no data for {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return series


def yahoo(cfg, ticker):
    """Daily closes (adjusted) of a Yahoo Finance ticker."""
    def fetch():
        import yfinance as yf
        closes = yf.download(ticker, start=cfg["data"]["start"], end=cfg["data"]["end"], progress=False,
                             auto_adjust=True)["Close"]
        return closes.squeeze("columns") if isinstance(closes, pd.DataFrame) else closes
    return _cached(cfg, ticker, fetch)


def fred(cfg, series_id):
    """A FRED series (daily, weekly or monthly), from its public CSV endpoint."""
    def fetch():
        raw = urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=60).read()
        df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return _cached(cfg, "fred_" + series_id, fetch)


def indicator(cfg, source, symbol):
    """An indicator level: a Yahoo or FRED series, or the 3-2-1 crack spread from crude, gasoline and heating oil
    futures, in dollars per barrel: (2 gasoline + 1 heating oil) x 42 gallons / 3 - crude."""
    if source == "yahoo":
        return yahoo(cfg, symbol)
    if source == "fred":
        return fred(cfg, symbol)
    if source == "crack":
        crude, gasoline, heating = (yahoo(cfg, s) for s in symbol.split())
        return ((2 * gasoline + heating) * 42 / 3 - crude).dropna()
    raise ValueError(f"unknown indicator source {source!r}")


def real_market(cfg):
    """Daily returns of the assets on the US trading calendar (other markets' holidays forward-filled, at most 3 days)
    and the levels of the configured indicators, each as of its last observation on or before every date."""
    R, D = cfg["real"], cfg["data"]
    prices = pd.DataFrame({a: yahoo(cfg, R["assets"][a]) for a in cfg["assets"]})
    calendar = yahoo(cfg, R["assets"][next(iter(cfg["assets"]))]).loc[D["start"]:D["end"]].index
    prices = prices.reindex(prices.index.union(calendar)).ffill(limit=3).reindex(calendar)
    returns = prices.pct_change().iloc[1:].dropna()
    raw = {}
    for fid, (source, symbol) in R["indicators"].items():
        if fid in cfg["features"]:
            level = indicator(cfg, source, symbol)
            raw[fid] = level.reindex(level.index.union(returns.index)).ffill().reindex(returns.index)
    return returns, pd.DataFrame(raw, index=returns.index)


def load_news(path):
    """Articles from a JSON file: a list of objects, or {"articles": [...]}, with publish_ts and received_ts (ISO 8601),
    source, headline and body. A missing file gives no articles (the LLM features are then neutral)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    data = json.loads(path.read_text())
    news = pd.DataFrame(data["articles"] if isinstance(data, dict) else data)
    for col in ("publish_ts", "received_ts"):
        news[col] = pd.to_datetime(news[col], utc=True).dt.tz_localize(None)
    if "id" not in news:
        news["id"] = range(len(news))
    news["body"] = news.get("body", news["headline"]).fillna(news["headline"])
    return news[NEWS_COLUMNS].sort_values("received_ts").reset_index(drop=True)
