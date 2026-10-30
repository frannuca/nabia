"""Weekly returns of the factor proxies (FRED series and market.db prices, combined as factors.json says), of the
portfolio assets, and of the news sentiment per factor."""

from __future__ import annotations

import csv
import io
import sys
import urllib.request

import numpy as np
import pandas as pd

from scoring import scoring_version

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
WEEK = "W-FRI"  # weeks end on Friday


def download_fred(db, factors: dict) -> None:
    """Downloads every FRED series the proxies use into the series table (the whole history: a few hundred KB)."""
    ids = sorted({key[5:] for f in factors.values() for key, _, _ in f.get("proxy", []) if key.startswith("fred:")})
    for sid in ids:
        text = urllib.request.urlopen(FRED.format(sid), timeout=120).read().decode()
        rows = [(d, float(v)) for d, v in list(csv.reader(io.StringIO(text)))[1:] if v not in ("", ".")]
        print(f"fred {sid:12} {db.save_series('fred:' + sid, rows)} observations, last {rows[-1][0]}", file=sys.stderr)


def levels(db, key: str) -> pd.Series:
    """A daily series of the database: 'price:<symbol>' or a downloaded series ('fred:<id>')."""
    rows = db.prices(key[6:]) if key.startswith("price:") else db.series(key)
    return pd.Series([v for _, v in rows], pd.to_datetime([d for d, _ in rows]), dtype=float)


def weekly(s: pd.Series, transform: str) -> pd.Series:
    """Friday-to-Friday change of the last observation of each week: log return or difference."""
    w = s.resample(WEEK).last()
    return np.log(w).diff() if transform == "logret" else w.diff()


def proxy_returns(db, factors: dict) -> pd.DataFrame:
    """One column per factor with a proxy, in the order of factors.json (the orthogonalization order)."""
    return pd.DataFrame({name: sum(w * weekly(levels(db, key), t) for key, t, w in f["proxy"])
                         for name, f in factors.items() if f.get("proxy")})


def asset_returns(db, assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame({a: weekly(levels(db, "price:" + a), "logret") for a in assets})


def article_scores(news_db, cfg: dict) -> pd.DataFrame:
    """One row per article and factor (received_at as a naive UTC timestamp, sentiment, confidence), from the scores of
    a single model and prompt version (scoring_version): scores of other versions are ignored, never averaged."""
    model, prompt = scoring_version(cfg)
    rows = pd.DataFrame(news_db.scores(model=model, prompt=prompt),
                        columns=["news_id", "received_at", "factor", "sentiment", "confidence"])
    rows = rows[rows.factor.isin(cfg["factors"])]
    return rows.assign(received_at=pd.to_datetime(rows.received_at, utc=True).dt.tz_localize(None))


def news_sentiment(news_db, cfg: dict) -> pd.DataFrame:
    """Weekly confidence-weighted mean sentiment per factor (NaN for weeks without news)."""
    factors = cfg["factors"]
    rows = article_scores(news_db, cfg)
    rows["week"] = rows.received_at.dt.to_period(WEEK).dt.end_time.dt.normalize()
    rows["ws"] = rows.sentiment * rows.confidence
    g = rows.groupby(["week", "factor"])[["ws", "confidence"]].sum()
    return (g.ws / g.confidence).unstack("factor").reindex(columns=list(factors))
