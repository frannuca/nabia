"""Real market data for the examples.

Prices are downloaded from Yahoo Finance (yfinance) and cached as CSV in
examples/.data/, so the examples run offline after the first time. Without a
connection and without a cache, the shared database data/market.db is used (filled by
``python -m nabia.marketdata``). The tickers below exist in both.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".data"
MARKET_DB = HERE.parents[1] / "data" / "market.db"  # shared with webui/

EQUITIES = ["^GSPC", "^NDX", "^RUT", "^STOXX50E", "^GDAXI", "^FTSE", "^N225"]
BONDS = ["ZT=F", "ZF=F", "ZN=F", "ZB=F"]
COMMODITIES = ["GC=F", "SI=F", "HG=F"]
UNIVERSE = EQUITIES + BONDS + COMMODITIES


def _download(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].set_axis(tickers, axis=1)
    if close.dropna(how="all").empty:
        raise RuntimeError("empty download")
    return close


def _from_market_db(tickers: list[str], start: str) -> pd.DataFrame:
    if not MARKET_DB.exists():
        raise RuntimeError(f"no connection and no {MARKET_DB}")
    with sqlite3.connect(MARKET_DB) as db:
        marks = ",".join("?" * len(tickers))
        long = pd.read_sql_query(
            f"SELECT symbol, date, close FROM prices WHERE symbol IN ({marks}) AND date >= ?", db,
            params=[*tickers, start])
    wide = long.pivot(index="date", columns="symbol", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide


def load_prices(tickers: list[str] = UNIVERSE, start: str = "2012-01-01") -> pd.DataFrame:
    """Daily closes, DatetimeIndex x tickers, on the dates where every ticker trades."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{'_'.join(t.strip('^').replace('=', '') for t in tickers)}_{start}.csv"
    if path.exists():
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        try:
            prices = _download(tickers, start)
            source = "Yahoo Finance"
        except Exception as error:  # offline, rate limited, yfinance missing...
            print(f"[data] download failed ({error}); using {MARKET_DB.name}")
            prices = _from_market_db(tickers, start)
            source = MARKET_DB.name
        prices.to_csv(path)
        print(f"[data] {len(prices)} days of {len(tickers)} series from {source}, cached in {path.name}")
    prices = prices[tickers].dropna()
    # Continuous futures can print non-positive prices around rolls; returns need positive prices.
    return prices[(prices > 0).all(axis=1)]


def load_returns(tickers: list[str] = UNIVERSE, start: str = "2012-01-01") -> pd.DataFrame:
    """Daily simple returns on common trading dates."""
    return load_prices(tickers, start).pct_change().dropna()


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
