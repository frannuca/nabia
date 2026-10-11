"""Market data sources: where daily prices come from."""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

import pandas as pd

#: Columns of the long-format frames returned by every source.
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


class MarketDataSource(Protocol):
    """Anything that returns daily prices in long format.

    ``fetch`` returns one row per (symbol, date) with the columns of
    :data:`PRICE_COLUMNS`; ``date`` holds ``datetime.date`` values, and fields a
    source does not provide are NaN. Symbols without data are simply absent.
    """

    name: str

    def fetch(self, symbols: Sequence[str], start: dt.date, end: dt.date | None = None) -> pd.DataFrame: ...


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS}).astype({"symbol": "object", "date": "object"})


class YahooFinanceSource:
    """Daily prices from Yahoo Finance through the ``yfinance`` package.

    Free and convenient, but best effort: Yahoo's terms restrict use to personal
    research, and futures tickers (``ZN=F``, ``CL=F``) are continuous
    front-month series that jump at each contract roll (they are not
    back-adjusted), which shows up as spurious returns on roll dates.
    """

    name = "yahoo"

    def __init__(self, batch_size: int = 20) -> None:
        self.batch_size = batch_size

    def fetch(self, symbols: Sequence[str], start: dt.date, end: dt.date | None = None) -> pd.DataFrame:
        import yfinance as yf  # optional dependency, imported on use

        frames = []
        for i in range(0, len(symbols), self.batch_size):
            batch = list(symbols[i : i + self.batch_size])
            raw = yf.download(
                batch,
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat() if end else None,  # yfinance's end is exclusive
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                progress=False,
                threads=True,
                multi_level_index=True,
            )
            frames.append(yahoo_to_long(raw, batch))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else empty_prices()


_YAHOO_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}


def yahoo_to_long(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Converts a ``yfinance.download(..., group_by="ticker")`` frame (columns
    (ticker, field)) to the long format of :data:`PRICE_COLUMNS`."""
    if raw is None or raw.empty:
        return empty_prices()
    rows = []
    for symbol in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                continue
            block = raw[symbol]
        else:
            block = raw
        block = block.rename(columns=_YAHOO_FIELDS)
        block = block.dropna(subset=[c for c in ("close", "adj_close") if c in block.columns], how="all")
        if block.empty:
            continue
        index = pd.DatetimeIndex(block.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        out = pd.DataFrame({"symbol": symbol, "date": index.date})
        for column in PRICE_COLUMNS[2:]:
            out[column] = block[column].to_numpy(dtype="float64") if column in block.columns else float("nan")
        rows.append(out)
    return pd.concat(rows, ignore_index=True)[PRICE_COLUMNS] if rows else empty_prices()


log = logging.getLogger(__name__)

_DAY_MS = 86_400_000


class CryptoSource:
    """Daily spot candles from a crypto exchange through the ``ccxt`` package.

    Public market data, no API key needed. Symbols use ccxt's ``BASE/QUOTE``
    form (``"BTC/USD"``). Candles are UTC days; ``adj_close`` equals ``close``
    and ``volume`` is in the base currency. The default exchange, Coinbase,
    serves the full daily history in pages of 300 candles; some exchanges only
    return a recent window (Kraken: the last 720 days), so prefer one with a
    long history for backtests.
    """

    name = "crypto"

    def __init__(self, exchange: str | Any = "coinbase", page_size: int = 300, retries: int = 3) -> None:
        """``exchange``: a ccxt exchange id, or an already configured exchange object."""
        self.exchange = exchange
        self.page_size = page_size
        self.retries = retries
        self._client: Any = None

    def _exchange(self) -> Any:
        if self._client is None:
            if isinstance(self.exchange, str):
                import ccxt  # optional dependency, imported on use

                self._client = getattr(ccxt, self.exchange)({"enableRateLimit": True})
            else:
                self._client = self.exchange
            self._client.load_markets()
        return self._client

    def _page(self, client: Any, symbol: str, since: int) -> list[list[float]]:
        for attempt in range(self.retries + 1):
            try:
                return client.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=self.page_size)
            except Exception as error:  # ccxt.NetworkError and friends; retry with backoff
                if attempt == self.retries:
                    raise
                log.warning("%s: %s, retrying", symbol, error)
                time.sleep(2.0**attempt)
        return []

    def fetch(self, symbols: Sequence[str], start: dt.date, end: dt.date | None = None) -> pd.DataFrame:
        client = self._exchange()
        start_ms = int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp() * 1000)
        last_day = end or dt.datetime.now(dt.UTC).date()
        stop_ms = int(dt.datetime.combine(last_day + dt.timedelta(days=1), dt.time(), dt.UTC).timestamp() * 1000)
        frames = []
        for symbol in symbols:
            if symbol not in client.markets:
                log.warning("%s is not listed on %s", symbol, getattr(client, "id", "the exchange"))
                continue
            candles: dict[int, list[float]] = {}
            since = start_ms
            while since < stop_ms:
                page = self._page(client, symbol, since)
                if not page:
                    # Nothing in this window: before the listing date (or a gap). Step
                    # forward a page; stop once past what has been collected so far.
                    if candles:
                        break
                    since += self.page_size * _DAY_MS
                    continue
                for candle in page:
                    if start_ms <= candle[0] < stop_ms:
                        candles[int(candle[0])] = candle
                next_since = int(page[-1][0]) + _DAY_MS
                if next_since <= since:  # no progress: the exchange ignores `since`
                    break
                since = next_since
            if not candles:
                continue
            rows = [candles[k] for k in sorted(candles)]
            frames.append(pd.DataFrame({
                "symbol": symbol,
                "date": [dt.datetime.fromtimestamp(r[0] / 1000, dt.UTC).date() for r in rows],
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
                "adj_close": [r[4] for r in rows],
                "volume": [r[5] for r in rows],
            }).astype({c: "float64" for c in PRICE_COLUMNS[2:]}))
        return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS] if frames else empty_prices()
