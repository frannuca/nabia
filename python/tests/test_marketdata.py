import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from nabia.marketdata import (
    DEFAULT_UNIVERSE,
    PRICE_COLUMNS,
    Asset,
    CryptoSource,
    MarketDataStore,
    YahooFinanceSource,
    find_assets,
    update_market_data,
)
from nabia.marketdata.sources import yahoo_to_long

ASSETS = [
    Asset("^GSPC", "S&P 500", "equity_index", "USD"),
    Asset("ZN=F", "US 10-Year T-Note future", "bond_future", "USD"),
    Asset("GC=F", "Gold future", "commodity", "USD"),
]


class FakeSource:
    """Deterministic prices on business days; records every request."""

    name = "fake"

    def __init__(self, revision: float = 0.0):
        self.calls: list[tuple[tuple[str, ...], dt.date, dt.date | None]] = []
        self.revision = revision

    def price(self, dates, k=0):
        """A fixed function of the date (plus the revision), whatever the request."""
        days = (pd.DatetimeIndex(dates) - pd.Timestamp("2024-01-01")).days.to_numpy()
        return 100.0 + 10 * k + 0.1 * days + self.revision

    def fetch(self, symbols, start, end=None):
        self.calls.append((tuple(symbols), start, end))
        dates = pd.bdate_range(start, end or dt.date(2024, 3, 29))
        rows = []
        for k, symbol in enumerate(symbols):
            close = self.price(dates, k)
            rows.append(pd.DataFrame({"symbol": symbol, "date": dates.date, "open": close, "high": close + 1,
                                      "low": close - 1, "close": close, "adj_close": close, "volume": 1000.0}))
        return pd.concat(rows, ignore_index=True)[PRICE_COLUMNS]


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(f"sqlite:///{tmp_path / 'market.db'}")


def test_universe_covers_the_three_asset_classes():
    classes = {a.asset_class for a in DEFAULT_UNIVERSE}
    assert classes == {"equity_index", "bond_future", "commodity", "crypto"}
    assert {a.source for a in DEFAULT_UNIVERSE if a.asset_class == "crypto"} == {"crypto"}
    assert {a.source for a in DEFAULT_UNIVERSE if a.asset_class != "crypto"} == {"yahoo"}
    assert len({a.symbol for a in DEFAULT_UNIVERSE}) == len(DEFAULT_UNIVERSE)
    assert [a.symbol for a in find_assets(["ZN=F", "^GSPC"])] == ["ZN=F", "^GSPC"]
    with pytest.raises(KeyError):
        find_assets(["NOPE"])


def test_store_upserts_and_loads_wide_prices(store):
    store.upsert_assets(ASSETS, source="fake")
    prices = FakeSource().fetch(["^GSPC", "ZN=F"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert store.upsert_prices(prices) == len(prices)
    assert store.upsert_prices(prices) == len(prices)  # idempotent

    wide = store.load_prices(["ZN=F", "^GSPC"])
    assert list(wide.columns) == ["ZN=F", "^GSPC"]
    assert isinstance(wide.index, pd.DatetimeIndex)
    assert len(wide) == len(pd.bdate_range("2024-01-01", "2024-01-31"))
    assert wide.loc["2024-01-01", "^GSPC"] == 100.0

    revised = prices.assign(close=prices["close"] + 1.0)  # a data revision overwrites
    store.upsert_prices(revised)
    assert store.load_prices(["^GSPC"]).loc["2024-01-01", "^GSPC"] == 101.0
    assert store.load_prices(["^GSPC"], start="2024-01-15", end="2024-01-19").shape == (5, 1)
    assert store.last_dates() == {"^GSPC": dt.date(2024, 1, 31), "ZN=F": dt.date(2024, 1, 31)}
    assert set(store.assets()["symbol"]) == {"^GSPC", "ZN=F", "GC=F"}
    assert store.load_prices(["GC=F"]).empty


def test_missing_values_are_stored_as_null_and_load_as_nan(store):
    store.upsert_assets(ASSETS[:2], source="fake")
    prices = FakeSource().fetch(["^GSPC", "ZN=F"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    prices.loc[(prices["symbol"] == "ZN=F") & (prices["date"] == dt.date(2024, 1, 3)), "close"] = np.nan
    store.upsert_prices(prices)
    wide = store.load_prices(["^GSPC", "ZN=F"])
    assert np.isnan(wide.loc["2024-01-03", "ZN=F"])
    assert wide["^GSPC"].notna().all()


def test_update_is_incremental_with_overlap(store):
    source = FakeSource()
    written = update_market_data(store, source, ASSETS, start=dt.date(2024, 1, 1), end=dt.date(2024, 2, 29))
    assert all(n > 0 for n in written.values())
    assert source.calls == [(("^GSPC", "ZN=F", "GC=F"), dt.date(2024, 1, 1), dt.date(2024, 2, 29))]

    # Second run: from 5 days before the last stored date, picking up revisions.
    revised = FakeSource(revision=0.5)
    update_market_data(store, revised, ASSETS, start=dt.date(2024, 1, 1), end=dt.date(2024, 3, 29))
    assert revised.calls == [(("^GSPC", "ZN=F", "GC=F"), dt.date(2024, 2, 24), dt.date(2024, 3, 29))]
    wide = store.load_prices(["^GSPC"])
    assert wide.index[-1] == pd.Timestamp("2024-03-29")
    base = FakeSource()
    assert wide.loc["2024-02-23", "^GSPC"] == pytest.approx(base.price(["2024-02-23"])[0])         # before the overlap: kept
    assert wide.loc["2024-02-26", "^GSPC"] == pytest.approx(base.price(["2024-02-26"])[0] + 0.5)   # overlap: revised
    assert wide.loc["2024-03-29", "^GSPC"] == pytest.approx(base.price(["2024-03-29"])[0] + 0.5)   # new dates

    full = FakeSource()
    update_market_data(store, full, ASSETS[:1], start=dt.date(2024, 1, 1), full=True)
    assert full.calls[0][1] == dt.date(2024, 1, 1)


def test_yahoo_frames_are_converted_to_long_format():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([["^GSPC", "ZN=F"], fields])
    raw = pd.DataFrame(np.arange(24, dtype=float).reshape(2, 12), index=dates, columns=columns)
    raw.loc[dates[1], ("ZN=F", "Close")] = np.nan
    raw.loc[dates[1], ("ZN=F", "Adj Close")] = np.nan  # no price that day: row dropped
    long = yahoo_to_long(raw, ["^GSPC", "ZN=F", "MISSING"])
    assert list(long.columns) == PRICE_COLUMNS
    assert len(long) == 3
    assert long.iloc[0].to_dict() == {"symbol": "^GSPC", "date": dt.date(2024, 1, 2), "open": 0.0, "high": 1.0,
                                      "low": 2.0, "close": 3.0, "adj_close": 4.0, "volume": 5.0}
    assert set(long["symbol"]) == {"^GSPC", "ZN=F"}


@pytest.mark.skipif(os.environ.get("NABIA_NETWORK_TESTS") != "1", reason="set NABIA_NETWORK_TESTS=1 to download")
def test_yahoo_download(store):  # pragma: no cover - network
    written = update_market_data(store, YahooFinanceSource(), find_assets(["^GSPC", "ZN=F", "GC=F"]),
                                 start=dt.date(2024, 1, 1), end=dt.date(2024, 3, 31))
    assert all(n > 40 for n in written.values())
    assert store.load_prices(["^GSPC", "ZN=F", "GC=F"]).dropna().shape[0] > 40


DAY_MS = 86_400_000


class FakeExchange:
    """ccxt-like exchange: daily candles from 2024-01-01, served in pages."""

    id = "fake"

    def __init__(self, ignore_since: bool = False, fail_first: int = 0, listed: str = "2024-01-01"):
        self.markets = {"BTC/USD": {}, "ETH/USD": {}}
        self.loaded = False
        self.requests: list[int] = []
        self.ignore_since = ignore_since
        self.fail_first = fail_first
        self.listed = int(pd.Timestamp(listed, tz="UTC").timestamp() * 1000)

    def load_markets(self):
        self.loaded = True

    def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None):
        assert timeframe == "1d"
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ConnectionError("temporary network error")
        self.requests.append(since)
        first = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        last = int(pd.Timestamp("2024-03-31", tz="UTC").timestamp() * 1000)
        t = first if self.ignore_since else max(since, first)
        if symbol == "ETH/USD":
            t = max(t, self.listed)
            if since + limit * DAY_MS <= self.listed:
                return []  # nothing before the listing date
        base = 40_000.0 if symbol == "BTC/USD" else 2_000.0
        page = []
        while t <= last and len(page) < limit:
            day = (t - first) // DAY_MS
            price = base + day
            page.append([t, price, price + 10, price - 10, price + 1, 5.0])
            t += DAY_MS
        return page


def test_crypto_source_pages_through_history():
    exchange = FakeExchange()
    source = CryptoSource(exchange, page_size=30)
    prices = source.fetch(["BTC/USD", "ETH/USD", "DOGE/USD"], dt.date(2024, 1, 10), dt.date(2024, 3, 15))
    assert exchange.loaded
    assert list(prices.columns) == PRICE_COLUMNS
    assert set(prices["symbol"]) == {"BTC/USD", "ETH/USD"}  # DOGE/USD is not listed: skipped
    btc = prices[prices["symbol"] == "BTC/USD"]
    assert btc["date"].iloc[0] == dt.date(2024, 1, 10)
    assert btc["date"].iloc[-1] == dt.date(2024, 3, 15)  # end is inclusive, nothing after it
    assert len(btc) == (dt.date(2024, 3, 15) - dt.date(2024, 1, 10)).days + 1  # every day, weekends included
    assert btc["date"].is_unique
    assert btc.iloc[0][["open", "close", "adj_close", "volume"]].tolist() == [40_009.0, 40_010.0, 40_010.0, 5.0]
    assert len(exchange.requests) >= 2 * 3  # several pages per symbol


def test_crypto_source_stops_when_the_exchange_ignores_since_and_retries_errors():
    exchange = FakeExchange(ignore_since=True, fail_first=1)
    source = CryptoSource(exchange, page_size=500, retries=2)
    prices = source.fetch(["BTC/USD"], dt.date(2024, 2, 1), dt.date(2024, 2, 29))
    assert len(prices) == 29
    assert len(exchange.requests) <= 3  # no endless loop on a source that always returns the same window


def test_update_routes_each_asset_to_its_source(store):
    crypto = CryptoSource(FakeExchange(), page_size=50)
    yahoo = FakeSource()
    assets = [ASSETS[0], Asset("BTC/USD", "Bitcoin", "crypto", "USD", source="crypto")]
    written = update_market_data(store, {"yahoo": yahoo, "crypto": crypto}, assets,
                                 start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31))
    assert written == {"^GSPC": len(pd.bdate_range("2024-01-01", "2024-01-31")), "BTC/USD": 31}
    assert yahoo.calls[0][0] == ("^GSPC",)
    assert dict(zip(store.assets()["symbol"], store.assets()["source"])) == {"^GSPC": "yahoo", "BTC/USD": "crypto"}
    wide = store.load_prices(["^GSPC", "BTC/USD"])
    assert wide["BTC/USD"].notna().sum() == 31
    assert wide["^GSPC"].isna().sum() == 31 - len(pd.bdate_range("2024-01-01", "2024-01-31"))  # weekends
    with pytest.raises(ValueError, match="no source"):
        update_market_data(store, {"yahoo": yahoo}, assets, start=dt.date(2024, 1, 1))


@pytest.mark.skipif(os.environ.get("NABIA_NETWORK_TESTS") != "1", reason="set NABIA_NETWORK_TESTS=1 to download")
def test_crypto_download(store):  # pragma: no cover - network
    written = update_market_data(store, {"crypto": CryptoSource()}, find_assets(["BTC/USD", "ETH/USD"]),
                                 start=dt.date(2023, 1, 1), end=dt.date(2024, 6, 30))
    assert written == {"BTC/USD": 547, "ETH/USD": 547}


def test_crypto_source_steps_over_the_time_before_listing():
    exchange = FakeExchange(listed="2024-03-01")
    prices = CryptoSource(exchange, page_size=20).fetch(["ETH/USD"], dt.date(2023, 6, 1), dt.date(2024, 3, 31))
    assert prices["date"].iloc[0] == dt.date(2024, 3, 1)
    assert len(prices) == 31
