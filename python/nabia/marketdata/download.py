"""Downloading market data into the store, incrementally."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping, Sequence

from .sources import MarketDataSource
from .store import MarketDataStore
from .universe import Asset

log = logging.getLogger(__name__)

#: Days re-downloaded before the last stored date, to pick up late revisions.
DEFAULT_OVERLAP_DAYS = 5


def update_market_data(
    store: MarketDataStore,
    source: MarketDataSource | Mapping[str, MarketDataSource],
    assets: Sequence[Asset],
    start: dt.date,
    end: dt.date | None = None,
    full: bool = False,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> dict[str, int]:
    """Downloads daily prices for ``assets`` and upserts them into ``store``.

    ``source`` is either one source for every asset, or a mapping from source
    name to source, in which case each asset is fetched from ``asset.source``
    (e.g. ``{"yahoo": YahooFinanceSource(), "crypto": CryptoSource()}``).

    Incremental by default: a symbol already in the store is fetched from
    ``overlap_days`` before its last stored date (so recent revisions are
    picked up); a new symbol from ``start``. ``full=True`` re-downloads
    everything from ``start``. Returns the number of rows written per symbol
    (0 for symbols the source returned no data for).
    """
    sources: dict[str, MarketDataSource]
    if isinstance(source, Mapping):
        sources = dict(source)
        unknown = {a.source for a in assets} - sources.keys()
        if unknown:
            raise ValueError(f"no source given for: {sorted(unknown)}")
        source_of = {a.symbol: a.source for a in assets}
    else:
        sources = {source.name: source}
        source_of = {a.symbol: source.name for a in assets}
    for name in sorted({source_of[a.symbol] for a in assets}):
        store.upsert_assets([a for a in assets if source_of[a.symbol] == name], source=name)
    last = {} if full else store.last_dates()

    # Group symbols by source and fetch start, so each group is one batched request.
    groups: dict[tuple[str, dt.date], list[str]] = {}
    for asset in assets:
        from_date = start
        if asset.symbol in last:
            from_date = max(start, last[asset.symbol] - dt.timedelta(days=overlap_days))
        groups.setdefault((source_of[asset.symbol], from_date), []).append(asset.symbol)

    written = {asset.symbol: 0 for asset in assets}
    for (name, from_date), symbols in sorted(groups.items()):
        log.info("fetching %d symbols from %s since %s", len(symbols), name, from_date)
        prices = sources[name].fetch(symbols, from_date, end)
        for symbol, rows in prices.groupby("symbol"):
            written[str(symbol)] = store.upsert_prices(rows)
            log.info("%s: %d prices stored (%s to %s)", symbol, written[str(symbol)], rows["date"].min(), rows["date"].max())
    for symbol, count in written.items():
        if count == 0:
            log.warning("no data for %s", symbol)
    return written
