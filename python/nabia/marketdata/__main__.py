"""Command line: python -m nabia.marketdata --db sqlite:///market.db --start 2010-01-01"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from .download import update_market_data
from .sources import CryptoSource, YahooFinanceSource
from .store import MarketDataStore
from .universe import DEFAULT_UNIVERSE, find_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nabia.marketdata", description=__doc__)
    parser.add_argument("--db", default="sqlite:///market.db", help="SQLAlchemy URL (default: sqlite:///market.db)")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2010, 1, 1), help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None, help="last date (default: today)")
    parser.add_argument("--symbols", nargs="+", default=None, help="subset of the default universe")
    parser.add_argument("--full", action="store_true", help="re-download everything from --start")
    parser.add_argument("--asset-class", nargs="+", default=None,
                        choices=["equity_index", "bond_future", "commodity", "crypto"],
                        help="only these asset classes of the default universe")
    parser.add_argument("--crypto-exchange", default="coinbase", help="ccxt exchange id for crypto (default: coinbase)")
    parser.add_argument("--list", action="store_true", help="list the default universe and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        for a in DEFAULT_UNIVERSE:
            print(f"{a.symbol:12} {a.asset_class:13} {a.currency:4} {a.source:7} {a.name}")
        return 0
    try:
        assets = find_assets(args.symbols)
    except KeyError as error:
        parser.error(str(error))
    if args.asset_class:
        assets = [a for a in assets if a.asset_class in args.asset_class]
    store = MarketDataStore(args.db)
    sources = {"yahoo": YahooFinanceSource(), "crypto": CryptoSource(args.crypto_exchange)}
    written = update_market_data(store, sources, assets, args.start, args.end, full=args.full)
    print(f"{sum(written.values())} rows written for {sum(1 for n in written.values() if n)} of {len(written)} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
