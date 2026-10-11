"""Market data: download daily prices and store them in a SQL database.

    from nabia.marketdata import MarketDataStore, YahooFinanceSource, DEFAULT_UNIVERSE, update_market_data

    store = MarketDataStore("sqlite:///market.db")          # later: "postgresql+psycopg://..."
    sources = {"yahoo": YahooFinanceSource(), "crypto": CryptoSource("coinbase")}
    update_market_data(store, sources, DEFAULT_UNIVERSE, start=datetime.date(2010, 1, 1))
    prices = store.load_prices(["^GSPC", "ZN=F", "GC=F"])   # wide DataFrame for nabia.backtest

From the command line:  python -m nabia.marketdata --db sqlite:///market.db --start 2010-01-01
"""

from .download import update_market_data
from .sources import PRICE_COLUMNS, CryptoSource, MarketDataSource, YahooFinanceSource
from .store import MarketDataStore
from .universe import BOND_FUTURES, COMMODITIES, CRYPTO, DEFAULT_UNIVERSE, EQUITY_INDICES, Asset, find_assets

__all__ = [
    "Asset",
    "BOND_FUTURES",
    "COMMODITIES",
    "CRYPTO",
    "CryptoSource",
    "DEFAULT_UNIVERSE",
    "EQUITY_INDICES",
    "MarketDataSource",
    "MarketDataStore",
    "PRICE_COLUMNS",
    "YahooFinanceSource",
    "find_assets",
    "update_market_data",
]
