"""The default asset universe: equity indices, bond futures, commodities and crypto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AssetClass = Literal["equity_index", "bond_future", "commodity", "crypto"]


@dataclass(frozen=True)
class Asset:
    """An instrument to download and store."""

    symbol: str
    """Identifier in the data source (Yahoo Finance ticker for the default universe)."""
    name: str
    asset_class: AssetClass
    currency: str
    source: str = "yahoo"
    """Name of the MarketDataSource that provides this asset ("yahoo", "crypto")."""


EQUITY_INDICES: tuple[Asset, ...] = (
    Asset("^GSPC", "S&P 500", "equity_index", "USD"),
    Asset("^NDX", "Nasdaq 100", "equity_index", "USD"),
    Asset("^RUT", "Russell 2000", "equity_index", "USD"),
    Asset("^STOXX50E", "Euro Stoxx 50", "equity_index", "EUR"),
    Asset("^GDAXI", "DAX", "equity_index", "EUR"),
    Asset("^FTSE", "FTSE 100", "equity_index", "GBP"),
    Asset("^N225", "Nikkei 225", "equity_index", "JPY"),
    Asset("^HSI", "Hang Seng", "equity_index", "HKD"),
)

# CBOT Treasury futures, continuous front month. Yahoo Finance has no Bund /
# Bobl / Schatz or JGB futures; add them with another data source.
BOND_FUTURES: tuple[Asset, ...] = (
    Asset("ZT=F", "US 2-Year T-Note future", "bond_future", "USD"),
    Asset("ZF=F", "US 5-Year T-Note future", "bond_future", "USD"),
    Asset("ZN=F", "US 10-Year T-Note future", "bond_future", "USD"),
    Asset("ZB=F", "US Treasury Bond future", "bond_future", "USD"),
    Asset("UB=F", "US Ultra Treasury Bond future", "bond_future", "USD"),
)

# Continuous front-month futures (energy, metals, grains).
COMMODITIES: tuple[Asset, ...] = (
    Asset("CL=F", "WTI crude oil future", "commodity", "USD"),
    Asset("BZ=F", "Brent crude oil future", "commodity", "USD"),
    Asset("NG=F", "Henry Hub natural gas future", "commodity", "USD"),
    Asset("GC=F", "Gold future", "commodity", "USD"),
    Asset("SI=F", "Silver future", "commodity", "USD"),
    Asset("HG=F", "Copper future", "commodity", "USD"),
    Asset("ZC=F", "Corn future", "commodity", "USD"),
    Asset("ZW=F", "Wheat future", "commodity", "USD"),
    Asset("ZS=F", "Soybean future", "commodity", "USD"),
)

# Spot crypto against USD, from an exchange through ccxt (symbols in ccxt's
# BASE/QUOTE form). Crypto trades every day, so its weekend prices are dropped
# when it is aligned with exchange-traded assets in a backtest.
CRYPTO: tuple[Asset, ...] = (
    Asset("BTC/USD", "Bitcoin", "crypto", "USD", source="crypto"),
    Asset("ETH/USD", "Ether", "crypto", "USD", source="crypto"),
    Asset("SOL/USD", "Solana", "crypto", "USD", source="crypto"),
    Asset("XRP/USD", "XRP", "crypto", "USD", source="crypto"),
    Asset("ADA/USD", "Cardano", "crypto", "USD", source="crypto"),
    Asset("LTC/USD", "Litecoin", "crypto", "USD", source="crypto"),
)

DEFAULT_UNIVERSE: tuple[Asset, ...] = EQUITY_INDICES + BOND_FUTURES + COMMODITIES + CRYPTO


def find_assets(symbols: list[str] | None = None, universe: tuple[Asset, ...] = DEFAULT_UNIVERSE) -> list[Asset]:
    """The assets of ``universe`` with the given symbols (all of them if None).

    Raises KeyError for symbols not in the universe.
    """
    if symbols is None:
        return list(universe)
    by_symbol = {a.symbol: a for a in universe}
    missing = [s for s in symbols if s not in by_symbol]
    if missing:
        raise KeyError(f"unknown symbols: {', '.join(missing)}")
    return [by_symbol[s] for s in symbols]
