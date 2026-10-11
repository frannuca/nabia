"""SQL storage of assets and daily prices (SQLite now, PostgreSQL later).

The store uses SQLAlchemy Core with portable types, so the same code runs on
both databases; only the URL changes:

    MarketDataStore("sqlite:///market.db")
    MarketDataStore("postgresql+psycopg://user:password@host/marketdata")   # pip install psycopg

Writes are upserts (insert, or update on a (symbol, date) conflict), so
re-downloading overlapping periods is safe and picks up data revisions.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

from .sources import PRICE_COLUMNS
from .universe import Asset

metadata = MetaData()

assets_table = Table(
    "assets",
    metadata,
    Column("symbol", String(32), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("asset_class", String(32), nullable=False),
    Column("currency", String(8), nullable=False),
    Column("source", String(32), nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

prices_table = Table(
    "prices",
    metadata,
    Column("symbol", String(32), ForeignKey("assets.symbol"), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", Float),
)

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


def upsert_rows(engine: Engine, table: Table, rows: list[dict], keys: Sequence[str], chunk: int = 500) -> None:
    """Insert rows, updating the non-key columns of rows whose keys already exist."""
    if not rows:
        return
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:  # pragma: no cover - other databases are not supported yet
        raise NotImplementedError(f"upsert is not implemented for {dialect}")
    with engine.begin() as connection:
        for i in range(0, len(rows), chunk):
            statement = insert(table).values(rows[i : i + chunk])
            updates = {c.name: statement.excluded[c.name] for c in table.columns if c.name not in keys}
            connection.execute(statement.on_conflict_do_update(index_elements=list(keys), set_=updates))


class MarketDataStore:
    """Assets and daily prices in a SQL database (tables are created if missing)."""

    def __init__(self, url: str, echo: bool = False) -> None:
        self.engine = create_engine(url, echo=echo)
        metadata.create_all(self.engine)

    def upsert_assets(self, assets: Iterable[Asset], source: str) -> None:
        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        rows = [
            {"symbol": a.symbol, "name": a.name, "asset_class": a.asset_class, "currency": a.currency,
             "source": source, "updated_at": now}
            for a in assets
        ]
        upsert_rows(self.engine, assets_table, rows, keys=["symbol"])

    def upsert_prices(self, prices: pd.DataFrame) -> int:
        """Stores long-format prices (columns of ``PRICE_COLUMNS``); returns the number of rows written.
        The symbols must already be registered with :meth:`upsert_assets`."""
        if prices.empty:
            return 0
        missing = set(PRICE_COLUMNS) - set(prices.columns)
        if missing:
            raise ValueError(f"prices is missing columns: {sorted(missing)}")
        frame = prices[PRICE_COLUMNS].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame.drop_duplicates(subset=["symbol", "date"], keep="last")
        frame = frame.astype({f: "float64" for f in PRICE_FIELDS}).astype(object)
        frame = frame.where(pd.notna(frame), None)  # NaN -> SQL NULL
        rows = frame.to_dict("records")
        upsert_rows(self.engine, prices_table, rows, keys=["symbol", "date"])
        return len(rows)

    def assets(self) -> pd.DataFrame:
        with self.engine.connect() as connection:
            result = connection.execute(select(assets_table).order_by(assets_table.c.asset_class, assets_table.c.symbol))
            return pd.DataFrame(result.mappings().all(), columns=[c.name for c in assets_table.columns])

    def coverage(self) -> pd.DataFrame:
        """Per stored symbol: first and last date and number of prices."""
        p = prices_table
        query = (select(p.c.symbol, func.min(p.c.date), func.max(p.c.date), func.count())
                 .group_by(p.c.symbol).order_by(p.c.symbol))
        with self.engine.connect() as connection:
            return pd.DataFrame(connection.execute(query).all(), columns=["symbol", "first_date", "last_date", "prices"])

    def last_dates(self) -> dict[str, dt.date]:
        """Most recent stored date per symbol."""
        query = select(prices_table.c.symbol, func.max(prices_table.c.date)).group_by(prices_table.c.symbol)
        with self.engine.connect() as connection:
            return {symbol: last for symbol, last in connection.execute(query)}

    def load_prices(
        self,
        symbols: Sequence[str] | None = None,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
        field: str = "close",
    ) -> pd.DataFrame:
        """Prices as a wide frame: DatetimeIndex ``date`` x one column per symbol.

        Dates where a symbol has no price are NaN (``nabia.backtest`` drops them).
        Columns follow the order of ``symbols`` when given.
        """
        if field not in PRICE_FIELDS:
            raise ValueError(f"field must be one of {PRICE_FIELDS}")
        column = prices_table.c[field]
        query = select(prices_table.c.symbol, prices_table.c.date, column)
        if symbols is not None:
            query = query.where(prices_table.c.symbol.in_(list(symbols)))
        if start is not None:
            query = query.where(prices_table.c.date >= pd.Timestamp(start).date())
        if end is not None:
            query = query.where(prices_table.c.date <= pd.Timestamp(end).date())
        with self.engine.connect() as connection:
            long = pd.DataFrame(connection.execute(query).all(), columns=["symbol", "date", field])
        if long.empty:
            wide = pd.DataFrame(columns=list(symbols or []), dtype="float64")
            wide.index = pd.DatetimeIndex([], name="date")
            return wide
        wide = long.pivot(index="date", columns="symbol", values=field)
        wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index), name="date")
        wide.columns.name = None
        wide = wide.sort_index().astype("float64")
        return wide.reindex(columns=list(symbols)) if symbols is not None else wide
