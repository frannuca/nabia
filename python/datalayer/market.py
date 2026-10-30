"""Market side of the database: the existing prices table (read only), downloaded time series (FRED), the
estimated asset exposures to the factors, the risk budgets computed from them and the portfolio weights."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (          -- downloaded time series, e.g. key fred:DGS2
    key   TEXT NOT NULL,
    date  TEXT NOT NULL,                     -- YYYY-MM-DD
    value REAL NOT NULL,
    PRIMARY KEY (key, date)
);
CREATE TABLE IF NOT EXISTS asset_exposures ( -- one row per estimation date, asset and factor (point in time)
    as_of     TEXT NOT NULL,                 -- end of the estimation window
    asset     TEXT NOT NULL,                 -- symbol of the prices table
    factor    TEXT NOT NULL,
    exposure  REAL NOT NULL,                 -- the mapping used downstream: 0 when not significant or stable
    raw       REAL,                          -- standardized beta before the significance and stability tests
    t_stat    REAL,
    stability REAL,                          -- share of recent windows with the same sign
    method    TEXT NOT NULL,                 -- proxy, news, group (borrowed from the factor's group) or none
    PRIMARY KEY (as_of, asset, factor)
);
CREATE TABLE IF NOT EXISTS risk_budgets (    -- one row per rebalance date and asset
    date       TEXT NOT NULL,
    asset      TEXT NOT NULL,
    base       REAL NOT NULL,                -- base budget (normalized)
    score      REAL NOT NULL,                -- news score of the asset, -1..1 (> 0: news bullish for the asset)
    multiplier REAL NOT NULL,                -- 1 + strength * score, within the configured bounds
    budget     REAL NOT NULL,                -- base * multiplier, normalized to sum 1 over the assets
    exposures  TEXT NOT NULL,                -- as_of of the exposure map used
    PRIMARY KEY (date, asset)
);
CREATE TABLE IF NOT EXISTS portfolio_weights ( -- risk budgeting weights, one row per rebalance date and asset
    date              TEXT NOT NULL,
    asset             TEXT NOT NULL,
    budget            REAL NOT NULL,         -- the risk budget targeted (risk_budgets.budget)
    weight            REAL NOT NULL,         -- long-only, sum 1
    risk_contribution REAL NOT NULL,         -- relative risk contribution reached (= budget when converged)
    portfolio_risk    REAL NOT NULL,         -- annualized risk of the portfolio (the measure of the run)
    converged         INTEGER NOT NULL,
    PRIMARY KEY (date, asset)
);
"""


class MarketDB:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.con.close()

    def prices(self, symbol: str) -> list[tuple[str, float]]:
        """(date, adjusted close or close) of a symbol of the prices table, by date."""
        rows = self.con.execute("SELECT date, coalesce(adj_close, close) FROM prices WHERE symbol = ? AND "
                                "coalesce(adj_close, close) > 0 ORDER BY date", (symbol,)).fetchall()
        if not rows:
            raise KeyError(f"no prices for {symbol} in the prices table")
        return [tuple(r) for r in rows]

    def save_series(self, key: str, rows) -> int:
        """Stores (date, value) observations of a series, replacing revised values; returns how many were given."""
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO series VALUES (?, ?, ?)", [(key, d, v) for d, v in rows])
        return len(rows)

    def series(self, key: str) -> list[tuple[str, float]]:
        rows = self.con.execute("SELECT date, value FROM series WHERE key = ? ORDER BY date", (key,)).fetchall()
        if not rows:
            raise KeyError(f"series {key} not downloaded (python -m exposures downloads the FRED series)")
        return [tuple(r) for r in rows]

    def save_exposures(self, rows: list[dict]) -> None:
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO asset_exposures VALUES (:as_of, :asset, :factor, :exposure, "
                                 ":raw, :t_stat, :stability, :method)", rows)

    def exposures(self, as_of: str | None = None) -> list[dict]:
        """The exposures of the latest estimation at or before as_of (default: the latest)."""
        date = self.con.execute("SELECT max(as_of) FROM asset_exposures WHERE as_of <= ?", (as_of or "9999",)).fetchone()[0]
        return [dict(r) for r in self.con.execute("SELECT * FROM asset_exposures WHERE as_of = ? ORDER BY asset, factor", (date,))]

    def save_budgets(self, rows: list[dict]) -> None:
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO risk_budgets VALUES (:date, :asset, :base, :score, :multiplier, "
                                 ":budget, :exposures)", rows)

    def budgets(self, date: str | None = None) -> list[dict]:
        """The risk budgets of the latest rebalance at or before date (default: the latest)."""
        day = self.con.execute("SELECT max(date) FROM risk_budgets WHERE date <= ?", (date or "9999",)).fetchone()[0]
        return [dict(r) for r in self.con.execute("SELECT * FROM risk_budgets WHERE date = ? ORDER BY asset", (day,))]

    def budget_history(self, start: str = "", end: str = "9999") -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT date, asset, budget FROM risk_budgets WHERE date >= ? AND date <= ? ORDER BY date, asset", (start, end))]

    def save_weights(self, rows: list[dict]) -> None:
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO portfolio_weights VALUES (:date, :asset, :budget, :weight, "
                                 ":risk_contribution, :portfolio_risk, :converged)", rows)

    def weights(self, date: str | None = None) -> list[dict]:
        """The portfolio weights of the latest rebalance at or before date (default: the latest)."""
        day = self.con.execute("SELECT max(date) FROM portfolio_weights WHERE date <= ?", (date or "9999",)).fetchone()[0]
        return [dict(r) for r in self.con.execute("SELECT * FROM portfolio_weights WHERE date = ? ORDER BY asset", (day,))]
