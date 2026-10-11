"""Shared plumbing of the MCP servers: configuration, data access, JSON cleaning."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ...marketdata.store import MarketDataStore
from ..config import AIConfig
from ..db import AgentStore


@dataclass
class ServerContext:
    """What every tool needs: settings, the run name and the two stores."""

    config: AIConfig
    run: str
    market: MarketDataStore
    store: AgentStore
    read_only: bool = False

    @classmethod
    def create(cls, config: AIConfig, run: str = "live", read_only: bool = False) -> ServerContext:
        config.validate()
        return cls(config, run, MarketDataStore(config.db_url), AgentStore(config.db_url), read_only)

    def prices(self, as_of: dt.date, symbols: list[str] | None = None, days: int | None = None) -> pd.DataFrame:
        """Close prices of the portfolio symbols (or `symbols`) up to and including as_of."""
        symbols = symbols or self.config.symbols
        unknown = set(symbols) - set(self.config.symbols)
        if unknown:
            raise ValueError(f"not in the portfolio universe: {sorted(unknown)}")
        start = as_of - dt.timedelta(days=days or self.config.stats_history_days)
        prices = self.market.load_prices(symbols, start=start, end=as_of)
        empty = [s for s in symbols if prices[s].notna().sum() == 0] if len(prices) else symbols
        if empty:
            raise ValueError(f"no prices up to {as_of} for: {empty}. Download market data first.")
        return prices


def parse_as_of(text: str) -> dt.date:
    """'YYYY-MM-DD' (or 'today'). The as-of date bounds every query: tools never
    return data dated after it, which keeps backtests free of look-ahead."""
    if text == "today":
        return dt.date.today()
    try:
        return dt.date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"as_of must be YYYY-MM-DD, got {text!r}") from error


def clean(value: Any, digits: int = 6) -> Any:
    """JSON-safe copy: NaN/inf -> None, floats rounded, pandas/numpy -> builtins."""
    if isinstance(value, pd.DataFrame):
        return {str(i): clean(row.to_dict(), digits) for i, row in value.iterrows()}
    if isinstance(value, pd.Series):
        return {str(k): clean(v, digits) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, digits) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def server_arguments(description: str, argv: list[str] | None = None) -> ServerContext:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="AIConfig JSON file")
    parser.add_argument("--run", default="live", help="run name: 'live' or a backtest name")
    parser.add_argument("--read-only", action="store_true", help="do not expose tools that write")
    args = parser.parse_args(argv)
    return ServerContext.create(AIConfig.load(args.config), args.run, args.read_only)


F = TypeVar("F", bound=Callable[..., Any])


def tool(server: MCPServer) -> Callable[[F], F]:
    """Registers a function as an MCP tool, reporting ValueError / KeyError to the
    model as tool errors with their message (MCP otherwise treats them as crashes
    and hides the text), so the agent can correct its call."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except (ValueError, KeyError) as error:
                raise ToolError(str(error)) from error

        # Resolved annotations, so schema generation does not depend on this module's globals.
        wrapper.__signature__ = inspect.signature(fn, eval_str=True)  # type: ignore[attr-defined]
        wrapper.__annotations__ = {**{k: p.annotation for k, p in wrapper.__signature__.parameters.items()},
                                   "return": wrapper.__signature__.return_annotation}
        server.tool()(wrapper)
        return fn

    return decorator
