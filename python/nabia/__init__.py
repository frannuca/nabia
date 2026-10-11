"""nabia: systematic trend strategy tools.

- ``nabia.backtest`` / ``nabia.latest_weights``: risk budgeting portfolio
  backtests and current target weights, computed by the C++ core (the
  ``nabia._nabia`` extension module).
- ``nabia.marketdata``: download market data (equity indices, bond futures,
  commodities) and store it in a SQL database (SQLite now, PostgreSQL later).

The C++ extension is imported lazily, so ``nabia.marketdata`` works even where
the extension has not been built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["backtest", "latest_weights", "BacktestResult", "TargetWeights"]

if TYPE_CHECKING:
    from .portfolio import BacktestResult, TargetWeights, backtest, latest_weights


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import portfolio

        return getattr(portfolio, name)
    raise AttributeError(f"module 'nabia' has no attribute {name!r}")
