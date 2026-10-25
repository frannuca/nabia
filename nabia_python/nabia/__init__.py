"""nabia: sentiment-modulated risk budgeting and portfolio tools in Python, on top of the nabia C++ library.

- ``nabia.sentiment_rb``: the sentiment risk-budgeting pipeline (LLM news sentiment and indicators -> elastic net
  per asset -> sentiment score -> modulated risk budgets, long-only or long-short -> backtest against ERC);
  ``python -m nabia.sentiment_rb``.
- ``nabia.backtest`` / ``nabia.latest_weights``: risk budgeting portfolio
  backtests and current target weights, computed by the C++ core (the
  ``nabia_python_bridge._rb_backtest`` extension module of the nabia_python_bridge wheel).
- ``nabia.analytics``: returns, volatilities, correlations, trend signals and budget
  projections on pandas price frames.
- ``nabia.marketdata``: download market data (equity indices, bond futures,
  commodities, crypto) and store it in a SQL database (SQLite or PostgreSQL).

The whole C++ library (optimizers, risk measures, forecasting, regression...) is in
``nabia_python_bridge``, installed from the wheel built by the C++ project (../cpp).
The extension is imported lazily, so ``nabia.marketdata`` works without the wheel.
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
