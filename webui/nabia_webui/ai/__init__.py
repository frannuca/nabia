"""AI agents for the trend portfolio.

- News analyzer: collects financial news from the web into the database.
- Stats analyzer: analyzes trends, volatilities and correlations of the assets.
- Execution agent: combines both into risk budgets and computes target weights.

The agents use tools served by local MCP servers (``nabia_webui.ai.servers``); see
``python -m nabia_webui.ai --help`` for the daily and backtest commands.
"""

from .config import AIConfig

__all__ = ["AIConfig"]
