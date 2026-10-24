"""MCP server: statistical analysis of the portfolio's underlying assets.

    python -m nabia_webui.ai.servers.stats --config ai_config.json --run live [--read-only]
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from nabia import analytics
from .common import ServerContext, clean, parse_as_of, server_arguments, tool


class AssetView(BaseModel):
    symbol: str
    trend: str = Field(description="e.g. strong up, up, flat, down, strong down")
    volatility_regime: str = Field(description="e.g. calm, normal, elevated, stressed")
    diversification: str = Field(description="How it correlates with the rest of the portfolio")
    budget_tilt: float = Field(ge=-1.0, le=1.0, description="Suggested tilt of its risk budget: -1 cut .. +1 raise")
    reason: str


class StatsReport(BaseModel):
    summary: str = Field(description="Cross-asset picture in a few sentences")
    regime: str = Field(description="e.g. risk-on trending, risk-off, high-correlation stress, choppy")
    assets: list[AssetView]
    key_risks: list[str]


def build_server(ctx: ServerContext) -> MCPServer:
    server = MCPServer(
        "nabia-stats",
        instructions="Statistics of the nabia portfolio assets. Every tool takes as_of (YYYY-MM-DD) and "
                     "uses only prices dated on or before it. Volatilities are annualized; returns are simple.",
    )

    @tool(server)
    def list_universe(as_of: str) -> dict:
        """Portfolio symbols with their asset class, name and price coverage up to as_of."""
        end = parse_as_of(as_of)
        assets = ctx.market.assets().set_index("symbol")
        prices = ctx.prices(end)
        out = []
        for s in ctx.config.symbols:
            p = prices[s].dropna()
            out.append({
                "symbol": s,
                "name": assets.at[s, "name"] if s in assets.index else None,
                "asset_class": assets.at[s, "asset_class"] if s in assets.index else None,
                "first_date": p.index[0].date().isoformat() if len(p) else None,
                "last_date": p.index[-1].date().isoformat() if len(p) else None,
                "last_price": float(p.iloc[-1]) if len(p) else None,
            })
        return clean({"as_of": end.isoformat(), "assets": out})

    @tool(server)
    def correlation_matrix(as_of: str, window: int = 60) -> dict:
        """Correlation of daily returns over the last `window` dates where all assets
        traded, plus the average pairwise correlation."""
        end = parse_as_of(as_of)
        corr = analytics.correlation_matrix(ctx.prices(end), window)
        return clean({"as_of": end.isoformat(), "window": window,
                      "average_correlation": analytics.average_correlation(corr), "matrix": corr}, 3)

    @tool(server)
    def rolling_volatility(as_of: str) -> dict:
        """Annualized volatility per asset over 20, 60 and 120 days, and the percentile
        of today's 20-day volatility within its last year (1.0 = highest of the year)."""
        end = parse_as_of(as_of)
        return clean({"as_of": end.isoformat(), "volatility": analytics.rolling_volatility(ctx.prices(end))}, 4)

    @tool(server)
    def trend_signals(as_of: str) -> dict:
        """Per asset: 1/3/6/12-month returns, distance from the 50/200-day moving
        averages, volatility-scaled 12-1 month momentum, current and 1-year max drawdown."""
        end = parse_as_of(as_of)
        return clean({"as_of": end.isoformat(), "signals": analytics.trend_signals(ctx.prices(end))}, 4)

    @tool(server)
    def market_regime(as_of: str) -> dict:
        """Cross-asset regime: average correlation now vs over a year, median volatility
        percentile, and the share of assets in an uptrend."""
        end = parse_as_of(as_of)
        return clean({"as_of": end.isoformat(), **analytics.market_regime(ctx.prices(end))}, 4)

    @tool(server)
    def get_stats_report(as_of: str) -> dict:
        """The latest stats report saved on or before as_of (null if none)."""
        end = parse_as_of(as_of)
        return {"report": ctx.store.stats_report(ctx.run, end)}

    if not ctx.read_only:
        @tool(server)
        def save_stats_report(as_of: str, report: StatsReport) -> dict:
            """Save the structured analysis for as_of (replaces an earlier one for the same date).
            Include every portfolio symbol in `assets`."""
            end = parse_as_of(as_of)
            missing = set(ctx.config.symbols) - {a.symbol for a in report.assets}
            if missing:
                raise ValueError(f"report.assets is missing: {sorted(missing)}")
            ctx.store.save_stats_report(ctx.run, end, report.model_dump())
            return {"saved": True, "as_of": end.isoformat()}

    return server


def main() -> None:
    build_server(server_arguments(__doc__)).run()


if __name__ == "__main__":
    main()
