"""MCP server: risk budget decisions and target weights (C++ engine).

    python -m nabia.ai.servers.portfolio --config ai_config.json --run live
"""

from __future__ import annotations

import datetime as dt

from mcp.server.mcpserver import MCPServer

from .. import analytics
from .common import ServerContext, clean, parse_as_of, server_arguments, tool


def build_server(ctx: ServerContext) -> MCPServer:
    server = MCPServer(
        "nabia-portfolio",
        instructions="Risk budgets and target weights of the nabia risk budgeting portfolio. "
                     "Budgets are each asset's share of portfolio risk; they are normalized to sum to 1.",
    )

    @tool(server)
    def get_portfolio_config() -> dict:
        """Universe, budget bounds and the portfolio engine settings (covariance window,
        rebalancing, volatility target, costs)."""
        c = ctx.config
        return {"run": ctx.run, "symbols": c.symbols, "min_budget": c.min_budget, "max_budget": c.max_budget,
                "equal_budget": 1 / len(c.symbols), "portfolio": c.portfolio}

    @tool(server)
    def get_previous_decision(as_of: str) -> dict:
        """The most recent risk budget decision strictly before as_of (null if none),
        with its rationale and resulting weights."""
        return {"decision": ctx.store.decision(ctx.run, parse_as_of(as_of), strictly_before=True)}

    @tool(server)
    def preview_budgets(budgets: dict[str, float]) -> dict:
        """Show how budgets would be normalized and bounded, without saving anything."""
        _check_symbols(ctx, budgets)
        projected = analytics.project_budgets(budgets, ctx.config.min_budget, ctx.config.max_budget)
        return clean({"budgets": projected}, 4)

    if not ctx.read_only:
        @tool(server)
        def set_risk_budgets(as_of: str, budgets: dict[str, float], rationale: str) -> dict:
            """Decide the risk budgets for as_of: one positive number per portfolio symbol
            (relative sizes; normalized to sum to 1 and bounded to [min_budget, max_budget]).
            The rationale should cite the evidence (stats, news) behind each tilt."""
            end = parse_as_of(as_of)
            _check_symbols(ctx, budgets)
            if len(rationale.strip()) < 20:
                raise ValueError("give a rationale that explains the budgets")
            projected = analytics.project_budgets(budgets, ctx.config.min_budget, ctx.config.max_budget)
            ctx.store.save_decision(ctx.run, end, projected, rationale.strip(), decided_by=f"agent:{ctx.config.model_label}")
            changed = {s: round(projected[s] - budgets[s] / sum(budgets.values()), 4) for s in projected}
            return clean({"as_of": end.isoformat(), "budgets": projected,
                          "adjusted_by_bounds": {s: d for s, d in changed.items() if abs(d) > 1e-6}}, 4)

        @tool(server)
        def compute_target_weights(as_of: str) -> dict:
            """Target portfolio weights from the budgets decided for as_of: risk budgeting on
            the trailing covariance (C++ engine), scaled to the volatility target. Call it
            after set_risk_budgets."""
            end = parse_as_of(as_of)
            decision = ctx.store.decision(ctx.run, end)
            if decision is None or decision["as_of"] != end.isoformat():
                raise ValueError(f"no risk budgets set for {end}; call set_risk_budgets first")
            target = compute_weights(ctx, end, decision["budgets"])
            return target

    return server


def _check_symbols(ctx: ServerContext, budgets: dict[str, float]) -> None:
    universe = set(ctx.config.symbols)
    if set(budgets) != universe:
        raise ValueError(f"budgets must name exactly the portfolio symbols: missing {sorted(universe - set(budgets))}, "
                         f"unknown {sorted(set(budgets) - universe)}")


def compute_weights(ctx: ServerContext, as_of: dt.date, budgets: dict[str, float]) -> dict:
    """Runs nabia.latest_weights on prices up to as_of and stores the weights with the decision."""
    import nabia  # the C++ extension, imported on use

    days = int(ctx.config.portfolio.get("covariance_window", 60)) * 3 + 30
    prices = ctx.prices(as_of, days=max(days, 120))
    target = nabia.latest_weights(prices, {**ctx.config.portfolio, "budgets": budgets})
    weights = {s: float(w) for s, w in target.weights.items()}
    ctx.store.save_weights(ctx.run, as_of, weights, target.leverage)
    return clean({"as_of": as_of.isoformat(), "priced_on": target.date.date().isoformat(),
                  "weights": weights, "risk_contributions": target.risk_contributions.to_dict(),
                  "leverage": target.leverage, "converged": target.converged}, 4)


def main() -> None:
    build_server(server_arguments(__doc__)).run()


if __name__ == "__main__":
    main()
