"""Tests of the AI agents' tools, runtime and pipeline. No Claude API calls:
the model is replaced by a scripted client, the MCP servers are real."""

import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from nabia.ai import analytics
from nabia.ai.agents.runtime import AgentRefusedError, AgentRuntime, MCPServerSpec, PORTFOLIO_SERVER
from nabia.ai.agents.team import AgentTeam
from nabia.ai.config import AIConfig
from nabia.ai.db import AgentStore
from nabia.ai.pipeline import decision_dates, run_backtest, trend_rule_decider
from nabia.ai.servers import news, portfolio, stats
from nabia.ai.servers.common import ServerContext
from nabia.marketdata import Asset, MarketDataStore

SYMBOLS = ["UP", "DOWN", "FLAT", "BTC/USD"]


def synthetic_prices(rows: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=rows)
    drift = np.array([0.0012, -0.0008, 0.0, 0.0015])
    vol = np.array([0.010, 0.012, 0.006, 0.030])
    r = drift + vol * rng.normal(0, 1, (rows, 4))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=SYMBOLS)


def load_market(url: str, prices: pd.DataFrame) -> None:
    market = MarketDataStore(url)
    market.upsert_assets([Asset("UP", "Up index", "equity_index", "USD"), Asset("DOWN", "Down future", "bond_future", "USD"),
                          Asset("FLAT", "Flat commodity", "commodity", "USD"),
                          Asset("BTC/USD", "Bitcoin", "crypto", "USD", source="crypto")], source="test")
    long = prices.stack().rename("close").reset_index()
    long.columns = ["date", "symbol", "close"]
    for c in ("open", "high", "low", "adj_close"):
        long[c] = long["close"]
    long["volume"] = 1.0
    market.upsert_prices(long)


@pytest.fixture
def config(tmp_path):
    cfg = AIConfig(db_url=f"sqlite:///{tmp_path / 'ai.db'}", symbols=SYMBOLS, min_budget=0.05, max_budget=0.5,
                   portfolio={"covariance_window": 60, "rebalance_frequency": 5, "target_volatility": 0.10,
                              "max_leverage": 3.0, "transaction_cost": 0.0005})
    load_market(cfg.db_url, synthetic_prices())
    return cfg


def call(server, name, args=None):
    """Calls a tool in-process and returns its JSON output."""
    result = asyncio.run(server.call_tool(name, args or {}))
    return json.loads(result.content[0].text)


# --- analytics -------------------------------------------------------------------

def test_budget_projection_normalizes_and_bounds():
    b = analytics.project_budgets({"a": 10.0, "b": 1.0, "c": 1.0, "d": 0.01}, 0.05, 0.5)
    assert sum(b.values()) == pytest.approx(1.0)
    assert b["a"] == pytest.approx(0.5) and b["d"] == pytest.approx(0.05)
    assert b["b"] == pytest.approx(b["c"]) == pytest.approx(0.225)
    assert analytics.project_budgets({"a": 1.0, "b": 3.0}, 0.0001, 1.0) == pytest.approx({"a": 0.25, "b": 0.75})
    with pytest.raises(ValueError):
        analytics.project_budgets({"a": 1.0, "b": -1.0}, 0.05, 0.5)
    with pytest.raises(ValueError, match="infeasible"):
        analytics.project_budgets({"a": 1.0, "b": 1.0}, 0.6, 0.9)


def test_signals_see_trends_and_tilt_budgets():
    prices = synthetic_prices()
    signals = analytics.trend_signals(prices)
    assert signals.loc["UP", "momentum_12_1"] > 0 > signals.loc["DOWN", "momentum_12_1"]
    assert signals.loc["UP", "vs_ma200"] > 0
    vols = analytics.rolling_volatility(prices)
    assert vols.loc["BTC/USD", "vol_60d"] > vols.loc["FLAT", "vol_60d"] * 3
    corr = analytics.correlation_matrix(prices, 60)
    assert corr.shape == (4, 4) and np.allclose(np.diag(corr), 1.0)
    budgets = analytics.trend_tilted_budgets(prices, 0.5, 0.05, 0.5)
    assert budgets["UP"] > 0.25 > budgets["DOWN"]
    assert sum(budgets.values()) == pytest.approx(1.0)


# --- MCP servers (in-process) ------------------------------------------------------

def test_news_server_stores_and_queries_point_in_time(config):
    server = news.build_server(ServerContext.create(config))
    article = {"url": "https://example.com/a", "published_date": "2023-03-10", "title": "Bank stress",
               "source": "Example", "topic": "credit", "asset_classes": ["equity_index"], "symbols": ["UP"],
               "sentiment": -0.6, "relevance": 0.9, "summary": "Regional bank failure."}
    out = call(server, "store_news_articles", {"articles": [
        article,
        {**article, "url": "https://example.com/b", "published_date": "2023-03-14", "sentiment": 0.2, "relevance": 0.5},
        {**article, "url": "https://example.com/c", "symbols": ["NOT_IN_UNIVERSE"]},
    ]})
    assert out["stored"] == 2 and len(out["rejected"]) == 1
    assert call(server, "query_news", {"as_of": "2023-03-12", "days": 7})["count"] == 1  # the 14th is in the future
    summary = call(server, "news_summary", {"as_of": "2023-03-14", "days": 7})
    assert summary["articles"] == 2
    assert summary["by_symbol"]["UP"]["weighted_sentiment"] == pytest.approx((-0.6 * 0.9 + 0.2 * 0.5) / 1.4, abs=1e-3)
    with pytest.raises(ToolError, match="sentiment"):  # schema validation reaches the model as a tool error
        asyncio.run(server.call_tool("store_news_articles", {"articles": [{**article, "sentiment": 3}]}))
    read_only = news.build_server(ServerContext.create(config, read_only=True))
    assert "store_news_articles" not in {t.name for t in asyncio.run(read_only.list_tools())}


def test_stats_tools_never_look_past_as_of(config, tmp_path):
    server = stats.build_server(ServerContext.create(config))
    as_of = "2023-06-30"
    before = {name: call(server, name, {"as_of": as_of})
              for name in ("trend_signals", "rolling_volatility", "correlation_matrix", "market_regime")}
    # Rewrite every price after as_of: the outputs must not change.
    prices = synthetic_prices()
    prices.loc[prices.index > as_of] *= 3.0
    load_market(config.db_url, prices)
    after = {name: call(server, name, {"as_of": as_of}) for name in before}
    assert after == before
    assert call(server, "list_universe", {"as_of": as_of})["assets"][0]["last_date"] == as_of

    report = {"summary": "s", "regime": "trending", "key_risks": [],
              "assets": [{"symbol": s, "trend": "up", "volatility_regime": "normal", "diversification": "ok",
                          "budget_tilt": 0.1, "reason": "r"} for s in SYMBOLS]}
    assert call(server, "save_stats_report", {"as_of": as_of, "report": report})["saved"]
    assert call(server, "get_stats_report", {"as_of": "2023-07-15"})["report"]["regime"] == "trending"
    assert call(server, "get_stats_report", {"as_of": "2023-06-29"})["report"] is None
    with pytest.raises(ToolError, match="missing"):
        asyncio.run(server.call_tool("save_stats_report", {"as_of": as_of, "report": {**report, "assets": report["assets"][:2]}}))


def test_portfolio_server_sets_budgets_and_weights(config):
    server = portfolio.build_server(ServerContext.create(config))
    assert call(server, "get_previous_decision", {"as_of": "2023-06-30"})["decision"] is None
    out = call(server, "set_risk_budgets", {"as_of": "2023-06-30", "budgets": {"UP": 6, "DOWN": 1, "FLAT": 2, "BTC/USD": 1},
                                            "rationale": "Tilt to the uptrend, cut the downtrend."})
    assert out["budgets"]["UP"] == pytest.approx(0.5)  # 0.6 capped at max_budget
    assert sum(out["budgets"].values()) == pytest.approx(1.0)
    weights = call(server, "compute_target_weights", {"as_of": "2023-06-30"})
    assert weights["converged"]
    assert weights["risk_contributions"]["UP"] == pytest.approx(out["budgets"]["UP"], abs=1e-3)
    stored = AgentStore(config.db_url).decision("live", dt.date(2023, 6, 30))
    assert stored["weights"] == pytest.approx(weights["weights"], abs=1e-4)
    assert call(server, "get_previous_decision", {"as_of": "2023-07-31"})["decision"]["as_of"] == "2023-06-30"
    with pytest.raises(ToolError, match="missing"):
        asyncio.run(server.call_tool("set_risk_budgets", {"as_of": "2023-07-31", "budgets": {"UP": 1},
                                                          "rationale": "incomplete budgets on purpose"}))
    with pytest.raises(ToolError, match="set_risk_budgets first"):
        asyncio.run(server.call_tool("compute_target_weights", {"as_of": "2023-07-31"}))


# --- agent runtime over real stdio MCP servers ------------------------------------------

class ScriptedClient:
    """Stands in for anthropic.AsyncAnthropic: returns scripted responses, records requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot: the list keeps growing
        stop_reason, blocks = self.script.pop(0)
        usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=usage, stop_details=None)


def tool_use(id, name, **input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def text(t):
    return SimpleNamespace(type="text", text=t)


def test_runtime_drives_mcp_tools_over_stdio(config, tmp_path):
    path = config.save(tmp_path / "ai_config.json")
    budgets = {"UP": 4, "DOWN": 1, "FLAT": 2, "BTC/USD": 1}
    client = ScriptedClient([
        ("tool_use", [text("Checking."), tool_use("t1", "get_portfolio_config"),
                      tool_use("t2", "preview_budgets", budgets={"UP": 1})]),  # parallel calls, one fails
        ("pause_turn", [SimpleNamespace(type="server_tool_use", id="s1", name="web_search", input={})]),
        ("tool_use", [tool_use("t3", "set_risk_budgets", as_of="2023-06-30", budgets=budgets,
                               rationale="Uptrend favoured, downtrend cut.")]),
        ("tool_use", [tool_use("t4", "compute_target_weights", as_of="2023-06-30")]),
        ("end_turn", [text("Budgets set.")]),
    ])
    runtime = AgentRuntime(config, path, run="live", client=client)
    run = asyncio.run(runtime.run("execution", "2023-06-30", "system prompt", "decide",
                                  [MCPServerSpec(PORTFOLIO_SERVER)], effort="high"))
    assert run.final_text == "Budgets set."
    assert [c.name for c in run.tool_calls] == ["get_portfolio_config", "preview_budgets", "set_risk_budgets",
                                                "compute_target_weights"]
    assert [c.is_error for c in run.tool_calls] == [False, True, False, False]
    assert "missing" in run.tool_calls[1].output  # the model sees why the call failed
    assert run.turns == 5 and run.server_tool_calls == 1 and run.usage["input_tokens"] == 500

    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["output_config"] == {"effort": "high"} and first["thinking"] == {"type": "adaptive"}
    assert {t["name"] for t in first["tools"]} >= {"set_risk_budgets", "compute_target_weights"}
    # Both results of the parallel calls go back in one user message.
    results = client.requests[1]["messages"][-1]
    assert results["role"] == "user" and [r["tool_use_id"] for r in results["content"]] == ["t1", "t2"]
    assert client.requests[2]["messages"][-1]["role"] == "assistant"  # pause_turn resumed as is
    stored = AgentStore(config.db_url).decision("live", dt.date(2023, 6, 30))
    assert stored["budgets"]["UP"] == pytest.approx(0.5) and stored["weights"] is not None


def test_runtime_raises_on_refusal(config, tmp_path):
    client = ScriptedClient([("refusal", [])])
    runtime = AgentRuntime(config, config.save(tmp_path / "c.json"), client=client)
    with pytest.raises(AgentRefusedError):
        asyncio.run(runtime.run("stats", "2023-06-30", "s", "p", [MCPServerSpec("nabia.ai.servers.stats")]))


def test_execution_agent_gets_read_only_news_and_stats(config, tmp_path):
    client = ScriptedClient([("end_turn", [text("nothing to do")])])
    team = AgentTeam(config, tmp_path / "work", run="bt", client=client)
    asyncio.run(team.decide(dt.date(2023, 6, 30)))
    request = client.requests[0]
    names = {t["name"] for t in request["tools"]}
    assert {"news_summary", "query_news", "get_stats_report", "set_risk_budgets", "compute_target_weights"} <= names
    assert not {"store_news_articles", "save_stats_report"} & names
    assert "Today is 2023-06-30" in request["system"]
    assert "historical simulation" in request["system"]  # backtest run: no web, point in time
    assert "UP: Up index, equity_index" in request["system"]


# --- pipeline ----------------------------------------------------------------------

def test_decision_dates_are_period_ends():
    index = pd.bdate_range("2023-01-02", "2023-04-28")
    assert decision_dates(index, dt.date(2023, 1, 15), dt.date(2023, 4, 28)) == [
        dt.date(2023, 1, 31), dt.date(2023, 2, 28), dt.date(2023, 3, 31), dt.date(2023, 4, 28)]
    daily = decision_dates(index.delete(5), dt.date(2023, 1, 2), dt.date(2023, 1, 13), "D")
    assert daily == [d.date() for d in pd.bdate_range("2023-01-02", "2023-01-13") if d != index[5]]  # trading dates only


def test_backtest_with_decided_budgets(config):
    calls = []
    rule = trend_rule_decider(config, "bt-rule")

    async def counting(as_of):
        calls.append(as_of)
        await rule(as_of)

    bt = asyncio.run(run_backtest(config, dt.date(2023, 1, 1), dt.date(2024, 4, 30), "bt-rule", counting))
    assert len(calls) == 16 and calls[0] == dt.date(2023, 1, 31)
    assert len(bt.budget_schedule) == 16
    np.testing.assert_allclose(bt.budget_schedule.sum(axis=1), 1.0)
    # At each rebalance the risk contributions match the budgets in force (the last decision before it).
    schedule = bt.budget_schedule
    for date, contributions in bt.result.risk_contributions.iterrows():
        in_force = schedule[schedule.index <= date]
        expected = in_force.iloc[-1] if len(in_force) else pd.Series(0.25, index=SYMBOLS)
        np.testing.assert_allclose(contributions.to_numpy(), expected.to_numpy(), atol=1e-5)
    assert set(bt.comparison().columns) == {"decided budgets", "equal risk"}

    # Resuming does not decide again.
    asyncio.run(run_backtest(config, dt.date(2023, 1, 1), dt.date(2024, 4, 30), "bt-rule", counting))
    assert len(calls) == 16
