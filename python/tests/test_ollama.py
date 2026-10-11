"""Local-model support: the Ollama adapter and the RSS news tool."""

import asyncio
import datetime as dt
import json
import os
from types import SimpleNamespace

import httpx
import pytest

from nabia.ai.agents.ollama import OllamaClient
from nabia.ai.config import AIConfig
from nabia.ai.servers import news
from nabia.ai.servers.common import ServerContext

from test_ai import SYMBOLS, call, load_market, synthetic_prices


def test_requests_and_replies_are_translated():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        if len(captured) == 1:
            return httpx.Response(200, json={"model": "m", "done_reason": "stop", "prompt_eval_count": 120, "eval_count": 9,
                                             "message": {"role": "assistant", "content": "Checking the config.", "tool_calls": [
                                                 {"id": "call_a", "function": {"name": "get_portfolio_config", "arguments": {}}},
                                                 {"function": {"name": "preview_budgets", "arguments": '{"budgets": {"A": 1}}'}}]}})
        return httpx.Response(200, json={"model": "m", "done_reason": "stop", "message": {"role": "assistant", "content": "Done."}})

    client = OllamaClient("qwen3.8:latest", options={"num_ctx": 8192}, transport=httpx.MockTransport(handler))
    tools = [{"name": "get_portfolio_config", "description": "config", "input_schema": {"type": "object", "properties": {}}},
             {"type": "web_search_20260209", "name": "web_search"}]  # hosted tool: dropped
    first = asyncio.run(client.beta.messages.create(
        model="ignored", max_tokens=1000, system="You are a PM.", tools=tools,
        messages=[{"role": "user", "content": "Decide."}], thinking={"type": "adaptive"}, betas=["x"], fallbacks="default"))
    body = captured[0]
    assert body["model"] == "qwen3.8:latest" and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "You are a PM."}, {"role": "user", "content": "Decide."}]
    assert [t["function"]["name"] for t in body["tools"]] == ["get_portfolio_config"]
    assert body["options"] == {"num_predict": 1000, "num_ctx": 8192}
    assert first.stop_reason == "tool_use"
    assert [b.type for b in first.content] == ["text", "tool_use", "tool_use"]
    assert first.content[2].input == {"budgets": {"A": 1}}  # JSON-string arguments are parsed
    assert first.usage.input_tokens == 120 and first.usage.output_tokens == 9

    # Second turn: the assistant's tool calls and the tool results are translated back.
    ids = [b.id for b in first.content if b.type == "tool_use"]
    second = asyncio.run(client.beta.messages.create(system="You are a PM.", tools=tools, messages=[
        {"role": "user", "content": "Decide."},
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ids[0], "content": '{"symbols": ["A"]}'},
                                     {"type": "tool_result", "tool_use_id": ids[1], "content": "missing B", "is_error": True}]},
    ]))
    sent = captured[1]["messages"]
    assert sent[2]["role"] == "assistant" and [c["function"]["name"] for c in sent[2]["tool_calls"]] == [
        "get_portfolio_config", "preview_budgets"]
    assert sent[3] == {"role": "tool", "content": '{"symbols": ["A"]}', "tool_name": "get_portfolio_config"}
    assert sent[4] == {"role": "tool", "content": "ERROR: missing B", "tool_name": "preview_budgets"}
    assert second.stop_reason == "end_turn" and second.content[0].text == "Done."


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Fed holds rates</title><link>https://ex.com/fed</link><pubDate>{today}</pubDate>
<description>&lt;p&gt;The Fed kept rates unchanged.&lt;/p&gt;</description></item>
<item><title>Old story</title><link>https://ex.com/old</link><pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
</channel></rss>"""
ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Bitcoin jumps</title><link href="https://ex.com/btc"/><updated>{iso}</updated><summary>BTC up 5%.</summary></entry>
</feed>"""


def test_news_feeds_tool(tmp_path, monkeypatch):
    now = dt.datetime.now(dt.UTC)
    pages = {
        "https://feeds/rss": RSS.format(today=now.strftime("%a, %d %b %Y %H:%M:%S GMT")),
        "https://feeds/atom": ATOM.format(iso=now.isoformat()),
    }

    def fake_get(url, **kwargs):
        if url not in pages:
            raise httpx.ConnectError("unreachable")
        return httpx.Response(200, content=pages[url].encode(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    cfg = AIConfig(db_url=f"sqlite:///{tmp_path / 'n.db'}", symbols=SYMBOLS, min_budget=0.05, max_budget=0.5,
                   news_feeds={"Wire": "https://feeds/rss", "Crypto": "https://feeds/atom", "Broken": "https://feeds/broken"})
    load_market(cfg.db_url, synthetic_prices(100))
    server = news.build_server(ServerContext.create(cfg))
    out = call(server, "fetch_news_feeds", {"max_age_days": 2})
    assert {i["title"] for i in out["items"]} == {"Fed holds rates", "Bitcoin jumps"}  # the 2024 story is too old
    fed = next(i for i in out["items"] if i["source"] == "Wire")
    assert fed["description"] == "The Fed kept rates unchanged." and fed["published"] == now.date().isoformat()
    assert out["feed_errors"] == ["Broken: ConnectError"]

    call(server, "store_news_articles", {"articles": [{
        "url": "https://ex.com/fed", "published_date": now.date().isoformat(), "title": "Fed holds rates", "source": "Wire",
        "topic": "monetary_policy", "asset_classes": ["bond_future"], "symbols": [], "sentiment": 0.1, "relevance": 0.8,
        "summary": "No change."}]})
    again = call(server, "fetch_news_feeds", {"max_age_days": 2})
    assert [i["title"] for i in again["items"]] == ["Bitcoin jumps"] and again["already_stored"] == 1


def test_runtime_uses_ollama_when_configured(tmp_path):
    from nabia.ai.agents.runtime import AgentRuntime

    cfg = AIConfig(provider="ollama", ollama_model="gemma4:26b", symbols=SYMBOLS, min_budget=0.05, max_budget=0.5)
    runtime = AgentRuntime(cfg, tmp_path / "c.json")
    assert isinstance(runtime.client, OllamaClient) and runtime.client.model == "gemma4:26b"
    assert cfg.model_label == "ollama:gemma4:26b"
    with pytest.raises(ValueError, match="provider"):
        AIConfig(provider="openai").validate()


@pytest.mark.skipif(os.environ.get("NABIA_OLLAMA_TESTS") != "1", reason="set NABIA_OLLAMA_TESTS=1 to run the local model")
def test_execution_agent_on_the_local_model(tmp_path):  # pragma: no cover - needs a running Ollama
    from nabia.ai.agents.team import AgentTeam
    from nabia.ai.db import AgentStore

    cfg = AIConfig(db_url=f"sqlite:///{tmp_path / 'local.db'}", symbols=SYMBOLS, min_budget=0.05, max_budget=0.5,
                   provider="ollama", ollama_model=os.environ.get("NABIA_OLLAMA_MODEL", "qwen3.8:latest"),
                   portfolio={"covariance_window": 60, "target_volatility": 0.10, "max_leverage": 3.0})
    load_market(cfg.db_url, synthetic_prices())
    team = AgentTeam(cfg, tmp_path / "work", run="local-test")
    run = asyncio.run(team.decide(dt.date(2023, 6, 30)))
    decision = AgentStore(cfg.db_url).decision("local-test", dt.date(2023, 6, 30))
    assert decision is not None, run.final_text
    assert decision["decided_by"] == f"agent:ollama:{cfg.ollama_model}"
    assert decision["weights"] is not None and sum(decision["budgets"].values()) == pytest.approx(1.0)
