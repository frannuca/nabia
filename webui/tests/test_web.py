"""Tests of the web app's API (no Claude API calls)."""

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from nabia_webui.ai.agents.runtime import AgentRun, ToolCall
from nabia_webui.ai.config import AIConfig
from nabia_webui.ai.db import AgentStore
from nabia_webui.web.app import create_app

from test_ai import SYMBOLS, load_market, synthetic_prices


@pytest.fixture
def config(tmp_path):
    cfg = AIConfig(db_url=f"sqlite:///{tmp_path / 'web.db'}", symbols=SYMBOLS, min_budget=0.05, max_budget=0.5,
                   portfolio={"covariance_window": 60, "rebalance_frequency": 5, "target_volatility": 0.10,
                              "max_leverage": 3.0, "transaction_cost": 0.0005})
    load_market(cfg.db_url, synthetic_prices())
    return cfg


@pytest.fixture
def client(config, tmp_path):
    with TestClient(create_app(config, tmp_path / "work")) as c:
        yield c


def wait_for_job(client, job_id, timeout=60):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_front_end_and_overview(client):
    page = client.get("/")
    assert page.status_code == 200 and "nabia" in page.text
    assert client.get("/static/app.js").status_code == 200
    o = client.get("/api/overview").json()
    assert [c["symbol"] for c in o["coverage"]] == SYMBOLS
    assert o["coverage"][0]["prices"] == 600 and o["coverage"][0]["name"] == "Up index"
    assert o["live_decision"] is None and o["news_last_7_days"] == 0


def test_news_filters_and_aggregates(client, config):
    store = AgentStore(config.db_url)
    base = {"source": "Example", "asset_classes": ["equity_index"], "symbols": ["UP"], "relevance": 0.8,
            "summary": "Stocks rally on rate cut hopes."}
    store.store_news([
        {**base, "url": "https://e.com/1", "published_date": dt.date(2024, 3, 1), "title": "Rally", "topic": "monetary_policy", "sentiment": 0.6},
        {**base, "url": "https://e.com/2", "published_date": dt.date(2024, 3, 2), "title": "Oil drops", "topic": "energy",
         "sentiment": -0.4, "relevance": 0.3, "symbols": [], "asset_classes": ["commodity"]},
    ])
    data = client.get("/api/news", params={"start": "2024-03-01", "end": "2024-03-31"}).json()
    assert data["total"] == 2 and data["per_day"] == {"2024-03-01": 1, "2024-03-02": 1}
    assert data["by_topic"]["energy"]["avg_sentiment"] == pytest.approx(-0.4)
    assert client.get("/api/news", params={"start": "2024-03-01", "end": "2024-03-31", "min_relevance": 0.5}).json()["total"] == 1
    assert client.get("/api/news", params={"start": "2024-03-01", "end": "2024-03-31", "q": "oil"}).json()["articles"][0]["title"] == "Oil drops"
    assert client.get("/api/news", params={"start": "2024-03-01", "end": "2024-03-31", "symbol": "UP"}).json()["total"] == 1


def test_backtest_job_runs_and_results_are_served(client):
    bad = client.post("/api/jobs", json={"kind": "backtest", "params": {"run": "live", "start": "2023-01-01", "end": "2023-06-30"}})
    assert bad.status_code == 400
    response = client.post("/api/jobs", json={"kind": "backtest", "params": {
        "run": "rule-test", "policy": "trend_rule", "start": "2023-01-01", "end": "2023-12-31", "frequency": "QE"}})
    assert response.status_code == 201
    job = wait_for_job(client, response.json()["id"])
    assert job["status"] == "done", job["error"]
    assert "4 decisions" in job["message"]

    runs = client.get("/api/runs").json()["runs"]
    entry = next(r for r in runs if r["run"] == "rule-test")
    assert entry["decisions"] == 4 and entry["backtest"]["policy"] == "trend_rule"
    assert entry["backtest"]["metrics"]["decided"]["rebalances"] > 0

    bt = client.get("/api/backtests/rule-test").json()
    assert bt["dates"][0] >= "2023-01-01" and len(bt["equity"]) == len(bt["dates"]) == len(bt["baseline_equity"])
    assert len(bt["schedule"]["dates"]) == 4 and set(bt["weights"]["values"]) == set(SYMBOLS)
    assert client.get("/api/backtests/unknown").status_code == 404

    decisions = client.get("/api/runs/rule-test/decisions").json()["decisions"]
    assert decisions[0]["decided_by"] == "trend_rule"


def test_agent_runs_are_listed_with_their_tool_calls(client, config):
    store = AgentStore(config.db_url)
    record = AgentRun(agent="execution", as_of="2024-03-01", final_text="done", turns=3,
                      tool_calls=[ToolCall("set_risk_budgets", {"as_of": "2024-03-01"}, False, "ok"),
                                  ToolCall("preview_budgets", {}, True, "missing symbols")],
                      usage={"input_tokens": 100, "output_tokens": 50})
    run_id = store.save_agent_run("live", record, dt.datetime(2024, 3, 1, 18, 0))
    listed = client.get("/api/agent_runs", params={"run": "live"}).json()["agent_runs"]
    assert listed[0]["tool_call_count"] == 2 and listed[0]["tool_errors"] == 1
    detail = client.get(f"/api/agent_runs/{run_id}").json()
    assert detail["tool_calls"][1]["output"] == "missing symbols" and detail["final_text"] == "done"
    assert client.get("/api/agent_runs/999").status_code == 404


def test_failed_jobs_report_their_error(client):
    response = client.post("/api/jobs", json={"kind": "download", "params": {"symbols": ["NOT_A_SYMBOL"]}})
    job = wait_for_job(client, response.json()["id"])
    assert job["status"] == "failed" and "NOT_A_SYMBOL" in job["error"]
    assert client.post("/api/jobs", json={"kind": "nope"}).status_code == 400


def test_job_logs_are_captured_and_tailed(client):
    import logging

    logging.getLogger("nabia.test").info("outside any job: not captured")
    response = client.post("/api/jobs", json={"kind": "backtest", "params": {
        "run": "log-test", "policy": "trend_rule", "start": "2023-01-01", "end": "2023-12-31", "frequency": "QE"}})
    job_id = response.json()["id"]
    assert wait_for_job(client, job_id)["status"] == "done"

    data = client.get(f"/api/jobs/{job_id}/logs").json()
    messages = [line["message"] for line in data["logs"]]
    assert data["job"]["status"] == "done"
    assert messages[0].startswith(f"job #{job_id}: backtest")
    assert sum("deciding" in m for m in messages) == 4  # one line per decision date
    assert sum(m.startswith("trend rule 2023-") for m in messages) == 4
    assert any("Sharpe" in m for m in messages)
    assert messages[-1].startswith(f"job #{job_id} done")
    assert not any("outside any job" in m for m in messages)
    ids = [line["id"] for line in data["logs"]]
    assert ids == sorted(ids)

    # Tailing: only lines after the last id the client has.
    tail = client.get(f"/api/jobs/{job_id}/logs", params={"after": ids[-3]}).json()["logs"]
    assert [line["id"] for line in tail] == ids[-2:]
    assert client.get("/api/jobs/9999/logs").status_code == 404


def test_failed_job_log_has_the_traceback(client):
    job_id = client.post("/api/jobs", json={"kind": "download", "params": {"symbols": ["NOT_A_SYMBOL"]}}).json()["id"]
    wait_for_job(client, job_id)
    logs = client.get(f"/api/jobs/{job_id}/logs").json()["logs"]
    error = next(line for line in logs if line["level"] == "ERROR")
    assert f"job #{job_id} failed" in error["message"] and "Traceback" in error["message"]
    assert "NOT_A_SYMBOL" in error["message"]


def test_daily_decision_frequency(client):
    assert client.post("/api/jobs", json={"kind": "backtest", "params": {
        "run": "bad-freq", "start": "2023-01-01", "end": "2023-02-01", "frequency": "hourly"}}).status_code == 400
    job_id = client.post("/api/jobs", json={"kind": "backtest", "params": {
        "run": "daily-rule", "policy": "trend_rule", "start": "2023-03-01", "end": "2023-03-31", "frequency": "D"}}).json()["id"]
    assert wait_for_job(client, job_id)["status"] == "done"
    decisions = client.get("/api/runs/daily-rule/decisions").json()["decisions"]
    assert len(decisions) == 23  # every trading day of March 2023


class FakeLocalModel:
    """Stands in for the Ollama model: the execution agent sets budgets, other agents just finish."""

    def __init__(self):
        from types import SimpleNamespace

        self.ns = SimpleNamespace
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        import re

        self.calls += 1
        usage = self.ns(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        as_of = re.search(r"Today is (\d{4}-\d{2}-\d{2})", kwargs["system"]).group(1)
        last = kwargs["messages"][-1]
        if "portfolio manager" in kwargs["system"] and isinstance(last["content"], str):
            call = self.ns(type="tool_use", id=f"c{self.calls}", name="set_risk_budgets",
                           input={"as_of": as_of, "budgets": {s: 1.0 for s in SYMBOLS}, "rationale": "Equal risk, local model."})
            return self.ns(content=[call], stop_reason="tool_use", usage=usage, stop_details=None)
        return self.ns(content=[self.ns(type="text", text="done")], stop_reason="end_turn", usage=usage, stop_details=None)


def test_backtest_with_the_local_model_policy(config, tmp_path):
    from nabia_webui.ai.pipeline import policy_config

    assert policy_config(config, "ollama").provider == "ollama"
    assert policy_config(config, "claude").provider == "anthropic"
    assert policy_config(config, "agents") is config

    with TestClient(create_app(config, tmp_path / "work", client=FakeLocalModel())) as c:
        overview = c.get("/api/overview").json()["config"]
        assert overview["ollama_model"] == config.ollama_model
        job_id = c.post("/api/jobs", json={"kind": "backtest", "params": {
            "run": "local-qe", "policy": "ollama", "start": "2023-01-01", "end": "2023-12-31", "frequency": "QE"}}).json()["id"]
        job = wait_for_job(c, job_id)
        assert job["status"] == "done", job["error"]
        decisions = c.get("/api/runs/local-qe/decisions").json()["decisions"]
        assert len(decisions) == 4
        assert {d["decided_by"] for d in decisions} == {f"agent:ollama:{config.ollama_model}"}
        runs = {r["run"]: r for r in c.get("/api/runs").json()["runs"]}
        assert runs["local-qe"]["backtest"]["policy"] == "ollama"
        agents = {a["agent"] for a in c.get("/api/agent_runs", params={"run": "local-qe"}).json()["agent_runs"]}
        assert agents == {"stats_analyzer", "execution"}
        logs = [l["message"] for l in c.get(f"/api/jobs/{job_id}/logs").json()["logs"]]
        assert any("model ollama:" in m for m in logs) and any("set_risk_budgets" in m for m in logs)
        assert c.post("/api/jobs", json={"kind": "backtest", "params": {
            "run": "x", "policy": "gpt", "start": "2023-01-01", "end": "2023-02-01"}}).status_code == 400
