"""Web app to manage the agents: news, decisions, agent runs, backtests and jobs.

    python -m nabia_webui.web --config ai_config.json      # http://127.0.0.1:8000
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..ai.config import AIConfig
from ..ai.db import AgentStore
from ..ai.pipeline import evaluate_decisions
from nabia.marketdata.store import MarketDataStore
from .jobs import JOB_KINDS, JobRunner

STATIC = Path(__file__).parent / "static"


def _clean(value: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy -> builtins."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _series(index: pd.Index) -> list[str]:
    return [d.date().isoformat() for d in pd.DatetimeIndex(index)]


class JobRequest(BaseModel):
    kind: str
    params: dict[str, Any] = {}


def create_app(config: AIConfig, workdir: Path, client: Any = None) -> FastAPI:
    config.validate()
    store = AgentStore(config.db_url)
    market = MarketDataStore(config.db_url)
    runner: JobRunner | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal runner
        runner = JobRunner(config, workdir, client=client)
        app.state.runner = runner
        yield
        await runner.wait_idle()
        runner.close()

    app = FastAPI(title="nabia agents", lifespan=lifespan)

    # --- Overview -----------------------------------------------------------------
    @app.get("/api/overview")
    def overview() -> dict:
        coverage = market.coverage()
        coverage = coverage[coverage["symbol"].isin(config.symbols)]
        assets = market.assets().set_index("symbol") if len(market.assets()) else pd.DataFrame()
        rows = []
        for s in config.symbols:
            c = coverage[coverage["symbol"] == s]
            rows.append({
                "symbol": s,
                "name": assets.at[s, "name"] if s in assets.index else None,
                "asset_class": assets.at[s, "asset_class"] if s in assets.index else None,
                "first_date": c["first_date"].iloc[0] if len(c) else None,
                "last_date": c["last_date"].iloc[0] if len(c) else None,
                "prices": int(c["prices"].iloc[0]) if len(c) else 0,
            })
        today = dt.date.today()
        week_news = store.news(today - dt.timedelta(days=6), today, limit=10_000)
        return _clean({
            "config": {"symbols": config.symbols, "model": config.model_label, "db_url": config.db_url,
                       "ollama_model": config.ollama_model, "claude_model": config.model,
                       "min_budget": config.min_budget, "max_budget": config.max_budget, "portfolio": config.portfolio},
            "coverage": rows,
            "live_decision": store.decision("live", today),
            "news_last_7_days": len(week_news),
            "runs": store.runs(),
            "jobs": store.jobs(limit=8),
        })

    # --- News ---------------------------------------------------------------------
    @app.get("/api/news")
    def news(start: dt.date | None = None, end: dt.date | None = None, topic: str | None = None,
             asset_class: str | None = None, symbol: str | None = None, min_relevance: float = 0.0,
             q: str | None = None, limit: int = Query(200, le=2000)) -> dict:
        end = end or dt.date.today()
        start = start or end - dt.timedelta(days=30)
        articles = store.news(start, end, topic or None, symbol or None, asset_class or None, limit=10_000)
        articles = [a for a in articles if a["relevance"] >= min_relevance]
        if q:
            needle = q.lower()
            articles = [a for a in articles if needle in a["title"].lower() or needle in a["summary"].lower()]
        by_topic: dict[str, list[float]] = defaultdict(list)
        for a in articles:
            by_topic[a["topic"]].append(a["sentiment"])
        per_day = Counter(a["published_date"] for a in articles)
        return _clean({
            "start": start, "end": end, "total": len(articles), "articles": articles[:limit],
            "by_topic": {t: {"articles": len(v), "avg_sentiment": sum(v) / len(v)} for t, v in sorted(by_topic.items())},
            "per_day": dict(sorted(per_day.items())),
        })

    # --- Runs, decisions and agent runs --------------------------------------------
    @app.get("/api/runs")
    def runs() -> dict:
        meta = {r["run"]: r for r in store.backtest_runs()}
        out = []
        for r in store.runs():
            out.append({**r, "backtest": meta.pop(r["run"], None)})
        out += [{"run": name, "decisions": 0, "first": None, "last": None, "backtest": m} for name, m in meta.items()]
        return {"runs": sorted(out, key=lambda r: (r["run"] != "live", r["run"]))}

    @app.get("/api/runs/{run}/decisions")
    def decisions(run: str) -> dict:
        return _clean({"run": run, "symbols": config.symbols, "decisions": store.decisions(run)})

    @app.get("/api/runs/{run}/stats_report")
    def stats_report(run: str, as_of: dt.date) -> dict:
        return _clean({"report": store.stats_report(run, as_of)})

    @app.get("/api/agent_runs")
    def agent_runs(run: str | None = None, limit: int = Query(100, le=1000)) -> dict:
        return {"agent_runs": store.agent_runs(run, limit)}

    @app.get("/api/agent_runs/{run_id}")
    def agent_run(run_id: int) -> dict:
        record = store.agent_run(run_id)
        if record is None:
            raise HTTPException(404, "no such agent run")
        return record

    # --- Backtests ----------------------------------------------------------------
    @app.get("/api/backtests/{run}")
    def backtest(run: str) -> dict:
        meta = store.backtest_run(run)
        decided = store.decisions(run)
        if not decided:
            raise HTTPException(404, f"run {run!r} has no decisions")
        start = dt.date.fromisoformat(meta["start"] if meta else decided[0]["as_of"])
        end = dt.date.fromisoformat(meta["end"] if meta else decided[-1]["as_of"])
        if run == "live":
            end = dt.date.today()
        try:
            bt = evaluate_decisions(config, run, start, end)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        r, b = bt.result, bt.baseline
        active = r.weights.index >= pd.Timestamp(start)
        weekly = r.weights[active].resample("W-FRI").last().dropna(how="all")
        return _clean({
            "run": run, "meta": meta, "start": start, "end": end, "symbols": config.symbols,
            "dates": _series(r.equity.index[active]),
            "equity": r.equity[active].tolist(),
            "baseline_equity": b.equity[active].tolist(),
            "drawdown": r.performance["drawdown"][active].tolist(),
            "baseline_drawdown": b.performance["drawdown"][active].tolist(),
            "schedule": {"dates": _series(bt.budget_schedule.index),
                         "budgets": {s: bt.budget_schedule[s].tolist() for s in config.symbols}},
            "weights": {"dates": _series(weekly.index), "values": {s: weekly[s].tolist() for s in config.symbols}},
            "metrics": {"decided": r.metrics, "equal_risk": b.metrics},
        })

    # --- Jobs -----------------------------------------------------------------------
    @app.get("/api/jobs")
    def jobs(limit: int = Query(50, le=500)) -> dict:
        return {"jobs": store.jobs(limit), "kinds": JOB_KINDS}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int) -> dict:
        record = store.job(job_id)
        if record is None:
            raise HTTPException(404, "no such job")
        return record

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: int, after: int = 0, limit: int = Query(1000, le=5000)) -> dict:
        """Log lines newer than `after` (the last id the client has), for live tailing."""
        record = store.job(job_id)
        if record is None:
            raise HTTPException(404, "no such job")
        return {"job": record, "logs": store.job_logs(job_id, after, limit)}

    @app.post("/api/jobs", status_code=201)
    async def submit(request: JobRequest) -> dict:  # async: jobs are scheduled on this event loop
        try:
            return {"id": app.state.runner.submit(request.kind, request.params)}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    # --- Front end ----------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
