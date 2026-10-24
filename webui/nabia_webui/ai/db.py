"""Storage of the agents' work: news articles, stats reports and risk budget decisions.

Lives in the same database as the market data (SQLite now, PostgreSQL later)
and uses the same portable SQLAlchemy Core types and upserts.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    select,
)

from nabia.marketdata.store import upsert_rows

metadata = MetaData()

NEWS_TOPICS = (
    "monetary_policy", "inflation", "growth", "labor", "earnings", "fiscal", "geopolitics",
    "energy", "metals", "agriculture", "crypto", "rates", "credit", "fx", "regulation", "other",
)
ASSET_CLASSES = ("equity_index", "bond_future", "commodity", "crypto")

news_table = Table(
    "news_articles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("url", String(1024), nullable=False, unique=True),
    Column("published_date", Date, nullable=False, index=True),
    Column("collected_at", DateTime, nullable=False),
    Column("title", String(512), nullable=False),
    Column("source", String(128), nullable=False),
    Column("topic", String(32), nullable=False, index=True),
    Column("asset_classes", String(128), nullable=False),  # comma-separated
    Column("symbols", String(256), nullable=False),        # comma-separated
    Column("sentiment", Float, nullable=False),            # -1 (bearish) .. +1 (bullish)
    Column("relevance", Float, nullable=False),            # 0 .. 1
    Column("summary", Text, nullable=False),
)

stats_reports_table = Table(
    "stats_reports",
    metadata,
    Column("run", String(64), primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("report", Text, nullable=False),  # JSON
)

decisions_table = Table(
    "risk_budget_decisions",
    metadata,
    Column("run", String(64), primary_key=True),   # "live", or a backtest name
    Column("as_of", Date, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("budgets", Text, nullable=False),        # JSON {symbol: budget}, normalized
    Column("rationale", Text, nullable=False),
    Column("weights", Text),                        # JSON {symbol: weight}, after compute_target_weights
    Column("leverage", Float),
    Column("decided_by", String(64), nullable=False),
)


agent_runs_table = Table(
    "agent_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run", String(64), nullable=False, index=True),
    Column("agent", String(32), nullable=False),
    Column("as_of", Date, nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime, nullable=False),
    Column("turns", Integer, nullable=False),
    Column("tool_calls", Text, nullable=False),   # JSON [{name, input, is_error, output}]
    Column("server_tool_calls", Integer, nullable=False),
    Column("usage", Text, nullable=False),        # JSON token counts
    Column("final_text", Text, nullable=False),
    Column("error", Text),                        # set when the run failed
)

jobs_table = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kind", String(32), nullable=False),
    Column("params", Text, nullable=False),       # JSON
    Column("status", String(16), nullable=False),  # queued, running, done, failed
    Column("created_at", DateTime, nullable=False),
    Column("finished_at", DateTime),
    Column("message", Text),                      # progress or result summary
    Column("error", Text),
)

job_logs_table = Table(
    "job_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", Integer, nullable=False, index=True),
    Column("ts", DateTime, nullable=False),
    Column("level", String(10), nullable=False),
    Column("logger", String(128), nullable=False),
    Column("message", Text, nullable=False),
)

backtest_runs_table = Table(
    "backtest_runs",
    metadata,
    Column("run", String(64), primary_key=True),
    Column("policy", String(32), nullable=False),
    Column("start", Date, nullable=False),
    Column("end", Date, nullable=False),
    Column("frequency", String(16), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("metrics", Text),                      # JSON {decided, equal_risk}
)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def _split(text: str) -> list[str]:
    return [t for t in text.split(",") if t]


class AgentStore:
    """Read/write access to the agents' tables (created if missing)."""

    def __init__(self, url: str) -> None:
        self.engine = create_engine(url)
        metadata.create_all(self.engine)

    # --- News -------------------------------------------------------------
    def store_news(self, articles: Sequence[dict[str, Any]]) -> int:
        """Upserts articles keyed by URL; returns the number written."""
        rows = []
        for a in articles:
            rows.append({
                "url": a["url"], "published_date": a["published_date"], "collected_at": _now(),
                "title": a["title"][:512], "source": a["source"][:128], "topic": a["topic"],
                "asset_classes": ",".join(a.get("asset_classes", [])),
                "symbols": ",".join(a.get("symbols", [])),
                "sentiment": float(a["sentiment"]), "relevance": float(a["relevance"]),
                "summary": a["summary"],
            })
        upsert_rows(self.engine, news_table, rows, keys=["url"])
        return len(rows)

    def known_urls(self, urls: Sequence[str]) -> set[str]:
        """The subset of `urls` already stored."""
        if not urls:
            return set()
        with self.engine.connect() as connection:
            rows = connection.execute(select(news_table.c.url).where(news_table.c.url.in_(list(urls))))
            return {r[0] for r in rows}

    def news(self, start: dt.date, end: dt.date, topic: str | None = None, symbol: str | None = None,
             asset_class: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Articles published in [start, end], most relevant and recent first."""
        conditions = [news_table.c.published_date >= start, news_table.c.published_date <= end]
        if topic:
            conditions.append(news_table.c.topic == topic)
        query = (select(news_table).where(and_(*conditions))
                 .order_by(news_table.c.published_date.desc(), news_table.c.relevance.desc()))
        with self.engine.connect() as connection:
            rows = [dict(r) for r in connection.execute(query).mappings()]
        out = []
        for r in rows:
            r["asset_classes"] = _split(r["asset_classes"])
            r["symbols"] = _split(r["symbols"])
            if symbol and symbol not in r["symbols"]:
                continue
            if asset_class and asset_class not in r["asset_classes"]:
                continue
            r.pop("id", None)
            r["published_date"] = r["published_date"].isoformat()
            r["collected_at"] = r["collected_at"].isoformat(timespec="seconds")
            out.append(r)
        return out[:limit]

    # --- Stats reports ----------------------------------------------------
    def save_stats_report(self, run: str, as_of: dt.date, report: dict[str, Any]) -> None:
        upsert_rows(self.engine, stats_reports_table,
                    [{"run": run, "as_of": as_of, "created_at": _now(), "report": json.dumps(report)}],
                    keys=["run", "as_of"])

    def stats_report(self, run: str, as_of: dt.date) -> dict[str, Any] | None:
        """The latest report of `run` dated on or before as_of."""
        t = stats_reports_table
        query = select(t).where(and_(t.c.run == run, t.c.as_of <= as_of)).order_by(t.c.as_of.desc()).limit(1)
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        if row is None:
            return None
        return {"as_of": row["as_of"].isoformat(), **json.loads(row["report"])}

    # --- Decisions --------------------------------------------------------
    def save_decision(self, run: str, as_of: dt.date, budgets: dict[str, float], rationale: str,
                      decided_by: str) -> None:
        upsert_rows(self.engine, decisions_table, [{
            "run": run, "as_of": as_of, "created_at": _now(), "budgets": json.dumps(budgets),
            "rationale": rationale, "weights": None, "leverage": None, "decided_by": decided_by,
        }], keys=["run", "as_of"])

    def save_weights(self, run: str, as_of: dt.date, weights: dict[str, float], leverage: float) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                decisions_table.update()
                .where(and_(decisions_table.c.run == run, decisions_table.c.as_of == as_of))
                .values(weights=json.dumps(weights), leverage=leverage))
        if result.rowcount == 0:
            raise KeyError(f"no decision for run {run!r} on {as_of}")

    def decision(self, run: str, as_of: dt.date, strictly_before: bool = False) -> dict[str, Any] | None:
        """The decision of `run` on as_of, or the latest one before it."""
        t = decisions_table
        date_condition = t.c.as_of < as_of if strictly_before else t.c.as_of <= as_of
        query = select(t).where(and_(t.c.run == run, date_condition)).order_by(t.c.as_of.desc()).limit(1)
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        return None if row is None else self._decision_dict(row)

    def decisions(self, run: str) -> list[dict[str, Any]]:
        t = decisions_table
        with self.engine.connect() as connection:
            rows = connection.execute(select(t).where(t.c.run == run).order_by(t.c.as_of)).mappings().all()
        return [self._decision_dict(r) for r in rows]

    def runs(self) -> list[dict[str, Any]]:
        """Every run with decisions: name, number of decisions, first/last date."""
        t = decisions_table
        from sqlalchemy import func

        query = select(t.c.run, func.count(), func.min(t.c.as_of), func.max(t.c.as_of)).group_by(t.c.run)
        with self.engine.connect() as connection:
            return [{"run": r, "decisions": n, "first": a.isoformat(), "last": b.isoformat()}
                    for r, n, a, b in connection.execute(query)]

    # --- Agent runs (audit trail) -------------------------------------------
    def save_agent_run(self, run: str, record: Any, started_at: dt.datetime, error: str | None = None) -> int:
        from dataclasses import asdict

        with self.engine.begin() as connection:
            result = connection.execute(agent_runs_table.insert().values(
                run=run, agent=record.agent, as_of=dt.date.fromisoformat(record.as_of), started_at=started_at,
                finished_at=_now(), turns=record.turns,
                tool_calls=json.dumps([asdict(c) for c in record.tool_calls], default=str),
                server_tool_calls=record.server_tool_calls, usage=json.dumps(record.usage),
                final_text=record.final_text, error=error))
            return int(result.inserted_primary_key[0])

    def agent_runs(self, run: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        t = agent_runs_table
        query = select(t).order_by(t.c.id.desc()).limit(limit)
        if run:
            query = query.where(t.c.run == run)
        with self.engine.connect() as connection:
            return [self._agent_run_dict(r, full=False) for r in connection.execute(query).mappings()]

    def agent_run(self, run_id: int) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(agent_runs_table).where(agent_runs_table.c.id == run_id)).mappings().first()
        return None if row is None else self._agent_run_dict(row, full=True)

    @staticmethod
    def _agent_run_dict(row: Any, full: bool) -> dict[str, Any]:
        calls = json.loads(row["tool_calls"])
        out = {
            "id": row["id"], "run": row["run"], "agent": row["agent"], "as_of": row["as_of"].isoformat(),
            "started_at": row["started_at"].isoformat(timespec="seconds"),
            "finished_at": row["finished_at"].isoformat(timespec="seconds"), "turns": row["turns"],
            "tool_call_count": len(calls), "tool_errors": sum(1 for c in calls if c["is_error"]),
            "server_tool_calls": row["server_tool_calls"], "usage": json.loads(row["usage"]), "error": row["error"],
        }
        if full:
            out["tool_calls"] = calls
            out["final_text"] = row["final_text"]
        return out

    # --- Jobs ---------------------------------------------------------------
    def create_job(self, kind: str, params: dict[str, Any]) -> int:
        with self.engine.begin() as connection:
            result = connection.execute(jobs_table.insert().values(
                kind=kind, params=json.dumps(params), status="queued", created_at=_now()))
            return int(result.inserted_primary_key[0])

    def update_job(self, job_id: int, **values: Any) -> None:
        if values.get("status") in ("done", "failed"):
            values["finished_at"] = _now()
        with self.engine.begin() as connection:
            connection.execute(jobs_table.update().where(jobs_table.c.id == job_id).values(**values))

    def jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(jobs_table).order_by(jobs_table.c.id.desc()).limit(limit)).mappings()
            return [self._job_dict(r) for r in rows]

    def job(self, job_id: int) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(jobs_table).where(jobs_table.c.id == job_id)).mappings().first()
        return None if row is None else self._job_dict(row)

    @staticmethod
    def _job_dict(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "kind": row["kind"], "params": json.loads(row["params"]), "status": row["status"],
                "created_at": row["created_at"].isoformat(timespec="seconds"),
                "finished_at": row["finished_at"].isoformat(timespec="seconds") if row["finished_at"] else None,
                "message": row["message"], "error": row["error"]}

    def add_job_log(self, job_id: int, level: str, logger: str, message: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(job_logs_table.insert().values(
                job_id=job_id, ts=_now(), level=level, logger=logger[:128], message=message[:8000]))

    def job_logs(self, job_id: int, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        """Log lines of a job with id > after, oldest first."""
        t = job_logs_table
        query = select(t).where(and_(t.c.job_id == job_id, t.c.id > after)).order_by(t.c.id).limit(limit)
        with self.engine.connect() as connection:
            return [{"id": r["id"], "ts": r["ts"].isoformat(timespec="seconds"), "level": r["level"],
                     "logger": r["logger"], "message": r["message"]} for r in connection.execute(query).mappings()]

    # --- Backtest runs ------------------------------------------------------
    def save_backtest_run(self, run: str, policy: str, start: dt.date, end: dt.date, frequency: str,
                          metrics: dict[str, Any] | None) -> None:
        upsert_rows(self.engine, backtest_runs_table, [{
            "run": run, "policy": policy, "start": start, "end": end, "frequency": frequency, "created_at": _now(),
            "metrics": json.dumps(metrics) if metrics is not None else None}], keys=["run"])

    def backtest_runs(self) -> list[dict[str, Any]]:
        t = backtest_runs_table
        with self.engine.connect() as connection:
            rows = connection.execute(select(t).order_by(t.c.created_at.desc())).mappings().all()
        return [{"run": r["run"], "policy": r["policy"], "start": r["start"].isoformat(), "end": r["end"].isoformat(),
                 "frequency": r["frequency"], "created_at": r["created_at"].isoformat(timespec="seconds"),
                 "metrics": json.loads(r["metrics"]) if r["metrics"] else None} for r in rows]

    def backtest_run(self, run: str) -> dict[str, Any] | None:
        return next((r for r in self.backtest_runs() if r["run"] == run), None)

    @staticmethod
    def _decision_dict(row: Any) -> dict[str, Any]:
        return {
            "run": row["run"], "as_of": row["as_of"].isoformat(), "budgets": json.loads(row["budgets"]),
            "rationale": row["rationale"], "weights": json.loads(row["weights"]) if row["weights"] else None,
            "leverage": row["leverage"], "decided_by": row["decided_by"],
        }
