"""Background jobs started from the web app: downloads, agent runs, backtests.

Jobs run one at a time (SQLite allows one writer, and agent runs are costly),
in the web server's event loop; their status lives in the ``jobs`` table so it
survives page reloads.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import logging
import traceback
from pathlib import Path
from typing import Any

from ..ai.agents.team import AgentTeam
from ..ai.config import AIConfig
from ..ai.db import AgentStore
from ..ai.pipeline import POLICIES, make_decider, run_backtest, run_daily
from ..marketdata import CryptoSource, MarketDataStore, YahooFinanceSource, find_assets, update_market_data

log = logging.getLogger(__name__)

JOB_KINDS = ("download", "news", "daily", "backtest")

#: The job the current code runs for. Context variables follow the job into
#: asyncio tasks and asyncio.to_thread() workers, so log records can be tagged.
current_job: contextvars.ContextVar[int | None] = contextvars.ContextVar("nabia_job", default=None)

# Chatty libraries: only their warnings and errors go to the job log.
_QUIET = ("httpx", "httpcore", "urllib3", "asyncio", "mcp", "uvicorn", "yfinance", "peewee", "multipart")


class JobLogHandler(logging.Handler):
    """Stores the log records emitted while a job runs, under that job's id."""

    def __init__(self, store: AgentStore) -> None:
        super().__init__(logging.INFO)
        self.store = store
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        job_id = current_job.get()
        if job_id is None:
            return
        if record.name.startswith(_QUIET) and record.levelno < logging.WARNING:
            return
        try:
            self.store.add_job_log(job_id, record.levelname, record.name, self.format(record))
        except Exception:  # logging must never break the job
            self.handleError(record)


def _date(params: dict[str, Any], key: str, default: dt.date | None = None) -> dt.date | None:
    value = params.get(key)
    return dt.date.fromisoformat(value) if value else default


class JobRunner:
    def __init__(self, config: AIConfig, workdir: Path, client: Any = None) -> None:
        self.config = config
        self.workdir = workdir
        self.client = client  # Claude client override (tests)
        self.store = AgentStore(config.db_url)
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._handler = JobLogHandler(self.store)
        logging.getLogger().addHandler(self._handler)
        logging.getLogger("nabia").setLevel(logging.INFO)  # INFO from our code even if logging is unconfigured

    def close(self) -> None:
        logging.getLogger().removeHandler(self._handler)

    def submit(self, kind: str, params: dict[str, Any]) -> int:
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        self._validate(kind, params)
        job_id = self.store.create_job(kind, params)
        task = asyncio.get_running_loop().create_task(self._run(job_id, kind, params))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def wait_idle(self) -> None:
        """Waits for every submitted job (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def _validate(self, kind: str, params: dict[str, Any]) -> None:
        if kind == "backtest":
            if not params.get("run") or not str(params["run"]).replace("-", "").replace("_", "").isalnum():
                raise ValueError("backtest needs a run name (letters, digits, - and _)")
            if params["run"] == "live":
                raise ValueError("'live' is reserved for daily decisions")
            if params.get("policy", "trend_rule") not in POLICIES:
                raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
            if params.get("frequency", "ME") not in ("D", "W-FRI", "ME", "QE"):
                raise ValueError("frequency must be D, W-FRI, ME or QE")
            start, end = _date(params, "start"), _date(params, "end")
            if start is None or end is None or end <= start:
                raise ValueError("backtest needs start < end")

    async def _run(self, job_id: int, kind: str, params: dict[str, Any]) -> None:
        async with self._lock:
            token = current_job.set(job_id)
            try:
                self.store.update_job(job_id, status="running", message="started")
                log.info("job #%s: %s %s", job_id, kind, params or "")
                message = await getattr(self, f"_{kind}")(job_id, params)
                log.info("job #%s done: %s", job_id, message)
                self.store.update_job(job_id, status="done", message=message)
            except Exception as error:
                log.exception("job #%s failed", job_id)  # with the traceback, in the job log
                self.store.update_job(job_id, status="failed", error="".join(
                    traceback.format_exception_only(error)).strip())
            finally:
                current_job.reset(token)

    async def _download(self, job_id: int, params: dict[str, Any]) -> str:
        symbols = params.get("symbols") or None
        assets = find_assets(symbols)
        if params.get("asset_classes"):
            assets = [a for a in assets if a.asset_class in params["asset_classes"]]
        sources = {"yahoo": YahooFinanceSource(), "crypto": CryptoSource(params.get("crypto_exchange", "coinbase"))}
        start = _date(params, "start", dt.date(2010, 1, 1))
        written = await asyncio.to_thread(update_market_data, MarketDataStore(self.config.db_url), sources, assets,
                                          start, None, bool(params.get("full")))
        return f"{sum(written.values())} rows for {sum(1 for n in written.values() if n)} of {len(written)} symbols"

    async def _news(self, job_id: int, params: dict[str, Any]) -> str:
        team = AgentTeam(self.config, self.workdir, client=self.client)
        run = await team.collect_news()
        return f"{len(run.tool_calls)} tool calls, {run.server_tool_calls} web calls"

    async def _daily(self, job_id: int, params: dict[str, Any]) -> str:
        result = await run_daily(self.config, self.workdir, _date(params, "as_of"),
                                 collect_news=params.get("collect_news", True), client=self.client)
        return f"budgets decided for {result.as_of} (leverage {result.decision['leverage']:.2f})"

    async def _backtest(self, job_id: int, params: dict[str, Any]) -> str:
        run, policy = params["run"], params.get("policy", "trend_rule")
        start, end = _date(params, "start"), _date(params, "end")
        frequency = params.get("frequency", "ME")
        decider = make_decider(self.config, policy, run, self.workdir, client=self.client)
        def progress(as_of: dt.date, k: int, n: int) -> None:
            self.store.update_job(job_id, message=f"deciding {as_of} ({k + 1}/{n})")

        bt = await run_backtest(self.config, start, end, run, decider, frequency, on_decision=progress, policy=policy)
        return (f"{len(bt.decisions)} decisions; Sharpe {bt.result.metrics['sharpe_ratio']:.2f} "
                f"vs {bt.baseline.metrics['sharpe_ratio']:.2f} for equal risk")
