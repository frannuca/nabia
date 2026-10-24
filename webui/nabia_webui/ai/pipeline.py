"""Daily runs and backtests of the agents' risk budgets."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nabia.marketdata.store import MarketDataStore
from nabia import analytics
from .agents.runtime import AgentRun
from .agents.team import AgentTeam
from .config import AIConfig
from .db import AgentStore

log = logging.getLogger(__name__)

#: A decision policy: given an as-of date, store the risk budgets for it.
Decider = Callable[[dt.date], Awaitable[Any]]


@dataclass
class DailyResult:
    as_of: dt.date
    decision: dict[str, Any]
    """Budgets, rationale, weights and leverage, as stored."""
    runs: list[AgentRun]


async def run_daily(config: AIConfig, workdir: str | Path, as_of: dt.date | None = None,
                    collect_news: bool = True, client: Any = None) -> DailyResult:
    """News -> stats -> execution for one date (today by default): the risk budgets
    and target weights to trade, stored under the run "live"."""
    as_of = as_of or dt.date.today()
    team = AgentTeam(config, workdir, run="live", client=client)
    runs = []
    if collect_news:
        runs.append(await team.collect_news(as_of))
    runs.append(await team.analyze(as_of))
    runs.append(await team.decide(as_of))
    decision = AgentStore(config.db_url).decision("live", as_of)
    if decision is None or decision["as_of"] != as_of.isoformat():
        raise RuntimeError(f"the execution agent did not set risk budgets for {as_of}")
    return DailyResult(as_of, decision, runs)


def decision_dates(index: pd.DatetimeIndex, start: dt.date, end: dt.date, frequency: str = "ME") -> list[dt.date]:
    """Decision dates within [start, end]: every trading date for "D" (daily),
    otherwise the last trading date of each period (pandas offset alias: "ME"
    month end, "W-FRI" weekly, "QE" quarter end)."""
    dates = pd.Series(index, index=index)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if frequency.upper() in ("D", "B"):
        return [d.date() for d in dates]
    return [d.date() for d in dates.groupby(pd.Grouper(freq=frequency)).max().dropna()]


def trend_rule_decider(config: AIConfig, run: str, strength: float = 0.5) -> Decider:
    """A transparent rule-based policy (equal risk tilted by 12-1 month momentum):
    a baseline for the agents, and a way to exercise the pipeline without API calls."""
    market = MarketDataStore(config.db_url)
    store = AgentStore(config.db_url)

    def decide_now(as_of: dt.date) -> None:
        prices = market.load_prices(config.symbols, start=as_of - dt.timedelta(days=config.stats_history_days),
                                    end=as_of)
        budgets = analytics.trend_tilted_budgets(prices, strength, config.min_budget, config.max_budget)
        log.info("trend rule %s: %s", as_of, ", ".join(f"{s} {b:.0%}" for s, b in budgets.items()))
        store.save_decision(run, as_of, budgets, f"trend rule (strength {strength})", decided_by="trend_rule")

    async def decide(as_of: dt.date) -> None:
        await asyncio.to_thread(decide_now, as_of)  # off the event loop: the web app stays responsive

    return decide


POLICIES = ("trend_rule", "ollama", "claude", "agents")
"""Backtest decision policies: the trend rule (no model), the agents on the local
Ollama model, the agents on Claude, or the agents on the config's provider."""


def policy_config(config: AIConfig, policy: str) -> AIConfig:
    """The config to run `policy` with: "ollama" / "claude" force that provider."""
    import dataclasses

    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if policy == "ollama":
        return dataclasses.replace(config, provider="ollama")
    if policy == "claude":
        return dataclasses.replace(config, provider="anthropic")
    return config


def make_decider(config: AIConfig, policy: str, run: str, workdir: str | Path, client: Any = None) -> Decider:
    """The decider for a backtest policy (see POLICIES)."""
    if policy == "trend_rule":
        return trend_rule_decider(config, run)
    return agent_decider(AgentTeam(policy_config(config, policy), workdir, run=run, client=client))


def agent_decider(team: AgentTeam) -> Decider:
    """The agents' policy: stats analyzer, then execution agent, as of each date."""

    async def decide(as_of: dt.date) -> None:
        await team.analyze(as_of)
        await team.decide(as_of)

    return decide


@dataclass
class AgentBacktest:
    result: Any  # nabia.BacktestResult with the decided budgets
    baseline: Any  # nabia.BacktestResult with equal risk budgets
    budget_schedule: pd.DataFrame
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def comparison(self) -> pd.DataFrame:
        keys = ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "annualized_turnover",
                "total_costs"]
        return pd.DataFrame({"decided budgets": {k: self.result.metrics[k] for k in keys},
                             "equal risk": {k: self.baseline.metrics[k] for k in keys}})


async def run_backtest(config: AIConfig, start: dt.date, end: dt.date, run: str, decider: Decider,
                       frequency: str = "ME", reuse: bool = True,
                       on_decision: Callable[[dt.date, int, int], None] | None = None,
                       policy: str = "custom") -> AgentBacktest:
    """Backtest of time-varying risk budgets decided by `decider` at each decision date.

    Decisions are stored under `run`; with ``reuse`` a date that already has one
    is not decided again, so an interrupted backtest resumes where it stopped.
    The budgets then drive one continuous backtest (``nabia.backtest`` with a
    budget schedule): budgets decided at the close of date d apply from the
    rebalance at d on. An equal-risk backtest over the same period is the baseline.
    The run's period, policy and metrics are recorded (``AgentStore.backtest_runs``).
    """
    config.validate()
    store = AgentStore(config.db_url)
    store.save_backtest_run(run, policy, start, end, frequency, None)
    prices = _backtest_prices(config, start, end)
    dates = decision_dates(prices.dropna().index, start, end, frequency)
    if not dates:
        raise ValueError(f"no trading dates between {start} and {end}")

    for k, as_of in enumerate(dates):
        existing = store.decision(run, as_of)
        if reuse and existing is not None and existing["as_of"] == as_of.isoformat():
            continue
        if on_decision:
            on_decision(as_of, k, len(dates))
        log.info("backtest %s: deciding %s (%d/%d)", run, as_of, k + 1, len(dates))
        await decider(as_of)
        made = store.decision(run, as_of)
        if made is None or made["as_of"] != as_of.isoformat():
            raise RuntimeError(f"no risk budgets were stored for {as_of}")

    log.info("backtest %s: %d decision dates; running the portfolio backtest", run, len(dates))
    bt = await asyncio.to_thread(evaluate_decisions, config, run, start, end, prices)
    log.info("backtest %s: Sharpe %.2f (equal risk %.2f), max drawdown %.1f%%", run, bt.result.metrics["sharpe_ratio"],
             bt.baseline.metrics["sharpe_ratio"], 100 * bt.result.metrics["max_drawdown"])
    store.save_backtest_run(run, policy, start, end, frequency,
                            {"decided": bt.result.metrics, "equal_risk": bt.baseline.metrics})
    return bt


def _backtest_prices(config: AIConfig, start: dt.date, end: dt.date) -> pd.DataFrame:
    window = int(config.portfolio.get("covariance_window", 60))
    warmup = dt.timedelta(days=int(window * 1.6) + 30)  # trading days -> calendar days, with margin
    return MarketDataStore(config.db_url).load_prices(config.symbols, start=start - warmup, end=end)


def evaluate_decisions(config: AIConfig, run: str, start: dt.date, end: dt.date,
                       prices: pd.DataFrame | None = None) -> AgentBacktest:
    """Backtest of the decisions already stored for `run` between start and end
    (no new decisions), against equal risk budgets."""
    import nabia

    prices = _backtest_prices(config, start, end) if prices is None else prices
    decisions = [d for d in AgentStore(config.db_url).decisions(run)
                 if start.isoformat() <= d["as_of"] <= end.isoformat()]
    if not decisions:
        raise ValueError(f"run {run!r} has no decisions between {start} and {end}")
    schedule = pd.DataFrame([d["budgets"] for d in decisions],
                            index=pd.DatetimeIndex([d["as_of"] for d in decisions], name="date"))[config.symbols]
    params = dict(config.portfolio)
    result = nabia.backtest(prices, params, budget_schedule=schedule)
    baseline = nabia.backtest(prices, params)
    return AgentBacktest(result, baseline, schedule, decisions)
