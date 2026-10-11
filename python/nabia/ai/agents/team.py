"""The three agents: news analyzer, stats analyzer and execution (risk budgets)."""

from __future__ import annotations

import datetime as dt
import logging
import traceback
from pathlib import Path
from typing import Any

from ..config import AIConfig
from ..db import AgentStore
from .runtime import NEWS_SERVER, PORTFOLIO_SERVER, STATS_SERVER, AgentRun, AgentRuntime, MCPServerSpec

log = logging.getLogger(__name__)

AS_OF_RULE = (
    "Today is {as_of}. Treat that date as the present: base every judgement on what the tools return "
    "(they only see data up to {as_of}) and never on knowledge of events after {as_of}."
)

NEWS_SYSTEM = """You are the news analyst of a systematic cross-asset trend portfolio.

{as_of_rule}

Portfolio universe (symbol: name, asset class):
{universe}

Your job: find the financial news published on {as_of} and the previous day that matters for these
assets, and store it in the news database. Cover central banks and rates, inflation and growth data,
equity markets, energy and metals, and crypto; prefer primary and reputable sources.

How to work:
1. Call query_news for {as_of} (days=2) to see what is already stored; do not store it again.
2. {research}
   Aim for 8-20 distinct, material articles, not every headline: skip stories with no plausible
   effect on these assets.
3. Store them with store_news_articles (in batches is fine). Use only URLs and dates you actually saw
   in the feed items or search results; never invent them. topic: the main driver. asset_classes and symbols: what the story
   moves (symbols only from the universe above). sentiment: expected price impact on the affected
   assets, -1 clearly bearish to +1 clearly bullish (0 neutral or mixed). relevance: 1 for
   portfolio-wide shocks, 0.5 for one asset class, lower for minor items.
4. Finish with five lines summarizing the dominant themes and their direction."""

STATS_SYSTEM = """You are the quantitative analyst of a systematic cross-asset trend portfolio that allocates
risk across the assets below with risk budgets (each asset's share of portfolio risk).

{as_of_rule}

Portfolio universe:
{universe}

Your job: analyze the statistical state of these assets as of {as_of} and save a structured report
for the portfolio manager.

How to work:
1. Look at market_regime, trend_signals, rolling_volatility and correlation_matrix (60-day, and
   252-day for context). Use get_stats_report to see the previous report for continuity.
2. For each asset judge: trend (sign, strength and persistence across horizons), volatility regime
   (level and percentile vs its last year), and diversification (correlation with the others,
   clusters of assets that move together).
3. Suggest a budget tilt per asset in [-1, 1]: positive for persistent uptrends with contained
   volatility and diversification value; negative for downtrends, stressed volatility or redundant
   exposure. Tilts should be proportionate to the strength of the evidence; 0 when unclear.
4. Save the report with save_stats_report (every symbol in assets, numbers cited in each reason).
5. Finish with a short summary of the regime and the main tilts."""

EXECUTION_SYSTEM = """You are the portfolio manager of a systematic cross-asset trend portfolio. The portfolio holds
the assets below with long-only risk budgeting: you set each asset's risk budget (its share of total
portfolio risk); the engine turns budgets into weights using the trailing covariance and scales the
portfolio to its volatility target.

{as_of_rule}

Portfolio universe:
{universe}

Your job: decide the risk budgets for {as_of} and compute the resulting target weights.

How to work:
1. Read get_portfolio_config (budget bounds, settings) and get_previous_decision.
2. Read the quantitative view with get_stats_report and its underlying signals if needed.
3. Read news_summary for {as_of} (last {news_days} days) and query_news for detail on material themes.
4. Decide budgets. Start from equal risk and tilt with evidence: primarily the statistical trend and
   risk picture; news as a secondary input that can confirm, temper or flag a risk the statistics do
   not show yet. Respect the bounds. Trading has costs: keep budgets close to the previous decision
   unless the evidence has changed materially, and explain any large change.
5. Call set_risk_budgets for {as_of} with a rationale that cites the evidence behind each tilt, then
   compute_target_weights for {as_of}.
6. Finish with a table of budgets and weights and a three-sentence rationale.
{mode_note}"""

BACKTEST_NOTE = """
This is a historical simulation: news is only what was stored before {as_of}, and there is no web
access. Decide exactly as you would have on {as_of}."""


class AgentTeam:
    """Runs the news, stats and execution agents for one run ("live" or a backtest name)."""

    def __init__(self, config: AIConfig, workdir: str | Path, run: str = "live", client: Any = None) -> None:
        config.validate()
        self.config = config
        self.run = run
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.config_path = config.save(workdir / "ai_config.json")  # read by the MCP server processes
        self.runtime = AgentRuntime(config, self.config_path, run=run, client=client)
        self.store = AgentStore(config.db_url)

    async def _run(self, agent: str, as_of: dt.date, **kwargs: Any) -> AgentRun:
        """Runs an agent and records its transcript summary (also when it fails)."""
        started = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        log.info("starting %s as of %s (run %s)", agent, as_of, self.run)
        try:
            record = await self.runtime.run(agent, as_of.isoformat(), **kwargs)
        except Exception as error:
            failed = AgentRun(agent=agent, as_of=as_of.isoformat(), final_text="")
            self.store.save_agent_run(self.run, failed, started,
                                      error="".join(traceback.format_exception_only(error)).strip())
            raise
        self.store.save_agent_run(self.run, record, started)
        log.info("%s done: %d turns, %d tool calls (%d errors), tokens %s", agent, record.turns,
                 len(record.tool_calls), sum(1 for c in record.tool_calls if c.is_error), record.usage)
        return record

    def _universe(self) -> str:
        from ..db import AgentStore  # noqa: F401  (ensures tables exist before servers start)
        from ...marketdata.store import MarketDataStore

        assets = MarketDataStore(self.config.db_url).assets()
        info = {r.symbol: (r.name, r.asset_class) for r in assets.itertuples()}
        return "\n".join(f"- {s}: {info.get(s, ('?', '?'))[0]}, {info.get(s, ('?', '?'))[1]}" for s in self.config.symbols)

    def _fill(self, template: str, as_of: dt.date, **extra: Any) -> str:
        values = {"as_of": as_of.isoformat(), "universe": self._universe(),
                  "news_days": self.config.news_lookback_days, **extra}
        values["as_of_rule"] = AS_OF_RULE.format(**values)
        return template.format(**values)

    async def collect_news(self, as_of: dt.date | None = None) -> AgentRun:
        """News analyzer: RSS feeds (and web search on Claude) -> news database.
        Live use only: feeds and the web are not point-in-time."""
        as_of = as_of or dt.date.today()
        if self.config.provider == "anthropic":
            web = [
                {"type": "web_search_20260209", "name": "web_search", "max_uses": self.config.news_max_searches},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": self.config.news_max_searches},
            ]
            research = ("Call fetch_news_feeds for the latest headlines from financial news and central-bank\n"
                        "   feeds, then use web_search (and web_fetch on promising URLs when a snippet is not\n"
                        "   enough) to fill gaps and confirm the most market-moving stories.")
        else:
            web = []  # a local model has no web search: the RSS feeds are the source
            research = ("Call fetch_news_feeds for the latest headlines from financial news and central-bank\n"
                        "   feeds, and pick the stories that matter, judging them from their title and description.")
        return await self._run(
            "news_analyzer", as_of, system=self._fill(NEWS_SYSTEM, as_of, research=research),
            prompt=f"Collect and store the market-moving news for {as_of}.",
            servers=[MCPServerSpec(NEWS_SERVER)], server_tools=web, effort=self.config.effort.get("news", "medium"))

    async def analyze(self, as_of: dt.date) -> AgentRun:
        """Stats analyzer: statistics -> saved stats report."""
        return await self._run(
            "stats_analyzer", as_of, system=self._fill(STATS_SYSTEM, as_of),
            prompt=f"Analyze the portfolio assets as of {as_of} and save the stats report.",
            servers=[MCPServerSpec(STATS_SERVER)], effort=self.config.effort.get("stats", "high"))

    async def decide(self, as_of: dt.date) -> AgentRun:
        """Execution agent: news + stats -> risk budgets -> target weights."""
        note = BACKTEST_NOTE.format(as_of=as_of.isoformat()) if self.run != "live" else ""
        return await self._run(
            "execution", as_of, system=self._fill(EXECUTION_SYSTEM, as_of, mode_note=note),
            prompt=f"Decide the risk budgets for {as_of} and compute the target weights.",
            servers=[MCPServerSpec(NEWS_SERVER, read_only=True), MCPServerSpec(STATS_SERVER, read_only=True),
                     MCPServerSpec(PORTFOLIO_SERVER)],
            effort=self.config.effort.get("execution", "high"))
