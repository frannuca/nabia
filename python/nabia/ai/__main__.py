"""Command line for the nabia agents.

    python -m nabia.ai init     --config ai_config.json [--db sqlite:///market.db]
    python -m nabia.ai news     --config ai_config.json
    python -m nabia.ai daily    --config ai_config.json [--as-of 2026-10-09] [--no-news]
    python -m nabia.ai backtest --config ai_config.json --start 2023-01-01 --end 2024-12-31 \\
                                --run bt-2024 [--policy agents|trend_rule] [--frequency ME] [--yes]

Agent commands call the Claude API (usage is billed); the trend_rule policy does not.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import sys
from pathlib import Path

from .agents.team import AgentTeam
from .config import AIConfig
from .pipeline import POLICIES, make_decider, policy_config, run_backtest, run_daily


def _print_run(run) -> None:
    errors = sum(1 for c in run.tool_calls if c.is_error)
    print(f"[{run.agent}] {run.turns} turns, {len(run.tool_calls)} tool calls ({errors} errors), "
          f"{run.server_tool_calls} web calls, tokens {run.usage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nabia.ai", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "news", "daily", "backtest"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--workdir", type=Path, default=Path(".nabia_ai"), help="scratch directory for the agents")
        if name == "init":
            p.add_argument("--db", default="sqlite:///market.db")
        if name == "daily":
            p.add_argument("--as-of", type=dt.date.fromisoformat, default=None)
            p.add_argument("--no-news", action="store_true", help="skip news collection")
        if name == "backtest":
            p.add_argument("--start", type=dt.date.fromisoformat, required=True)
            p.add_argument("--end", type=dt.date.fromisoformat, required=True)
            p.add_argument("--run", required=True, help="name under which decisions are stored (resumable)")
            p.add_argument("--policy", choices=POLICIES, default="agents",
                           help="trend_rule (no model), ollama (local model), claude, or agents (the config's provider)")
            p.add_argument("--frequency", default="ME",
                           help="decision frequency: D (every trading day), W-FRI, ME, QE, ... (pandas)")
            p.add_argument("--output", type=Path, default=None, help="CSV of daily weights and PnL")
            p.add_argument("--yes", action="store_true", help="do not ask before calling the API")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "init":
        if args.config.exists():
            parser.error(f"{args.config} exists")
        AIConfig(db_url=args.db).save(args.config)
        print(f"wrote {args.config}; edit symbols, budget bounds and portfolio settings as needed")
        return 0

    config = AIConfig.load(args.config)
    if args.command == "news":
        run = asyncio.run(AgentTeam(config, args.workdir).collect_news())
        _print_run(run)
        print(run.final_text)
    elif args.command == "daily":
        result = asyncio.run(run_daily(config, args.workdir, args.as_of, collect_news=not args.no_news))
        for run in result.runs:
            _print_run(run)
        d = result.decision
        print(f"\nRisk budgets and target weights for {result.as_of} (leverage {d['leverage']:.2f}):")
        for s in config.symbols:
            print(f"  {s:10} budget {d['budgets'][s]:7.2%}   weight {d['weights'][s]:8.2%}")
        print(f"\nRationale: {d['rationale']}")
    else:
        decider = make_decider(config, args.policy, args.run, args.workdir)
        if args.policy != "trend_rule" and not args.yes:
            model = policy_config(config, args.policy).model_label
            cost = "billed" if model.startswith("claude") else "local, can take hours"
            answer = input(f"Run the stats and execution agents on {model} at each {args.frequency} decision date "
                           f"between {args.start} and {args.end} (two agent runs per date, {cost})? [y/N] ")
            if answer.strip().lower() != "y":
                return 1
        bt = asyncio.run(run_backtest(config, args.start, args.end, args.run, decider, args.frequency,
                                      on_decision=lambda d, k, n: print(f"deciding {d} ({k + 1}/{n})"),
                                      policy=args.policy))
        print(f"\n{len(bt.decisions)} decisions")
        print(bt.comparison().to_string(float_format="{:.4f}".format))
        if args.output:
            bt.result.weights.join(bt.result.performance).to_csv(args.output)
            print(f"daily weights and PnL written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
