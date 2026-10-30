"""python -m scoring [config.json] [--since D] [--until D] [--limit N] [--provider P] [--workers N] [--dry-run]"""

import argparse

from datalayer import load_config

from .score import score

p = argparse.ArgumentParser(prog="scoring", description="Score the stored news on the economic factors with the LLM.")
p.add_argument("config", nargs="?", help="configuration file (default: config/config.json)")
p.add_argument("--since", help="first received day")
p.add_argument("--until", help="end received day, excluded")
p.add_argument("--limit", type=int, help="at most this many articles")
p.add_argument("--provider", choices=["ollama", "openai", "anthropic"], help="overrides llm.provider")
p.add_argument("--workers", type=int, help="overrides llm.workers: parallel requests (useful with the APIs)")
p.add_argument("--dry-run", action="store_true", help="only count the articles to score and their prompt tokens")
args = p.parse_args()
cfg = load_config(args.config)
cfg["llm"]["provider"] = args.provider or cfg["llm"]["provider"]
cfg["llm"]["workers"] = args.workers or cfg["llm"]["workers"]
score(cfg, args.since, args.until, args.limit, args.dry_run)
