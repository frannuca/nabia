"""python -m nabia.news fetch | stats  (see nabia.news)"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import __doc__ as overview
from .sources import FEEDS, GDELT_QUERIES, fetch_gdelt, fetch_gkg, fetch_rss, now_utc
from .store import NewsStore


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nabia.news", description=overview,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download news and append the new articles to the store")
    fetch.add_argument("--file", type=Path, default=Path("data/news.json"), help="the news store (default data/news.json)")
    fetch.add_argument("--feeds", nargs="*", help=f"RSS feeds to read (default all: {', '.join(FEEDS)})")
    fetch.add_argument("--no-rss", action="store_true", help="skip the RSS feeds")
    fetch.add_argument("--gdelt", action="store_true", help="also query GDELT (history: use --since/--until)")
    fetch.add_argument("--query", nargs="*", help="GDELT queries (default: one per asset class)")
    fetch.add_argument("--gkg", action="store_true", help="history from GDELT's raw GKG files (needs --since; no API limit)")
    fetch.add_argument("--per-day", type=int, default=10, help="GKG: articles kept per day (default 10)")
    fetch.add_argument("--weekends", action="store_true", help="GKG: also read weekend days")
    fetch.add_argument("--workers", type=int, default=4, help="GKG: days downloaded in parallel (default 4)")
    fetch.add_argument("--since", type=dt.date.fromisoformat, help="GDELT start date (default: yesterday)")
    fetch.add_argument("--until", type=dt.date.fromisoformat, help="GDELT end date, exclusive (default: today)")
    score = sub.add_parser("score", help="score the stored news with an LLM (default: local Ollama) into a scores file")
    score.add_argument("--file", type=Path, default=Path("data/news.json"))
    score.add_argument("--out", type=Path, default=Path("data/news_scores.json"), help="default data/news_scores.json")
    score.add_argument("--provider", choices=["ollama", "anthropic"], default="ollama")
    score.add_argument("--model", help="Ollama model (default: llm.ollama_model, gemma4:26b)")
    score.add_argument("--workers", type=int, help="articles scored concurrently (default: llm.workers)")
    score.add_argument("--config", type=Path, help="JSON overrides of the sentiment pipeline's CONFIG")
    stats = sub.add_parser("stats", help="summarize the store")
    stats.add_argument("--file", type=Path, default=Path("data/news.json"))
    args = ap.parse_args(argv)

    if args.command == "score":
        from ..sentiment_rb.config import load_config, merge
        from .scoring import score_news
        llm = {"provider": args.provider, **({"ollama_model": args.model} if args.model else {}),
               **({"workers": args.workers} if args.workers else {})}
        score_news(merge(load_config(args.config), {"llm": llm}), args.file, args.out)
        return 0
    store = NewsStore(args.file)
    if args.command == "stats":
        print(json.dumps(store.stats(), indent=2))
        return 0
    added, failed = 0, []
    if not args.no_rss:
        received = now_utc()
        for name in args.feeds or FEEDS:
            url, weight = FEEDS[name]
            try:
                n = store.add(fetch_rss(name, url, weight, received))
                print(f"{name:18} +{n}")
                added += n
            except Exception as error:  # a feed down must not stop the others
                failed.append(name)
                print(f"{name:18} failed: {type(error).__name__}: {str(error)[:80]}", file=sys.stderr)
    if args.gdelt:
        today = dt.datetime.now(dt.timezone.utc).date()
        start = dt.datetime.combine(args.since or today - dt.timedelta(days=1), dt.time(), dt.timezone.utc)
        end = dt.datetime.combine(args.until or today, dt.time(), dt.timezone.utc)
        for query in args.query or GDELT_QUERIES:
            try:
                n = store.add(fetch_gdelt(query, start, end))
                print(f"GDELT {query[:50]:50} +{n}")
                added += n
            except Exception as error:
                failed.append(query)
                print(f"GDELT {query[:50]} failed: {type(error).__name__}: {str(error)[:80]}", file=sys.stderr)
            store.save()  # keep what was collected so far
    if args.gkg:
        if not args.since:
            ap.error("--gkg needs --since")
        from concurrent.futures import ThreadPoolExecutor
        until = args.until or dt.datetime.now(dt.timezone.utc).date()
        days = [args.since + dt.timedelta(days=k) for k in range((until - args.since).days)]
        days = [d for d in days if args.weekends or d.weekday() < 5]

        def one(day):
            try:
                return day, fetch_gkg(day, per_day=args.per_day), None
            except Exception as error:
                return day, [], error

        with ThreadPoolExecutor(max_workers=args.workers) as pool:  # a few files at a time from GDELT's file server
            for k, (day, articles, error) in enumerate(pool.map(one, days), 1):
                if error:
                    failed.append(str(day))
                    print(f"GKG {day} failed: {type(error).__name__}: {str(error)[:80]}", file=sys.stderr, flush=True)
                    continue
                n = store.add(articles)
                added += n
                print(f"GKG {day} +{n} ({len(store.articles)} stored)", flush=True)
                if k % 20 == 0:
                    store.save()  # keep what was collected so far
    store.save()
    print(f"{added} new articles, {len(store.articles)} in {args.file}" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed and not added else 0


if __name__ == "__main__":
    sys.exit(main())
