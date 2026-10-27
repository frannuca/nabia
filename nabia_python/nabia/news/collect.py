"""Collecting articles into a news store: RSS feeds (today's items) and GDELT GKG files (history)."""

from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor

from .sources import FEEDS, fetch_gkg, fetch_rss, now_utc


def collect_gkg(store, since, until=None, per_day=10, weekends=False, workers=4):
    """GDELT GKG history into the store, `per_day` finance articles per day: (articles added, failed days)."""
    added, failed = 0, []
    until = until or dt.datetime.now(dt.timezone.utc).date()
    days = [since + dt.timedelta(days=k) for k in range((until - since).days)]
    days = [d for d in days if weekends or d.weekday() < 5]

    def one(day):
        try:
            return day, fetch_gkg(day, per_day=per_day), None
        except Exception as error:
            return day, [], error

    with ThreadPoolExecutor(max_workers=workers) as pool:  # a few files at a time from GDELT's file server
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
    return added, failed


def collect_rss(store, feeds=None):
    """Today's items of the RSS feeds into the store: (articles added, failed feeds)."""
    added, failed, received = 0, [], now_utc()
    for name in feeds or FEEDS:
        url, weight = FEEDS[name]
        try:
            n = store.add(fetch_rss(name, url, weight, received))
            print(f"{name:18} +{n}")
            added += n
        except Exception as error:  # a feed down must not stop the others
            failed.append(name)
            print(f"{name:18} failed: {type(error).__name__}: {str(error)[:80]}", file=sys.stderr)
    return added, failed
