"""Collecting news into the database: today's RSS items and the GDELT GKG history."""

from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor

from datalayer import NewsDB

from .sources import fetch_gkg, fetch_rss


def collect_rss(db: NewsDB, feeds: dict) -> int:
    """Every feed's current items; a feed that fails is reported and skipped."""
    added, received = 0, dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    for name, url in feeds.items():
        try:
            added += (n := db.add_news(fetch_rss(name, url, received)))
            print(f"rss {name:18} +{n}", flush=True)
        except Exception as error:
            print(f"rss {name:18} failed: {type(error).__name__}: {str(error)[:80]}", file=sys.stderr)
    return added


def collect_gkg(db: NewsDB, since: dt.date, until: dt.date, c: dict) -> int:
    """GKG articles of the days in [since, until), a few days downloaded at a time. Days that fail are reported and
    can be collected again with --since (articles already stored are skipped)."""
    days = [since + dt.timedelta(k) for k in range((until - since).days)]
    days = [d for d in days if c["weekends"] or d.weekday() < 5]

    def one(day):
        try:
            return day, fetch_gkg(day, c["gkg_slices"], c["gkg_per_day"]), None
        except Exception as error:
            return day, [], error

    added = 0
    with ThreadPoolExecutor(c["workers"]) as pool:
        for day, articles, error in pool.map(one, days):  # results in day order; writes from this thread only
            if error:
                print(f"gkg {day} failed: {type(error).__name__}: {str(error)[:80]}", file=sys.stderr, flush=True)
                continue
            added += (n := db.add_news(articles))
            print(f"gkg {day} +{n}", flush=True)
    return added


def collect(cfg: dict, since: str | None = None, until: str | None = None) -> int:
    """RSS and GKG per the configuration's collect section. Without since, GKG resumes after the last GKG day
    stored (from collect.since on an empty database)."""
    c, today = cfg["collect"], dt.datetime.now(dt.timezone.utc).date()
    with NewsDB(cfg["db"]) as db:
        added = collect_rss(db, c["feeds"]) if c["rss"] else 0
        if c["gkg"]:
            last = db.last_received("gdelt-gkg:")
            start = dt.date.fromisoformat(since) if since else max(
                dt.date.fromisoformat(c["since"]), dt.date.fromisoformat(last[:10]) + dt.timedelta(1) if last else dt.date.min)
            added += collect_gkg(db, start, dt.date.fromisoformat(until or c["until"] or str(today)), c)
        print(f"{added} new articles; database: {db.stats()}")
    return added
