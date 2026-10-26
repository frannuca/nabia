"""The news store: one JSON file, {"description": ..., "articles": [...]}, in the format the pipeline reads
(nabia.sentiment_rb.data.load_news), deduplicated by URL and by normalized headline on the same received day
(the pipeline weights repeats of a story over the following days by novelty), written atomically."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

DESCRIPTION = ("News collected by nabia.news: publish_ts and received_ts in UTC (received_ts = download time for RSS, "
               "first-seen time for GDELT), source = weight class (wire, paper, blog), feed, headline, body, url.")


def _key(headline: str, received) -> str:
    """Normalized headline and received day."""
    return " ".join(re.findall(r"[a-z0-9]+", headline.lower())) + "|" + _iso(received)[:10]


def _iso(t) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(t, dt.datetime) else str(t)


class NewsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.articles = data.get("articles", []) if isinstance(data, dict) else data
        self.urls = {a.get("url") for a in self.articles if a.get("url")}
        self.headlines = {_key(a["headline"], a["received_ts"]) for a in self.articles}

    def add(self, articles) -> int:
        """Appends the articles not yet stored (same URL, or same headline the same day); returns how many."""
        added = 0
        for a in sorted(articles, key=lambda a: a["received_ts"]):
            key = _key(a["headline"], a["received_ts"])
            if (a.get("url") and a["url"] in self.urls) or key in self.headlines:
                continue
            self.articles.append({"id": len(self.articles), **{k: _iso(v) for k, v in a.items()}})
            self.urls.add(a.get("url")), self.headlines.add(key)
            added += 1
        return added

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"description": DESCRIPTION, "articles": self.articles}, ensure_ascii=False))
        tmp.replace(self.path)

    def stats(self) -> dict:
        received = sorted(a["received_ts"] for a in self.articles)
        return {"articles": len(self.articles), "first": received[0] if received else None,
                "last": received[-1] if received else None,
                "by_feed": dict(Counter(a.get("feed", "?") for a in self.articles).most_common()),
                "by_month": dict(sorted(Counter(r[:7] for r in received).items()))}
