"""SQLite data layer of the news pipeline. Three tables, added next to the others of the database (data/market.db):
  news          one row per article, deduplicated by URL and by normalized headline on the same received day;
  news_scoring  one row per article, model and prompt version once scored (also when the article was irrelevant);
  news_scores   the sentiment and confidence of a scored article on each economic factor it concerns.
All times are UTC ISO 8601 strings, so they sort and compare as text."""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id           INTEGER PRIMARY KEY,
    url          TEXT UNIQUE,              -- NULL when the source gives none
    headline     TEXT NOT NULL,
    body         TEXT NOT NULL,            -- the headline again when the source has no text
    source       TEXT NOT NULL,            -- feed or outlet, e.g. CNBC, gdelt-gkg:reuters.com
    published_at TEXT NOT NULL,
    received_at  TEXT NOT NULL,            -- when the article became available (point in time)
    dedup_key    TEXT NOT NULL UNIQUE      -- normalized headline | received day
);
CREATE INDEX IF NOT EXISTS ix_news_received ON news (received_at);
CREATE TABLE IF NOT EXISTS news_scoring (
    news_id   INTEGER NOT NULL REFERENCES news (id),
    model     TEXT NOT NULL,               -- provider:model, e.g. ollama:gemma4:26b
    prompt    TEXT NOT NULL,               -- prompt version (hash of the prompt and the factor list)
    relevant  INTEGER NOT NULL,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (news_id, model, prompt)
);
CREATE TABLE IF NOT EXISTS news_scores (
    news_id    INTEGER NOT NULL,
    model      TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    factor     TEXT NOT NULL,
    sentiment  REAL NOT NULL,              -- -1..1, sign as defined by the factor's "positive"
    confidence REAL NOT NULL,              -- 0..1
    PRIMARY KEY (news_id, model, prompt, factor),
    FOREIGN KEY (news_id, model, prompt) REFERENCES news_scoring (news_id, model, prompt)
);
"""


def iso(t) -> str:
    """A datetime (naive = UTC) or an ISO string as YYYY-MM-DDTHH:MM:SSZ."""
    if isinstance(t, str):
        t = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
    t = t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now() -> str:
    return iso(dt.datetime.now(dt.timezone.utc))


def dedup_key(headline: str, received_at: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", headline.lower())) + "|" + received_at[:10]


class NewsDB:
    """Reads and writes of the news tables. Use from one thread (the scorer writes from its main thread)."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.con.close()

    def add_news(self, articles) -> int:
        """Inserts articles (dicts: headline, body, source, published_at, received_at, url) not stored yet; returns
        how many were new."""
        rows = []
        for a in articles:
            received = iso(a["received_at"])
            rows.append((a.get("url") or None, a["headline"], a.get("body") or a["headline"], a["source"],
                         min(iso(a.get("published_at") or received), received), received, dedup_key(a["headline"], received)))
        before = self.con.total_changes
        with self.con:  # one transaction; duplicates (url or dedup_key) are skipped
            self.con.executemany("INSERT OR IGNORE INTO news (url, headline, body, source, published_at, received_at, "
                                 "dedup_key) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return self.con.total_changes - before

    def last_received(self, source_prefix: str = "") -> str | None:
        """The latest received_at of the articles whose source starts with source_prefix."""
        return self.con.execute("SELECT max(received_at) FROM news WHERE source LIKE ?", (source_prefix + "%",)).fetchone()[0]

    def unscored(self, model: str, prompt: str, since=None, until=None, limit=None) -> list[dict]:
        """Articles without a scoring by this model and prompt version, oldest first, received in [since, until)."""
        rows = self.con.execute(
            "SELECT n.* FROM news n LEFT JOIN news_scoring s ON s.news_id = n.id AND s.model = ? AND s.prompt = ? "
            "WHERE s.news_id IS NULL AND n.received_at >= ? AND n.received_at < ? ORDER BY n.received_at LIMIT ?",
            (model, prompt, since or "", until or "9999", -1 if limit is None else limit))
        return [dict(r) for r in rows]

    def save_scoring(self, news_id: int, model: str, prompt: str, relevant: bool, scores) -> None:
        """Stores (replacing) the scoring of an article: scores = [(factor, sentiment, confidence)]."""
        key = (news_id, model, prompt)
        with self.con:
            self.con.execute("DELETE FROM news_scores WHERE news_id = ? AND model = ? AND prompt = ?", key)
            self.con.execute("INSERT OR REPLACE INTO news_scoring VALUES (?, ?, ?, ?, ?)", (*key, int(relevant), now()))
            self.con.executemany("INSERT INTO news_scores VALUES (?, ?, ?, ?, ?, ?)", [(*key, *s) for s in scores])

    def scores(self, model=None, prompt=None, since=None, until=None) -> list[dict]:
        """Factor scores with their article's times and headline, by received time: the input of later models."""
        rows = self.con.execute(
            "SELECT n.id AS news_id, n.received_at, n.published_at, n.source, n.headline, s.model, s.prompt, s.factor, "
            "s.sentiment, s.confidence FROM news_scores s JOIN news n ON n.id = s.news_id "
            "WHERE s.model LIKE ? AND s.prompt LIKE ? AND n.received_at >= ? AND n.received_at < ? ORDER BY n.received_at",
            (model or "%", prompt or "%", since or "", until or "9999"))
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        q = lambda sql: self.con.execute(sql).fetchone()
        first, last, n = q("SELECT min(received_at), max(received_at), count(*) FROM news")
        return {"news": n, "first": first, "last": last,
                "scored": dict(self.con.execute("SELECT model || ' ' || prompt, count(*) FROM news_scoring GROUP BY 1").fetchall()),
                "relevant": q("SELECT count(*) FROM news_scoring WHERE relevant")[0],
                "factor_scores": q("SELECT count(*) FROM news_scores")[0]}
