"""MCP server: financial news store (write articles, query, summarize).

    python -m nabia_webui.ai.servers.news --config ai_config.json [--read-only]
"""

from __future__ import annotations

import datetime as dt
import email.utils
import html
import re
import xml.etree.ElementTree as ET
from typing import Literal

import httpx

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from ..db import NEWS_TOPICS
from .common import ServerContext, clean, parse_as_of, server_arguments, tool

Topic = Literal[NEWS_TOPICS]  # type: ignore[valid-type]
AssetClass = Literal["equity_index", "bond_future", "commodity", "crypto"]


class NewsArticle(BaseModel):
    """One news item, as extracted by the news analyzer."""

    url: str = Field(description="Canonical URL of the article (the unique key)")
    published_date: dt.date = Field(description="Publication date, YYYY-MM-DD")
    title: str
    source: str = Field(description="Publisher, e.g. Reuters, Bloomberg, FT")
    topic: Topic = Field(description="Main topic")
    asset_classes: list[AssetClass] = Field(description="Asset classes the news affects")
    symbols: list[str] = Field(default_factory=list, description="Portfolio symbols directly affected")
    sentiment: float = Field(ge=-1.0, le=1.0, description="Price impact on the affected assets: -1 bearish .. +1 bullish")
    relevance: float = Field(ge=0.0, le=1.0, description="Importance for a cross-asset portfolio: 0 .. 1")
    summary: str = Field(description="Two or three factual sentences: what happened and why it matters")


def build_server(ctx: ServerContext) -> MCPServer:
    server = MCPServer(
        "nabia-news",
        instructions="Financial news database of the nabia trend portfolio. Dates are YYYY-MM-DD; "
                     "queries never return articles published after as_of.",
    )

    if not ctx.read_only:
        @tool(server)
        def store_news_articles(articles: list[NewsArticle]) -> dict:
            """Store news articles (upsert by URL, so storing an article twice updates it).

            Only store articles with a real URL and publication date you have seen.
            Symbols must be portfolio symbols (see the news_summary universe)."""
            today = dt.date.today()
            universe = set(ctx.config.symbols)
            rejected = []
            accepted = []
            for a in articles:
                if a.published_date > today:
                    rejected.append({"url": a.url, "reason": "published_date is in the future"})
                elif not set(a.symbols) <= universe:
                    rejected.append({"url": a.url, "reason": f"unknown symbols {sorted(set(a.symbols) - universe)}"})
                else:
                    accepted.append(a.model_dump())
            stored = ctx.store.store_news(accepted)
            return {"stored": stored, "rejected": rejected}

    if not ctx.read_only:
        @tool(server)
        def fetch_news_feeds(max_age_days: int = 2, max_items: int = 60) -> dict:
            """Latest headlines from the configured financial news RSS feeds (markets,
            economy, central banks, energy, crypto), newest first, skipping articles
            already stored. Use it to find candidate stories to classify and store."""
            cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=max_age_days)
            items, errors = [], []
            for source, url in ctx.config.news_feeds.items():
                try:
                    items += [i for i in _read_feed(source, url) if i["published"] is None or i["published"] >= cutoff]
                except Exception as error:  # one broken feed must not hide the others
                    errors.append(f"{source}: {type(error).__name__}")
            known = ctx.store.known_urls([i["url"] for i in items])
            fresh, seen = [], set()
            for i in sorted(items, key=lambda i: i["published"] or cutoff, reverse=True):
                if i["url"] in known or i["url"] in seen:
                    continue
                seen.add(i["url"])
                fresh.append({**i, "published": i["published"].date().isoformat() if i["published"] else None})
            return {"items": fresh[:max_items], "already_stored": len(known), "feed_errors": errors}

    @tool(server)
    def query_news(as_of: str, days: int = 7, topic: Topic | None = None, symbol: str | None = None,
                   asset_class: AssetClass | None = None, limit: int = 30) -> dict:
        """Stored articles published in the `days` days up to and including as_of,
        most recent and relevant first, optionally filtered by topic, symbol or asset class."""
        end = parse_as_of(as_of)
        articles = ctx.store.news(end - dt.timedelta(days=days - 1), end, topic, symbol, asset_class, limit)
        return {"as_of": end.isoformat(), "count": len(articles), "articles": articles}

    @tool(server)
    def news_summary(as_of: str, days: int = 7) -> dict:
        """Aggregate view of the news in the `days` days up to as_of: article counts and
        relevance-weighted sentiment per topic, asset class and symbol, plus the top headlines."""
        end = parse_as_of(as_of)
        articles = ctx.store.news(end - dt.timedelta(days=days - 1), end, limit=10_000)

        def aggregate(key_of) -> dict:
            groups: dict[str, list[dict]] = {}
            for a in articles:
                for key in key_of(a):
                    groups.setdefault(key, []).append(a)
            out = {}
            for key, items in sorted(groups.items()):
                weight = sum(i["relevance"] for i in items) or 1.0
                out[key] = {
                    "articles": len(items),
                    "weighted_sentiment": sum(i["sentiment"] * i["relevance"] for i in items) / weight,
                    "max_relevance": max(i["relevance"] for i in items),
                }
            return out

        top = sorted(articles, key=lambda a: a["relevance"], reverse=True)[:10]
        return clean({
            "as_of": end.isoformat(),
            "days": days,
            "universe": ctx.config.symbols,
            "articles": len(articles),
            "by_topic": aggregate(lambda a: [a["topic"]]),
            "by_asset_class": aggregate(lambda a: a["asset_classes"]),
            "by_symbol": aggregate(lambda a: a["symbols"]),
            "top_headlines": [{k: a[k] for k in ("published_date", "title", "source", "topic", "symbols",
                                                 "sentiment", "relevance", "summary")} for a in top],
        }, 3)

    return server


_ATOM = "{http://www.w3.org/2005/Atom}"


def _text(element: ET.Element | None, *tags: str) -> str:
    for tag in tags:
        value = element.findtext(tag) if element is not None else None
        if value:
            return value.strip()
    return ""


def _parse_date(text: str) -> dt.datetime | None:
    if not text:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(text)  # RSS: RFC 822
    except (TypeError, ValueError):
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))  # Atom: ISO 8601
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _read_feed(source: str, url: str) -> list[dict]:
    """Items of an RSS or Atom feed: source, title, url, published (UTC datetime), description."""
    response = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (nabia news)"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    out = []
    for item in root.findall(".//item") + root.findall(f".//{_ATOM}entry"):
        link = _text(item, "link")
        if not link:
            atom_link = item.find(f"{_ATOM}link")
            link = atom_link.get("href", "") if atom_link is not None else ""
        description = re.sub(r"<[^>]+>", " ", html.unescape(_text(item, "description", f"{_ATOM}summary")))
        title = html.unescape(_text(item, "title", f"{_ATOM}title"))
        if link and title:
            out.append({"source": source, "title": title, "url": link,
                        "published": _parse_date(_text(item, "pubDate", f"{_ATOM}updated", f"{_ATOM}published")),
                        "description": " ".join(description.split())[:400]})
    return out


def main() -> None:
    build_server(server_arguments(__doc__)).run()


if __name__ == "__main__":
    main()
