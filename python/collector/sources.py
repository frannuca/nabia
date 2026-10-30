"""News sources. Each returns articles as dicts: headline, body, source, published_at and received_at (UTC datetimes),
url. received_at is when the article became available: the download time for RSS, GDELT's processing time for GKG."""

from __future__ import annotations

import csv
import datetime as dt
import email.utils
import html
import io
import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

HEADERS = {"User-Agent": "Mozilla/5.0 (nabia news collector)"}
ATOM = "{http://www.w3.org/2005/Atom}"

#: Finance and macro topics, matched in GKG page titles.
GKG_TOPICS = re.compile(
    r"\b(stocks?|stock market|equit(y|ies)|wall street|s&p|nasdaq|dow jones|euro stoxx|ftse|dax|nikkei|oil|crude|brent|"
    r"wti|opec|natural gas|gold|silver|copper|bullion|metals?|treasur(y|ies)|bonds?|yields?|federal reserve|fed|ecb|"
    r"central bank|rate (cut|hike)s?|interest rates?|inflation|cpi|recession|gdp|jobs report|payrolls|unemployment|"
    r"junk bonds?|high.yield|credit spreads?|commodit(y|ies)|tariffs?|dollar|euro|yen|yuan|currenc(y|ies)|forex|"
    r"earnings|markets?)\b", re.I)
#: Financial outlets ranked first when a day has more matching titles than it keeps.
GKG_PREFERRED = ("reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "ft.com", "wsj.com", "apnews.com",
                 "investing.com", "finance.yahoo.com", "barrons.com", "fxstreet.com", "oilprice.com", "kitco.com",
                 "economist.com", "businessinsider.com", "forbes.com", "fortune.com", "morningstar.com", "nasdaq.com",
                 "seekingalpha.com", "thestreet.com", "theguardian.com", "bbc.co.uk", "nytimes.com")


def get(url: str, timeout: float = 60) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout).read()


def clean(text: str | None) -> str:
    """Plain text: tags removed, entities decoded, spaces collapsed."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def parse_date(text: str | None) -> dt.datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom); naive times are UTC."""
    for parse in (email.utils.parsedate_to_datetime, lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))):
        try:
            d = parse((text or "").strip())
            return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
        except (TypeError, ValueError):
            pass
    return None


def fetch_rss(name: str, url: str, received: dt.datetime) -> list[dict]:
    """The items of an RSS 2.0 or Atom feed (feeds carry no history: only today's items)."""
    articles = []
    for item in ET.fromstring(get(url, 30)).iter():
        if item.tag == "item":
            title, link, body = item.findtext("title"), item.findtext("link"), item.findtext("description")
            date = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
        elif item.tag == ATOM + "entry":
            title, body = item.findtext(ATOM + "title"), item.findtext(ATOM + "summary") or item.findtext(ATOM + "content")
            link = (item.find(ATOM + "link") if item.find(ATOM + "link") is not None else {}).get("href")
            date = item.findtext(ATOM + "published") or item.findtext(ATOM + "updated")
        else:
            continue
        if headline := clean(title):
            articles.append({"headline": headline, "body": clean(body) or headline, "source": name, "url": (link or "").strip(),
                             "published_at": parse_date(date) or received, "received_at": received})
    return articles


def fetch_gkg(day: dt.date, slices=("143000", "203000"), per_day: int = 10) -> list[dict]:
    """History from GDELT's Global Knowledge Graph: files every 15 minutes listing the web articles GDELT processed,
    with their page titles. Reads the given 15-minute slices (HHMMSS, UTC) of the day, keeps the finance titles,
    ranks the preferred outlets first and returns at most per_day articles (titles only: GKG has no text)."""
    found, seen = [], set()
    csv.field_size_limit(1 << 30)
    for hhmmss in slices:
        try:
            raw = get(f"https://data.gdeltproject.org/gdeltv2/{day:%Y%m%d}{hhmmss}.gkg.csv.zip", 180)
        except urllib.error.HTTPError as error:
            if error.code == 404:  # a missing slice (GDELT outage)
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
        for row in csv.reader(io.StringIO(text), delimiter="\t"):  # columns: 1 date, 3 domain, 4 url, 26 extras
            if len(row) < 27 or not (m := re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", row[26])):
                continue
            headline, domain = clean(m.group(1)), row[3].lower()
            key = " ".join(re.findall(r"[a-z0-9]+", headline.lower()))
            if len(headline) < 25 or key in seen or not GKG_TOPICS.search(headline):
                continue
            seen.add(key)
            ts = dt.datetime.strptime(row[1], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
            rank = next((k for k, d in enumerate(GKG_PREFERRED) if domain == d or domain.endswith("." + d)), len(GKG_PREFERRED))
            found.append((rank, ts, {"headline": headline, "body": headline, "source": "gdelt-gkg:" + domain,
                                     "url": row[4], "published_at": ts, "received_at": ts}))
    return [f[2] for f in sorted(found, key=lambda f: f[:2])[:per_day]]


def read_json(path: str) -> list[dict]:
    """Articles of a news.json file of the former Python project ({"articles": [{headline, body, feed, publish_ts,
    received_ts, url}]}), to import them without downloading them again."""
    data = json.loads(open(path).read())
    return [{"headline": a["headline"], "body": a.get("body"), "source": a.get("feed") or a["source"], "url": a.get("url"),
             "published_at": a["publish_ts"], "received_at": a["received_ts"]}
            for a in (data["articles"] if isinstance(data, dict) else data)]
