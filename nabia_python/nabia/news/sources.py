"""News sources: RSS / Atom feeds and the GDELT DOC 2.0 API. Each returns articles as dicts with publish_ts and
received_ts (UTC datetimes), source (wire, paper or blog: the weight class of the pipeline), feed, headline, body, url.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import csv
import io
import zipfile

#: name: (url, weight class). Central banks are official statements (wire); media are paper; OilPrice a blog.
FEEDS = {
    "MarketWatch": ("https://feeds.content.dowjones.io/public/rss/mw_topstories", "paper"),
    "CNBC": ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "paper"),
    "CNBC Economy": ("https://www.cnbc.com/id/20910258/device/rss/rss.html", "paper"),
    "Financial Times": ("https://www.ft.com/markets?format=rss", "paper"),
    "Investing.com": ("https://www.investing.com/rss/news.rss", "paper"),
    "OilPrice": ("https://oilprice.com/rss/main", "blog"),
    "Federal Reserve": ("https://www.federalreserve.gov/feeds/press_all.xml", "wire"),
    "ECB": ("https://www.ecb.europa.eu/rss/press.html", "wire"),
    "Bank of England": ("https://www.bankofengland.co.uk/rss/news", "wire"),
}

#: GDELT queries covering the portfolio's asset classes (English sources).
GDELT_QUERIES = [
    '("stock market" OR equities OR "wall street" OR "s&p 500")',
    '("oil prices" OR crude OR brent OR opec)',
    '("gold prices" OR bullion)',
    '("treasury yields" OR "bond market" OR "federal reserve" OR "central bank")',
    '("high yield" OR "junk bonds" OR "credit spreads")',
    '("european stocks" OR "euro stoxx" OR "ecb")',
]

USER_AGENT = {"User-Agent": "Mozilla/5.0 (nabia news downloader)"}
ATOM = "{http://www.w3.org/2005/Atom}"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse_date(text: str | None) -> dt.datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom, some feeds); naive times are taken as UTC."""
    if not text:
        return None
    text = text.strip()
    try:
        d = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return (d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)).astimezone(dt.timezone.utc)


def clean(text: str | None) -> str:
    """Plain text: tags removed, entities decoded, spaces collapsed."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def fetch_rss(name: str, url: str, weight_class: str, received: dt.datetime | None = None, timeout: float = 30):
    """The items of an RSS or Atom feed; received_ts is the download time (the feed carries no history)."""
    received = received or now_utc()
    root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=timeout).read())
    articles = []
    for item in root.iter():
        if item.tag == "item":  # RSS 2.0
            title, link, desc, date = (item.findtext("title"), item.findtext("link"), item.findtext("description"),
                                       item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date"))
        elif item.tag == ATOM + "entry":
            link_el = item.find(ATOM + "link")
            title, desc = item.findtext(ATOM + "title"), item.findtext(ATOM + "summary") or item.findtext(ATOM + "content")
            link = link_el.get("href") if link_el is not None else None
            date = item.findtext(ATOM + "published") or item.findtext(ATOM + "updated")
        else:
            continue
        headline = clean(title)
        if not headline:
            continue
        published = parse_date(date) or received
        articles.append({"publish_ts": min(published, received), "received_ts": received, "source": weight_class,
                         "feed": name, "headline": headline, "body": clean(desc) or headline, "url": (link or "").strip()})
    return articles


def _get_json(url: str, retries: int = 5, wait: float = 10.0, timeout: float = 60):
    """GET a JSON document, backing off on HTTP 429 / 5xx (GDELT asks for at most one request every 5 seconds)."""
    for attempt in range(retries):
        try:
            body = urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=timeout).read()
            return json.loads(body) if body.strip().startswith(b"{") else {}
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500 or attempt == retries - 1:
                raise
            time.sleep(wait * 2 ** attempt)
    return {}


def fetch_gdelt(query: str, start: dt.datetime, end: dt.datetime, max_records: int = 250, pause: float = 6.0):
    """GDELT DOC 2.0 article list for an English-language query between start and end (UTC), queried day by day.
    publish_ts = received_ts = GDELT's first-seen time, when the article became publicly indexed."""
    articles, day = [], start
    while day < end:
        until = min(day + dt.timedelta(days=1), end)
        params = urllib.parse.urlencode({"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
                                         "maxrecords": max_records, "sort": "datedesc",
                                         "startdatetime": day.strftime("%Y%m%d%H%M%S"),
                                         "enddatetime": until.strftime("%Y%m%d%H%M%S")})
        for a in _get_json("https://api.gdeltproject.org/api/v2/doc/doc?" + params).get("articles", []):
            seen = dt.datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
            headline = clean(a.get("title"))
            if headline:
                articles.append({"publish_ts": seen, "received_ts": seen, "source": "paper",
                                 "feed": "gdelt:" + a.get("domain", ""), "headline": headline, "body": headline,
                                 "url": a.get("url", "")})
        day = until
        if day < end:
            time.sleep(pause)
    return articles


#: Finance topics of the portfolio, matched in GKG page titles (English).
GKG_TOPICS = re.compile(
    r"\b(stocks?|stock market|equit(y|ies)|wall street|s&p|nasdaq|dow jones|euro stoxx|ftse|dax|oil|crude|brent|wti|opec|"
    r"gold|bullion|precious metals?|treasur(y|ies)|bonds?|yields?|federal reserve|fed|ecb|central bank|rate (cut|hike)s?|"
    r"interest rates?|inflation|cpi|recession|gdp|jobs report|payrolls|junk bonds?|high.yield|credit spreads?|"
    r"commodit(y|ies)|energy prices?|markets?)\b", re.I)
#: Financial outlets ranked first when a day has more matching titles than it keeps.
GKG_PREFERRED = ("reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "ft.com", "wsj.com", "apnews.com",
                 "investing.com", "finance.yahoo.com", "barrons.com", "fxstreet.com", "oilprice.com", "kitco.com",
                 "economist.com", "businessinsider.com", "forbes.com", "fortune.com", "morningstar.com", "nasdaq.com",
                 "seekingalpha.com", "thestreet.com", "proactiveinvestors.co.uk", "theguardian.com", "bbc.co.uk", "nytimes.com")


def fetch_gkg(day: dt.date, slices=("143000", "203000"), per_day: int = 10, timeout: float = 180):
    """Historical news from GDELT's raw GKG files (every 15 minutes, English web articles with their page titles).
    Reads the given 15-minute slices of the day (UTC), keeps the titles about the portfolio's topics, ranks the
    preferred financial outlets first and returns at most per_day articles. publish_ts = received_ts = the GKG
    timestamp, when GDELT processed the article (point in time)."""
    found, seen = [], set()
    for hhmmss in slices:
        url = f"https://data.gdeltproject.org/gdeltv2/{day:%Y%m%d}{hhmmss}.gkg.csv.zip"
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=timeout).read()
        except urllib.error.HTTPError as error:
            if error.code == 404:  # a missing slice (GDELT outage)
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
        csv.field_size_limit(1 << 30)
        for row in csv.reader(io.StringIO(text), delimiter="\t"):
            if len(row) < 27 or not (m := re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", row[26])):
                continue
            headline = clean(m.group(1))
            key = " ".join(re.findall(r"[a-z0-9]+", headline.lower()))
            if len(headline) < 25 or key in seen or not GKG_TOPICS.search(headline):
                continue
            seen.add(key)
            ts = dt.datetime.strptime(row[1], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
            domain = row[3].lower()
            rank = next((k for k, d in enumerate(GKG_PREFERRED) if domain == d or domain.endswith("." + d)), len(GKG_PREFERRED))
            found.append((rank, ts, {"publish_ts": ts, "received_ts": ts, "source": "paper", "feed": "gdelt-gkg:" + domain,
                                     "headline": headline, "body": headline, "url": row[4]}))
    found.sort(key=lambda f: (f[0], f[1]))
    return [f[2] for f in found[:per_day]]
