"""nabia.news offline: feed parsing, GDELT GKG files, the news store, and news scores files."""

import datetime as dt
import io
import zipfile

import pandas as pd
import pytest

from nabia.news import NewsStore, fetch_gkg, fetch_rss, sources
from nabia.sentiment_rb.data import load_news
from nabia.sentiment_rb.data.scores import load_scores, write_scores
from nabia.sentiment_rb.llm.schemas import Digest

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Oil prices surge as OPEC+ cuts output</title><link>https://x.com/a</link>
<description>&lt;p&gt;Brent jumped 4%.&lt;/p&gt;</description><pubDate>Thu, 22 Oct 2026 16:40:09 GMT</pubDate></item>
<item><title>Gold steady</title><link>https://x.com/b</link><pubDate>2026-10-22 18:00:00</pubDate></item>
</channel></rss>"""
ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>ECB holds rates</title>
<link href="https://ecb.eu/1"/><summary>Rates unchanged.</summary><updated>2026-10-24T08:00:00+02:00</updated></entry></feed>"""


def fake_urlopen(payload):
    return lambda request, timeout=None: io.BytesIO(payload)


@pytest.mark.parametrize("text, expected", [
    ("Thu, 22 Oct 2026 16:40:09 GMT", "2026-10-22 16:40:09"),
    ("Sun, 25 Oct 2026 14:00:00 -0500", "2026-10-25 19:00:00"),
    ("2026-10-24T08:00:00+02:00", "2026-10-24 06:00:00"),
    ("2026-10-25 19:24:28", "2026-10-25 19:24:28"),
])
def test_dates_are_parsed_to_utc(text, expected):
    assert sources.parse_date(text).strftime("%Y-%m-%d %H:%M:%S") == expected


def test_rss_and_atom_items(monkeypatch):
    received = dt.datetime(2026, 10, 25, 12, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen(RSS))
    items = fetch_rss("Feed", "https://x.com/rss", "paper", received)
    assert [a["headline"] for a in items] == ["Oil prices surge as OPEC+ cuts output", "Gold steady"]
    assert items[0]["body"] == "Brent jumped 4%." and items[0]["received_ts"] == received and items[0]["source"] == "paper"
    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen(ATOM))
    [entry] = fetch_rss("ECB", "https://ecb.eu/rss", "wire", received)
    assert entry["url"] == "https://ecb.eu/1" and entry["publish_ts"].hour == 6


def test_gkg_keeps_finance_titles_preferred_outlets_first(monkeypatch):
    def row(domain, title):
        cells = [""] * 27
        cells[1], cells[3], cells[4] = "20240102143000", domain, f"https://{domain}/{abs(hash(title))}"
        cells[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
        return "\t".join(cells)
    lines = [row("blog.example", "Stocks rally as investors cheer the new year"), row("reuters.com", "Oil prices jump on Red Sea tensions today"),
             row("news.example", "Local football team wins the championship game"), row("cnbc.com", "Treasury yields rise ahead of the jobs report")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("x.gkg.csv", "\n".join(lines))
    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen(buffer.getvalue()))
    articles = fetch_gkg(dt.date(2024, 1, 2), slices=("143000",), per_day=2)
    assert [a["feed"] for a in articles] == ["gdelt-gkg:reuters.com", "gdelt-gkg:cnbc.com"]
    assert articles[0]["received_ts"] == dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc)


def test_store_appends_without_duplicates_and_the_pipeline_reads_it(tmp_path):
    t = dt.datetime(2026, 10, 25, 12, tzinfo=dt.timezone.utc)
    a = {"publish_ts": t, "received_ts": t, "source": "paper", "feed": "F", "headline": "Oil prices surge", "body": "b", "url": "u1"}
    store = NewsStore(tmp_path / "news.json")
    assert store.add([a, {**a, "url": "u2"}, {**a, "headline": "Gold steady", "url": "u1"}]) == 1  # same headline, same URL
    assert store.add([{**a, "url": "u3", "received_ts": t + dt.timedelta(days=1)}]) == 1  # same headline another day
    store.save()
    again = NewsStore(tmp_path / "news.json")
    assert len(again.articles) == 2 and again.add([a]) == 0
    news = load_news(tmp_path / "news.json")
    assert list(news.headline) == ["Oil prices surge", "Oil prices surge"] and news.received_ts.dt.tz is None


def test_scores_file_round_trip_maps_to_trading_days(tmp_path):
    details = {0: {"id": 0, "received_ts": "2024-01-06 15:00:00", "headline": "h", "relevant": True,  # a Saturday
                   "scores": [{"node_id": "L3.WTI", "sentiment": 0.5, "weight": 0.6}]}}
    digest = Digest(risk_appetite=0.2, growth_tone=0.1, inflation_tone=0.0, geopolitical_stress=0.3, credit_tone=0.0,
                    cb_tone={"FED": None}, commodity_supply_tone={"energy": 0.4}, item_count_used=1, key_drivers=["x"])
    write_scores(tmp_path / "s.json", details, {pd.Timestamp("2024-01-05"): digest}, {"model": "m"})
    dates = pd.bdate_range("2024-01-01", periods=10)
    records, digests = load_scores(tmp_path / "s.json", dates)
    assert records.day.iloc[0] == pd.Timestamp("2024-01-08") and records.weight.iloc[0] == 0.6  # next trading day
    assert digests[pd.Timestamp("2024-01-05")].commodity_supply_tone["energy"] == 0.4
