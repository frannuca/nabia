"""nabia.news: download financial news and persist it in the JSON format the sentiment pipeline reads.

- RSS / Atom feeds (financial media, central banks): only recent items, so run the download daily; received_ts is
  the download time.
- GDELT DOC 2.0 (a global news index, with history): received_ts is GDELT's first-seen time. Its API rate-limits
  hard (one request every 5 seconds, longer blocks after bursts).
- GDELT's raw GKG files (every 15 minutes since 2015, English web articles with page titles): history without the API
  limit; a few slices per day, finance titles only, financial outlets first, received_ts = the file's timestamp.
- A JSON news store, deduplicated by URL and headline, appended in place.

    python -m nabia.news fetch                       # every RSS feed -> data/news.json
    python -m nabia.news fetch --gdelt --since 2024-01-01 --until 2024-03-31
    python -m nabia.news fetch --no-rss --gkg --since 2024-01-01 --per-day 10   # history from GDELT's raw files
    python -m nabia.news score                       # LLM scores (local Ollama) -> data/news_scores.json
    python -m nabia.news stats
"""

from .sources import FEEDS, GDELT_QUERIES, fetch_gdelt, fetch_gkg, fetch_rss
from .store import NewsStore

__all__ = ["FEEDS", "GDELT_QUERIES", "NewsStore", "fetch_gdelt", "fetch_gkg", "fetch_rss"]
