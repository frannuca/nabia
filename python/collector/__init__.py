"""News collector: RSS feeds and GDELT GKG history into the news database."""

from .collect import collect, collect_gkg, collect_rss

__all__ = ["collect", "collect_gkg", "collect_rss"]
