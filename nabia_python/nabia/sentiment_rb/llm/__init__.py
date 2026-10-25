"""LLM news pipeline: schemas, client (mock or Anthropic), dedup, scoring, node features."""

from .client import LLM
from .news import node_features, run_news

__all__ = ["LLM", "node_features", "run_news"]
