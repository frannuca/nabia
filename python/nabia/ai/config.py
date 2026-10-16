"""Configuration shared by the agents, the MCP servers and the pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SYMBOLS = ["^GSPC", "^STOXX50E", "^N225", "ZN=F", "ZB=F", "BZ=F", "GC=F", "HG=F", "BTC/USD"]

# Public RSS feeds the news analyzer reads (works with any model; the only
# news source for local models, which have no web search).
DEFAULT_NEWS_FEEDS = {
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "Financial Times": "https://www.ft.com/markets?format=rss",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "OilPrice": "https://oilprice.com/rss/main",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "Bank of England": "https://www.bankofengland.co.uk/rss/news",
}

DEFAULT_PORTFOLIO = {
    "covariance_window": 120,
    "covariance_estimator": "exponential",
    "ewma_halflife": 40,
    "rebalance_frequency": 21,
    "risk_measure": "volatility",
    "target_volatility": 0.10,
    "max_leverage": 3.0,
    "transaction_cost": 0.0005,
    "initial_capital": 1_000_000,
}


@dataclass
class AIConfig:
    """Everything the agents need to know. Saved as JSON so the MCP server
    processes (started by the agents) read the same settings."""

    db_url: str = "sqlite:///market.db"
    """Database with market data (nabia.marketdata), news, stats reports and decisions."""
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    """Portfolio universe: one risk budget per symbol."""
    portfolio: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PORTFOLIO))
    """nabia.backtest parameters (without budgets, which the execution agent sets)."""
    min_budget: float = 0.02
    """Lower bound on each normalized risk budget, enforced by set_risk_budgets."""
    max_budget: float = 0.40
    """Upper bound on each normalized risk budget, enforced by set_risk_budgets."""
    provider: str = "anthropic"
    """Model provider: "anthropic" (Claude API) or "ollama" (a local model)."""
    model: str = "claude-opus-5-5"
    """Claude model, when provider is "anthropic"."""
    effort: dict[str, str] = field(default_factory=lambda: {"news": "medium", "stats": "high", "execution": "high"})
    """Claude effort level per agent (news, stats, execution)."""
    ollama_model: str = "qwen3.8:latest"
    """Local model, when provider is "ollama" (it must support tool calling: `ollama show <model>`)."""
    ollama_host: str = "http://localhost:11434"
    """URL of the Ollama server."""
    ollama_options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.2, "num_ctx": 32768})
    """Ollama model options; num_ctx must hold the prompts, tool schemas and tool results."""
    ollama_think: bool = False
    """Let thinking models reason before answering (slower, sometimes better tool use)."""
    news_feeds: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_NEWS_FEEDS))
    """RSS / Atom feeds read by fetch_news_feeds, by source name."""
    news_max_searches: int = 10
    """Cap on web searches per news collection run."""
    news_lookback_days: int = 7
    """Window of stored news the execution agent reviews."""
    stats_history_days: int = 400
    """Calendar days of prices loaded by the stats tools (covers 1y signals)."""

    @property
    def model_label(self) -> str:
        """The model in use, e.g. "claude-opus-5-5" or "ollama:qwen3.8:latest"."""
        return self.model if self.provider == "anthropic" else f"ollama:{self.ollama_model}"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> AIConfig:
        data = json.loads(Path(path).read_text())
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def validate(self) -> None:
        if len(self.symbols) < 2 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must list at least two distinct assets")
        if not 0 < self.min_budget <= 1 / len(self.symbols) <= self.max_budget <= 1:
            raise ValueError("need 0 < min_budget <= 1/len(symbols) <= max_budget <= 1")
        if self.provider not in ("anthropic", "ollama"):
            raise ValueError('provider must be "anthropic" or "ollama"')
        if "budgets" in self.portfolio:
            raise ValueError("portfolio must not set budgets: the execution agent decides them")
