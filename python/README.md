# nabia (Python)

Python interface to the nabia C++ core, plus market data and AI agents for a
systematic cross-asset trend portfolio.

| Module | What it does |
|---|---|
| `nabia.backtest`, `nabia.latest_weights` | Risk budgeting backtests and today's target weights on pandas data (C++ engine) |
| `nabia.marketdata` | Downloads equity indices, bond futures, commodities (Yahoo Finance) and crypto (exchanges through ccxt) into SQLite / PostgreSQL |
| `nabia.ai` | Three Claude agents (news, statistics, execution) that set the risk budgets, with MCP servers for their tools, and a backtest mode |
| `nabia.web` | Local web app to manage it all: news, executions, backtests and jobs |

## Install

From the repository root:

```sh
python3 -m venv python/.venv
python/.venv/bin/pip install "./python[marketdata,ai,web,test]"   # builds the C++ extension (CMake + a C++20 compiler)
```

For development, build the extension in place and use the sources directly:

```sh
python/.venv/bin/pip install numpy pandas pybind11 sqlalchemy yfinance ccxt "anthropic[mcp]" mcp fastapi uvicorn pytest httpx
cmake --preset python && cmake --build --preset python    # writes python/nabia/_nabia*.so
PYTHONPATH=python python/.venv/bin/python -m pytest python/tests
```

## Backtests from pandas

```python
import nabia

result = nabia.backtest(prices, {           # prices: DataFrame, DatetimeIndex x assets; NaN dates are dropped
    "covariance_window": 120, "rebalance_frequency": 21,
    "budgets": {"^GSPC": 0.4, "ZN=F": 0.4, "GC=F": 0.2},
    "target_volatility": 0.10, "transaction_cost": 0.0005, "initial_capital": 1_000_000,
})
result.weights          # weights held each day
result.performance      # gross/net return, costs, turnover, pnl, equity, drawdown
result.metrics          # Sharpe, max drawdown, turnover, ...

nabia.latest_weights(prices, params).weights    # the position to hold from the last date on
nabia.backtest(prices, params, budget_schedule=budgets_by_date)   # time-varying risk budgets
```

Parameters use the same keys as the `nabia_rb_backtest` command line tool
(`apps/rb_backtest/README.md`).

## Market data

```sh
python -m nabia.marketdata --list                                  # the default universe
python -m nabia.marketdata --db sqlite:///market.db --start 2010-01-01
python -m nabia.marketdata --db sqlite:///market.db --asset-class crypto --crypto-exchange coinbase
```

Downloads are incremental: re-running fetches only recent days (with a 5-day
overlap that picks up revisions). Each asset records its source (`yahoo` or
`crypto`). Moving to PostgreSQL later only changes the URL
(`postgresql+psycopg://user:pass@host/db`, after `pip install "./python[postgres]"`).

Caveats:
- Yahoo futures tickers are front-month continuous series. They jump at
  contract rolls, and WTI (`CL=F`) went negative in April 2020, where returns are undefined.
- Crypto trades every day. Aligned with exchange-traded assets, its weekend
  prices are dropped, and returns span the gap.

## AI agents

| Agent | Tools (MCP server) | Output |
|---|---|---|
| News analyzer | web search and fetch (Claude server tools), `nabia.ai.servers.news` | Articles stored per date with topic, asset classes, symbols, sentiment and relevance |
| Stats analyzer | `nabia.ai.servers.stats`: correlations, rolling volatilities, trend signals, market regime | A structured report with a suggested budget tilt per asset |
| Execution agent | news and stats (read-only), `nabia.ai.servers.portfolio` | Risk budgets with a rationale (normalized and bounded), and the target weights from the C++ engine |

By default they run on Claude (`claude-opus-5-5`, with adaptive thinking and
server-side refusal fallbacks). Authentication: `ANTHROPIC_API_KEY`, or a profile
from `ant auth login`. They can also run on a local model through Ollama, which is
free and keeps all data on your machine (see below).

```sh
python -m nabia.ai init  --config ai_config.json --db sqlite:///market.db   # then edit symbols, bounds, settings
python -m nabia.ai news  --config ai_config.json                            # collect today's news
python -m nabia.ai daily --config ai_config.json                            # news -> stats -> budgets and weights
```

### Local models with Ollama

Set the provider in `ai_config.json`:

```json
"provider": "ollama",
"ollama_model": "qwen3.8:latest",
"ollama_host": "http://localhost:11434",
"ollama_options": {"temperature": 0.2, "num_ctx": 32768},
"ollama_think": false
```

- **Model requirements:** the model must support tool calling. `ollama show <model>` lists
  `tools` under its capabilities. A model of about 27B parameters (qwen3.8, gemma4:26b,
  glm-4.7-flash) handles the multi-step tool use well. Small 7-8B models often skip
  steps or send malformed arguments; the tools reject those with an explanation, but
  the agent may not recover.
- **Context window:** `num_ctx` must hold the instructions, the tool definitions and the tool results.
  32k is enough for the default universe.
- **What is different:** a local model has no web search, so the news analyzer reads public RSS
  feeds (markets, economy, central banks, energy, crypto; edit `news_feeds` in the
  config) through the news server's `fetch_news_feeds` tool. Claude uses those feeds too, plus web
  search. Thinking effort, prompt caching and refusal fallbacks are Claude-only and are ignored.
- **Speed:** expect minutes per agent run on a laptop. A backtest runs the stats and
  execution agents at every decision date.
- **Model attribution:** decisions record which model made them (`agent:ollama:qwen3.8:latest`), so
  runs of different models can be compared in the web app.

### Backtest mode

```sh
python -m nabia.ai backtest --config ai_config.json --start 2023-01-01 --end 2024-12-31 \
    --run agents-2024 --policy agents --frequency ME --output agents-2024.csv
python -m nabia.ai backtest ... --run trend-rule --policy trend_rule        # rule-based baseline, no API calls
```

At each decision date (here every month end), the stats and execution agents
run as of that date. Every tool returns only data dated on or before it, news
comes only from the database (web access is off), and the prompt tells the
model to treat that date as today. The decisions become a budget schedule for
one continuous backtest, which is compared with equal risk budgets. Decisions
are stored per run, so an interrupted backtest resumes where it stopped.

**Cost:** each decision date costs two agent runs, billed. Start with a short
period or a quarterly frequency (`--frequency QE`).

**Look-ahead:** the tools can't see the future, but the model itself was trained
on data up to its knowledge cutoff. Over periods it knows about, it may
recognize the market history, so treat backtest results inside that window as
optimistic. They are fully out-of-sample only after the cutoff.

### Web app

```sh
python -m nabia.web --config ai_config.json        # http://127.0.0.1:8000 (local only by default)
```

| Page | Contents |
|---|---|
| Overview | Latest live decision (budgets, weights, rationale), market data coverage with stale-data flags, recent jobs, buttons for the common jobs |
| News | Collected articles with filters (dates, topic, asset class, symbol, relevance, text), articles per day, sentiment by topic, article detail |
| Executions | Per run: risk budgets over time and every decision, with its budgets vs weights, rationale, stats report, and the agent runs behind it (each tool call's input and output, token usage, errors) |
| Backtests | Starts backtests (trend rule or agents) and compares each run with equal risk: equity, drawdown, metrics, decided budgets, weekly weights |
| Jobs | Market data downloads, news collection and daily decisions, with status, progress and errors |

Jobs run one at a time in the background, and their status is stored in the
database. Install with `pip install "./python[web]"`.

### Using the MCP servers from Claude Code or other clients

Each server runs over stdio and works with any MCP client. For Claude Code, in `.mcp.json`:

```json
{
  "mcpServers": {
    "nabia-stats": {
      "command": "/path/to/nabia/python/.venv/bin/python",
      "args": ["-m", "nabia.ai.servers.stats", "--config", "/path/to/ai_config.json", "--read-only"],
      "env": {"PYTHONPATH": "/path/to/nabia/python"}
    },
    "nabia-news": {
      "command": "/path/to/nabia/python/.venv/bin/python",
      "args": ["-m", "nabia.ai.servers.news", "--config", "/path/to/ai_config.json"],
      "env": {"PYTHONPATH": "/path/to/nabia/python"}
    }
  }
}
```

`--read-only` drops the tools that write. `--run <name>` selects a backtest's
reports and decisions (`live` by default).
