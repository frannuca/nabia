# nabia web UI

The agents, their MCP tool servers and the web application of nabia, as an independent project. Open
this folder (`webui/`) in PyCharm. Every computation goes through the `nabia` package of `../python`
(backtests and target weights from the C++ engine, analytics, market data), which runs on the
`nabia_python_bridge` wheel of `../cpp`.

| Package | What it does |
|---|---|
| `nabia_webui.ai` | Three agents (news, statistics, execution) that set the risk budgets, the pipeline (daily runs and backtests) and its command line |
| `nabia_webui.ai.servers` | The MCP servers that give the agents their tools: news, stats, portfolio |
| `nabia_webui.web` | The web app: FastAPI back end (`app.py`, `jobs.py`) and the static front end (`static/`) |

## Setup

```sh
../cpp/scripts/build_wheels.sh 3.14   # once: the C++ wheel, into ../dist (needs uv)
scripts/setup_env.sh                  # .venv: the wheel, ../python (editable) and this package (editable)
.venv/bin/python -m pytest            # 24 tests; NABIA_OLLAMA_TESTS=1 adds a local-model test
```

In PyCharm, select `webui/.venv` as the interpreter and keep `webui/` as the working directory of run
configurations: `ai_config.json` is here, and its database URL (`sqlite:///../data/market.db`, shared
with `../python`) is relative to it.

On Windows, use the PowerShell scripts: `..\cpp\scripts\build_wheels.ps1 -Versions 3.12`, then
`scripts\setup_env.ps1` (interpreter: `.venv\Scripts\python.exe`).

## Running the web app

1. **Start the model.**
   - Local model (`"provider": "ollama"`): pull the model once with `ollama pull qwen3.8:latest`, then
     keep `ollama serve` running (or the Ollama app).
   - Claude (`"provider": "anthropic"`): set `ANTHROPIC_API_KEY` instead. Usage is billed.
   - The trend rule policy needs neither.
2. **Start the web app.**
   ```sh
   .venv/bin/python -m nabia_webui.web --config ai_config.json      # or: .venv/bin/nabia-web --config ai_config.json
   ```
   Then open http://127.0.0.1:8000. Add `--port <n>` to use another port, or `--host 0.0.0.0` to expose
   it on the network (it has no authentication).
3. **First run, from the Jobs tab:** download market data, collect news, run a daily decision; or start a
   backtest from the Backtests tab. Click a job to follow its live log.

You don't start the MCP servers yourself. Each agent run launches the ones it needs as subprocesses
(`python -m nabia_webui.ai.servers.<name>`) and stops them at the end. If port 8000 is already in use,
stop the old server with `lsof -ti :8000 | xargs kill`.

## AI agents

| Agent | Tools (MCP server) | Output |
|---|---|---|
| News analyzer | web search and fetch (Claude server tools), `nabia_webui.ai.servers.news` | Articles stored per date with topic, asset classes, symbols, sentiment and relevance |
| Stats analyzer | `nabia_webui.ai.servers.stats`: correlations, rolling volatilities, trend signals, market regime | A structured report with a suggested budget tilt per asset |
| Execution agent | news and stats (read-only), `nabia_webui.ai.servers.portfolio` | Risk budgets with a rationale (normalized and bounded), and the target weights from the C++ engine |

By default they run on Claude (`claude-opus-5-5`, with adaptive thinking and
server-side refusal fallbacks). Authentication: `ANTHROPIC_API_KEY`, or a profile
from `ant auth login`. They can also run on a local model through Ollama, which is
free and keeps all data on your machine (see below).

```sh
.venv/bin/python -m nabia_webui.ai init  --config ai_config.json --db sqlite:///../data/market.db   # then edit symbols, bounds, settings
.venv/bin/python -m nabia_webui.ai news  --config ai_config.json                            # collect today's news
.venv/bin/python -m nabia_webui.ai daily --config ai_config.json                            # news -> stats -> budgets and weights
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
.venv/bin/python -m nabia_webui.ai backtest --config ai_config.json --start 2023-01-01 --end 2024-12-31 \
    --run agents-2024 --policy agents --frequency ME --output agents-2024.csv
.venv/bin/python -m nabia_webui.ai backtest ... --run trend-rule --policy trend_rule        # rule-based baseline, no API calls
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
.venv/bin/python -m nabia_webui.web --config ai_config.json   # http://127.0.0.1:8000 (local only by default)
```

| Page | Contents |
|---|---|
| Overview | Latest live decision (budgets, weights, rationale), market data coverage with stale-data flags, recent jobs, buttons for the common jobs |
| News | Collected articles with filters (dates, topic, asset class, symbol, relevance, text), articles per day, sentiment by topic, article detail |
| Executions | Per run: risk budgets over time and every decision, with its budgets vs weights, rationale, stats report, and the agent runs behind it (each tool call's input and output, token usage, errors) |
| Backtests | Starts backtests (trend rule or agents) and compares each run with equal risk: equity, drawdown, metrics, decided budgets, weekly weights |
| Jobs | Market data downloads, news collection and daily decisions, with status, progress and errors |

Jobs run one at a time in the background, and their status is stored in the
database.

### Using the MCP servers from Claude Code or other clients

Each server runs over stdio and works with any MCP client. For Claude Code, in `.mcp.json`:

```json
{
  "mcpServers": {
    "nabia-stats": {
      "command": "/path/to/nabia/webui/.venv/bin/python",
      "args": ["-m", "nabia_webui.ai.servers.stats", "--config", "/path/to/nabia/webui/ai_config.json", "--read-only"]
    },
    "nabia-news": {
      "command": "/path/to/nabia/webui/.venv/bin/python",
      "args": ["-m", "nabia_webui.ai.servers.news", "--config", "/path/to/nabia/webui/ai_config.json"]
    }
  }
}
```

A client may start the servers in another working directory, so give them a configuration whose
`db_url` is absolute (`sqlite:////path/to/nabia/data/market.db`). `--read-only` drops the tools that write. `--run <name>` selects a backtest's
reports and decisions (`live` by default).

## Documentation

[`doc/nabia_webui_reference.pdf`](doc/nabia_webui_reference.pdf): this project's reference — the agents
with their complete prompts, every MCP tool with its description and input schema, the agent loop, how
decisions become positions, news sentiment and relevance, and the web app. The prompts, tool catalogue
and configuration table are extracted from the current sources. Rebuild it with `doc/build.sh` (needs
`pdflatex`).
