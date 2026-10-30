#!/usr/bin/env bash
# Collects the news into the database, scores the new articles with the LLM, re-estimates the asset exposures
# (FRED proxies and prices, quick), then computes the risk budgets and the portfolio weights. The first run creates .venv with numpy, pandas and
# the nabia_python_bridge wheel of ../dist (cpp/scripts/build_wheels.sh or fetch_wheels.sh).
#   python/run.sh                         # config/config.json
#   python/run.sh my_config.json          # another configuration
#   PYTHON=python3.12 python/run.sh       # interpreter that creates .venv (default python3)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
config="$(cd "$(dirname "${1:-config/config.json}")" && pwd)/$(basename "${1:-config/config.json}")"
[[ -x .venv/bin/python ]] || { "${PYTHON:-python3}" -m venv .venv && .venv/bin/pip install -q --find-links ../dist -e .; }
py=.venv/bin/python

# A local model needs a running Ollama server.
read -r provider host < <("$py" -c 'import sys; from datalayer import load_config
L = load_config(sys.argv[1])["llm"]; print(L["provider"], L["ollama"]["host"])' "$config")
if [[ "$provider" == ollama ]] && ! curl -sf "$host/api/version" >/dev/null; then
    echo "Ollama is not answering at $host: start it (brew services start ollama) or change llm.provider" >&2; exit 1
fi

"$py" -m collector "$config"
"$py" -m scoring "$config"
"$py" -m exposures "$config"
"$py" -m budgets "$config"
