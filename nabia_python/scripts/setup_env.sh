#!/usr/bin/env bash
# Sets up nabia_python/.venv for the Python project (the interpreter to select in PyCharm):
#   1. installs the nabia_python_bridge wheel built by the C++ project (../dist, see
#      cpp/scripts/build_wheels.sh) - reinstalled every time, so a rebuilt wheel is picked up;
#   2. installs this nabia package in editable mode, with the market data and test extras (add llm for --live).
#
#   nabia_python/scripts/setup_env.sh                 # uses python3 to create .venv if it does not exist
#   PYTHON=python3.12 nabia_python/scripts/setup_env.sh
#   NABIA_WHEEL_DIR=/path/to/wheels nabia_python/scripts/setup_env.sh
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wheels="${NABIA_WHEEL_DIR:-$(dirname "$project")/dist}"
venv="$project/.venv"

[[ -x "$venv/bin/python" ]] || "${PYTHON:-python3}" -m venv "$venv"
py="$venv/bin/python"
"$py" -m pip install --quiet --upgrade pip

# pip picks the wheel matching this interpreter and platform among those in $wheels.
"$py" -m pip install --force-reinstall --no-deps --no-index --find-links "$wheels" nabia_python_bridge || {
    echo "no nabia_python_bridge wheel for $("$py" -V) in $wheels: run cpp/scripts/build_wheels.sh first" >&2
    exit 1; }
"$py" -m pip install --quiet --find-links "$wheels" -e "$project[marketdata,test]"

"$py" -c 'import nabia, nabia_python_bridge as nb; print("nabia", nabia.__file__); print("nabia_python_bridge", nb.__version__, nb.__file__)'
