#!/usr/bin/env bash
# Sets up webui/.venv for the web app project (the interpreter to select in PyCharm):
#   1. installs the nabia_python_bridge wheel built by the C++ project (../dist, see
#      cpp/scripts/build_wheels.sh) - reinstalled every time, so a rebuilt wheel is picked up;
#   2. installs the nabia package of ../python in editable mode (changes there are seen at once);
#   3. installs this nabia_webui package in editable mode, with the test extra.
#
#   webui/scripts/setup_env.sh                 # uses python3 to create .venv if it does not exist
#   PYTHON=python3.12 webui/scripts/setup_env.sh
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(dirname "$project")"
wheels="${NABIA_WHEEL_DIR:-$root/dist}"
venv="$project/.venv"

[[ -x "$venv/bin/python" ]] || "${PYTHON:-python3}" -m venv "$venv"
py="$venv/bin/python"
"$py" -m pip install --quiet --upgrade pip

"$py" -m pip install --force-reinstall --no-deps --no-index --find-links "$wheels" nabia_python_bridge || {
    echo "no nabia_python_bridge wheel for $("$py" -V) in $wheels: run cpp/scripts/build_wheels.sh first" >&2
    exit 1; }
"$py" -m pip install --quiet --find-links "$wheels" -e "$root/python[marketdata]" -e "$project[test]"

"$py" -c 'import nabia, nabia_webui; print("nabia", nabia.__file__); print("nabia_webui", nabia_webui.__file__)'
