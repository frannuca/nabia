#!/usr/bin/env bash
# Builds doc/news_risk_budgeting.pdf: figures and tables from the database (figures.py, needs matplotlib in ../.venv),
# then pdflatex through latexmk.
#   python/doc/build.sh                 # config/config.json
#   python/doc/build.sh my_config.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
../.venv/bin/python figures.py ${1:+"$1"}
latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=build news_risk_budgeting.tex >build/latexmk.log 2>&1 \
    || { tail -30 build/latexmk.log; exit 1; }
cp build/news_risk_budgeting.pdf .
echo "doc/news_risk_budgeting.pdf"
