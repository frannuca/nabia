#!/usr/bin/env bash
# Builds doc/nabia.pdf, the single nabia document (needs pdflatex):
#   1. --figures: reruns the sentiment pipeline (long-only, long-short, both news-quality sweeps) with the Python
#      project's .venv, and copies its figures and result tables into doc/figures;
#   2. extract_snippets.py writes the code excerpts, their file:line locations and the packaging file pieces from
#      the current sources into doc/build/snippets, so the PDF shows the code that runs;
#   3. pdflatex runs three times (labels, table of contents, final references); the PDF is copied to doc/.
#
#   doc/build.sh               # the document, with the figures already in doc/figures
#   doc/build.sh --figures     # regenerate the pipeline's figures and tables first (a few minutes)
set -euo pipefail

doc="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$doc")"
out="$doc/build"
mkdir -p "$out" "$doc/figures"

if [[ "${1:-}" == "--figures" ]]; then
    py="$root/nabia_python/.venv/bin/python"
    [[ -x "$py" ]] || { echo "run nabia_python/scripts/setup_env.sh first" >&2; exit 1; }
    ls_cfg="$out/long_short.json"
    echo '{"budgets": {"mode": "long_short"}}' > "$ls_cfg"
    (cd "$root/nabia_python"
     "$py" -m nabia.sentiment_rb --out output/sentiment_rb
     "$py" -m nabia.sentiment_rb --config "$ls_cfg" --out output/sentiment_rb_long_short
     "$py" -m nabia.sentiment_rb --ic-sweep 0 0.05 0.1 0.2 0.3 0.5 --out output/sentiment_rb_sweep
     "$py" -m nabia.sentiment_rb --config "$ls_cfg" --ic-sweep 0 0.05 0.1 0.2 0.3 0.5 --out output/sentiment_rb_long_short_sweep)
fi

# figures and tables of the last runs (kept in doc/figures, so the PDF builds without rerunning)
o="$root/nabia_python/output"
copy() { [[ -f "$1" ]] && cp "$1" "$doc/figures/$2" || true; }
copy "$o/sentiment_rb/portfolio.png" lo_portfolio.png
copy "$o/sentiment_rb/budgets.png" lo_budgets.png
copy "$o/sentiment_rb/model.png" lo_model.png
copy "$o/sentiment_rb_long_short/budgets.png" ls_budgets.png
copy "$o/sentiment_rb_long_short/portfolio.png" ls_portfolio.png
copy "$o/sentiment_rb_sweep/sweep.png" sweep_long_only.png
copy "$o/sentiment_rb_long_short_sweep/sweep.png" sweep_long_short.png
python3 - "$o" "$doc/figures" <<'PY'
import csv, sys
from pathlib import Path
o, figures = Path(sys.argv[1]), Path(sys.argv[2])
def table(src, dst, header, rows):
    if src.exists():
        dst.write_text("\\begin{tabular}{@{}" + "l" + "r" * (len(header) - 1) + "@{}}\n\\toprule\n" + " & ".join(header) +
                       " \\\\\n\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
for name in ["long_only", "long_short"]:
    src = o / ("sentiment_rb" if name == "long_only" else "sentiment_rb_long_short") / "comparison.csv"
    if src.exists():
        r = list(csv.reader(src.open()))
        table(src, figures / f"comparison_{name}.tex", ["metric", "ERC ($b^0$)", "sentiment overlay"],
              [[x[0].replace("_", " "), f"{float(x[1]):.4f}", f"{float(x[2]):.4f}"] for x in r[1:]])
    src = o / f"sentiment_rb{'' if name == 'long_only' else '_long_short'}_sweep" / "sweep.csv"
    if src.exists():
        r = list(csv.DictReader(src.open()))
        table(src, figures / f"sweep_{name}.tex", ["news IC", "ERC Sharpe", "overlay Sharpe", "excess return", "IC of $S$", "IC of LLM level"],
              [[x["sentiment_ic"], f"{float(x['erc_sharpe']):.2f}", f"{float(x['overlay_sharpe']):.2f}",
                f"{100 * float(x['excess_return']):+.2f}\\%", f"{float(x['ic_S']):.2f}", f"{float(x['ic_llm_level']):.2f}"] for x in r])
PY

python3 "$doc/extract_snippets.py" "$root" "$out/snippets"
for pass in 1 2 3; do
    (cd "$doc" && TEXINPUTS="$doc/common:$out/snippets:" pdflatex -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$out" nabia.tex > "$out/nabia.pass$pass.txt") || {
        grep -m5 -A3 "^!\|:[0-9]*:" "$out/nabia.log" >&2; exit 1; }
done
cp "$out/nabia.pdf" "$doc/nabia.pdf"
grep -E "Overfull|undefined|Warning: Reference|Warning: Citation" "$out/nabia.log" | sort | uniq -c | sort -rn | head -20 || true
echo "$doc/nabia.pdf ($(grep -o 'Output written on .*(\([0-9]*\) pages' "$out/nabia.log" | grep -o '[0-9]* pages'))"
