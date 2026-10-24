#!/usr/bin/env bash
# Builds doc/nabia_python_reference.pdf from doc/nabia_python_reference.tex (needs pdflatex):
#   1. extract_snippets.py writes the code excerpts and their source locations from the current
#      sources into doc/build/snippets, so the PDF always shows the code that runs;
#   2. pdflatex runs three times (labels, table of contents, final cross-references);
#   3. the PDF is copied next to its source. Warnings are printed from the LaTeX log.
#
#   python/doc/build.sh
set -euo pipefail

doc="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(dirname "$doc")"
name=nabia_python_reference
out="$doc/build"

mkdir -p "$out"
python3 "$doc/extract_snippets.py" "$project" "$out/snippets"
for pass in 1 2 3; do
    TEXINPUTS="$doc/common:$out/snippets:" pdflatex -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$out" "$doc/$name.tex" > "$out/$name.pass$pass.txt" || {
        tail -30 "$out/$name.log" >&2; exit 1; }
done
cp "$out/$name.pdf" "$doc/$name.pdf"
grep -E "Overfull|undefined|Warning" "$out/$name.log" || true
echo "$doc/$name.pdf"
