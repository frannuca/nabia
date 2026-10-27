#!/usr/bin/env bash
# Runs the staged sentiment pipeline from one configuration file, step after step, stopping at the first failure:
#   1 collect_news_1      articles (RSS, GDELT GKG history)         -> data.news_file
#   2 digest_news_2       LLM daily macro digests                   -> data.digests_file
#   3 score_news_3        LLM triage and node scores of every article -> data.scores_file
#   4 calibrate_betas_4   elastic-net betas per refit and asset      -> data.betas_file
#   5 backtest_5          portfolio with the stored betas, against ERC -> backtest.out
# Each step's output is also written to <logs>/<step>.log. Steps 2 and 3 cache every LLM response (llm.cache), so a
# run stopped there resumes where it was when started again.
#
#   nabia_python/scripts/pipeline/run_pipeline.sh                          # configs/real.json, steps 1 to 5
#   nabia_python/scripts/pipeline/run_pipeline.sh configs/my_run.json
#   nabia_python/scripts/pipeline/run_pipeline.sh configs/real.json 4 5    # only steps 4 and 5 (from, to)
#   LOGS=output/my_logs nabia_python/scripts/pipeline/run_pipeline.sh
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project"  # the configuration's relative paths (data/..., output/...) are relative to nabia_python
config="${1:-configs/real.json}"
from="${2:-1}"
to="${3:-${2:-5}}"
[[ $# -eq 2 ]] && to=5  # one step number: from there to the end
py="$project/.venv/bin/python"
logs="${LOGS:-output/pipeline_logs}"
steps=(collect_news_1 digest_news_2 score_news_3 calibrate_betas_4 backtest_5)

[[ -x "$py" ]] || { echo "no $py: run scripts/setup_env.sh first" >&2; exit 1; }
[[ -f "$config" ]] || { echo "no configuration $config" >&2; exit 1; }
[[ "$from" =~ ^[1-5]$ && "$to" =~ ^[1-5]$ && "$from" -le "$to" ]] || {
    echo "steps: from and to in 1..5, from <= to (got $from $to)" >&2; exit 1; }

# The LLM steps need a running Ollama server when the configuration uses it.
if [[ "$from" -le 3 && "$to" -ge 2 ]]; then
    read -r provider host < <("$py" -c 'import sys
from pathlib import Path
from nabia.sentiment_rb.config import load_config
L = load_config(Path(sys.argv[1]))["llm"]
print(L["provider"], L["ollama_host"])' "$config")
    if [[ "$provider" == "ollama" ]] && ! curl -sf "$host/api/version" >/dev/null; then
        echo "Ollama is not answering at $host: start it (brew services start ollama) or change llm.provider" >&2
        exit 1
    fi
fi

mkdir -p "$logs"
echo "pipeline: $config, steps $from..$to, logs in $logs"
start=$SECONDS
for ((k = from; k <= to; k++)); do
    step="${steps[k - 1]}"
    echo
    echo "=== step $k: $step ($(date '+%H:%M:%S'))"
    t=$SECONDS
    "$py" -u "scripts/pipeline/$step.py" "$config" 2>&1 | tee "$logs/$step.log"
    echo "=== step $k done in $(((SECONDS - t) / 60)) min $(((SECONDS - t) % 60)) s"
done
echo
echo "pipeline done in $(((SECONDS - start) / 60)) min; logs in $logs"
