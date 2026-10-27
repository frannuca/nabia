"""Step 3 of the staged sentiment pipeline: the LLM's triage and node scores of every article.

    .venv/bin/python scripts/pipeline/score_news_3.py configs/real.json

Reads data.news_file; with llm.provider routes each article to the factor-tree nodes it concerns and scores its
sentiment per node, writing data.scores_file. Responses are cached in llm.cache, so an interrupted run resumes.

The steps: collect_news_1 -> digest_news_2 -> score_news_3 -> calibrate_betas_4 -> backtest_5 (nabia/sentiment_rb/stages.py).
"""

import sys
from pathlib import Path

from nabia.sentiment_rb import stages
from nabia.sentiment_rb.config import load_config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: score_news_3.py CONFIG.json")
    stages.score(load_config(Path(sys.argv[1])))
