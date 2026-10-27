"""Step 2 of the staged sentiment pipeline: the LLM's daily macro digests.

    .venv/bin/python scripts/pipeline/digest_news_2.py configs/real.json

Reads data.news_file; with llm.provider (ollama, anthropic, or mock offline) writes one digest per day (risk
appetite, growth, inflation, geopolitics, credit, central banks, commodity supply) to data.digests_file. Responses
are cached in llm.cache, so an interrupted run resumes.

The steps: collect_news_1 -> digest_news_2 -> score_news_3 -> calibrate_betas_4 -> backtest_5 (nabia/sentiment_rb/stages.py).
"""

import sys
from pathlib import Path

from nabia.sentiment_rb import stages
from nabia.sentiment_rb.config import load_config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: digest_news_2.py CONFIG.json")
    stages.digest(load_config(Path(sys.argv[1])))
