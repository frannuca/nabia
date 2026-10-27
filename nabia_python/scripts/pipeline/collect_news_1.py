"""Step 1 of the staged sentiment pipeline: collect news articles.

    .venv/bin/python scripts/pipeline/collect_news_1.py configs/real.json

Reads collect (rss, feeds, gkg, since, until, per_day, weekends, workers); appends the new articles,
deduplicated, to data.news_file. Run it again (e.g. daily) to add the latest items.

The steps: collect_news_1 -> digest_news_2 -> score_news_3 -> calibrate_betas_4 -> backtest_5 (nabia/sentiment_rb/stages.py).
"""

import sys
from pathlib import Path

from nabia.sentiment_rb import stages
from nabia.sentiment_rb.config import load_config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: collect_news_1.py CONFIG.json")
    stages.collect(load_config(Path(sys.argv[1])))
