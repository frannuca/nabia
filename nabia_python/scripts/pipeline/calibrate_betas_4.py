"""Step 4 of the staged sentiment pipeline: calibrate the elastic-net betas per asset.

    .venv/bin/python scripts/pipeline/calibrate_betas_4.py configs/real.json

Reads the market data (Yahoo Finance, FRED), data.scores_file and data.digests_file; builds every asset's candidate
features from its configured factors and fits the purged walk-forward elastic net at each refit date; writes the
betas per refit date and asset to data.betas_file.

The steps: collect_news_1 -> digest_news_2 -> score_news_3 -> calibrate_betas_4 -> backtest_5 (nabia/sentiment_rb/stages.py).
"""

import sys
from pathlib import Path

from nabia.sentiment_rb import stages
from nabia.sentiment_rb.config import load_config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_betas_4.py CONFIG.json")
    stages.calibrate(load_config(Path(sys.argv[1])))
