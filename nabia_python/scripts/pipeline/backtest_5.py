"""Step 5 of the staged sentiment pipeline: backtest the portfolio with the stored betas.

    .venv/bin/python scripts/pipeline/backtest_5.py configs/real.json

Reads the market data, the scores and digests files and data.betas_file; the betas map the aggregated sentiment
to each asset's score S and its normalized s; budgets and directions follow (budgets section), and the portfolio
is backtested against plain ERC. Writes the report (tables, figures) and the positions over the lookback window
(backtest: lookback_days, as_of, out).

The steps: collect_news_1 -> digest_news_2 -> score_news_3 -> calibrate_betas_4 -> backtest_5 (nabia/sentiment_rb/stages.py).
"""

import sys
from pathlib import Path

from nabia.sentiment_rb import stages
from nabia.sentiment_rb.config import load_config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: backtest_5.py CONFIG.json")
    stages.backtest(load_config(Path(sys.argv[1])))
