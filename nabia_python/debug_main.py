"""Runs the sentiment pipeline's command line (python -m nabia.sentiment_rb) from a script, e.g. to debug it.

Arguments given on the command line are used as they are. Without any, the run uses DEBUG_CONFIG (written to
<out>/debug_config.json and passed with --config) and DEBUG_OUT.
"""

import json
import sys
from pathlib import Path

from nabia.sentiment_rb.__main__ import main

DEBUG_OUT = Path("output/debug")
DEBUG_CONFIG = {
    "budgets": {"mode": "long_short", "short_below": 0.3, "long_above": 0.7, "modulate_long_short": True},
    "synthetic": {"sentiment_ic": 0.3},
    "model": {"threads": 1},  # fit the assets one after the other: no worker threads under the debugger
}

if __name__ == "__main__":
    if len(sys.argv) == 1:
        DEBUG_OUT.mkdir(parents=True, exist_ok=True)
        config = DEBUG_OUT / "debug_config.json"
        config.write_text(json.dumps(DEBUG_CONFIG, indent=2))
        sys.argv += ["--config", str(config), "--out", str(DEBUG_OUT)]
    main()
