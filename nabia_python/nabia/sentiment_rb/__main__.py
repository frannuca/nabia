"""Command line: python -m nabia.sentiment_rb [--live] [--config c.json] [--out dir] [--show] [--ic-sweep IC ...]"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __doc__ as overview
from . import report
from .config import load_config, merge
from .pipeline import run


def main():
    ap = argparse.ArgumentParser(prog="python -m nabia.sentiment_rb", description=overview,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="score the last live_days of news with the Anthropic API")
    ap.add_argument("--config", type=Path, help="JSON file overriding CONFIG keys")
    ap.add_argument("--out", type=Path, default=Path("output") / "sentiment_rb", help="default: output/sentiment_rb")
    ap.add_argument("--show", action="store_true", help="open the figures (they are always saved as PNG)")
    ap.add_argument("--ic-sweep", type=float, nargs="+", metavar="IC",
                    help="rerun with these synthetic news ICs (same market data) and plot alpha against news quality")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not args.ic_sweep:
        run(cfg, args.out, args.live, args.show)
        return
    rows = []
    for ic in args.ic_sweep:
        print(f"--- sentiment_ic = {ic}")
        rows.append(run(merge(load_config(args.config), {"synthetic": {"sentiment_ic": ic}}), args.out / f"ic_{ic:g}",
                        verbose=False))
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
    sweep = pd.DataFrame(rows).set_index("sentiment_ic")
    sweep.to_csv(args.out / "sweep.csv")
    print(f"\n== alpha against news quality ({cfg['budgets']['mode']}) ==\n", sweep.drop(columns="mode").round(4))
    report.plot_sweep(args.out, sweep, args.show)
    print(f"\nper-IC outputs in {args.out}/ic_*, summary in {args.out}/sweep.csv and sweep.png")


if __name__ == "__main__":
    main()
