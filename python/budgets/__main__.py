"""python -m budgets [config.json] [--start D] [--end D]: risk budgets, then the risk budgeting weights; prints the latest."""

import argparse

import pandas as pd

from datalayer import load_config

from .compute import compute, contributions
from .weights import compute_weights

p = argparse.ArgumentParser(prog="budgets", description="Risk budgets per asset from the factor news and exposures.")
p.add_argument("config", nargs="?", help="configuration file (default: config/config.json)")
p.add_argument("--start", help="first rebalance date (default: budgets.start)")
p.add_argument("--end", help="last rebalance date (default: budgets.end, else today)")
args = p.parse_args()
cfg = load_config(args.config)
rows = compute(cfg, args.start, args.end)
if rows.empty:
    raise SystemExit("no exposures estimated before these dates: run python -m exposures first")
last = rows[rows.date == rows.date.max()].set_index("asset")
print(f"{rows.date.nunique()} rebalances, {rows.date.min()} to {rows.date.max()}; budgets at {rows.date.max()} "
      f"(exposures of {last.exposures.iloc[0]}):")
with pd.option_context("display.width", 200):
    print(last[["base", "score", "multiplier", "budget"]].round(4).to_string())
    c = contributions(cfg, rows.date.max())
    top = c.loc[:, c.abs().max() > 1e-4]
    print("\nscore contributions (exposure x factor signal / sum |exposure|):")
    print(top.round(4).to_string() if not top.empty else "  none: no factor has news before this date")
w = compute_weights(cfg, rows.date.min(), rows.date.max())
if w.empty:
    raise SystemExit("no weights: not enough price history before these dates")
lw = w[w.date == w.date.max()].set_index("asset")
print(f"\nweights at {w.date.max()} ({cfg['weights']['risk_measure']}, portfolio risk {lw.portfolio_risk.iloc[0]:.2%}, "
      f"{w.date.nunique()} rebalances, {w.groupby('date').converged.min().mean():.0%} converged):")
with pd.option_context("display.width", 200):
    print(lw[["budget", "weight", "risk_contribution"]].round(4).to_string())
