"""python -m exposures [config.json] [--no-download]: estimates the asset exposures and prints the latest map."""

import argparse

import pandas as pd

from datalayer import load_config

from .estimate import estimate

p = argparse.ArgumentParser(prog="exposures", description="Map the portfolio assets to the factors from data.")
p.add_argument("config", nargs="?", help="configuration file (default: config/config.json)")
p.add_argument("--no-download", action="store_true", help="use the FRED series already in the database")
args = p.parse_args()
rows = estimate(load_config(args.config), not args.no_download)
last = rows[rows.as_of == rows.as_of.max()]
print(f"{rows.as_of.nunique()} estimation dates, {rows.as_of.min()} to {rows.as_of.max()}; exposures at {rows.as_of.max()}:")
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(last.pivot(index="factor", columns="asset", values="exposure").reindex(rows.factor.unique()).round(2))
    print("\nmethods:", last.groupby("method").size().to_dict())
