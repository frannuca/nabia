"""python -m collector [config.json] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--import news.json]"""

import argparse

from datalayer import NewsDB, load_config

from .collect import collect
from .sources import read_json

p = argparse.ArgumentParser(prog="collector", description="Download news (RSS, GDELT GKG) into the database.")
p.add_argument("config", nargs="?", help="configuration file (default: config/config.json)")
p.add_argument("--since", help="first GKG day (default: the day after the last one stored)")
p.add_argument("--until", help="GKG end day, excluded (default: today)")
p.add_argument("--import", dest="json", help="only import the articles of a news.json file of the former project")
args = p.parse_args()
cfg = load_config(args.config)
if args.json:
    with NewsDB(cfg["db"]) as db:
        print(f"imported {db.add_news(read_json(args.json))} new articles; database: {db.stats()}")
else:
    collect(cfg, args.since, args.until)
