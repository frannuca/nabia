"""News scores files: the LLM's triage and node scores of every article, and the daily digests, computed once
(python -m nabia.news score) and read by the pipeline instead of calling a model (data.scores_file / --scores)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..llm.schemas import Digest


def write_scores(path, details, digests, meta, complete=True):
    """details: {row: article dict with its scores}; digests: {day: Digest}. Written atomically; complete = False
    marks an intermediate file of a run still going (it can be used as it is)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**meta, "complete": complete, "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "articles": [details[i] for i in sorted(dict(details))],
            "digests": {str(day.date()): d.model_dump() for day, d in sorted(dict(digests).items())}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2))  # readable
    tmp.replace(path)


def load_scores(path, dates):
    """(records, digests) as run_news returns them, each article and digest mapped to the first trading date at or
    after its day: records (day, node, sentiment, weight), digests {day: Digest}."""
    data = json.loads(Path(path).read_text())
    to_day = lambda ts: dates[min(dates.searchsorted(pd.Timestamp(ts).normalize()), len(dates) - 1)]
    records = [(to_day(a["received_ts"]), s["node_id"], s["sentiment"], s["weight"])
               for a in data["articles"] for s in a["scores"]]
    digests = {}
    for day, d in data["digests"].items():
        digests[to_day(day)] = Digest.model_validate(d)
    return pd.DataFrame(records, columns=["day", "node", "sentiment", "weight"]), digests
