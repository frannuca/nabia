"""Files of the staged pipeline, computed once and read by the later steps instead of calling a model again:
scores files (the LLM's triage and node scores of every article, and/or the daily digests: data.scores_file,
data.digests_file) and betas files (the calibrated elastic-net coefficients per refit and asset: data.betas_file)."""

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


def _to_day(dates):
    return lambda ts: dates[min(dates.searchsorted(pd.Timestamp(ts).normalize()), len(dates) - 1)]


def load_digests(path, dates):
    """{trading day: Digest} of a scores or digests file."""
    to_day = _to_day(dates)
    return {to_day(day): Digest.model_validate(d) for day, d in json.loads(Path(path).read_text())["digests"].items()}


def write_betas(path, betas, in_sample, features, meta):
    """betas: {(refit date, asset): Series over the asset's candidate features}. Stores each refit's non-zero
    coefficients (the others are 0) and the candidate features of every asset."""
    rows = [{"refit": str(day.date()), "asset": a, "in_sample_ic": None if pd.isna(in_sample.get((day, a))) else float(in_sample[(day, a)]),
             "betas": {k: float(v) for k, v in b.items() if v}} for (day, a), b in sorted(betas.items())]
    data = {**meta, "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "features": features, "refits": rows}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_betas(path, X_by_asset):
    """({(refit date, asset): betas over the asset's current candidate features}, {(refit date, asset): in-sample IC}).
    Fails when a stored feature is not a candidate any more (the configuration changed since the calibration)."""
    data = json.loads(Path(path).read_text())
    betas, in_sample = {}, {}
    for r in data["refits"]:
        cols = X_by_asset[r["asset"]].columns
        unknown = set(r["betas"]) - set(cols)
        if unknown:
            raise ValueError(f"{path}: features of {r['asset']} not in this run: {sorted(unknown)}; calibrate again")
        key = (pd.Timestamp(r["refit"]), r["asset"])
        betas[key] = pd.Series(r["betas"], dtype=float).reindex(cols).fillna(0.0)
        in_sample[key] = float("nan") if r["in_sample_ic"] is None else r["in_sample_ic"]
    return betas, in_sample


def load_scores(path, dates):
    """(records, digests) as run_news returns them, each article and digest mapped to the first trading date at or
    after its day: records (day, node, sentiment, weight), digests {day: Digest}."""
    data = json.loads(Path(path).read_text())
    to_day = _to_day(dates)
    records = [(to_day(a["received_ts"]), s["node_id"], s["sentiment"], s["weight"])
               for a in data.get("articles", []) for s in a["scores"]]
    digests = {to_day(day): Digest.model_validate(d) for day, d in data.get("digests", {}).items()}
    return pd.DataFrame(records, columns=["day", "node", "sentiment", "weight"]), digests
