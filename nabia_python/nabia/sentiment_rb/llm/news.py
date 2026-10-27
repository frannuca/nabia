"""News pipeline: dedup and novelty, triage, scoring, daily digests, and the 5 LLM features per node."""

from __future__ import annotations

import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from nabia_python_bridge import signal

from ..features.tree import rollup
from .client import fill


def embed(text, dim=256):
    v = np.zeros(dim)
    for tok in re.findall(r"[a-z0-9&]+", text.lower()):
        v[zlib.crc32(tok.encode()) % dim] += 1
    return v / (np.linalg.norm(v) or 1)


def run_all(work, items, llm, label, checkpoint=None):
    """work(item) for every item, in order: in parallel threads for live calls (llm.workers), sequentially for the
    offline mock (whose random draws must keep their order). Reports progress on long live runs, and calls
    checkpoint() every 25 items (to save intermediate results)."""
    workers = llm.cfg["llm"]["workers"] if llm.provider != "mock" else 1
    if workers <= 1 and llm.provider == "mock":
        return [work(item) for item in items]
    start, results = time.perf_counter(), []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, result in enumerate(pool.map(work, items), 1):
            results.append(result)
            if n % 25 == 0 or n == len(items):
                if checkpoint:
                    checkpoint()
                rate = n / (time.perf_counter() - start)
                print(f"  {label}: {n}/{len(items)}, {rate:.2f}/s, {(len(items) - n) / rate / 60:.0f} min left", flush=True)
    return results


def run_news(cfg, news, dates, llm, live_from, truth=None, details=None, digests_out=None, checkpoint=None,
             stages=("articles", "digests")):
    """Dedup -> triage -> scoring -> per-node weighted scores by day; daily digest. Returns (scores, digests).

    Articles are assigned to the first date at or after the day they were received. Days from live_from on go to the
    live model (through the response cache); with llm.batch, their uncached requests are first submitted as Message
    Batches (triage, then scoring, then digests). `truth` (synthetic runs) feeds the offline mock of the digest.
    `details` and `digests_out`, dicts, receive each article's triage and scores and each day's digest as they are
    computed; `checkpoint()` is called every 25 articles or digests (the scoring command saves its file there).
    `stages` runs only the article scores ("articles") or only the digests ("digests").
    """
    tax, L = cfg["taxonomy"], cfg["llm"]
    if news.empty:
        return pd.DataFrame(columns=["day", "node", "sentiment", "weight"]), {}
    alias = {v["name"]: f"Entity {i}" for i, v in enumerate(cfg["assets"].values())} if L["anonymize"] else {}  # asset names
    anon = lambda s: re.sub("|".join(map(re.escape, alias)), lambda m: alias[m[0]], s) if alias else s
    taxonomy_lines = "\n".join(f"{n} | {v[1]} | {v[3]} | {v[2] or 'mixed'} | {tax[n][0] or '-'}" for n, v in tax.items())
    news = news.sort_values("received_ts").reset_index(drop=True)
    news["day"] = dates[np.minimum(dates.searchsorted(news.received_ts.dt.normalize()), len(dates) - 1)]
    base = {i: {"publish_ts_utc": a.publish_ts, "source_name": a.source, "headline": anon(a.headline),
                "body": anon(a.body), "body_truncated_to_N_tokens": anon(a.body)[:4000]} for i, a in news.iterrows()}
    triage_fill = lambda i: (lambda t: fill(t, base[i], taxonomy_lines))
    node_lines = lambda nodes: "\n".join(f"{m.node_id} | {tax[m.node_id][1]} | {tax[m.node_id][2] or 'mixed'} | {m.channel}" for m in nodes)
    scoring_fill = lambda i, nodes: (lambda t: fill(t, base[i], node_lines(nodes)))
    def canonical(node_id):
        """The taxonomy node an answer means: models sometimes copy the whole prompt line ("US.EQ | US equities | ...")."""
        return node_id.split("|")[0].strip()

    def relevant_nodes(tri):
        if not (tri and tri.relevant):
            return []
        nodes = {}
        for m in tri.nodes:
            if (node := canonical(m.node_id)) in tax and node not in nodes:
                nodes[node] = m.model_copy(update={"node_id": node})
        return list(nodes.values())
    if llm.client is not None and L["batch"] and "articles" in stages:  # backfill the live days' triage and scoring
        live_rows = [i for i in news.index if news.day[i] >= live_from]
        llm.prefill("triage", [triage_fill(i) for i in live_rows])
        cached = {i: llm.cache.get(llm.request("triage", triage_fill(i))[1]) for i in live_rows}
        found = {i: relevant_nodes(llm.SCHEMAS["triage"].model_validate(raw)) for i, raw in cached.items() if raw}
        llm.prefill("scoring", [scoring_fill(i, nodes) for i, nodes in found.items() if nodes])
    vecs, cluster, novelty, recent = [embed(h) for h in news.headline], {}, {}, []
    for i, a in news.iterrows():  # dedup clusters and novelty, in order of receipt
        recent = [j for j in recent if a.received_ts - news.received_ts[j] < pd.Timedelta(hours=L["dedup_hours"])]
        same = [j for j in recent if vecs[i] @ vecs[j] > L["dedup_cosine"]]
        cluster[i] = cluster[same[0]] if same else i
        novelty[i] = 1 / (1 + sum(cluster[j] == cluster[i] for j in recent))
        recent.append(i)

    def article(i):
        """Triage and scoring of one article: its (day, node, sentiment, weight) records."""
        a = news.loc[i]
        live = a.day >= live_from
        tri = llm("triage", triage_fill(i), {"article": a, "ref": int(a.id)}, live)
        nodes = relevant_nodes(tri)
        sc = llm("scoring", scoring_fill(i, nodes), {"article": a, "nodes": nodes, "ref": int(a.id)}, live) if nodes else None
        rel = {m.node_id: m.relevance for m in nodes}
        scores = [s.model_copy(update={"node_id": canonical(s.node_id)}) for s in (sc.scores if sc else [])]
        scores = list({s.node_id: s for s in scores if s.node_id in rel}.values())
        if details is not None:
            details[i] = {"id": int(a.id), "received_ts": str(a.received_ts), "source": a.source, "headline": a.headline,
                          "relevant": bool(nodes), "event_type": tri.event_type if tri else None,
                          "scores": [{**s.model_dump(), "relevance": rel[s.node_id], "novelty": novelty[i],
                                      "channel": next(m.channel for m in nodes if m.node_id == s.node_id),
                                      "weight": rel[s.node_id] * s.confidence * novelty[i] * L["source_weight"].get(a.source, 0.5)}
                                     for s in scores]}
        return [(a.day, s.node_id, s.sentiment, rel[s.node_id] * s.confidence * novelty[i] * L["source_weight"].get(a.source, 0.5))
                for s in scores]

    records = [r for rs in run_all(article, list(news.index), llm, "articles", checkpoint) for r in rs] if "articles" in stages else []
    days = list(news.groupby("day"))
    digest_fill = lambda day, g: (lambda t: fill(t, {"yyyy-mm-dd": day.date()}, "\n".join(
        f"{r.publish_ts} | {r.source} | {anon(r.headline)} | {anon(r.body)}" for r in g.itertuples())))
    if "digests" not in stages:
        return pd.DataFrame(records, columns=["day", "node", "sentiment", "weight"]), {}
    llm.prefill("digest", [digest_fill(day, g) for day, g in days if day >= live_from])

    def digest(item):
        day, g = item
        ctx = {"factors": None if truth is None else truth.loc[day], "headlines": list(g.headline), "n": len(g),
               "ref": str(day.date())}
        d = llm("digest", digest_fill(day, g), ctx, day >= live_from)
        if digests_out is not None and d:
            digests_out[day] = d
        return day, d

    digests = {day: d for day, d in run_all(digest, days, llm, "digests", checkpoint) if d}
    return pd.DataFrame(records, columns=["day", "node", "sentiment", "weight"]), digests


def ema0(x, half_life):
    """Zero-started EWMA (nabia): y_t ∝ sum_j (1-a)^(t-j) x_j, so ratios of two ema0 are decay-weighted means."""
    a = 1 - 0.5 ** (1 / half_life)
    return signal.exponential_moving_average(np.r_[0.0, np.asarray(x, float)], a)[1:]


def node_features(cfg, scores, dates):
    """The 5 LLM features of every node: decay-weighted level, volume, dispersion, 5-day momentum, surprise."""
    out, decay = {}, cfg["llm"]["rollup"]
    if decay and len(scores):  # each score also counts for the same-class ancestors, weighted decay**level
        scores = pd.DataFrame([(d, a, s, w * decay**k) for d, n, s, w in scores[["day", "node", "sentiment", "weight"]].itertuples(index=False)
                               for a, k in (rollup(cfg, n) if n in cfg["taxonomy"] else [(n, 0)])], columns=["day", "node", "sentiment", "weight"])
    for node in cfg["taxonomy"]:
        g = scores[scores.node == node].assign(ws=lambda d: d.weight * d.sentiment).groupby("day")
        day = g.agg(ws=("ws", "sum"), w=("weight", "sum"), n=("sentiment", "size"), sd=("sentiment", "std")).reindex(dates)
        num, den = ema0(day.ws.fillna(0), cfg["llm"]["half_life"]), ema0(day.w.fillna(0), cfg["llm"]["half_life"])
        level = pd.Series(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12), dates)
        out |= {f"llm_level@{node}": level, f"llm_volume@{node}": np.log1p(day.n.fillna(0)),
                f"llm_dispersion@{node}": day.sd.fillna(0), f"llm_mom5@{node}": level - level.shift(5),
                f"llm_surprise@{node}": level - level.rolling(60, min_periods=20).mean()}
    return pd.DataFrame(out)
