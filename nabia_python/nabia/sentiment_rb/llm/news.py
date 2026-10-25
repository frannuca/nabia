"""News pipeline: dedup and novelty, triage, scoring, daily digests, and the 5 LLM features per node."""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

from nabia_python_bridge import signal

from .client import fill


def embed(text, dim=256):
    v = np.zeros(dim)
    for tok in re.findall(r"[a-z0-9&]+", text.lower()):
        v[zlib.crc32(tok.encode()) % dim] += 1
    return v / (np.linalg.norm(v) or 1)


def run_news(cfg, news, f, llm, live_from):
    """Dedup -> triage -> scoring -> per-node weighted scores by day; daily digest. Returns (scores, digests)."""
    tax, L = cfg["taxonomy"], cfg["llm"]
    alias = {v[1]: f"Entity {i}" for i, v in enumerate(tax.values())} if L["anonymize"] else {}
    anon = lambda s: re.sub("|".join(map(re.escape, alias)), lambda m: alias[m[0]], s) if alias else s
    taxonomy_lines = "\n".join(f"{n} | {anon(v[1])} | {v[2]} | {anon(v[1])}" for n, v in tax.items())
    news = news.sort_values("received_ts").reset_index(drop=True)
    news["day"] = f.index[np.minimum(f.index.searchsorted(news.received_ts.dt.normalize()), len(f) - 1)]
    vecs, cluster, records, recent = [embed(h) for h in news.headline], {}, [], []
    for i, a in news.iterrows():
        recent = [j for j in recent if a.received_ts - news.received_ts[j] < pd.Timedelta(hours=L["dedup_hours"])]
        same = [j for j in recent if vecs[i] @ vecs[j] > L["dedup_cosine"]]
        cluster[i] = cluster[same[0]] if same else i
        novelty = 1 / (1 + sum(cluster[j] == cluster[i] for j in recent))
        recent.append(i)
        live, base = a.day >= live_from, {"publish_ts_utc": a.publish_ts, "source_name": a.source,
                                          "headline": anon(a.headline), "body": anon(a.body),
                                          "body_truncated_to_N_tokens": anon(a.body)[:4000]}
        tri = llm("triage", lambda t: fill(t, base, taxonomy_lines), {"article": a, "cfg": cfg, "ref": int(a.id)}, live)
        if not tri or not tri.relevant or not (nodes := [m for m in tri.nodes if m.node_id in tax]):
            continue
        node_lines = "\n".join(f"{m.node_id} | {anon(tax[m.node_id][1])} | {tax[m.node_id][2]} | {m.channel}" for m in nodes)
        sc = llm("scoring", lambda t: fill(t, base, node_lines), {"article": a, "nodes": nodes, "ref": int(a.id)}, live)
        rel = {m.node_id: m.relevance for m in nodes}
        for s in sc.scores if sc else []:
            if s.node_id in rel:
                w = rel[s.node_id] * s.confidence * novelty * L["source_weight"].get(a.source, 0.5)
                records.append((a.day, s.node_id, s.sentiment, w))
    digests = {}
    for day, g in news.groupby("day"):
        items = "\n".join(f"{r.publish_ts} | {r.source} | {anon(r.headline)} | {anon(r.body)}" for r in g.itertuples())
        d = llm("digest", lambda t: fill(t, {"yyyy-mm-dd": day.date()}, items),
                {"factors": f.loc[day], "n": len(g), "ref": str(day.date())}, day >= live_from)
        if d:
            digests[day] = d
    return pd.DataFrame(records, columns=["day", "node", "sentiment", "weight"]), digests


def ema0(x, half_life):
    """Zero-started EWMA (nabia): y_t ∝ sum_j (1-a)^(t-j) x_j, so ratios of two ema0 are decay-weighted means."""
    a = 1 - 0.5 ** (1 / half_life)
    return signal.exponential_moving_average(np.r_[0.0, np.asarray(x, float)], a)[1:]


def node_features(cfg, scores, dates):
    """The 5 LLM features of every node: decay-weighted level, volume, dispersion, 5-day momentum, surprise."""
    out = {}
    for node in cfg["taxonomy"]:
        g = scores[scores.node == node].assign(ws=lambda d: d.weight * d.sentiment).groupby("day")
        day = g.agg(ws=("ws", "sum"), w=("weight", "sum"), n=("sentiment", "size"), sd=("sentiment", "std")).reindex(dates)
        num, den = ema0(day.ws.fillna(0), cfg["llm"]["half_life"]), ema0(day.w.fillna(0), cfg["llm"]["half_life"])
        level = pd.Series(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12), dates)
        out |= {f"llm_level@{node}": level, f"llm_volume@{node}": np.log1p(day.n.fillna(0)),
                f"llm_dispersion@{node}": day.sd.fillna(0), f"llm_mom5@{node}": level - level.shift(5),
                f"llm_surprise@{node}": level - level.rolling(60, min_periods=20).mean()}
    return pd.DataFrame(out)
