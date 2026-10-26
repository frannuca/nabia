"""Real-data mode, offline: lexicon scoring, news files, example news, cached loaders, the LLM cache and batches."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nabia.sentiment_rb.config import load_config, merge
from nabia.sentiment_rb.data import load_news, real_market, synthetic_market
from nabia.sentiment_rb.data.example_news import make_example_news, write_news
from nabia.sentiment_rb.features import vol_scaled_target
from nabia.sentiment_rb.llm import LLM, lexicon, node_features, run_news


@pytest.fixture
def cfg(tmp_path):
    return merge(load_config(), {"days": 1200, "data": {"cache_dir": str(tmp_path / "cache")},
                                 "llm": {"cache": str(tmp_path / "llm_cache.jsonl")}})


def test_lexicon_reads_the_example_vocabulary_and_finds_the_most_specific_node(cfg):
    phrases = ["slumps on severe downturn fears warning", "weakens as outlook is cut", "steady with mixed signals",
               "gains on improving demand outlook", "surges after strong upside surprise"]
    tones = [lexicon.tone(p) for p in phrases]
    assert tones == sorted(tones) and tones[0] < -0.5 and tones[2] == 0 and tones[-1] > 0.5
    assert lexicon.match_node(cfg, "US high yield gains") == "L3.HY"
    assert lexicon.match_node(cfg, "High yield credit weakens") == "L2.FI.HY"
    assert lexicon.match_node(cfg, "Brent crude jumps") == "L3.WTI"
    assert lexicon.match_node(cfg, "Unrelated headline") is None
    assert lexicon.is_recap("Gold moved +1.2% in today's session")


def test_example_news_carries_its_skill_through_the_offline_scorer(cfg, tmp_path):
    _, returns, _ = synthetic_market(cfg, np.random.default_rng(1))
    news = make_example_news(cfg, returns, np.random.default_rng(2), ic=0.6)
    write_news(news, tmp_path / "news.json", {"ic": 0.6})
    loaded = load_news(tmp_path / "news.json")
    assert list(loaded.columns) == ["id", "publish_ts", "received_ts", "source", "headline", "body"] and len(loaded) == len(news)
    dates = returns.index
    scores, digests = run_news(cfg, loaded, dates, LLM(cfg, tmp_path, False, np.random.default_rng(3)), dates[-1] + pd.Timedelta(days=1))
    level = node_features(cfg, scores, dates)["llm_level@L3.SPX"]
    forward = vol_scaled_target(cfg, returns["SPX"]).shift(-cfg["llm"]["lag"])
    assert level.corr(forward) > 0.1 and digests  # the tone read from the text predicts the asset's forward return


def test_real_market_reads_the_csv_cache(cfg):
    cache, dates = Path(cfg["data"]["cache_dir"]), pd.bdate_range("2014-01-01", periods=400)
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = [t.replace("^", "_").replace("=", "_") for t in cfg["real"]["assets"].values()]
    names += ["_VIX", "_MOVE", "DX-Y.NYB", "CL_F", "RB_F", "HO_F"]
    for name in names:
        pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), dates, name="Close").to_csv(cache / f"{name}.csv")
    for fred in ["BAMLH0A0HYM2", "NFCI", "THREEFYTP10"]:
        pd.Series(rng.normal(size=80), dates[::5], name=fred).to_csv(cache / f"fred_{fred}.csv")
    returns, raw = real_market(cfg)
    assert list(returns.columns) == list(cfg["assets"]) and len(returns) == len(dates) - 1
    assert set(raw.columns) == set(cfg["real"]["indicators"]) and raw.loc[dates[10]:].notna().all().all()


class FakeClient:
    """The Anthropic client's messages and batches APIs, answering every request with a valid triage."""
    answer = {"relevant": True, "reason": "new_information", "event_type": "other",
              "nodes": [{"node_id": "L3.SPX", "relevance": 0.9, "channel": "direct"}]}

    def __init__(self):
        self.calls, self.batched = 0, []
        message = lambda: SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.answer)])
        self.messages = SimpleNamespace(create=self.create, batches=SimpleNamespace(
            create=lambda requests: self.batched.extend(requests) or SimpleNamespace(id="b1"),
            retrieve=lambda _id: SimpleNamespace(id="b1", processing_status="ended"),
            results=lambda _id: [SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=message()))
                                 for r in self.batched]))
        self._message = message

    def create(self, **params):
        self.calls += 1
        return self._message()


def live_llm(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = LLM(cfg, tmp_path, True, np.random.default_rng(0))
    llm.client = FakeClient()
    return llm


def test_live_responses_are_cached_across_runs(cfg, tmp_path, monkeypatch):
    llm = live_llm(cfg, tmp_path, monkeypatch)
    ask = lambda model: model("triage", lambda t: t.replace("{headline}", "S&P 500 gains"), {"ref": 0}, True)
    assert ask(llm).relevant and ask(llm).relevant
    assert llm.client.calls == 1  # the second call is a cache hit
    again = live_llm(cfg, tmp_path, monkeypatch)  # a new run reads the cache file
    assert ask(again).relevant and again.client.calls == 0


def test_batch_prefill_makes_the_calls_cache_hits(cfg, tmp_path, monkeypatch):
    llm = live_llm(merge(cfg, {"llm": {"batch": True}}), tmp_path, monkeypatch)
    fills = [lambda t, h=h: t.replace("{headline}", h) for h in ["S&P 500 gains", "Gold slumps", "Oil surges"]]
    llm.prefill("triage", fills)
    assert len(llm.client.batched) == 3 and all(len(r["custom_id"]) <= 64 for r in llm.client.batched)
    assert all(llm("triage", f, {"ref": 0}, True).relevant for f in fills) and llm.client.calls == 0
    llm.prefill("triage", fills)  # nothing left to submit
    assert len(llm.client.batched) == 3
    assert json.loads(Path(cfg["llm"]["cache"]).read_text().splitlines()[0])["key"]
