"""Scoring the stored news with the LLM: one call per article returns its sentiment on the economic factors it
concerns, validated against the factor list and written to the database. Articles already scored by the same
model and prompt version are skipped, so an interrupted run resumes where it stopped."""

from __future__ import annotations

import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datalayer import NewsDB

from .llm import LLM

PROMPT = Path(__file__).with_name("prompt.md")


class Prompt:
    """The system prompt (instructions and factor list, the same for every article), the article template, the JSON
    schema of the answer and the version: a hash of both prompts, so editing the prompt or the factors rescores."""

    def __init__(self, factors: dict):
        system, self.user = PROMPT.read_text().split("\n=== USER ===\n")
        self.factors = factors
        self.system = system.format(factors="\n".join(f"{k} | {f['name']} | {f['positive']}" for k, f in factors.items()))
        self.version = hashlib.sha256((self.system + self.user).encode()).hexdigest()[:12]
        item = {"type": "object", "additionalProperties": False, "required": ["factor", "sentiment", "confidence"],
                "properties": {"factor": {"type": "string", "enum": list(factors)},
                               "sentiment": {"type": "number"}, "confidence": {"type": "number"}}}
        self.schema = {"type": "object", "additionalProperties": False, "required": ["relevant", "scores"],
                       "properties": {"relevant": {"type": "boolean"}, "scores": {"type": "array", "items": item}}}

    def article(self, a: dict) -> str:
        return self.user.format(published_at=a["published_at"], source=a["source"], headline=a["headline"],
                                body=a["body"][:4000])

    def parse(self, raw: dict) -> tuple[bool, list]:
        """(relevant, [(factor, sentiment, confidence)]): unknown factors dropped, values clipped to their range, one
        score per factor. Raises on a malformed answer (retried by the caller)."""
        clip = lambda x, lo: min(1.0, max(lo, float(x)))
        scores = {s["factor"]: (s["factor"], clip(s["sentiment"], -1), clip(s["confidence"], 0))
                  for s in raw["scores"] if s["factor"] in self.factors}
        relevant = bool(raw["relevant"]) and bool(scores)
        return relevant, list(scores.values()) if relevant else []


def scoring_version(cfg: dict) -> tuple[str, str]:
    """(model, prompt version) of the scores that the signals and exposures read: score.model and score.prompt when
    set, else the configured provider's model and the current prompt. Versions are never mixed."""
    L, s = cfg["llm"], cfg["score"]
    return (s.get("model") or f"{L['provider']}:{L[L['provider']]['model']}",
            s.get("prompt") or Prompt(cfg["factors"]).version)


def score(cfg: dict, since=None, until=None, limit=None, dry_run=False) -> int:
    """Scores the unscored articles received in [since, until) (default: the configuration's score section), at most
    limit; returns how many were scored. llm.workers calls run in parallel threads; writes stay in this thread.
    dry_run only reports the articles and their input tokens (about 4 characters per token), to size an API run."""
    s, L = cfg["score"], cfg["llm"]
    llm, prompt = (LLM(cfg) if not dry_run else None), Prompt(cfg["factors"])
    name = llm.name if llm else scoring_version({**cfg, "score": {}})[0]

    def one(a):
        for attempt in range(1 + L["max_retries"]):
            try:
                return prompt.parse(llm(prompt.system, prompt.article(a), prompt.schema))
            except Exception as error:  # invalid JSON or schema, network error
                last = error
        print(f"article {a['id']} skipped: {type(last).__name__}: {str(last)[:120]}", file=sys.stderr, flush=True)

    with NewsDB(cfg["db"]) as db:
        todo = db.unscored(name, prompt.version, since or s["since"], until or s["until"], limit or s["limit"])
        if dry_run:
            user = sum(len(prompt.article(a)) for a in todo) / 4
            print(f"{len(todo)} articles to score with {name}, prompt {prompt.version}: system prompt "
                  f"~{len(prompt.system) / 4:.0f} tokens per call (cacheable), articles ~{user:,.0f} tokens in all")
            return 0
        print(f"scoring {len(todo)} articles with {name}, prompt {prompt.version}, {L['workers']} workers", flush=True)
        start, done = time.perf_counter(), 0
        with ThreadPoolExecutor(max(1, L["workers"])) as pool:
            for n, (a, result) in enumerate(zip(todo, pool.map(one, todo)), 1):
                if result is not None:
                    db.save_scoring(a["id"], name, prompt.version, *result)
                    done += 1
                if n % 25 == 0 or n == len(todo):
                    rate = n / (time.perf_counter() - start)
                    print(f"  {n}/{len(todo)}, {rate:.2f}/s, {(len(todo) - n) / rate / 60:.0f} min left", flush=True)
        print(f"{done} scored; database: {db.stats()}")
    return done
