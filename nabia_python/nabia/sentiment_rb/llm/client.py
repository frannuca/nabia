"""The LLM client: versioned prompts, offline mock or Anthropic API, schema validation, retries, audit log, and a
response cache shared by all runs (each live request is paid for once; Message Batches fill it in bulk)."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.request
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ..config import PACKAGE
from . import lexicon
from .schemas import Digest, Scores, Triage


def mock_response(kind, ctx, rng, attempt):
    """Stands in for the model offline: from the synthetic truth when the article has one, otherwise by reading the
    text (lexicon_response). 1% of first attempts are malformed, so the retry path is exercised."""
    bad = attempt == 0 and rng.random() < 0.01
    if (kind != "digest" and "_tone" not in ctx["article"]) or (kind == "digest" and ctx.get("factors") is None):
        return lexicon_response(kind, ctx)
    if kind == "triage":
        a = ctx["article"]
        nodes = [{"node_id": a["_node"], "relevance": float(rng.uniform(0.6, 1)), "channel": "direct"}]
        if parent := ctx["cfg"]["taxonomy"][a["_node"]][0]:
            nodes.append({"node_id": parent, "relevance": 0.4, "channel": "spillover"})
        return {"relevant": not a["_recap"], "reason": "price_recap" if a["_recap"] else "new_information",
                "event_type": "other", "nodes": [] if a["_recap"] else nodes}
    if kind == "scoring":
        a = ctx["article"]
        return {"scores": [{"node_id": m.node_id, "sentiment": 1.7 if bad else float(np.clip(
            (1 if m.channel == "direct" else 0.5) * a["_tone"] + rng.normal(0, 0.15), -1, 1)),
            "confidence": float(rng.uniform(0.6, 0.95)), "magnitude": "medium", "horizon": "days",
            "surprise_vs_expectations": "not_stated", "fact_type": "confirmed", "rationale": "synthetic"}
            for m in ctx["nodes"]]}
    f, th = ctx["factors"], lambda x: float(np.tanh(x + rng.normal(0, 0.3)))
    return {"risk_appetite": th(f["L0.GLOBAL"]), "growth_tone": th(0.7 * f["L0.GLOBAL"]),
            "inflation_tone": th(0.5 * f["L1.CMD"]), "geopolitical_stress": float(1 / (1 + np.exp(2 * f["L0.GLOBAL"]))),
            "credit_tone": th(f["L2.FI.HY"]), "cb_tone": {"FED": th(-f["L1.FI"]), "ECB": th(-0.5 * f["L1.FI"])},
            "commodity_supply_tone": {"energy": th(f["L2.CMD.ENERGY"]), "metals": th(f["L2.CMD.PRECIOUS"])},
            "item_count_used": ctx["n"], "key_drivers": ["synthetic"]}


def lexicon_response(kind, ctx):
    """The offline answer for real-format news: nodes named in the headline, sentiment from the word lexicon."""
    cfg = ctx["cfg"]
    if kind == "triage":
        text = ctx["article"]["headline"]
        node = None if lexicon.is_recap(text) else lexicon.match_node(cfg, text)
        nodes = [] if node is None else [{"node_id": node, "relevance": 0.8, "channel": "direct"}]
        if node and (parent := cfg["taxonomy"][node][0]):
            nodes.append({"node_id": parent, "relevance": 0.4, "channel": "spillover"})
        reason = "price_recap" if lexicon.is_recap(text) else "new_information" if nodes else "irrelevant"
        return {"relevant": bool(nodes), "reason": reason, "event_type": "other", "nodes": nodes}
    if kind == "scoring":
        t = lexicon.tone(ctx["article"]["headline"])
        return {"scores": [{"node_id": m.node_id, "sentiment": round((1 if m.channel == "direct" else 0.5) * t, 4),
                            "confidence": 0.8, "magnitude": "medium", "horizon": "days", "surprise_vs_expectations": "not_stated",
                            "fact_type": "confirmed", "rationale": "lexicon"} for m in ctx["nodes"]]}
    by_node = {}  # digest: the day's tone per node, from the headlines that name it
    for h in ctx["headlines"]:
        if not lexicon.is_recap(h) and (node := lexicon.match_node(cfg, h)):
            by_node.setdefault(node, []).append(lexicon.tone(h))
    mean = lambda n, scale=1.0: float(np.clip(scale * np.mean(by_node[n]), -1, 1)) if n in by_node else 0.0
    risk = mean("L0.GLOBAL")
    return {"risk_appetite": risk, "growth_tone": mean("L0.GLOBAL", 0.7), "inflation_tone": mean("L1.CMD", 0.5),
            "geopolitical_stress": float(np.clip((1 - risk) / 2, 0, 1)), "credit_tone": mean("L2.FI.HY"),
            "cb_tone": {"FED": -mean("L1.FI"), "ECB": -mean("L1.FI", 0.5)},
            "commodity_supply_tone": {"energy": mean("L2.CMD.ENERGY"), "metals": mean("L2.CMD.PRECIOUS")},
            "item_count_used": ctx["n"], "key_drivers": ["lexicon"]}


class LLM:
    SCHEMAS = {"triage": Triage, "scoring": Scores, "digest": Digest}
    MODEL = {"triage": "triage_model", "scoring": "scoring_model", "digest": "digest_model"}

    def __init__(self, cfg, out, live, rng):
        self.cfg, self.rng, self.live, self.prompts = cfg, rng, live, {}
        for kind, file in [("triage", "triage_v1"), ("scoring", "scoring_v1"), ("digest", "digest_v1")]:
            text = (PACKAGE / "prompts" / f"{file}.md").read_text()
            self.prompts[kind] = (*text.split("\n=== USER ===\n"), f"{file}:{hashlib.sha256(text.encode()).hexdigest()[:12]}")
        self.audit, self.lock = open(out / "llm_audit.jsonl", "w"), threading.Lock()  # calls may run in threads
        self.cache_path = Path(cfg["llm"]["cache"])
        self.cache = {}  # key -> raw response of a valid live call
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["response"]
        self.client, self.provider = None, cfg["llm"]["provider"] if live else "mock"
        if self.provider == "anthropic":
            import anthropic  # retries 429/5xx with exponential backoff
            self.client = anthropic.Anthropic(max_retries=5)

    def model(self, kind):
        """The model answering `kind` with the current provider."""
        return self.cfg["llm"]["ollama_model"] if self.provider == "ollama" else self.cfg["llm"][self.MODEL[kind]]

    def request(self, kind, fill):
        """(user prompt, cache key) of a call: the key covers the prompt version, the model and the request."""
        system, template, version = self.prompts[kind]
        user = fill(template)
        return user, hashlib.sha256(f"{version}|{self.model(kind)}|{user}".encode()).hexdigest()

    def params(self, kind, user):
        """Messages API parameters of a call: temperature 0, output forced through a tool with the pydantic schema,
        system prompt cached."""
        system = self.prompts[kind][0]
        return dict(model=self.cfg["llm"][self.MODEL[kind]], max_tokens=2048, temperature=0,
                    messages=[{"role": "user", "content": user}],
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    tools=[{"name": "emit", "description": "Return the result.",
                            "input_schema": self.SCHEMAS[kind].model_json_schema()}],
                    tool_choice={"type": "tool", "name": "emit"})

    def store(self, key, raw):
        """Adds a valid live response to the cache (memory and file)."""
        with self.lock:
            self.cache[key] = raw
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "response": raw}, default=str) + "\n")

    def prefill(self, kind, fills):
        """With llm.batch and live calls: submits the uncached requests of `fills` as one Message Batch, waits for
        it and caches the valid responses, so the calls that follow are cache hits."""
        if self.client is not None and self.cfg["llm"]["batch"] and fills:
            from .batch import submit
            submit(self, kind, [self.request(kind, f) for f in fills])

    def __call__(self, kind, fill, ctx, live=False):
        _, _, version = self.prompts[kind]
        user, key = self.request(kind, fill)
        schema, model = self.SCHEMAS[kind], self.model(kind)
        for attempt in range(1 + self.cfg["llm"]["max_retries"]):
            cached = live and key in self.cache
            if cached:
                raw = self.cache[key]
            else:
                raw = (self._ollama(kind, user) if self.provider == "ollama" else self._anthropic(kind, user)) if live \
                    else mock_response(kind, {**ctx, "cfg": self.cfg}, self.rng, attempt)
            try:
                result = schema.model_validate(raw)
            except ValidationError:
                result = None
            if live and result is not None and not cached:
                self.store(key, raw)
            line = json.dumps({"kind": kind, "model": model if live else "mock", "prompt": version,
                               "request_sha": key[:16], "ref": ctx.get("ref"), "attempt": attempt,
                               "cached": cached, "valid": result is not None, "response": raw}, default=str) + "\n"
            with self.lock:
                self.audit.write(line)
                self.audit.flush()  # visible while a long live run is going
            if result is not None:
                return result
        return None  # logged and skipped

    def _ollama(self, kind, user):
        """A local model through Ollama's chat API, its output constrained to the pydantic schema (JSON)."""
        L = self.cfg["llm"]
        body = {"model": L["ollama_model"], "stream": False, "format": self.SCHEMAS[kind].model_json_schema(),
                "think": L["ollama_think"], "keep_alive": "30m", "options": L["ollama_options"],
                "messages": [{"role": "system", "content": self.prompts[kind][0]}, {"role": "user", "content": user}]}
        request = urllib.request.Request(L["ollama_host"].rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        content = json.loads(urllib.request.urlopen(request, timeout=L["ollama_timeout"]).read())["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content  # invalid: retried by the caller

    def _anthropic(self, kind, user):
        r = self.client.messages.create(**self.params(kind, user))
        return next(b.input for b in r.content if b.type == "tool_use")


def fill(template, values, lines):
    """Prompt template: {placeholders} from `values`, list placeholders ({node_id |...}, {ts |...}) by `lines`."""
    out = [lines if re.match(r"\{(node_id|ts) \|", l) else l for l in template.splitlines()]
    text = "\n".join(out)
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text
