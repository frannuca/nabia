"""The LLM client: versioned prompts, offline mock or Anthropic API, schema validation, retries, audit log."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from pydantic import ValidationError

from ..config import PACKAGE
from .schemas import Digest, Scores, Triage


def mock_response(kind, ctx, rng, attempt):
    """Stands in for the model offline, from the synthetic truth; 1% of first attempts are malformed (retry path)."""
    bad = attempt == 0 and rng.random() < 0.01
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


class LLM:
    SCHEMAS = {"triage": Triage, "scoring": Scores, "digest": Digest}
    MODEL = {"triage": "triage_model", "scoring": "scoring_model", "digest": "digest_model"}

    def __init__(self, cfg, out, live, rng):
        self.cfg, self.rng, self.live, self.prompts = cfg, rng, live, {}
        for kind, file in [("triage", "triage_v1"), ("scoring", "scoring_v1"), ("digest", "digest_v1")]:
            text = (PACKAGE / "prompts" / f"{file}.md").read_text()
            self.prompts[kind] = (*text.split("\n=== USER ===\n"), f"{file}:{hashlib.sha256(text.encode()).hexdigest()[:12]}")
        self.audit = open(out / "llm_audit.jsonl", "w")
        if live:
            import anthropic  # retries 429/5xx with exponential backoff
            self.client = anthropic.Anthropic(max_retries=5)

    def __call__(self, kind, fill, ctx, live=False):
        system, template, version = self.prompts[kind]
        user, schema, model = fill(template), self.SCHEMAS[kind], self.cfg["llm"][self.MODEL[kind]]
        for attempt in range(1 + self.cfg["llm"]["max_retries"]):
            raw = self._anthropic(model, system, user, schema) if live else mock_response(kind, ctx, self.rng, attempt)
            try:
                result = schema.model_validate(raw)
            except ValidationError:
                result = None
            self.audit.write(json.dumps({"kind": kind, "model": model if live else "mock", "prompt": version,
                                         "request_sha": hashlib.sha256(user.encode()).hexdigest()[:16], "ref": ctx.get("ref"),
                                         "attempt": attempt, "valid": result is not None, "response": raw}, default=str) + "\n")
            if result is not None:
                return result
        return None  # logged and skipped

    def _anthropic(self, model, system, user, schema):
        r = self.client.messages.create(
            model=model, max_tokens=2048, temperature=0, messages=[{"role": "user", "content": user}],
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[{"name": "emit", "description": "Return the result.", "input_schema": schema.model_json_schema()}],
            tool_choice={"type": "tool", "name": "emit"})
        return next(b.input for b in r.content if b.type == "tool_use")


def fill(template, values, lines):
    """Prompt template: {placeholders} from `values`, list placeholders ({node_id |...}, {ts |...}) by `lines`."""
    out = [lines if re.match(r"\{(node_id|ts) \|", l) else l for l in template.splitlines()]
    text = "\n".join(out)
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text
