"""One interface to the language models: a local Ollama model, or the OpenAI and Anthropic APIs with the key read
from the environment variable named in the configuration. Each call returns the JSON object the model produced,
constrained to a JSON schema (Ollama format, OpenAI structured outputs, Anthropic forced tool use). Standard
library only: the calls are plain HTTPS requests."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request


class LLM:
    def __init__(self, cfg: dict):
        L = cfg["llm"]
        self.provider, self.p, self.timeout, self.temperature = L["provider"], L[L["provider"]], L["timeout"], L["temperature"]
        self.name = f"{self.provider}:{self.p['model']}"  # stored with every scoring
        self.key = os.environ.get(self.p["api_key_env"]) if "api_key_env" in self.p else None
        if self.provider != "ollama" and not self.key:
            raise RuntimeError(f"set {self.p['api_key_env']} to use {self.provider}")
        self.interval = 60 / L["requests_per_minute"] if L.get("requests_per_minute") else 0
        self.lock, self.next_slot = threading.Lock(), 0.0

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        if self.interval:  # shared by the worker threads: at most requests_per_minute calls start per minute
            with self.lock:
                wait, self.next_slot = self.next_slot - time.monotonic(), max(self.next_slot, time.monotonic()) + self.interval
            time.sleep(max(wait, 0))
        return getattr(self, "_" + self.provider)(system, user, schema)

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        """POST JSON, retrying rate limits and server errors: after the server's Retry-After, else with exponential
        backoff."""
        request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
        for attempt in range(5):
            try:
                return json.loads(urllib.request.urlopen(request, timeout=self.timeout).read())
            except urllib.error.HTTPError as error:
                with error:
                    if (error.code != 429 and error.code < 500) or attempt == 4:
                        raise RuntimeError(f"{url}: HTTP {error.code} {error.read()[:300]!r}") from None
                    wait = float(error.headers.get("retry-after") or 2 ** attempt * 5)
                time.sleep(wait)

    def _ollama(self, system, user, schema):
        """Ollama chat API. The system prompt (instructions and factor list) is the same for every article, so the
        server reuses its processed prefix and only the article is evaluated."""
        body = {"model": self.p["model"], "stream": False, "format": schema, "think": self.p["think"], "keep_alive": "30m",
                "options": {"temperature": self.temperature, **self.p["options"]},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        return json.loads(self._post(self.p["host"].rstrip("/") + "/api/chat", body, {})["message"]["content"])

    def _openai(self, system, user, schema):
        """OpenAI Chat Completions with a strict JSON schema (also works with OpenAI-compatible servers via base_url)."""
        body = {"model": self.p["model"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "scores", "strict": True, "schema": schema}}}
        if self.temperature is not None:  # reasoning models accept only their default: set llm.temperature to null
            body["temperature"] = self.temperature
        r = self._post(self.p["base_url"].rstrip("/") + "/chat/completions", body, {"Authorization": f"Bearer {self.key}"})
        return json.loads(r["choices"][0]["message"]["content"])

    def _anthropic(self, system, user, schema):
        """Anthropic Messages API: the answer is forced through a tool whose input is the schema; the system prompt is
        cached between calls."""
        body = {"model": self.p["model"], "max_tokens": self.p["max_tokens"], "messages": [{"role": "user", "content": user}],
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "tools": [{"name": "emit", "description": "Return the scores.", "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": "emit"}}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        r = self._post(self.p["base_url"].rstrip("/") + "/messages", body,
                       {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        return next(b["input"] for b in r["content"] if b["type"] == "tool_use")
