"""Run the agents on a local model served by Ollama.

``OllamaClient`` exposes the one method the agent runtime uses,
``beta.messages.create(...)``, with Anthropic Messages-shaped requests and
responses, and translates them to Ollama's native chat API (``/api/chat``):

- the system prompt becomes a system message; tools become function tools;
- assistant ``tool_use`` blocks become ``tool_calls``, and ``tool_result``
  blocks become ``tool`` messages;
- the reply's ``tool_calls`` come back as ``tool_use`` blocks, with
  ``stop_reason`` "tool_use", "end_turn" or "max_tokens".

Claude-specific options (adaptive thinking, effort, prompt caching, refusal
fallbacks) have no Ollama equivalent and are ignored. Anthropic-hosted tools
(web search / fetch) cannot run locally and are dropped; the news agent uses
the news server's RSS tool instead.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from typing import Any

import httpx


def _get(block: Any, key: str, default: Any = None) -> Any:
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


class _Messages:
    def __init__(self, owner: OllamaClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        return await self._owner.create(**kwargs)


class OllamaClient:
    """Anthropic-style client backed by a local Ollama model."""

    def __init__(self, model: str, host: str = "http://localhost:11434", options: dict[str, Any] | None = None,
                 think: bool = False, timeout: float = 1800.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = dict(options or {})
        self.think = think
        self.timeout = timeout
        self.transport = transport  # tests: httpx.MockTransport
        self.beta = SimpleNamespace(messages=_Messages(self))
        self._ids = itertools.count(1)
        self._tool_names: dict[str, str] = {}  # tool_use id -> tool name, for tool messages

    # --- request -------------------------------------------------------------------
    def _tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            if "input_schema" not in t:  # Anthropic-hosted tool (web search...): not available locally
                continue
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}})
        return out

    def _messages(self, system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            text = system if isinstance(system, str) else "\n\n".join(_get(b, "text", "") for b in system)
            out.append({"role": "system", "content": text})
        for message in messages:
            role, content = message["role"], message["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                text = "\n".join(_get(b, "text", "") for b in content if _get(b, "type") == "text")
                calls = [{"id": _get(b, "id"), "function": {"name": _get(b, "name"), "arguments": dict(_get(b, "input") or {})}}
                         for b in content if _get(b, "type") == "tool_use"]
                entry: dict[str, Any] = {"role": "assistant", "content": text}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
            else:  # user turn: tool results and/or text
                texts = []
                for b in content:
                    kind = _get(b, "type")
                    if kind == "tool_result":
                        result = _get(b, "content")
                        if not isinstance(result, str):
                            result = "\n".join(_get(c, "text", "") for c in result)
                        if _get(b, "is_error"):
                            result = f"ERROR: {result}"
                        out.append({"role": "tool", "content": result,
                                    "tool_name": self._tool_names.get(_get(b, "tool_use_id"), "")})
                    elif kind == "text":
                        texts.append(_get(b, "text", ""))
                if texts:
                    out.append({"role": "user", "content": "\n".join(texts)})
        return out

    # --- call --------------------------------------------------------------------------
    async def create(self, *, model: str | None = None, max_tokens: int = 16_000, system: Any = None,
                     messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                     **_ignored: Any) -> SimpleNamespace:
        body = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "tools": self._tools(tools or []),
            "stream": False,
            "think": self.think,
            "options": {"num_predict": max_tokens, **self.options},
        }
        async with httpx.AsyncClient(base_url=self.host, timeout=self.timeout, transport=self.transport) as http:
            response = await http.post("/api/chat", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama {response.status_code}: {response.text[:500]}")
        return self._response(response.json())

    def _response(self, data: dict[str, Any]) -> SimpleNamespace:
        message = data.get("message", {})
        content: list[SimpleNamespace] = []
        text = (message.get("content") or "").strip()
        if text:
            content.append(SimpleNamespace(type="text", text=text))
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):  # some models return a JSON string
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_unparsed": arguments}
            call_id = call.get("id") or f"call_{next(self._ids)}"
            self._tool_names[call_id] = function.get("name", "")
            content.append(SimpleNamespace(type="tool_use", id=call_id, name=function.get("name", ""), input=arguments))
        if any(b.type == "tool_use" for b in content):
            stop_reason = "tool_use"
        elif data.get("done_reason") == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
        usage = SimpleNamespace(input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0),
                                cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, stop_details=None,
                               model=data.get("model", self.model))
