"""Run the agents on a local model served by Ollama.

``OllamaClient`` exposes the one method the agent runtime uses,
``beta.messages.create(...)``, with Anthropic Messages-shaped requests and
responses, and translates them to Ollama's native chat API (``/api/chat``):

- the system prompt becomes a system message; tools become function tools;
- assistant ``tool_use`` blocks become ``tool_calls``, and ``tool_result``
  blocks become ``tool`` messages;
- the reply's ``tool_calls`` come back as ``tool_use`` blocks, with
  ``stop_reason`` "tool_use", "end_turn" or "max_tokens".

``beta.messages.stream(...)`` streams the reply like the Anthropic SDK's
stream helper: iterating it yields ``content_block_delta`` events (text and
thinking deltas) as the model writes, and ``get_final_message()`` returns the
complete response.

Claude-specific options (adaptive thinking, effort, prompt caching, refusal
fallbacks) have no Ollama equivalent and are ignored. Anthropic-hosted tools
(web search / fetch) cannot run locally and are dropped; the news agent uses
the news server's RSS tool instead.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator
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

    def stream(self, **kwargs: Any) -> _Stream:
        return _Stream(self._owner, kwargs)


class _Stream:
    """A streamed reply: an async context manager and iterator of delta events."""

    def __init__(self, owner: OllamaClient, kwargs: dict[str, Any]) -> None:
        self._owner = owner
        self._kwargs = kwargs
        self._final: SimpleNamespace | None = None

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        data: dict[str, Any] = {}
        async for chunk in self._owner._chat(self._kwargs, stream=True):
            message = chunk.get("message") or {}
            for kind, key in (("thinking", "thinking"), ("text", "content")):
                if message.get(key):
                    yield SimpleNamespace(type="content_block_delta",
                                          delta=SimpleNamespace(type=f"{kind}_delta", **{kind: message[key]}))
            _merge(data, chunk)
        self._final = self._owner._response(data)

    async def get_final_message(self) -> SimpleNamespace:
        if self._final is None:
            async for _ in self:
                pass
        assert self._final is not None
        return self._final


def _merge(data: dict[str, Any], chunk: dict[str, Any]) -> None:
    """Accumulates one streamed chunk: message text and tool calls append, the rest is replaced."""
    message = data.setdefault("message", {"role": "assistant", "content": "", "thinking": ""})
    piece = chunk.get("message") or {}
    message["content"] += piece.get("content") or ""
    message["thinking"] += piece.get("thinking") or ""
    if piece.get("tool_calls"):
        message.setdefault("tool_calls", []).extend(piece["tool_calls"])
    data.update({k: v for k, v in chunk.items() if k != "message"})


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
    async def create(self, **kwargs: Any) -> SimpleNamespace:
        data: dict[str, Any] = {}
        async for chunk in self._chat(kwargs, stream=False):
            _merge(data, chunk)
        return self._response(data)

    async def _chat(self, kwargs: dict[str, Any], stream: bool) -> AsyncIterator[dict[str, Any]]:
        """Posts one chat request; yields the reply, or its chunks as they arrive (one JSON object per line)."""
        body = {
            "model": self.model,
            "messages": self._messages(kwargs.get("system"), kwargs["messages"]),
            "tools": self._tools(kwargs.get("tools") or []),
            "stream": stream,
            "think": self.think,
            "options": {"num_predict": kwargs.get("max_tokens", 16_000), **self.options},
        }
        async with httpx.AsyncClient(base_url=self.host, timeout=self.timeout, transport=self.transport) as http:
            async with http.stream("POST", "/api/chat", json=body) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama {response.status_code}: {(await response.aread()).decode()[:500]}")
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise RuntimeError(f"Ollama: {chunk['error']}")
                        yield chunk

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
