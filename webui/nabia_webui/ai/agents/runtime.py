"""Agent loop: Claude + tools from local MCP servers (stdio) + Claude's server tools.

A manual async loop rather than the SDK's tool runner, because one agent talks
to several MCP server processes at once, mixes their tools with Anthropic-hosted
tools (web search / fetch, which can pause a turn), and records every tool call
for the audit trail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from ..config import AIConfig

log = logging.getLogger(__name__)
#: The model's output as it streams (its own logger, so the job log shows it as "model").
model_log = logging.getLogger("nabia_webui.ai.agents.model")

MAX_TOKENS = 16_000


def _preview(value: Any, size: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= size else text[:size] + "…"


@dataclass(frozen=True)
class MCPServerSpec:
    """A nabia MCP server to start for an agent (``python -m <module> --config ... --run ...``)."""

    module: str
    read_only: bool = False


NEWS_SERVER = "nabia_webui.ai.servers.news"
STATS_SERVER = "nabia_webui.ai.servers.stats"
PORTFOLIO_SERVER = "nabia_webui.ai.servers.portfolio"


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    is_error: bool
    output: str


@dataclass
class AgentRun:
    """Transcript summary of one agent run."""

    agent: str
    as_of: str
    final_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    server_tool_calls: int = 0
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)


class LiveOutput:
    """Logs a streamed reply while it is written: a line per paragraph, or every
    few seconds / few hundred characters for long lines, so a job log follows
    the model as it thinks and writes."""

    def __init__(self, agent: str, max_chars: int = 400, max_seconds: float = 3.0) -> None:
        self.agent = agent
        self.max_chars = max_chars
        self.max_seconds = max_seconds
        self.kind = ""
        self.buffer = ""
        self.last = time.monotonic()
        self.chars = 0

    def add(self, kind: str, text: str) -> None:
        if kind != self.kind:
            self.flush()
            self.kind = kind
        self.buffer += text
        self.chars += len(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._emit(line)
        slow = len(self.buffer) >= self.max_chars // 5 and time.monotonic() - self.last >= self.max_seconds
        if len(self.buffer) >= self.max_chars or slow:  # long lines in pieces; slow streams every few seconds
            self._emit(self.buffer)
            self.buffer = ""

    def flush(self) -> None:
        if self.buffer:
            self._emit(self.buffer)
        self.buffer = ""

    def _emit(self, text: str) -> None:
        self.last = time.monotonic()
        if text.strip():
            label = "thinking" if self.kind == "thinking" else "writes"
            model_log.info("%s %s: %s", self.agent, label, text.strip())


class AgentRefusedError(RuntimeError):
    """The model declined the request (stop_reason "refusal")."""


class AgentRuntime:
    """Runs agents against the nabia MCP servers.

    ``client`` is an ``anthropic.AsyncAnthropic`` (created on first use if not
    given); tests pass a scripted stand-in with the same ``beta.messages.create``.
    """

    def __init__(self, config: AIConfig, config_path: str | Path, run: str = "live", client: Any = None,
                 max_turns: int = 40) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.run_name = run
        self.max_turns = max_turns
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if self.config.provider == "ollama":
                from .ollama import OllamaClient

                self._client = OllamaClient(self.config.ollama_model, self.config.ollama_host,
                                            self.config.ollama_options, self.config.ollama_think)
            else:
                import anthropic

                self._client = anthropic.AsyncAnthropic()
        return self._client

    @property
    def local(self) -> bool:
        return self.config.provider == "ollama"

    def _server_parameters(self, spec: MCPServerSpec) -> StdioServerParameters:
        args = ["-m", spec.module, "--config", str(self.config_path), "--run", self.run_name]
        if spec.read_only:
            args.append("--read-only")
        env = get_default_environment()
        # The servers import nabia: give them the same module search path as this process.
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        return StdioServerParameters(command=sys.executable, args=args, env=env)

    async def run(self, agent: str, as_of: str, system: str, prompt: str, servers: list[MCPServerSpec],
                  server_tools: list[dict[str, Any]] | None = None, effort: str = "high") -> AgentRun:
        """Runs one agent to completion and returns its transcript summary."""
        record = AgentRun(agent=agent, as_of=as_of)
        failure: Exception | None = None
        async with AsyncExitStack() as stack:
            sessions: dict[str, ClientSession] = {}
            tools: list[dict[str, Any]] = []
            log.info("%s (as of %s): starting MCP servers %s", agent, as_of, ", ".join(s.module.rsplit(".", 1)[-1]
                     + (" (read-only)" if s.read_only else "") for s in servers))
            for spec in servers:
                read, write = await stack.enter_async_context(
                    stdio_client(self._server_parameters(spec), errlog=sys.stderr))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                for t in (await session.list_tools()).tools:
                    if t.name in sessions:
                        raise RuntimeError(f"tool {t.name} is provided by two servers")
                    sessions[t.name] = session
                    tools.append({"name": t.name, "description": t.description or "", "input_schema": t.input_schema})
            tools += server_tools or []
            log.info("%s: %d tools available; model %s", agent, len(tools), self.config.model_label)
            try:
                await self._loop(record, system, prompt, tools, sessions, effort)
            except Exception as error:
                # Close the MCP sessions cleanly first: raised inside them, the error
                # would reach the caller wrapped in an ExceptionGroup by their task groups.
                failure = error
        if failure is not None:
            raise failure
        return record

    async def _loop(self, record: AgentRun, system: str, prompt: str, tools: list[dict[str, Any]],
                    sessions: dict[str, ClientSession], effort: str) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for turn in range(1, self.max_turns + 1):
            log.info("%s: turn %d, waiting for the model…", record.agent, turn)
            started = time.monotonic()
            live = LiveOutput(record.agent)
            response = await self._create(live, dict(
                model=self.config.model,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                cache_control={"type": "ephemeral"},  # caches the stable prefix: tools, system, history
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # re-run a refused request on a fallback model
            ))
            record.turns += 1
            self._add_usage(record, response.usage)
            log.info("%s: turn %d -> %s (%s in / %s out tokens, %.1fs)", record.agent, turn, response.stop_reason,
                     getattr(response.usage, "input_tokens", "?"), getattr(response.usage, "output_tokens", "?"),
                     time.monotonic() - started)
            for block in response.content:
                if block.type == "text" and block.text.strip() and not live.chars and response.stop_reason == "tool_use":
                    log.info("%s says: %s", record.agent, _preview(block.text, 300))  # not streamed
                elif block.type == "server_tool_use":
                    log.info("%s -> %s %s", record.agent, block.name, _preview(getattr(block, "input", {})))
            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                raise AgentRefusedError(f"{record.agent} refused: {getattr(details, 'category', None)}")
            messages.append({"role": "assistant", "content": response.content})
            record.server_tool_calls += sum(1 for b in response.content if b.type == "server_tool_use")
            if response.stop_reason == "pause_turn":  # a long server-tool turn: resend to resume it
                continue
            if response.stop_reason == "max_tokens":
                raise RuntimeError(f"{record.agent}: response hit max_tokens ({MAX_TOKENS})")
            if response.stop_reason != "tool_use":
                record.final_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                log.info("%s finished after %d turns, %d tool calls: %s", record.agent, record.turns,
                         len(record.tool_calls), _preview(record.final_text, 400))
                return
            calls = [b for b in response.content if b.type == "tool_use"]
            results = await asyncio.gather(*(self._call_tool(record, sessions, b) for b in calls))
            messages.append({"role": "user", "content": list(results)})  # all results in one message
        raise RuntimeError(f"{record.agent}: no final answer after {self.max_turns} turns")

    async def _create(self, live: LiveOutput, request: dict[str, Any]) -> Any:
        """One model turn, streamed when the client supports it, so the output is
        logged as it is written; clients with only ``create`` (tests) are awaited whole."""
        messages_api = self.client.beta.messages
        if not hasattr(messages_api, "stream"):
            return await messages_api.create(**request)
        async with messages_api.stream(**request) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        live.add("text", delta.text)
                    elif delta.type == "thinking_delta":
                        live.add("thinking", delta.thinking)
            response = await stream.get_final_message()
        live.flush()
        return response

    async def _call_tool(self, record: AgentRun, sessions: dict[str, ClientSession], block: Any) -> dict[str, Any]:
        session = sessions.get(block.name)
        log.info("%s -> %s(%s)", record.agent, block.name, _preview(dict(block.input)))
        try:
            if session is None:
                raise ValueError(f"unknown tool {block.name}")
            result = await session.call_tool(block.name, dict(block.input))
            text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
            if not text and getattr(result, "structured_content", None) is not None:
                text = json.dumps(result.structured_content)
            is_error = bool(result.is_error)
        except Exception as error:  # a broken server must not kill the agent: report it as a tool error
            text, is_error = f"{type(error).__name__}: {error}", True
        record.tool_calls.append(ToolCall(block.name, dict(block.input), is_error, text[:2000]))
        if is_error:
            log.warning("%s <- %s failed: %s", record.agent, block.name, _preview(text, 300))
        else:
            log.info("%s <- %s: %s", record.agent, block.name, _preview(text))
        return {"type": "tool_result", "tool_use_id": block.id, "content": text or "(no output)", "is_error": is_error}

    @staticmethod
    def _add_usage(record: AgentRun, usage: Any) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            record.usage[key] = record.usage.get(key, 0) + int(getattr(usage, key, 0) or 0)
