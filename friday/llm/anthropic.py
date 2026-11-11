"""Anthropic Messages client — the second real backend, in the same shape as the first.

The mature line shipped one client (`OpenAICompatibleClient`, stdlib `urllib`) and
a mock. The HUD line shipped a streaming Anthropic provider built on `httpx` and
its own `Message` type. This module keeps the *capability* and drops the parallel
universe: it implements the mature `Client` protocol, uses stdlib `urllib` like
every other client here, and returns the same `Response`/`ToolCall` objects the
agent loop already understands.

Two protocol mismatches are handled explicitly rather than papered over:

  * **Tool results.** Anthropic requires a `tool_result` block to answer a
    preceding `tool_use` block with the same id. FRIDAY's transcript keeps only
    `{"role": "tool", "name", "content"}` (the core does not replay assistant
    tool-call turns). The converter therefore reconstructs a paired
    `assistant(tool_use)` → `user(tool_result)` exchange. Tool inputs are `{}` in
    that reconstruction: the *content* of the result is what the model needs to
    continue, and the inputs were already executed.
  * **Streaming tool calls.** `arguments` arrive as `input_json_delta` fragments
    that are not valid JSON until the block closes. They are accumulated per
    content-block index and parsed once, exactly like the OpenAI reader.

No key configured means this client is never selected. `alive()` is a capability
check, not a network call — a HUD that hangs for 2 s on every `/api/config` would
be worse than one that says "not configured".
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .client import Response, ToolCall

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL = os.environ.get("FRIDAY_ANTHROPIC_MODEL", "claude-sonnet-4-5")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


class AnthropicClient:
    """`Client` protocol over the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_max_tokens = max_tokens
        self.name = f"anthropic:{self.model}"
        #: Cached so the HUD can say "anthropic, not configured" without a request.
        self._capable = bool(self.api_key)

    # ── capability ─────────────────────────────────────────────────────────────
    def alive(self) -> bool:
        return self._capable

    # ── conversion ─────────────────────────────────────────────────────────────
    @staticmethod
    def convert(messages: list[dict]) -> tuple[str, list[dict]]:
        """OpenAI-ish message list → (system, Anthropic messages).

        Consecutive `tool` messages are grouped into ONE assistant turn with N
        `tool_use` blocks followed by ONE user turn with N `tool_result` blocks,
        because that is the shape the API will accept.
        """
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        pending: list[dict] = []

        def push(role: str, blocks: list[dict]) -> None:
            if out and out[-1]["role"] == role == "user":
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        def flush_tools() -> None:
            nonlocal pending
            if not pending:
                return
            uses, results = [], []
            for i, m in enumerate(pending):
                tid = m.get("tool_call_id") or f"toolu_{i}_{abs(hash(m.get('name', 'tool'))) % 10**8}"
                uses.append({"type": "tool_use", "id": tid,
                             "name": m.get("name") or "tool", "input": {}})
                results.append({"type": "tool_result", "tool_use_id": tid,
                                "content": m.get("content") or ""})
            out.append({"role": "assistant", "content": uses})
            out.append({"role": "user", "content": results})
            pending = []

        for m in messages:
            role = m.get("role")
            if role == "system":
                text = _text_of(m.get("content"))
                if text:
                    system_parts.append(text)
                continue
            if role == "tool":
                pending.append(m)
                continue
            flush_tools()
            if role == "user":
                push("user", [{"type": "text", "text": _text_of(m.get("content")) or " "}])
            elif role == "assistant":
                blocks: list[dict] = []
                text = _text_of(m.get("content"))
                if text:
                    blocks.append({"type": "text", "text": text})
                for call in m.get("tool_calls") or []:
                    blocks.append({
                        "type": "tool_use",
                        "id": call.get("id") or "toolu_call",
                        "name": call.get("name") or "tool",
                        "input": call.get("arguments") or {},
                    })
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
        flush_tools()

        if not out or out[0]["role"] != "user":
            out.insert(0, {"role": "user",
                           "content": [{"type": "text", "text": "(session opened)"}]})
        return "\n\n".join(system_parts), out

    @staticmethod
    def tool_schema(tools: list[dict] | None) -> list[dict]:
        out = []
        for tool in tools or []:
            fn = tool.get("function", tool)
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    # ── request ────────────────────────────────────────────────────────────────
    def chat(self, messages, *, tools=None, temperature: float = 0.2,
             max_tokens: int = 768,
             on_token: Callable[[str], None] | None = None) -> Response:
        system, converted = self.convert(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": min(max_tokens, self.default_max_tokens) if max_tokens else 1024,
            "messages": converted,
            "temperature": max(0.0, min(1.0, temperature)),
            "stream": on_token is not None,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self.tool_schema(tools)

        req = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                if on_token is not None:
                    return self._read_stream(r, on_token)
                return self._parse_response(json.loads(r.read().decode()))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            raise RuntimeError(f"Anthropic HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach Anthropic ({e.reason})") from e

    # ── responses ──────────────────────────────────────────────────────────────
    def _parse_response(self, data: dict) -> Response:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(name=block.get("name", ""),
                                      arguments=block.get("input") or {},
                                      id=block.get("id", "")))
        usage = data.get("usage") or {}
        return Response(
            text="".join(text_parts),
            tool_calls=calls,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            raw=data,
        )

    def _read_stream(self, response, on_token: Callable[[str], None]) -> Response:
        import time

        t0 = time.perf_counter()
        text_parts: list[str] = []
        blocks: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        model = self.model
        ttft: int | None = None
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "message_start":
                info = event.get("message") or {}
                model = info.get("model", model)
                usage.update(info.get("usage") or {})
            elif kind == "content_block_start":
                idx = int(event.get("index", 0))
                block = event.get("content_block") or {}
                blocks[idx] = {"type": block.get("type"), "text": "",
                               "id": block.get("id", ""), "name": block.get("name", ""),
                               "json": ""}
            elif kind == "content_block_delta":
                idx = int(event.get("index", 0))
                delta = event.get("delta") or {}
                slot = blocks.setdefault(idx, {"type": delta.get("type"), "text": "",
                                               "id": "", "name": "", "json": ""})
                if delta.get("type") == "text_delta":
                    piece = delta.get("text", "")
                    if piece:
                        if ttft is None:
                            ttft = int((time.perf_counter() - t0) * 1000)
                        text_parts.append(piece)
                        on_token(piece)
                elif delta.get("type") == "input_json_delta":
                    slot["json"] += delta.get("partial_json", "")
            elif kind == "message_delta":
                usage.update((event.get("delta") or {}).get("usage")
                             or event.get("usage") or {})

        calls: list[ToolCall] = []
        for idx in sorted(blocks):
            block = blocks[idx]
            if block.get("type") != "tool_use":
                continue
            try:
                args = json.loads(block["json"]) if block["json"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(name=block.get("name", ""), arguments=args,
                                  id=block.get("id", "")))
        return Response(
            text="".join(text_parts),
            tool_calls=calls,
            model=model,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            ttft_ms=ttft,
            total_ms=int((time.perf_counter() - t0) * 1000),
            raw={"streamed": True},
        )
