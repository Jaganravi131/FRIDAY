"""Anthropic Messages API provider with streaming and tool use."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from friday.llm.base import (
    Delta,
    FinishDelta,
    LLMProvider,
    ProviderError,
    TextDelta,
    ToolCallDelta,
    UsageDelta,
)
from friday.types import Message, ToolSchema, Usage

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings):
        super().__init__(settings)
        self.base_url = (settings.base_url or "https://api.anthropic.com/v1").rstrip("/")

    # ------------------------------------------------------------- conversion
    @staticmethod
    def _convert(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if converted and converted[-1]["role"] == role == "user":
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == "user":
                push("user", [{"type": "text", "text": message.content or " "}])

            elif message.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.parsed_arguments(),
                        }
                    )
                if blocks:
                    converted.append({"role": "assistant", "content": blocks})

            elif message.role == "tool":
                push(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or "",
                            "content": message.content or "",
                            "is_error": message.meta.get("ok") is False,
                        }
                    ],
                )

        # Anthropic requires the conversation to begin with a user turn.
        if not converted or converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": [{"type": "text", "text": "(session opened)"}]})
        return "\n\n".join(system_parts), converted

    # ---------------------------------------------------------------- request
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Delta]:
        if not self.settings.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set", status=401)
        if not self.model:
            raise ProviderError("no Anthropic model configured")

        system, converted = self._convert(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "messages": converted,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]

        headers = {
            "x-api-key": self.settings.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        timeout = httpx.Timeout(self.settings.request_timeout, connect=20.0)

        # Map Anthropic content-block index -> tool-call index used by the agent.
        tool_slots: dict[int, int] = {}
        tool_count = 0
        input_tokens = 0
        output_tokens = 0

        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", f"{self.base_url}/messages", headers=headers, json=payload) as response,
            ):
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"anthropic HTTP {response.status_code}: {_short(body)}",
                        status=response.status_code,
                        retryable=response.status_code in {408, 409, 429, 500, 502, 503, 504},
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    kind = event.get("type")
                    if kind == "message_start":
                        usage = (event.get("message") or {}).get("usage") or {}
                        input_tokens = int(usage.get("input_tokens") or 0)

                    elif kind == "content_block_start":
                        block = event.get("content_block") or {}
                        index = int(event.get("index", 0))
                        if block.get("type") == "tool_use":
                            tool_slots[index] = tool_count
                            yield ToolCallDelta(
                                index=tool_count,
                                id=block.get("id"),
                                name=block.get("name"),
                                arguments="",
                            )
                            tool_count += 1

                    elif kind == "content_block_delta":
                        index = int(event.get("index", 0))
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            yield TextDelta(str(delta.get("text") or ""))
                        elif delta.get("type") == "input_json_delta":
                            slot = tool_slots.get(index)
                            if slot is not None:
                                yield ToolCallDelta(
                                    index=slot, arguments=str(delta.get("partial_json") or "")
                                )

                    elif kind == "message_delta":
                        usage = event.get("usage") or {}
                        output_tokens = int(usage.get("output_tokens") or 0)
                        reason = (event.get("delta") or {}).get("stop_reason")
                        if reason:
                            yield FinishDelta(reason=reason)

                    elif kind == "error":
                        error = event.get("error") or {}
                        raise ProviderError(
                            f"anthropic error: {error.get('message') or 'unknown'}", retryable=False
                        )

                yield UsageDelta(Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens))
        except httpx.TimeoutException as exc:
            raise ProviderError(f"anthropic timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic connection failed: {exc}", retryable=True) from exc


def _short(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")
