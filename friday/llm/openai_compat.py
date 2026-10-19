"""OpenAI-compatible streaming provider (OpenAI, OpenRouter, Groq, Ollama, LM Studio…)."""

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

_NATIVE_USAGE = {"openai", "openrouter", "groq"}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings, *, name: str | None = None):
        super().__init__(settings)
        self.name = name or settings.provider
        self.base_url = settings.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Jaganravi131/FRIDAY"
            headers["X-Title"] = "FRIDAY"
        return headers

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
            "stream": True,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_wire() for t in tools]
            payload["tool_choice"] = "auto"
        if self.name in _NATIVE_USAGE:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Delta]:
        if not self.model:
            raise ProviderError(f"no model configured for provider '{self.name}'")
        if not self.base_url:
            raise ProviderError(f"no base_url configured for provider '{self.name}'")

        url = f"{self.base_url}/chat/completions"
        payload = self._payload(messages, tools, temperature, max_tokens)
        timeout = httpx.Timeout(self.settings.request_timeout, connect=20.0)

        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, headers=self._headers(), json=payload) as response,
            ):
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"{self.name} HTTP {response.status_code}: {_short(body)}",
                        status=response.status_code,
                        retryable=response.status_code in {408, 409, 429, 500, 502, 503, 504},
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"] or {}
                        yield UsageDelta(
                            Usage(
                                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                completion_tokens=int(usage.get("completion_tokens") or 0),
                            )
                        )

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    content = delta.get("content")
                    if content:
                        yield TextDelta(str(content))

                    for call in delta.get("tool_calls") or []:
                        function = call.get("function") or {}
                        yield ToolCallDelta(
                            index=int(call.get("index", 0)),
                            id=call.get("id"),
                            name=function.get("name"),
                            arguments=function.get("arguments") or "",
                        )

                    if choice.get("finish_reason"):
                        yield FinishDelta(reason=choice["finish_reason"])
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name} timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} connection failed: {exc}", retryable=True) from exc


def _short(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")
