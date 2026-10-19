"""Core data types shared by the agent, providers, memory and server."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A model-requested tool invocation (OpenAI wire format)."""

    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


@dataclass
class Message:
    """Internal, provider-neutral chat message."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            wire["content"] = self.content or None
            wire["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        elif self.role == "tool":
            wire["content"] = self.content
            wire["tool_call_id"] = self.tool_call_id
            if self.name:
                wire["name"] = self.name
        else:
            wire["content"] = self.content
        return wire

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        calls = [
            ToolCall(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                arguments=str(item.get("arguments", "{}")),
            )
            for item in data.get("tool_calls") or []
            if isinstance(item, dict)
        ]
        return cls(
            role=data.get("role", "user"),
            content=data.get("content") or "",
            tool_calls=calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            meta=data.get("meta") or {},
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @staticmethod
    def estimate(messages: list[Message]) -> Usage:
        """Rough token estimate used when a provider does not report usage."""
        chars = 0
        for message in messages:
            chars += len(message.content or "")
            for call in message.tool_calls:
                chars += len(call.name) + len(call.arguments) + 8
        return Usage(prompt_tokens=max(1, chars // 4))


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_message(self, call: ToolCall) -> Message:
        return Message(
            role="tool",
            tool_call_id=call.id,
            name=call.name,
            content=self.output,
            meta={"ok": self.ok, **self.data},
        )
