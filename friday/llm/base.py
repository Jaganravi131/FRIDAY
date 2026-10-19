"""Provider contract and streaming delta types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from friday.config import Settings
from friday.types import Message, ToolSchema, Usage


class ProviderError(RuntimeError):
    """A model backend failed. `retryable` is advisory for the agent loop."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class UsageDelta:
    usage: Usage


@dataclass
class FinishDelta:
    reason: str | None = None


Delta = TextDelta | ToolCallDelta | UsageDelta | FinishDelta


class LLMProvider(ABC):
    """A streaming chat-completions backend with tool-calling support."""

    name: str = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.model

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Delta]:
        """Yield deltas for one assistant turn. Must be an async generator."""
        raise NotImplementedError
