"""Language-model providers."""

from __future__ import annotations

from friday.config import Settings
from friday.llm.anthropic import AnthropicProvider
from friday.llm.base import (
    Delta,
    FinishDelta,
    LLMProvider,
    ProviderError,
    TextDelta,
    ToolCallDelta,
    UsageDelta,
)
from friday.llm.openai_compat import OpenAIProvider
from friday.llm.simulator import SimulatorProvider

OPENAI_COMPATIBLE = {"openai", "openrouter", "groq", "ollama"}


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider named by settings, falling back to simulation."""
    if settings.is_simulation or not settings.provider:
        return SimulatorProvider(settings)
    if settings.provider in ("anthropic", "claude"):
        if not settings.api_key:
            return SimulatorProvider(settings)
        return AnthropicProvider(settings)
    if settings.provider in OPENAI_COMPATIBLE:
        if settings.provider != "ollama" and not settings.api_key:
            return SimulatorProvider(settings)
        return OpenAIProvider(settings, name=settings.provider)
    # Custom OpenAI-compatible endpoint (vLLM, LM Studio, Together, DeepSeek, …)
    return OpenAIProvider(settings, name=settings.provider)


__all__ = [
    "AnthropicProvider",
    "Delta",
    "FinishDelta",
    "LLMProvider",
    "OpenAIProvider",
    "ProviderError",
    "SimulatorProvider",
    "TextDelta",
    "ToolCallDelta",
    "UsageDelta",
    "build_provider",
]
