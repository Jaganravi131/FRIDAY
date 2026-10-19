"""Tool contract, registry, argument validation and execution guardrails."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from friday.config import Settings
from friday.memory import MemoryStore
from friday.types import ToolResult, ToolSchema

RISK_LEVELS = {"low", "medium", "high"}


class ToolError(RuntimeError):
    """Raised by tools for expected, user-facing failures."""


@dataclass
class ToolContext:
    """Everything a tool may touch. No globals, no ambient state."""

    settings: Settings
    memory: MemoryStore
    session_id: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def workspace(self):
        return self.settings.workspace


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    risk: str = "low"
    timeout: float = 30.0

    def schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    def is_dangerous(self) -> bool:
        return self.risk in {"medium", "high"}

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def summarize(self, args: dict[str, Any]) -> str:
        """One-line, human-readable intent shown in approvals and the timeline."""
        bits = ", ".join(f"{k}={_short_repr(v)}" for k, v in list(args.items())[:4])
        return f"{self.name}({bits})" if bits else self.name


def _short_repr(value: Any, limit: int = 70) -> str:
    text = repr(value)
    return text[:limit] + "…" if len(text) > limit else text


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Small JSON-Schema subset validator. Returns a list of human-readable errors."""
    errors: list[str] = []
    properties: dict[str, Any] = schema.get("properties") or {}
    required = schema.get("required") or []

    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]

    for key in required:
        if key not in args or args[key] in (None, ""):
            errors.append(f"missing required argument '{key}'")

    types = {
        "string": str,
        "boolean": bool,
        "array": list,
        "object": dict,
        "number": (int, float),
        "integer": int,
    }
    for key, value in args.items():
        spec = properties.get(key)
        if not isinstance(spec, dict) or value is None:
            continue
        expected = spec.get("type")
        if expected in types:
            py_type = types[expected]
            # bool is an int subclass; keep boolean validation strict.
            if (
                expected in ("number", "integer")
                and isinstance(value, bool)
                or not isinstance(value, py_type)
            ):
                errors.append(f"argument '{key}' must be {expected}")
        enum = spec.get("enum")
        if enum and value not in enum:
            errors.append(f"argument '{key}' must be one of {enum}")
    return errors


class ToolRegistry:
    """Ordered collection of tools with safe execution semantics."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        self._order: list[str] = []
        for tool in tools or []:
            self.register(tool)

    # ------------------------------------------------------------- management
    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("register() expects a Tool instance")
        if not tool.name:
            raise ValueError("tool must declare a name")
        if tool.risk not in RISK_LEVELS:
            raise ValueError(f"tool '{tool.name}' has invalid risk level '{tool.risk}'")
        if tool.name not in self._tools:
            self._order.append(tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        if name in self._order:
            self._order.remove(name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self._order]

    def names(self) -> list[str]:
        return list(self._order)

    def schemas(self, disabled: list[str] | None = None) -> list[ToolSchema]:
        disabled = set(disabled or [])
        return [tool.schema() for tool in self.all() if tool.name not in disabled]

    # -------------------------------------------------------------- execution
    async def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output=f"Unknown tool '{name}'. Available: {', '.join(self._order)}")

        errors = validate_arguments(tool.parameters, args)
        if errors:
            return ToolResult(ok=False, output="Invalid arguments: " + "; ".join(errors))

        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=tool.timeout)
        except TimeoutError:
            return ToolResult(ok=False, output=f"Tool '{name}' timed out after {tool.timeout:.0f}s")
        except ToolError as exc:
            return ToolResult(ok=False, output=str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            return ToolResult(ok=False, output=f"Tool '{name}' failed: {type(exc).__name__}: {exc}")

        if not isinstance(result, ToolResult):
            return ToolResult(ok=False, output=f"Tool '{name}' returned an invalid result")
        return result
