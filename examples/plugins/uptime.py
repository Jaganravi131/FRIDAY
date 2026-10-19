"""Example FRIDAY plugin: host uptime.

Enable it with:

    FRIDAY_PLUGIN_DIRS=./examples/plugins friday serve

Then ask FRIDAY for the host uptime. See docs/PLUGINS.md for the full contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from friday.tools.base import Tool, ToolContext, ToolError
from friday.types import ToolResult


class UptimeTool(Tool):
    name = "uptime"
    description = (
        "Return how long this host has been running. Use when the user asks about host uptime, "
        "time since boot, or general system freshness."
    )
    risk = "low"
    timeout = 10.0
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "unit": {
                "type": "string",
                "enum": ["hours", "days"],
                "description": "Preferred unit for the answer.",
                "default": "hours",
            }
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            seconds = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError) as exc:
            raise ToolError(f"Uptime is not available on this platform: {exc}") from exc

        unit = str(args.get("unit") or "hours")
        hours = seconds / 3600
        value = hours / 24 if unit == "days" else hours
        return ToolResult(
            ok=True,
            output=f"Host uptime: {value:.1f} {unit} ({seconds:.0f}s).",
            data={"seconds": seconds, "unit": unit},
        )

    def summarize(self, args: dict[str, Any]) -> str:
        return f"uptime({args.get('unit', 'hours')})"


TOOLS = [UptimeTool()]
