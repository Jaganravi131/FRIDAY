"""System telemetry, clock and safe arithmetic."""

from __future__ import annotations

import ast
import math
import os
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from friday.tools.base import Tool, ToolContext, ToolError
from friday.types import ToolResult


class ClockTool(Tool):
    name = "clock"
    description = "Current date and time (UTC by default, or an IANA timezone such as Asia/Kolkata)."
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "IANA timezone name.", "default": "UTC"}
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        zone = str(args.get("timezone") or "UTC")
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(zone)
        except Exception as exc:  # noqa: BLE001 - bad tz name is user error
            raise ToolError(f"Unknown timezone '{zone}'.") from exc
        now = datetime.now(tz)
        return ToolResult(
            ok=True,
            output=f"{now.strftime('%A, %d %B %Y — %H:%M:%S %Z')} ({zone})",
            data={"iso": now.isoformat(), "timezone": zone},
        )


class SystemStatusTool(Tool):
    name = "system_status"
    description = "Live CPU, memory, disk, uptime, platform and FRIDAY process telemetry."
    risk = "low"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        lines: list[str] = []
        total_mem, available_mem = _memory_info()
        if total_mem:
            used = total_mem - available_mem
            lines.append(f"Memory   : {_gb(used)} / {_gb(total_mem)} used ({used / total_mem * 100:.0f}%)")
        load = _load_average()
        if load is not None:
            lines.append(f"Load avg : {load:.2f} (1 min) on {os.cpu_count() or 1} cores")
        else:
            lines.append(f"CPU cores: {os.cpu_count() or 1}")
        disk = shutil.disk_usage(ctx.workspace)
        lines.append(
            f"Disk     : {_gb(disk.used)} / {_gb(disk.total)} used ({disk.used / disk.total * 100:.0f}%)"
        )
        if os.name == "posix":
            uptime = _uptime_seconds()
            if uptime:
                lines.append(f"Uptime   : {_duration(uptime)}")
        lines.append(f"Platform : {sys.platform} · Python {sys.version.split()[0]}")
        lines.append(f"Host     : {socket.gethostname()}")
        lines.append(f"Workspace: {ctx.workspace}")
        db = ctx.settings.db_file
        if db.exists():
            lines.append(f"Memory DB: {db} ({db.stat().st_size / 1024:.1f} KB)")
        lines.append(f"Provider : {ctx.settings.provider} · {ctx.settings.model}")
        return ToolResult(ok=True, output="\n".join(lines))


class MathTool(Tool):
    name = "math"
    description = (
        "Evaluate an arithmetic expression safely (no code execution). Supports + - * / // % **, "
        "sqrt, log, exp, sin, cos, tan, abs, round, min, max, pi and e."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '(17 * 42 + 9) / 3'"}},
        "required": ["expression"],
    }

    _FUNCTIONS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "floor": math.floor,
        "ceil": math.ceil,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
    }
    _CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        expression = str(args["expression"]).strip()
        if len(expression) > 500:
            raise ToolError("Expression too long.")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"Could not parse expression: {exc.msg}") from exc
        value = self._eval(tree.body)
        display = int(value) if isinstance(value, float) and value.is_integer() else value
        return ToolResult(ok=True, output=f"{expression} = {display}", data={"value": display})

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self._CONSTANTS:
                return self._CONSTANTS[node.id]
            raise ToolError(f"Unknown name '{node.id}'.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a**b,
            }
            op = ops.get(type(node.op))
            if op is None:
                raise ToolError("Operator not allowed.")
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ToolError("Exponent too large.")
            return op(left, right)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._FUNCTIONS:
                raise ToolError("Function not allowed.")
            if node.keywords:
                raise ToolError("Keyword arguments are not supported.")
            return self._FUNCTIONS[node.func.id](*[self._eval(arg) for arg in node.args])
        raise ToolError("Expression contains unsupported syntax.")


def telemetry() -> dict[str, Any]:
    """Lightweight system snapshot for the HUD; never throws."""
    total, available = _memory_info()
    try:
        disk = shutil.disk_usage(Path.cwd())
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "used_pct": round(disk.used / disk.total * 100, 1) if disk.total else 0.0,
        }
    except OSError:
        disk_info = {"total": 0, "used": 0, "free": 0, "used_pct": 0.0}
    used = max(0, total - available)
    return {
        "cpu_count": os.cpu_count() or 1,
        "load_1m": _load_average(),
        "memory": {
            "total": total,
            "available": available,
            "used": used,
            "used_pct": round(used / total * 100, 1) if total else 0.0,
        },
        "disk": disk_info,
        "uptime_seconds": _uptime_seconds(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
    }


def _memory_info() -> tuple[int, int]:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in {"MemTotal", "MemAvailable"}:
                    info[key] = int(rest.strip().split()[0]) * 1024
        if "MemTotal" in info:
            return info["MemTotal"], info.get("MemAvailable", 0)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource  # noqa: PLC0415

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return 0, int(usage.ru_maxrss) * 1024
    except Exception:  # noqa: BLE001
        return 0, 0


def _load_average() -> float | None:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def _uptime_seconds() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _gb(value: int) -> str:
    return f"{value / 1024**3:.1f} GB"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
