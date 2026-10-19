"""Tool registry assembly and plugin loading."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from friday.config import Settings
from friday.tools.base import Tool, ToolContext, ToolError, ToolRegistry, validate_arguments
from friday.tools.files import EditFileTool, GrepFilesTool, ListFilesTool, ReadFileTool, WriteFileTool
from friday.tools.memory_tools import ForgetMemoryTool, RecallMemoryTool, RememberTool, SearchHistoryTool
from friday.tools.shell import PythonExecTool, ShellTool
from friday.tools.system import ClockTool, MathTool, SystemStatusTool
from friday.tools.web import FetchUrlTool, WebSearchTool

BUILTIN_TOOLS: list[type[Tool]] = [
    ClockTool,
    SystemStatusTool,
    MathTool,
    ListFilesTool,
    ReadFileTool,
    GrepFilesTool,
    WriteFileTool,
    EditFileTool,
    ShellTool,
    PythonExecTool,
    FetchUrlTool,
    WebSearchTool,
    RememberTool,
    RecallMemoryTool,
    ForgetMemoryTool,
    SearchHistoryTool,
]


def load_plugins(registry: ToolRegistry, directories: list[str]) -> list[str]:
    """Import `*.py` plugin modules; each may expose a module-level TOOLS list."""
    loaded: list[str] = []
    for directory in directories:
        path = Path(directory).expanduser()
        if not path.is_dir():
            continue
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            module_name = f"friday_plugin_{file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not kill the agent
                print(f"[friday] plugin {file} failed to load: {exc}", file=sys.stderr)
                sys.modules.pop(module_name, None)
                continue
            tools = getattr(module, "TOOLS", None) or []
            for tool in tools:
                if isinstance(tool, Tool):
                    registry.register(tool)
                    loaded.append(tool.name)
    return loaded


def build_registry(settings: Settings) -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry([cls() for cls in BUILTIN_TOOLS])
    loaded = load_plugins(registry, settings.plugin_dirs)
    for name in settings.disabled_tools:
        registry.unregister(name)
    return registry, loaded


__all__ = [
    "BUILTIN_TOOLS",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "build_registry",
    "load_plugins",
    "validate_arguments",
]
