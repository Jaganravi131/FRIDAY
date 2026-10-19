"""Jailed filesystem tools: everything is confined to the configured workspace."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from friday.tools.base import Tool, ToolContext, ToolError
from friday.types import ToolResult

MAX_READ_BYTES = 400_000
MAX_WRITE_BYTES = 2_000_000
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".friday",
    "dist",
    "build",
    ".ruff_cache",
}


def resolve_in_workspace(workspace: Path, raw: str, *, must_exist: bool = False) -> Path:
    """Resolve a user-supplied path and guarantee it stays inside the workspace."""
    root = Path(workspace).resolve()
    candidate = Path(os.path.expanduser(str(raw or ".")))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolError(f"Path '{raw}' is outside the workspace ({root}). Access denied.")
    if must_exist and not resolved.exists():
        raise ToolError(f"Path '{raw}' does not exist in the workspace.")
    return resolved


def relative_display(workspace: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(workspace).resolve())) or "."
    except ValueError:
        return str(path)


def _human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{num}B"


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List files and directories inside the workspace. Use this to explore the project "
        "before reading or editing anything."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory relative to the workspace root.",
                "default": ".",
            },
            "depth": {"type": "integer", "description": "Recursion depth, 1-4.", "default": 1},
            "pattern": {"type": "string", "description": "Optional glob filter, e.g. '*.py'."},
            "limit": {"type": "integer", "description": "Maximum entries to return (default 200)."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = resolve_in_workspace(ctx.workspace, args.get("path", "."), must_exist=True)
        if not root.is_dir():
            raise ToolError(f"'{args.get('path')}' is not a directory.")
        depth = max(1, min(int(args.get("depth", 1)), 4))
        pattern = args.get("pattern")
        limit = max(1, min(int(args.get("limit", 200)), 1000))

        lines: list[str] = []
        entries = 0
        truncated = False
        base_depth = len(root.parts)

        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            level = len(current_path.parts) - base_depth
            if level >= depth:
                dirs[:] = []
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                if pattern and not fnmatch.fnmatch(name, pattern):
                    continue
                path = current_path / name
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{relative_display(ctx.workspace, path):<60} {_human_size(size):>9}")
                entries += 1
                if entries >= limit:
                    truncated = True
                    dirs[:] = []
                    break
            if truncated:
                break

        if not lines:
            return ToolResult(
                ok=True,
                output=f"No matching files under '{relative_display(ctx.workspace, root)}'.",
                data={"count": 0},
            )
        header = f"{relative_display(ctx.workspace, root)} — {entries} entr{'y' if entries == 1 else 'ies'}"
        if truncated:
            header += f" (truncated at {limit})"
        return ToolResult(ok=True, output=header + "\n" + "\n".join(lines), data={"count": entries})


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Returns numbered lines. Use start_line/end_line "
        "for large files."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "start_line": {"type": "integer", "description": "1-based first line.", "default": 1},
            "end_line": {"type": "integer", "description": "1-based last line (inclusive)."},
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_in_workspace(ctx.workspace, args["path"], must_exist=True)
        if path.is_dir():
            raise ToolError(f"'{args['path']}' is a directory. Use list_files instead.")
        if path.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"File is larger than {_human_size(MAX_READ_BYTES)}; refusing to load it whole.")

        raw = path.read_bytes()
        if b"\x00" in raw[:8000]:
            raise ToolError(f"'{args['path']}' looks like a binary file.")
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()

        start = max(1, int(args.get("start_line", 1)))
        end = int(args.get("end_line") or len(lines))
        end = min(len(lines), max(start, end))
        chunk = lines[start - 1 : end]

        numbered = "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(chunk))
        note = ""
        if end < len(lines):
            note = f"\n… {len(lines) - end} more lines (call again with start_line={end + 1})."
        return ToolResult(
            ok=True,
            output=f"{relative_display(ctx.workspace, path)} ({len(lines)} lines)\n{numbered}{note}",
            data={"path": relative_display(ctx.workspace, path), "lines": len(lines)},
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or overwrite a text file inside the workspace. For small, surgical changes prefer "
        "edit_file. Requires user approval in 'dangerous' and 'always' approval modes."
    )
    risk = "high"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "content": {"type": "string", "description": "Full file content to write."},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
        },
        "required": ["path", "content"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        content = str(args["content"])
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ToolError(f"Refusing to write more than {_human_size(MAX_WRITE_BYTES)} in one call.")
        path = resolve_in_workspace(ctx.workspace, args["path"])
        if path.exists() and path.is_dir():
            raise ToolError(f"'{args['path']}' is a directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = args.get("mode", "overwrite")
        existed = path.exists()
        with open(path, "a" if mode == "append" else "w", encoding="utf-8") as handle:
            handle.write(content)
        verb = "Appended to" if mode == "append" and existed else "Wrote"
        return ToolResult(
            ok=True,
            output=f"{verb} {relative_display(ctx.workspace, path)} ({len(content)} chars).",
            data={"path": relative_display(ctx.workspace, path)},
        )


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a workspace text file. Safer than write_file for small edits. "
        "The old_text must appear exactly once unless replace_all is true."
    )
    risk = "high"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text (empty string deletes)."},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_text", "new_text"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_in_workspace(ctx.workspace, args["path"], must_exist=True)
        if path.is_dir():
            raise ToolError(f"'{args['path']}' is a directory.")
        text = path.read_text(encoding="utf-8", errors="replace")
        old = str(args["old_text"])
        new = str(args["new_text"])
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolError("old_text was not found in the file. Read the file and match it exactly.")
        if occurrences > 1 and not args.get("replace_all"):
            raise ToolError(
                f"old_text appears {occurrences} times. Add more context or set replace_all=true."
            )
        updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Edited {relative_display(ctx.workspace, path)} ({occurrences} replacement(s) available, "
            f"{updated.count(new) if new else 0} present after edit).",
            data={"path": relative_display(ctx.workspace, path)},
        )


class GrepFilesTool(Tool):
    name = "grep_files"
    description = (
        "Search workspace files with a regular expression. Returns file, line number and matching line. "
        "Useful for finding symbols before editing."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "glob": {"type": "string", "description": "Filename filter, e.g. '*.py'."},
            "path": {"type": "string", "description": "Directory relative to workspace.", "default": "."},
            "limit": {"type": "integer", "description": "Maximum matches (default 60)."},
            "ignore_case": {"type": "boolean", "default": True},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            regex = re.compile(str(args["pattern"]), re.I if args.get("ignore_case", True) else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regular expression: {exc}") from exc
        root = resolve_in_workspace(ctx.workspace, args.get("path", "."), must_exist=True)
        glob = args.get("glob") or "*"
        limit = max(1, min(int(args.get("limit", 60)), 400))

        matches: list[str] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                if not fnmatch.fnmatch(name, glob):
                    continue
                path = Path(current) / name
                if path.stat().st_size > 800_000:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="strict")
                except (UnicodeDecodeError, OSError):
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        matches.append(
                            f"{relative_display(ctx.workspace, path)}:{number}: {line.strip()[:200]}"
                        )
                        if len(matches) >= limit:
                            break
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break

        if not matches:
            return ToolResult(ok=True, output=f"No matches for /{args['pattern']}/.", data={"count": 0})
        return ToolResult(ok=True, output="\n".join(matches), data={"count": len(matches)})
