"""Guarded shell and Python execution tools.

These run with the privileges of the FRIDAY process. They are marked high-risk so the
approval gate stops them in 'dangerous' and 'always' modes, and they refuse a known set
of destructive commands outright. This is a seatbelt, not a hyperbaric chamber: run
FRIDAY in a container if you are exposing it to untrusted prompts.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import tempfile
from typing import Any

from friday.tools.base import Tool, ToolContext, ToolError
from friday.tools.files import resolve_in_workspace
from friday.types import ToolResult

MAX_OUTPUT = 20_000

DENY_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*(/|/\*|~|~/\*|\$HOME)(\s|$)",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b[^\n]*\bof=/dev/",
    r"\b(shutdown|reboot|poweroff|halt|init\s+0)\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bchmod\s+(-R\s+)?777\s+/(\s|$)",
    r"\b(chown|chgrp)\s+[^\n]*\s/(\s|$)",
    r">\s*/dev/(sd[a-z]|nvme\d|hd[a-z])",
    r"\b(useradd|userdel|passwd|visudo)\b",
    r"\b(curl|wget)\b[^\n|]*\|\s*(ba|z|da)?sh\b",
    r"\bhistory\s+-c\b",
    r"\bkill(all)?\s+-9\s+1\b",
    r"/etc/(shadow|passwd|sudoers)",
    r"\bsudo\b",
]

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I)


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command inside the workspace and return stdout/stderr and exit code. "
        "High-risk and approval-gated. Not for long-running servers; use a short timeout. "
        "Never use it for destructive filesystem operations."
    )
    risk = "high"
    timeout = 130.0
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "description": "Seconds, max 120.", "default": 30},
            "cwd": {"type": "string", "description": "Working directory relative to the workspace."},
        },
        "required": ["command"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args["command"]).strip()
        if not command:
            raise ToolError("Empty command.")
        if len(command) > 4000:
            raise ToolError("Command too long (max 4000 chars).")
        for pattern in DENY_PATTERNS:
            if re.search(pattern, command, re.I):
                raise ToolError(f"Command blocked by safety policy (matched /{pattern}/).")

        cwd = resolve_in_workspace(ctx.workspace, args.get("cwd") or ".", must_exist=True)
        if not cwd.is_dir():
            raise ToolError("cwd must be a directory inside the workspace.")
        timeout = max(1, min(int(args.get("timeout", 30)), 120))

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", tempfile.gettempdir()),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            "CI": "1",
        }

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise ToolError(f"Command timed out after {timeout}s and was killed.") from None

        out = _clean(stdout.decode("utf-8", "replace"))
        err = _clean(stderr.decode("utf-8", "replace"))
        code = process.returncode or 0
        parts = [f"$ {command}", f"exit_code: {code}"]
        if out:
            parts.append(f"--- stdout ---\n{out}")
        if err:
            parts.append(f"--- stderr ---\n{err}")
        if not out and not err:
            parts.append("(no output)")
        return ToolResult(ok=code == 0, output="\n".join(parts), data={"exit_code": code})


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute Python code in an isolated subprocess in the workspace and return its output. "
        "Use for calculations, data wrangling and quick scripts. High-risk and approval-gated. "
        "No network or file access is technically restricted — treat it as running code as yourself."
    )
    risk = "high"
    timeout = 130.0
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
            "timeout": {"type": "integer", "description": "Seconds, max 120.", "default": 30},
        },
        "required": ["code"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = str(args["code"])
        if not code.strip():
            raise ToolError("Empty code.")
        if len(code) > 60_000:
            raise ToolError("Code too large (max 60k chars).")
        timeout = max(1, min(int(args.get("timeout", 30)), 120))

        runtime = ctx.extras.get("python_executable") or shutil.which("python3") or "python3"
        scratch = ctx.settings.data_dir / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", dir=scratch, delete=False, encoding="utf-8"
        ) as handle:
            handle.write(code)
            path = handle.name

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(scratch),
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            process = await asyncio.create_subprocess_exec(
                runtime,
                "-I",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ctx.workspace),
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.communicate()
                raise ToolError(f"Python execution timed out after {timeout}s.") from None
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

        out = _clean(stdout.decode("utf-8", "replace"))
        err = _clean(stderr.decode("utf-8", "replace"))
        code_rc = process.returncode or 0
        parts = [f"exit_code: {code_rc}"]
        if out:
            parts.append(f"--- stdout ---\n{out}")
        if err:
            parts.append(f"--- stderr ---\n{err}")
        if not out and not err:
            parts.append("(no output)")
        return ToolResult(ok=code_rc == 0, output="\n".join(parts), data={"exit_code": code_rc})


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n… output truncated ({len(text) - MAX_OUTPUT} chars omitted)"
    return text
