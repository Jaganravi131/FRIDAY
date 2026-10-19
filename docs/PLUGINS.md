# Writing a FRIDAY tool

A tool is a class with four promises: a name the model can call, a description a model can
understand, a JSON schema for its arguments, and an `async run()` that returns a `ToolResult`.

Plugins are ordinary Python files loaded from directories listed in `FRIDAY_PLUGIN_DIRS`. Each module
may expose a `TOOLS` list. They run in-process with the server's privileges — **only install plugins
you trust**.

---

## 1. Minimal example

```python
# examples/plugins/uptime.py
from typing import Any

from friday.tools.base import Tool, ToolContext, ToolError
from friday.types import ToolResult


class UptimeTool(Tool):
    name = "uptime"
    description = "Return how long the host has been running, in human units."
    risk = "low"                      # low | medium | high
    timeout = 10.0                    # seconds

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "unit": {
                "type": "string",
                "enum": ["hours", "days"],
                "description": "Preferred unit for the answer.",
            }
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        from pathlib import Path

        try:
            seconds = float(Path("/proc/uptime").read_text().split()[0])
        except (OSError, ValueError) as exc:
            raise ToolError(f"Uptime unavailable on this platform: {exc}") from exc

        hours = seconds / 3600
        value = hours / 24 if args.get("unit") == "days" else hours
        return ToolResult(
            ok=True,
            output=f"Host uptime: {value:.1f} {args.get('unit') or 'hours'}",
            data={"seconds": seconds},
        )

    def summarize(self, args: dict[str, Any]) -> str:
        return f"uptime(unit={args.get('unit', 'hours')})"


TOOLS = [UptimeTool()]
```

Enable it:

```bash
FRIDAY_PLUGIN_DIRS=./examples/plugins friday serve
```

```bash
friday tools | grep uptime
```

---

## 2. The contract

| Attribute | Required | Notes |
|---|---|---|
| `name` | yes | Snake case; must be unique (later registrations replace earlier ones) |
| `description` | yes | The model sees this. Say *when* to use it, not just what it does |
| `parameters` | yes | JSON Schema subset: `type`, `properties`, `required`, `enum` |
| `risk` | no | `low` (default), `medium`, `high` — drives the approval gate |
| `timeout` | no | Seconds; the registry cancels and reports a timeout |

`run(args, ctx)` must return a `ToolResult`:

```python
ToolResult(ok=True, output="text the model reads", data={"any": "structured extras"})
```

- `output` is what the model sees. Keep it compact; the engine truncates at 30 000 characters.
- `data` is passed into the persisted message metadata for the UI and future tooling.
- Raise `ToolError("…")` for expected failures (bad path, missing API key). The message goes to the
  model verbatim. Any other exception is caught by the registry and reported as a tool failure — the
  agent loop survives either way.

`ToolContext` gives you:

```python
ctx.settings    # typed Settings (workspace, timeouts, API keys, …)
ctx.memory      # MemoryStore — remember(), recall(), search_messages()
ctx.session_id  # current session
ctx.workspace   # Path, already resolved
ctx.extras      # dict for server-injected collaborators
```

Use `friday.tools.files.resolve_in_workspace(ctx.workspace, path)` for any filesystem access; it
enforces the jail and raises `ToolError` on escape.

---

## 3. Approval and risk

`risk` decides whether a human is asked, according to the active policy:

| risk | `dangerous` (default) | `always` | `never` |
|---|---|---|---|
| low | runs | approval | runs |
| medium | approval | approval | runs |
| high | approval | approval | runs |

If you are writing something that changes state outside the workspace, deletes data, spends money or
talks to a privileged service, mark it **high**. The gate is the point.

---

## 4. Testing a plugin

```python
import asyncio
from friday.memory import MemoryStore
from friday.tools.base import ToolContext
from my_plugin import UptimeTool


def test_uptime(settings, tmp_path):
    memory = MemoryStore(tmp_path / "db.sqlite")
    ctx = ToolContext(settings=settings, memory=memory, session_id="test")
    result = asyncio.run(UptimeTool().run({}, ctx))
    assert result.ok
    assert "uptime" in result.output.lower()
```

Run `friday tools` to confirm registration, then exercise it through the HUD — the tool timeline will
show its arguments, duration and output exactly as the model saw them.

---

## 5. Publishing

Plugins are not a packaging system yet: a directory of `.py` files is enough. If you want to share
one, include the class, a `TOOLS` list, a test, and a one-line install note. Keep secrets in
environment variables, never in the plugin source.
