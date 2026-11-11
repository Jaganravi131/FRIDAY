"""Four tools the HUD line had and the cognition line did not.

Deliberately not a second toolkit. The core already owns memory, retrieval, web
and (at phase 2) the confined workspace. What was genuinely missing:

  * **`file_edit`** — `ws_write` can only overwrite a whole file. Asking a model to
    re-emit a 300-line file to change one line is how you lose the rest of it.
    This replaces one exact substring, refuses if the match is not unique, and
    refuses if the file is outside the workspace.
  * **`file_grep`** — search the workspace without `ws_shell`'s single-command
    allowlist and without loading every file into the prompt.
  * **`math`** — deterministic arithmetic. A 2.6B model gets `17*42+9` wrong often
    enough that this is not a luxury; the mock client already special-cases
    arithmetic for the same reason.
  * **`clock`** — the loop injects a clock reading for `clock` intent, but a model
    that needs the time mid-reasoning should not have to hope the intent
    classifier noticed. This is the explicit call.

Every path goes through `workspace._confined`, the same resolve-then-contain
check `ws_read` uses, so traversal and symlink escape are covered once, not twice.
`file_edit` maps to the `workspace.files` sense in policy — offering it is not
granting it.
"""

from __future__ import annotations

import ast
import math as _math
import operator
import re
from datetime import datetime

from .. import paths
from .workspace import _confined

MAX_EDIT_BYTES = 400_000
MAX_GREP_RESULTS = 60
MAX_GREP_FILE_BYTES = 512_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "artifacts",
             ".mypy_cache", ".ruff_cache", "dist", "build"}

FILE_EDIT_DOC = (
    "Replace ONE exact substring in a file inside the workspace. `old` must match "
    "exactly once — if it matches zero times or several, nothing is written and the "
    "error says which. Use this instead of rewriting the whole file. The file stays "
    "inside the workspace; memory/ and config/ are never editable through this tool."
)
FILE_GREP_DOC = (
    "Search the workspace with a regular expression. Returns file:line:text for up "
    "to 60 matches. Use it to find where something is defined or used without "
    "reading every file."
)
MATH_DOC = (
    "Evaluate an arithmetic expression exactly. Supports + - * / // % **, "
    "parentheses, abs/round/min/max/sqrt/log/exp, pi and e. Use it for any "
    "calculation whose answer must be right; do not do arithmetic in your head."
)
CLOCK_DOC = (
    "The current local date and time, with timezone. Use it when the answer depends "
    "on 'now' and the compiled context does not already contain a reading."
)


# ── file_edit ──────────────────────────────────────────────────────────────────

def file_edit(ctx, path: str, old: str, new: str) -> dict:
    if not old:
        return {"error": "`old` is empty — an edit needs something to find"}
    real, err = _confined(path)
    if err:
        return err
    assert real is not None
    if not real.is_file():
        return {"error": f"no file '{path}' in the workspace"}
    if real.stat().st_size > MAX_EDIT_BYTES:
        return {"error": f"file exceeds {MAX_EDIT_BYTES} bytes — use ws_write in pieces"}
    try:
        text = real.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return {"error": f"could not read: {e}"}

    count = text.count(old)
    if count == 0:
        return {"error": "`old` not found — no change made. Re-read the file and "
                         "copy the exact text, including indentation."}
    if count > 1:
        return {"error": f"`old` matches {count} times — make it unique by including "
                         f"more surrounding lines. No change made."}
    updated = text.replace(old, new, 1)
    try:
        real.write_text(updated, encoding="utf-8", newline="\n")
    except OSError as e:
        return {"error": f"could not write: {e}"}
    return {"path": path, "replaced_chars": len(old), "with_chars": len(new),
            "bytes": len(updated.encode("utf-8"))}


# ── file_grep ──────────────────────────────────────────────────────────────────

def file_grep(ctx, pattern: str, path: str = ".", max_results: int = MAX_GREP_RESULTS) -> dict:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return {"error": f"invalid regular expression: {e}"}
    real, err = _confined(path)
    if err:
        return err
    assert real is not None
    if not real.exists():
        return {"error": f"no such path in workspace: '{path}'"}
    limit = max(1, min(int(max_results or MAX_GREP_RESULTS), MAX_GREP_RESULTS))

    files = [real] if real.is_file() else sorted(real.rglob("*"))
    hits: list[dict] = []
    scanned = 0
    for f in files:
        if len(hits) >= limit:
            break
        if not f.is_file() or any(part in SKIP_DIRS for part in f.parts):
            continue
        try:
            if f.stat().st_size > MAX_GREP_FILE_BYTES:
                continue
            body = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        rel = f.relative_to(paths.ROOT) if f.is_relative_to(paths.ROOT) else f
        for lineno, line in enumerate(body.splitlines(), 1):
            if rx.search(line):
                hits.append({"file": str(rel), "line": lineno,
                             "text": line.strip()[:240]})
                if len(hits) >= limit:
                    break
    return {"matches": hits, "count": len(hits), "files_scanned": scanned,
            "truncated": len(hits) >= limit}


# ── math ───────────────────────────────────────────────────────────────────────

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": _math.sqrt, "log": _math.log, "log10": _math.log10,
    "exp": _math.exp, "sin": _math.sin, "cos": _math.cos, "tan": _math.tan,
    "floor": _math.floor, "ceil": _math.ceil,
}
_CONSTS = {"pi": _math.pi, "e": _math.e, "tau": _math.tau}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 512 or abs(left) > 1e6):
            raise ValueError("exponent or base too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"unknown function '{node.func.id}'")
        return fn(*[_eval(a) for a in node.args])
    raise ValueError("only arithmetic, the listed functions and pi/e are allowed")


def math_tool(ctx, expression: str) -> dict:
    expr = (expression or "").strip()
    if not expr:
        return {"error": "expression is required"}
    if len(expr) > 500:
        return {"error": "expression too long"}
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return {"error": f"could not evaluate: {e}"}
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return {"expression": expr, "result": value,
            "repr": repr(value) if isinstance(value, float) else str(value)}


# ── clock ──────────────────────────────────────────────────────────────────────

def clock(ctx) -> dict:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": now.tzname() or "",
        "epoch": int(now.timestamp()),
    }
