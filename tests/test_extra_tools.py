"""math / clock / file_edit / file_grep.

The file tools are tested through `workspace._confined`, so escape attempts are
part of the suite rather than a note in a docstring.
"""

from __future__ import annotations

import pytest

from friday.agent.extra_tools import clock, file_edit, file_grep, math_tool
from friday.agent.workspace import workspace_root

# ── math ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expected", [
    ("17*42+9", 723),
    ("(2+3)**2", 25),
    ("7 // 2", 3),
    ("7 % 2", 1),
    ("sqrt(81)", 9),
    ("pi", pytest.approx(3.141592653589793)),
    ("round(2.6)", 3),
])
def test_math_is_exact(expr, expected):
    assert math_tool(None, expr)["result"] == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "[].__class__",
    "lambda: 1",
    "1/0",
])
def test_math_refuses_anything_that_is_not_arithmetic(expr):
    out = math_tool(None, expr)
    assert "error" in out


def test_math_rejects_an_absurd_exponent():
    out = math_tool(None, "9**9999")
    assert "error" in out


# ── clock ────────────────────────────────────────────────────────────────────

def test_clock_returns_a_parseable_local_reading():
    out = clock(None)
    assert set(out) == {"iso", "date", "time", "weekday", "timezone", "epoch"}
    assert len(out["date"]) == 10
    assert out["epoch"] > 0


# ── file_edit / file_grep ────────────────────────────────────────────────────

@pytest.fixture
def workspace(root):
    ws = workspace_root()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_file_edit_replaces_exactly_one_match(workspace):
    (workspace / "notes.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    out = file_edit(None, "notes.md", "beta", "BETA")
    assert "error" not in out
    assert (workspace / "notes.md").read_text() == "alpha\nBETA\ngamma\n"


def test_file_edit_refuses_an_ambiguous_match(workspace):
    (workspace / "dup.md").write_text("x\nx\n", encoding="utf-8")
    out = file_edit(None, "dup.md", "x", "y")
    assert "error" in out and "2 times" in out["error"]
    assert (workspace / "dup.md").read_text() == "x\nx\n"


def test_file_edit_refuses_a_missing_match(workspace):
    (workspace / "f.md").write_text("hello\n", encoding="utf-8")
    assert "error" in file_edit(None, "f.md", "absent", "x")


def test_file_edit_cannot_escape_the_workspace(workspace, root):
    (root / "secret.txt").write_text("do not touch", encoding="utf-8")
    out = file_edit(None, "../../secret.txt", "do not", "DID")
    assert "error" in out
    assert (root / "secret.txt").read_text() == "do not touch"


def test_file_grep_finds_lines_and_skips_binaries(workspace):
    (workspace / "a.py").write_text("def reactor():\n    pass\n", encoding="utf-8")
    (workspace / "b.txt").write_text("nothing here\n", encoding="utf-8")
    out = file_grep(None, r"def \w+\(", ".")
    assert out["count"] == 1
    assert out["matches"][0]["file"].endswith("a.py")
    assert out["matches"][0]["line"] == 1


def test_file_grep_rejects_an_invalid_regex(workspace):
    assert "error" in file_grep(None, "([", ".")


def test_file_grep_cannot_escape_the_workspace(workspace):
    assert "error" in file_grep(None, "root", "../..")
