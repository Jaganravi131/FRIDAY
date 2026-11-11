"""Silent failure is a design choice that must stay visible.

`except …: pass` is how a missing audit row, a failed cleanup or a missing backend
disappears without a trace. Sites that are allowed to swallow an error call
`friday.util.swallowed()` instead, which records it. This file pins that rule.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from friday import util

PACKAGE = Path(__file__).resolve().parents[1] / "friday"


_CATCH_ALL = {"Exception", "BaseException"}


def _catches_everything(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:                       # bare `except:`
        return True
    names = [handler.type] if not isinstance(handler.type, ast.Tuple) else handler.type.elts
    return any(isinstance(n, ast.Name) and n.id in _CATCH_ALL for n in names)


def _bare_except_pass_sites() -> list[str]:
    """Catch-everything handlers whose whole body is `pass`. A narrow `except
    FileNotFoundError: pass` is idiomatic and is not what this rule is about."""
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _catches_everything(node) \
                    and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                hits.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno}")
    return hits


def test_no_bare_except_pass_in_the_package():
    """Every swallowed error goes through `swallowed()`. A new bare `pass` fails here, with
    its location, so the author has to decide whether the error is allowed to disappear."""
    assert _bare_except_pass_sites() == [], _bare_except_pass_sites()


def test_swallowed_records_the_location_and_the_exception():
    util._SWALLOWED.clear()
    util.swallowed("friday/x.py:12", ValueError("boom"))
    assert util.swallowed_recent()[-1] == ("friday/x.py:12", "ValueError", "boom")


def test_the_record_is_bounded():
    util._SWALLOWED.clear()
    for i in range(util._SWALLOWED.maxlen + 50):
        util.swallowed(f"f.py:{i}", RuntimeError("x"))
    assert len(util.swallowed_recent()) == util._SWALLOWED.maxlen


def test_audit_failures_are_logged_at_warning(caplog):
    util._SWALLOWED.clear()
    with caplog.at_level(logging.DEBUG, logger="friday"):
        util.swallowed("friday/memory/watcher.py:318", OSError("disk full"), warn=True)
        util.swallowed("friday/bench.py:192", KeyError("predicate"))
    levels = {r.getMessage().split(" at ")[1].split(":")[0]: r.levelno for r in caplog.records}
    assert levels["friday/memory/watcher.py"] == logging.WARNING
    assert levels["friday/bench.py"] == logging.DEBUG
