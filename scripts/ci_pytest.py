#!/usr/bin/env python3
"""Run the test suite in CI and make failures readable through the GitHub API.

WHY THIS EXISTS INSTEAD OF `run: python -m pytest tests/ -q`

A failed job's log lives on `results-receiver.actions.githubusercontent.com` and its
artifacts on `pipelines.actions.githubusercontent.com`. Neither is `api.github.com`, so
in a restricted network — a sandbox, a corporate proxy, a CI viewer that only has API
access — the log is simply unreachable: `gh run view --log` returns a TLS `EOF` and
there is nothing to read. What *is* reachable through the ordinary API is the check-run
ANNOTATIONS endpoint, because annotations are stored on the commit, not in the log
blob:

    gh api repos/{owner}/{repo}/check-runs/{job_id}/annotations

So this script turns test failures into `::error::` workflow commands, which the runner
records as annotations. That is the difference between "X tests (py3.11,
windows-latest)" with no way to find out why, and the failing test names and assertion
messages readable from any machine that can reach the API.

WHY PYTHON AND NOT SHELL

`run:` on `windows-latest` executes PowerShell; on `ubuntu-latest`, bash. Anything
non-trivial — pipes, exit-code propagation, `${PIPESTATUS[0]}` — has to be written
twice and then diverges. This is a Python repository being tested on three Python
versions across two operating systems, and the CI glue is the one place where the two
platforms must behave IDENTICALLY or the comparison is meaningless. Windows is the
primary deployment target, so a diagnostic that only works on the platform nobody ships
to is worse than none: it reports green where the product is broken.

Annotation payloads are escaped the way the workflow-command grammar requires
(`%` -> `%25`, CR -> `%0D`, LF -> `%0A`), because an unescaped newline terminates the
command and silently swallows the rest of the message.
"""
from __future__ import annotations

import platform
import re
import subprocess
import sys

#: Enough to see a pattern, few enough that a badly broken run does not bury the job in
#: annotations. `--maxfail=5` bounds pytest anyway; this bounds the reporting.
MAX_ANNOTATIONS = 25


def _esc(text: str) -> str:
    """Escape a string for a GitHub workflow command's message field."""
    return (text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))


def _emit(level: str, message: str, *, title: str = "") -> None:
    """Print one `::level::` workflow command."""
    head = f"::{level}"
    if title:
        head += f" title={_esc(title)}"
    print(f"{head}::{_esc(message)}", flush=True)


#: `FAILED tests/test_x.py::test_y - AssertionError: …` / `ERROR tests/test_x.py`
_FAILURE = re.compile(r"^(FAILED|ERROR)\s+(?P<node>\S+?)(?:\s+-\s+(?P<why>.+))?$")


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else ["tests/", "-q", "--maxfail=5", "-rf", "--tb=line"]

    # Platform facts first, as a notice: when a suite passes on one OS and fails on
    # another, the first question is what differed, and the answer should be in the job
    # record rather than requiring someone to know which runner image was current that
    # week.
    _emit("notice",
          f"python {platform.python_version()} · {platform.system()} "
          f"{platform.release()} · {sys.platform} · sqlite "
          f"{__import__('sqlite3').sqlite_version}",
          title="environment")

    proc = subprocess.run([sys.executable, "-m", "pytest", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    # The full output still goes to the log for anyone who CAN read it. This script adds
    # a channel; it does not replace one.
    print(out, flush=True)

    failures: list[tuple[str, str]] = []
    for line in out.splitlines():
        m = _FAILURE.match(line.strip())
        if m:
            failures.append((m.group("node"), (m.group("why") or "").strip()))

    if failures:
        for node, why in failures[:MAX_ANNOTATIONS]:
            _emit("error", why or "failed (see log for the traceback)", title=node)
        if len(failures) > MAX_ANNOTATIONS:
            _emit("error",
                  f"{len(failures) - MAX_ANNOTATIONS} further failures are in the log only",
                  title="truncated")
        _emit("error", f"{len(failures)} failing test(s) on {sys.platform}",
              title="test suite failed")
    elif proc.returncode != 0:
        # A nonzero exit with no FAILED/ERROR lines means pytest itself broke — a
        # collection error, an internal crash, a bad exit code from a plugin. That is
        # worth saying explicitly, because "no annotations" would otherwise read as
        # "nothing went wrong".
        tail = "\n".join(out.splitlines()[-15:])
        _emit("error", tail or f"pytest exited {proc.returncode} with no parseable summary",
              title=f"pytest exited {proc.returncode} without a test-failure summary")

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
