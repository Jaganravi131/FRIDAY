"""Entry point for `python -m friday`, and the target of the `friday` console script.

`pyproject.toml` declares `friday = "friday.__main__:main"`, so `main` has to live
here rather than only in `cli.py` — otherwise `pip install -e .` produces a console
script that dies with an ImportError on first run, which is the worst possible moment
to discover it.

Everything else is `friday.cli`. This module exists to make three invocations behave
identically:

    python -m friday ask "what is my rent"     # __main__.py, via runpy
    friday ask "what is my rent"               # console script -> __main__:main
    from friday.cli import main; main([...])   # in-process, and what the tests use

Exit status is the CLI's return value, propagated through SystemExit so shell
scripting and `scripts/needle_test.py`-style gates can branch on it.
"""

from __future__ import annotations

from .cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
