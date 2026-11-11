"""The unified HUD (FastAPI + SSE) that sits on top of the cognition core.

Importing this package is cheap and stdlib-only: `friday.web.app` is where FastAPI
is imported, so a bare `pip install friday` keeps working with no web stack.
"""

__all__ = ["create_app", "run"]


def create_app(*args, **kwargs):        # pragma: no cover - thin re-export
    from .app import create_app as _create

    return _create(*args, **kwargs)
