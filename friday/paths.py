"""Repository layout, in one place. See docs/architecture/00-BLUEPRINT.md §7."""

from __future__ import annotations

import os
from pathlib import Path

#: Repo root. Override with $FRIDAY_ROOT (tests and multi-instance use).
ROOT = Path(os.environ.get("FRIDAY_ROOT", Path(__file__).resolve().parent.parent))

# ── truth: Markdown, git-versioned, hand-editable (Law 7) ──────────────────────
SOUL = ROOT / "soul"
MEMORY = ROOT / "memory"
TRACES = MEMORY / "traces"        # S0  append-only raw      YYYY-MM-DD.jsonl
EPISODES = MEMORY / "episodes"    # S1  summarised days      YYYY-MM-DD.md
FACTS = MEMORY / "facts"          # S2  bi-temporal facts    <domain>.md
DAILY = MEMORY / "daily"          # ephemeral log (NOT auto-injected)
SKILLS = ROOT / "skills"          # S3  agentskills.io format
CONFIG = ROOT / "config"
EVAL = ROOT / "eval"              # THE LOCKED SUITE — read-only to the agent

# ── artifacts: build outputs, never committed (rm -rf artifacts && rebuild) ────
ARTIFACTS = ROOT / "artifacts"
DB_PATH = ARTIFACTS / "friday.db"

TRUTH_DIRS = (SOUL, TRACES, EPISODES, FACTS, DAILY, SKILLS, CONFIG, EVAL)
ARTIFACT_DIRS = (ARTIFACTS,)


def rebind(new_root) -> None:
    """Re-point every path constant at `new_root`, IN THIS MODULE AND IN ANY MODULE
    THAT ALREADY IMPORTED IT.

    Why this exists rather than just reassigning the globals: `ROOT` is read from
    `$FRIDAY_ROOT` once, at import time, and other modules do `from .. import paths`
    and then use `paths.ARTIFACTS` at call time. Rebinding the globals fixes those,
    because they look the attribute up on the live module object. But pytest imports
    every test module during COLLECTION — before any fixture has run — so a test
    module that did `from friday.ledger import ladder` at the top holds a reference
    to the `ladder` module as it was bound to the REAL repository root. Reloading
    `friday.*` afterwards creates a second, correct `ladder`, and the first one
    quietly keeps writing into the developer's actual working tree.

    That is how `artifacts/offloaded/` accumulated files during a test run: not a
    logic error in the ladder, an aliasing error in the harness. Tests must not be
    able to write outside their own temporary root, so rebind walks `sys.modules`
    and re-points anything that has already captured this module.

    Safe to call repeatedly; a no-op if nothing has imported paths yet.
    """
    import sys

    global ROOT, SOUL, MEMORY, TRACES, EPISODES, FACTS, DAILY, SKILLS, CONFIG, EVAL
    global ARTIFACTS, DB_PATH, TRUTH_DIRS, ARTIFACT_DIRS

    new_root = Path(new_root)
    ROOT = new_root
    SOUL = new_root / "soul"
    MEMORY = new_root / "memory"
    TRACES = MEMORY / "traces"
    EPISODES = MEMORY / "episodes"
    FACTS = MEMORY / "facts"
    DAILY = MEMORY / "daily"
    SKILLS = new_root / "skills"
    CONFIG = new_root / "config"
    EVAL = new_root / "eval"
    ARTIFACTS = new_root / "artifacts"
    DB_PATH = ARTIFACTS / "friday.db"
    TRUTH_DIRS = (SOUL, TRACES, EPISODES, FACTS, DAILY, SKILLS, CONFIG, EVAL)
    ARTIFACT_DIRS = (ARTIFACTS,)

    # Re-point every module that holds a friday PATHS reference at THIS module object.
    #
    # This has to catch STALE duplicates too, not just `is this_module`. A test
    # module imported at collection time binds `ladder.paths` to whatever
    # `friday.paths` existed then; if another test file later deletes friday.* from
    # sys.modules and re-imports, a second paths module is created and `ladder` may
    # still be holding the first. Rebinding only the live one leaves the stale one
    # pointing at the real repository — which is why this failure was
    # order-dependent: test_ledger passed alone and failed in a full run.
    #
    # ⚠️ RECOGNISE BY SHAPE, NEVER BY ATTRIBUTE NAME. The first version of this
    # swept anything called `paths` and clobbered unrelated attributes — including
    # `pathlib.Path` values and locals named `paths` — which broke fixtures across
    # four test modules with TypeErrors that had nothing to do with paths. A friday
    # paths module is the thing that has ROOT/ARTIFACTS Paths AND ensure_layout.
    me = sys.modules.get(__name__)

    def _is_friday_paths(m) -> bool:
        if m is None or m is me or not hasattr(m, "ensure_layout"):
            return False
        return isinstance(getattr(m, "ROOT", None), Path) and \
            isinstance(getattr(m, "ARTIFACTS", None), Path)

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for attr in ("paths", "friday_paths"):
            held = getattr(mod, attr, None)
            if held is not None and held is not me and _is_friday_paths(held):
                setattr(mod, attr, me)


def ensure_layout() -> None:
    """Create every directory the pipeline expects. Idempotent."""
    for d in (*TRUTH_DIRS, *ARTIFACT_DIRS):
        d.mkdir(parents=True, exist_ok=True)
    (EVAL / "suites").mkdir(parents=True, exist_ok=True)


def today_stem(now=None) -> str:
    """`YYYY-MM-DD` in local time — the partition key for traces/episodes/daily."""
    from datetime import datetime

    return (now or datetime.now()).strftime("%Y-%m-%d")
