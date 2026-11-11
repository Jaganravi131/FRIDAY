"""Shared fixtures. Every test gets an isolated repo root via $FRIDAY_ROOT."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ⭐ Point FRIDAY at a scratch root BEFORE pytest imports the test modules. Test
# modules are imported during collection, which happens before any fixture runs, so
# a module-level `from friday.ledger import ladder` would otherwise capture the real
# repository root and every `paths.ARTIFACTS` write in it would land in the
# developer's working tree. The per-test `root` fixture below then rebinds to an
# isolated tmp_path, and `paths.rebind` re-points modules already holding a
# reference. Belt and braces: neither alone is sufficient.
# `tempfile.gettempdir()`, NOT "/tmp". Windows is the primary deployment platform and
# the CI matrix runs it; "/tmp/friday-collection-root" becomes C:\tmp\... there, which
# only exists if something already created it and the runner may write to the drive root.
# A collection-time scratch directory has no reason to be pinned to a POSIX path.
_BOOT_ROOT = Path(os.environ.get("FRIDAY_ROOT")
                  or str(Path(tempfile.gettempdir()) / "friday-collection-root"))
os.environ["FRIDAY_ROOT"] = str(_BOOT_ROOT)


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """⭐ EVERY test gets an isolated FRIDAY root, whether or not it asked for one.

    Autouse and unconditional, because the alternative leaks. Two failure modes were
    observed while getting this right, and both come from the same mistake:

      1. A test module imported at COLLECTION time (pytest imports all test modules
         before any fixture runs) binds `friday.ledger.ladder.paths` to whatever
         `friday.paths` existed then. If that module still points at the real
         repository, `rung3_offload` writes into the developer's working tree during
         a test run. Observed: six stray files in artifacts/offloaded/.

      2. The obvious fix — delete friday.* from sys.modules and re-import — creates a
         SECOND set of module objects. The collection-time `ladder` is then an
         ORPHAN: it is no longer in sys.modules, so nothing that walks sys.modules can
         find and re-point it. It keeps the stale paths forever. Observed as an
         order-dependent failure: test_ledger passed alone and failed in a full run,
         with `ladder.paths.ARTIFACTS` pointing at some *other* test's tmp_path.

    So: never delete and re-import. MUTATE THE ONE paths MODULE IN PLACE. Every
    holder does `from .. import paths` and then reads `paths.ARTIFACTS` at call time,
    which looks the attribute up on the live module object — so rebinding the globals
    reaches all of them, including orphans, because the orphan's `paths` attribute is
    still that same object.
    """
    monkeypatch.setenv("FRIDAY_ROOT", str(tmp_path))
    paths = sys.modules.get("friday.paths") or importlib.import_module("friday.paths")
    paths.rebind(tmp_path)
    paths.ensure_layout()

    # Any module that captured a DIFFERENT friday.paths object (an orphan's holder)
    # is re-pointed by rebind's shape sweep; assert the ones we care about agree.
    for name in ("friday.ledger.ladder", "friday.agent.tools", "friday.memory.facts"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "paths", None) is not None:
            assert mod.paths is paths, f"{name} holds a stale paths module"
            assert mod.paths.ARTIFACTS == tmp_path / "artifacts"
    yield tmp_path


@pytest.fixture
def root(isolated_paths):
    """The isolated root, for tests that need the Path itself."""
    return isolated_paths


@pytest.fixture
def friday(isolated_paths):
    """The whole package, bound to the isolated root."""
    return importlib.import_module("friday")


@pytest.fixture
def conn(root):
    from friday.store import db

    c = db.connect(root / "artifacts" / "friday.db")
    yield c
    c.close()


@pytest.fixture
def seeded(conn, root):
    """A root with starter soul/ + memory/ files, compiled into the DB."""
    from friday import seed
    from friday.memory import compiler

    seed.write_all()
    stats = compiler.compile_all(conn)
    return conn, stats
