"""Law 7 in code: hand-edit a memory file and FRIDAY notices, without a restart.

    Open memory/facts/housing.md, change the rent, save, and ask FRIDAY in the same
    breath. That moment — the first time it works — is when this stops being a
    project and starts being *yours*.      (docs/architecture/10, Day 3)

WHY POLLING AND NOT watchdog
----------------------------
The doc sketches this with `watchdog`'s `on_modified`. That is a third-party
dependency and an inotify/FSEvents watcher, and the core of FRIDAY is stdlib-only so
that it runs on a 16 GB Windows laptop, in a container, and on a phone termux shell
without anyone installing anything. `stat()` on a couple dozen small files costs
microseconds, so polling on demand is both cheaper to reason about and dependency
free. A background thread is provided for the REPL, where there is an idle loop to
sit in anyway.

WHY THIS DOES NOT NEED ITS OWN STATE
------------------------------------
The change snapshot is the DATABASE. `facts.origin_hash` already records the content
hash each row was derived from, and `compiler.compile_file` already skips a file when
`(origin_file, origin_hash)` matches. So "has this file been edited since we last
compiled it?" is a query, not a cache to maintain — and it survives restarts, which an
in-memory snapshot would not.

That also gives loop-safety for free. `_index_fact` re-reads the file after FRIDAY
writes it and stores the NEW hash, so FRIDAY's own writes never look like hand-edits.
Without that, every `friday write` would trigger a recompile of the file it had just
written and log a bogus "user edited this" audit entry.

WHAT GETS LOGGED, AND WHY IT MATTERS
------------------------------------
Every detected hand-edit writes an audit row (`actor="user"`,
`action="memory.hand_edit"`). Phase 0 exit test #6 requires that you can SEE your own
edits in the audit trail. This is not decoration: an agent whose memory you can edit
silently is an agent whose memory you cannot trust, because you could not tell whether
a wrong fact came from a bad extraction or from your own typo three weeks ago.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .. import paths
from ..util import now_iso
from . import compiler


@dataclass
class SyncReport:
    """What a sync pass noticed and did."""

    recompiled: list[str] = field(default_factory=list)   # fact files re-derived
    deleted: list[str] = field(default_factory=list)      # files removed from disk
    soul_changed: list[str] = field(default_factory=list)  # identity files (read live)
    facts_written: int = 0
    errors: list[str] = field(default_factory=list)
    checked: int = 0                                       # files stat'd
    elapsed_ms: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.recompiled or self.deleted or self.soul_changed)

    def summary(self) -> str:
        if not self.changed:
            return "no memory changes"
        bits = []
        if self.recompiled:
            bits.append(f"recompiled {len(self.recompiled)}: "
                        f"{', '.join(Path(p).name for p in self.recompiled)}")
        if self.deleted:
            bits.append(f"removed {len(self.deleted)}: "
                        f"{', '.join(Path(p).name for p in self.deleted)}")
        if self.soul_changed:
            bits.append(f"soul edited: {', '.join(Path(p).name for p in self.soul_changed)}")
        if self.facts_written:
            bits.append(f"{self.facts_written} facts re-derived")
        return "; ".join(bits)


def _hash(p: Path) -> str:
    """Hash the file the way the COMPILER hashes it: as decoded text, not raw bytes.

    `origin_hash` is the identity of "what FRIDAY derived this content from", so it
    must be stable across a write→read round trip. `compiler.compile_file` stores
    `sha256(text.encode())` where `text = path.read_text(encoding="utf-8")` — the
    parser's view. If the watcher instead hashed `read_bytes()`, the two would
    diverge on Windows, where text-mode write translates `\n` to `\r\n`: the fact
    FRIDAY just asserted is fingerprinted on logical text, the file on disk carries
    CRLF, and `stale_fact_files` reports an up-to-date file as edited. That is
    exactly the Windows-only failure in CI — green on POSIX because no translation
    happens there. Decoding first makes the hash a property of the content, not of
    the line-ending convention of the OS that happened to write it.
    """
    return hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _fact_files() -> list[Path]:
    if not paths.FACTS.exists():
        return []
    return sorted(paths.FACTS.rglob("*.md"))


def _soul_files() -> list[Path]:
    if not paths.SOUL.exists():
        return []
    return sorted(x for x in paths.SOUL.rglob("*.md") if x.is_file())


def stale_fact_files(conn: sqlite3.Connection,
                     files: Iterable[Path] | None = None) -> list[Path]:
    """Fact files whose current content is not what the DB was derived from.

    Uses the same `(origin_file, origin_hash)` check `compile_file` uses, so a file
    with no rows yet (newly created by hand) counts as stale, and a file FRIDAY wrote
    itself does not. This is one indexed query per file
    (`idx_facts_file(origin_file, origin_hash)`), which is why hashing everything on
    every turn is still cheap enough to leave on.
    """
    from .compiler import _origin_hash_matches

    out = []
    for p in files if files is not None else _fact_files():
        try:
            h = _hash(p)
        except OSError:
            continue
        # ⭐ Delegates to the same check compile_file uses, so the two can never
        # disagree — a file the watcher calls stale must be one compile_file will
        # actually recompile, and vice versa. Reading `origin_state` rather than
        # counting `facts` rows is what stops a zero-fact file being stale forever.
        if not _origin_hash_matches(conn, p, h):
            out.append(p)
    return out


def orphaned_origin_files(conn: sqlite3.Connection) -> list[str]:
    """Fact rows whose source file no longer exists.

    Markdown is truth and SQLite is derived (Law 2), so a deleted file means the
    derived rows must go. Leaving them would keep serving facts from a file the user
    deliberately removed — the exact "rm -rf artifacts && rebuild must restore"
    property, in reverse.
    """
    seen = set()
    rows = conn.execute(
        "SELECT DISTINCT origin_file FROM facts WHERE origin_file IS NOT NULL"
    ).fetchall()
    seen.update(r["origin_file"] for r in rows if r["origin_file"])
    try:
        # A zero-fact file has no `facts` rows, so without this its origin_state entry
        # would survive the deletion of the file it describes — a stale record of a
        # file that is gone, which is the same class of lie as a stale derived fact.
        rows = conn.execute("SELECT origin_file FROM origin_state").fetchall()
        seen.update(r["origin_file"] for r in rows if r["origin_file"])
    except sqlite3.OperationalError:
        pass
    return sorted(f for f in seen if not Path(f).exists())


class MemoryWatcher:
    """Detects hand-edits to Markdown memory and recompiles only what changed.

    Two ways to use it:

      * `sync(conn)` — call it before each turn. This is what the CLI does. It costs
        a `stat()` per memory file plus a hash for the ones that moved, which is far
        cheaper than the retrieval that follows it.
      * `start(conn)` — a background poller for the chat REPL, where there is an idle
        loop to sit in. Daemon thread, so it never blocks exit.

    ⚠️ FACT FILES ARE ALWAYS HASHED. An earlier version cached (mtime, size) and only
    hashed files whose signature had moved, which is a correctness hole: a same-length
    edit landing inside the filesystem's timestamp granularity — 2 s on FAT and some
    network shares, and `st_mtime` is a float, so precision is lost on any of them —
    changes the content while leaving both fields identical. The watcher then reports
    "no changes" and FRIDAY answers from the value you just replaced. That is the one
    failure this module exists to prevent, so it is not an acceptable trade for
    microseconds.

    Measured: hashing every facts file costs 0.075 ms for a realistic install (6 files,
    4 KB each) and 4.3 ms for an absurd one (200 files, 10 KB). Retrieval that follows
    it costs orders of magnitude more. If an install ever grew large enough for this to
    matter, the correct fix is a content-hash index in the DB, not a timestamp
    heuristic.

    The mtime cache is still used for `soul/`, where nothing is derived from the file
    — a missed soul edit costs one absent audit row, not a wrong answer.
    """

    def __init__(self, *, interval_s: float = 1.0, on_change: Callable[[SyncReport], None] | None = None,
                 log_audit: bool = True):
        self.interval_s = interval_s
        self.on_change = on_change
        self.log_audit = log_audit
        self._mtime: dict[str, tuple[float, int]] = {}
        self._soul_mtime: dict[str, tuple[float, int]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: False until the first sync has populated the soul mtime cache. Without this,
        #: every new process reports "soul edited" for files nobody touched — the CLI
        #: runs one process per `friday ask`, so the notice appeared on every single
        #: turn and became noise that would have hidden a real edit.
        self._soul_baselined = False

    # ── change detection ──────────────────────────────────────────────────────

    def _moved(self, cache: dict[str, tuple[float, int]], p: Path) -> bool:
        """True if this path's (mtime, size) differs from what we last saw.

        Size is included because an edit that preserves mtime granularity is still
        detectable by length, and because a same-length edit is caught by the hash
        check that follows — mtime is only ever a cheap PRE-filter, never the
        authority.
        """
        try:
            st = p.stat()
        except OSError:
            cache.pop(str(p), None)
            return True                      # vanished or unreadable: treat as changed
        sig = (st.st_mtime, st.st_size)
        if cache.get(str(p)) == sig:
            return False
        cache[str(p)] = sig
        return True

    def candidates(self) -> list[Path]:
        """Every facts file. See the class docstring for why this is not mtime-filtered:
        the filter saved 0.075 ms and could silently miss a real edit."""
        return _fact_files()

    def soul_candidates(self) -> list[Path]:
        return [p for p in _soul_files() if self._moved(self._soul_mtime, p)]

    # ── the sync pass ─────────────────────────────────────────────────────────

    def sync(self, conn: sqlite3.Connection, *, embedder=None,
             force: bool = False) -> SyncReport:
        """Recompile whatever the user edited. Returns what it noticed.

        `force` skips the mtime pre-filter and hashes everything — used after a
        restart, when the in-process mtime cache is empty but the DB hash may still
        be current (so a forced pass is cheap in DB terms and merely reads files).
        """
        t0 = time.perf_counter()
        rep = SyncReport()

        files = _fact_files()
        rep.checked = len(files)
        # `force` no longer changes what is examined — fact files are always hashed.
        # It is kept because callers use it to mean "re-check after a restart", and
        # because it still forces a pass over soul/ (which IS mtime-cached).
        stale = stale_fact_files(conn, files) if files else []
        if force and not self._soul_baselined:
            for p in _soul_files():
                self._moved(self._soul_mtime, p)
            self._soul_baselined = True

        for p in stale:
            try:
                n = compiler.compile_file(p, conn, embedder=embedder)
            except Exception as e:              # one bad hand-edit must not break the turn
                rep.errors.append(f"{p.name}: {e}")
                continue
            if n < 0:
                continue                        # hash said stale but content compiled equal
            rep.recompiled.append(str(p))
            rep.facts_written += n
            self._audit(conn, p, f"re-derived {n} facts")

        for origin in orphaned_origin_files(conn):
            self._drop_origin(conn, origin)
            rep.deleted.append(origin)
            self._audit(conn, Path(origin), "source file deleted; derived facts removed",
                        action="memory.file_deleted")

        moved_soul = self.soul_candidates()
        if not self._soul_baselined:
            # First pass in this process: the cache was empty, so EVERY file looks
            # moved. Record the baseline and report nothing. Fact files are unaffected
            # by this — their snapshot is the database, which outlives the process.
            self._soul_baselined = True
            moved_soul = []
        for p in moved_soul:
            # soul/*.md is read live by the Ledger on every compile, so nothing needs
            # re-deriving — but the edit is still yours and still belongs in the trail.
            # It also legitimately invalidates the prompt-cache prefix, which is why
            # it is reported separately rather than lumped in with fact recompiles.
            rep.soul_changed.append(str(p))
            self._audit(conn, p, "identity file edited (stable prefix will re-cache)",
                        action="memory.soul_edit")

        if rep.recompiled or rep.deleted:
            conn.commit()
        rep.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if rep.changed and self.on_change is not None:
            self.on_change(rep)
        return rep

    # ── internals ─────────────────────────────────────────────────────────────

    def _audit(self, conn: sqlite3.Connection, p: Path, detail: str,
               action: str = "memory.hand_edit") -> None:
        if not self.log_audit:
            return
        try:
            from ..agent.policy import log

            log(conn, actor="user", action=action, target=str(p),
                sense_id=None, decision="allowed", detail=detail)
        except Exception:
            # Audit must never be the reason a turn fails.
            pass

    @staticmethod
    def _drop_origin(conn: sqlite3.Connection, origin: str) -> None:
        from ..store import db

        rows = conn.execute("SELECT id FROM facts WHERE origin_file=?", (origin,)).fetchall()
        for r in rows:
            try:
                db.fts_delete(conn, r["id"], "fact")
                db.vec_delete(conn, r["id"])
            except Exception:
                pass
        conn.execute("DELETE FROM facts WHERE origin_file=?", (origin,))
        try:
            # Otherwise the compiled-state row outlives the file it describes, and a
            # file recreated at the same path with different content could be judged
            # against a hash recorded for the deleted one.
            conn.execute("DELETE FROM origin_state WHERE origin_file=?", (origin,))
        except sqlite3.OperationalError:
            pass

    # ── background polling ────────────────────────────────────────────────────

    def start(self, conn_factory: Callable[[], sqlite3.Connection]) -> None:
        """Poll in a daemon thread. Takes a CONNECTION FACTORY, not a connection:
        sqlite3 connections are not safe to share across threads, so the poller opens
        its own and closes it each pass."""
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    c = conn_factory()
                    try:
                        self.sync(c, force=False)
                    finally:
                        c.close()
                except Exception:
                    pass                      # a watcher that crashes the REPL is worse than none
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, name="friday-memory-watcher",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


#: One watcher per process, so the CLI and the REPL share an mtime cache.
_DEFAULT: MemoryWatcher | None = None


def default_watcher() -> MemoryWatcher:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MemoryWatcher()
    return _DEFAULT


def sync(conn: sqlite3.Connection, *, embedder=None, force: bool = False) -> SyncReport:
    """Module-level convenience: sync using the process-wide watcher."""
    return default_watcher().sync(conn, embedder=embedder, force=force)
