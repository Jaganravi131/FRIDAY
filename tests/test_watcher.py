"""Tests for Law 7 hot-reload: hand-edit a memory file and FRIDAY notices.

    Open memory/facts/housing.md, change the rent, save, and ask FRIDAY in the same
    breath.      (docs/architecture/10 Day 3)

Three properties are load-bearing and each gets its own test:

1. A HAND-EDIT IS DETECTED without a restart, and only the edited file is re-derived.
2. FRIDAY'S OWN WRITES ARE NOT MISTAKEN FOR HAND-EDITS. If they were, every
   `friday write` would recompile the file it had just written and log a bogus
   "user edited this" audit row — the trail would be full of edits you never made,
   which is worse than no trail.
3. THE EDIT IS AUDITED AS YOURS. Exit test #6 requires it. An agent whose memory you
   can silently edit is an agent whose memory you cannot trust, because you could
   not tell a bad extraction from your own typo three weeks ago.
"""
from __future__ import annotations

import os
import time

import pytest

from friday.memory import watcher
from friday.memory.compiler import compile_all
from friday.memory.facts import Fact, assert_fact
from friday.store import db


HOUSING = """---
domain: housing
---

# Housing

## Facts
- lease_amount_monthly: **18000 INR** [f_h1 · user_edit · conf 1.00]
  <!-- src: my rent is 18000 -->
"""


@pytest.fixture
def store(conn, isolated_paths):
    """A root with one hand-written facts file, compiled."""
    f = isolated_paths / "memory" / "facts" / "housing.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(HOUSING, encoding="utf-8")
    compile_all(conn)
    return conn, f


def _bump_mtime(p):
    """Force the mtime to move. Some filesystems have a coarse clock, and a test that
    passes only when the write happens to land on a new tick is a test that fails on
    a fast machine."""
    st = p.stat()
    os.utime(p, (st.st_atime - 5, st.st_mtime - 5))


def _rent(conn):
    row = conn.execute(
        "SELECT object FROM facts WHERE predicate='lease_amount_monthly' "
        "AND retracted_at IS NULL ORDER BY asserted_at DESC LIMIT 1").fetchone()
    return row["object"] if row else None


# ── detection ─────────────────────────────────────────────────────────────────


def test_a_hand_edit_is_picked_up_without_a_restart(store):
    """⭐ Exit test #6. Edit, save, ask — same process, no `friday build`."""
    conn, f = store
    assert _rent(conn) == "18000 INR"

    f.write_text(HOUSING.replace("18000 INR", "31500 INR"), encoding="utf-8")
    _bump_mtime(f)
    f.write_text(HOUSING.replace("18000 INR", "31500 INR"), encoding="utf-8")

    w = watcher.MemoryWatcher()
    rep = w.sync(conn)
    assert rep.changed, "the edit must be noticed"
    assert rep.recompiled == [str(f)]
    assert _rent(conn) == "31500 INR", "and the index must reflect it"


def test_an_unchanged_file_costs_nothing_and_reports_nothing(store):
    conn, f = store
    w = watcher.MemoryWatcher()
    first = w.sync(conn)
    assert not first.changed
    second = w.sync(conn)
    assert not second.changed
    assert second.summary() == "no memory changes"
    assert second.checked >= 1, "it still looked"


def test_a_same_length_edit_is_still_detected(store):
    """mtime+size is only a PRE-filter; the content hash is the authority. An edit
    that preserves both must still be caught, or the cheap path has become a
    correctness hole."""
    conn, f = store
    w = watcher.MemoryWatcher()
    w.sync(conn)
    f.write_text(HOUSING.replace("18000 INR", "26000 INR"), encoding="utf-8")
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime))          # deliberately do NOT move mtime
    rep = w.sync(conn, force=True)                    # force skips the mtime pre-filter
    assert rep.recompiled, "a same-mtime edit must be caught by the hash check"
    assert _rent(conn) == "26000 INR"


def test_only_the_edited_file_is_rederived(store, isolated_paths):
    """A hand-edit to housing.md must never disturb work.md. Recompiling everything
    would be slow and would churn ids and timestamps in files you did not touch."""
    conn, f = store
    work = isolated_paths / "memory" / "facts" / "work.md"
    work.write_text("# Work\n\n## Facts\n- employer: **Acme** [f_w1 · user_edit]\n",
                    encoding="utf-8")
    compile_all(conn)
    before = {r["id"]: r["asserted_at"] for r in
              conn.execute("SELECT id, asserted_at FROM facts").fetchall()}

    f.write_text(HOUSING.replace("18000 INR", "19000 INR"), encoding="utf-8")
    _bump_mtime(f)
    f.write_text(HOUSING.replace("18000 INR", "19000 INR"), encoding="utf-8")
    rep = watcher.MemoryWatcher().sync(conn)

    assert rep.recompiled == [str(f)], "work.md was not touched"
    work_rows = conn.execute(
        "SELECT id FROM facts WHERE origin_file=?", (str(work),)).fetchall()
    assert work_rows, "work.md's facts must survive a housing.md edit"


def test_a_broken_hand_edit_does_not_take_the_turn_down(store):
    """You will eventually save a file with a typo in the front matter. That must
    produce an error in the report, not an exception in the middle of answering."""
    conn, f = store
    f.write_text("---\ndomain: housing\n\n# unclosed front matter\n- : :\n",
                 encoding="utf-8")
    _bump_mtime(f)
    f.write_text("---\ndomain: housing\n\n# unclosed\n- : :\n", encoding="utf-8")
    rep = watcher.MemoryWatcher().sync(conn)
    # either it parsed permissively or it recorded an error — but it must return
    assert isinstance(rep.errors, list)
    assert _rent(conn) is not None or rep.recompiled or rep.errors


# ── loop safety ───────────────────────────────────────────────────────────────


def test_fridays_own_write_is_not_reported_as_a_hand_edit(store):
    """⭐ THE LOOP. `assert_fact` writes Markdown AND updates `origin_hash`, so the
    watcher must see no change. If it did, every write would trigger a recompile of
    the file just written and log an audit row claiming YOU edited it."""
    conn, f = store
    w = watcher.MemoryWatcher()
    w.sync(conn)

    assert_fact(conn, Fact(subject="user", predicate="lives_in", object="Chennai",
                           source_kind="user_edit", source_quote="I live in Chennai"))
    conn.commit()

    rep = w.sync(conn)
    assert not rep.recompiled, \
        f"FRIDAY's own write looked like a hand-edit: {rep.summary()}"
    bogus = conn.execute(
        "SELECT COUNT(*) FROM audit WHERE action='memory.hand_edit'").fetchone()[0]
    assert bogus == 0, "no audit row may claim the user edited anything"


def test_repeated_sync_never_loops(store):
    """Sync twice with no edit in between: the second must be a no-op. A watcher that
    re-derives on every pass would rewrite timestamps forever and spin a thread at
    100% for nothing."""
    conn, f = store
    w = watcher.MemoryWatcher()
    w.sync(conn)
    a = w.sync(conn)
    b = w.sync(conn)
    assert not a.changed and not b.changed


# ── deletions ─────────────────────────────────────────────────────────────────


def test_deleting_a_facts_file_removes_its_derived_rows(store, isolated_paths):
    """Markdown is truth, SQLite is derived. A file you deliberately deleted must stop
    being served — otherwise FRIDAY keeps quoting a memory you removed."""
    conn, f = store
    tmp = isolated_paths / "memory" / "facts" / "temp.md"
    tmp.write_text("# Temp\n\n## Facts\n- mood: **tired** [f_t1 · stated]\n",
                   encoding="utf-8")
    watcher.MemoryWatcher().sync(conn, force=True)
    assert conn.execute("SELECT COUNT(*) FROM facts WHERE predicate='mood'").fetchone()[0]

    tmp.unlink()
    rep = watcher.MemoryWatcher().sync(conn, force=True)
    assert rep.deleted == [str(tmp)]
    assert conn.execute("SELECT COUNT(*) FROM facts WHERE predicate='mood'").fetchone()[0] == 0
    actions = [r["action"] for r in conn.execute("SELECT action FROM audit").fetchall()]
    assert "memory.file_deleted" in actions, actions


# ── the audit trail ───────────────────────────────────────────────────────────


def test_the_edit_is_audited_as_the_users(store):
    """Exit test #6: "Audit log shows your edit." The actor is `user`, not `friday` —
    the distinction is the whole point of the trail."""
    conn, f = store
    f.write_text(HOUSING.replace("18000 INR", "20000 INR"), encoding="utf-8")
    _bump_mtime(f)
    f.write_text(HOUSING.replace("18000 INR", "20000 INR"), encoding="utf-8")
    watcher.MemoryWatcher().sync(conn)

    row = conn.execute(
        "SELECT actor, action, target, decision, detail FROM audit "
        "WHERE action='memory.hand_edit' ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["actor"] == "user", "this was YOUR edit, not FRIDAY's"
    assert row["target"] == str(f)
    assert row["decision"] == "allowed"
    assert "re-derived" in row["detail"]


def test_a_soul_edit_is_reported_separately_and_needs_no_recompile(store, isolated_paths):
    """soul/*.md is read live by the Ledger on every compile, so nothing is derived —
    but it IS your edit, and it legitimately invalidates the cached prompt prefix, so
    it is reported in its own bucket rather than lumped in with fact recompiles."""
    conn, f = store
    soul = isolated_paths / "soul" / "SOUL.md"
    soul.parent.mkdir(parents=True, exist_ok=True)
    soul.write_text("I am FRIDAY.\n", encoding="utf-8")

    w = watcher.MemoryWatcher()
    w.sync(conn)                                  # baseline the soul file
    soul.write_text("I am FRIDAY. I am terse.\n", encoding="utf-8")
    _bump_mtime(soul)
    soul.write_text("I am FRIDAY. I am terse.\n", encoding="utf-8")

    rep = w.sync(conn)
    assert rep.soul_changed == [str(soul)]
    assert not rep.recompiled, "nothing is derived from soul/, so nothing recompiles"
    row = conn.execute(
        "SELECT action FROM audit WHERE action='memory.soul_edit'").fetchone()
    assert row is not None


def test_audit_never_breaks_a_turn(store, monkeypatch):
    """The audit write is wrapped deliberately. If the audit table is missing or
    locked, the turn must still succeed — a memory refresh that crashes because it
    could not log itself is strictly worse than one that does not log."""
    conn, f = store

    def boom(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    import sqlite3
    from friday.agent import policy
    monkeypatch.setattr(policy, "log", boom)

    f.write_text(HOUSING.replace("18000 INR", "21000 INR"), encoding="utf-8")
    _bump_mtime(f)
    f.write_text(HOUSING.replace("18000 INR", "21000 INR"), encoding="utf-8")
    rep = watcher.MemoryWatcher().sync(conn)
    assert rep.recompiled, "the recompile must happen even if the audit row cannot"
    assert _rent(conn) == "21000 INR"


# ── the background poller ─────────────────────────────────────────────────────


def test_the_poller_takes_a_connection_factory_not_a_connection(store, isolated_paths):
    """sqlite3 connections are not shareable across threads, so the poller must open
    its own. Passing a live connection would raise ProgrammingError from the thread
    and kill the watcher silently."""
    conn, f = store
    from friday import paths

    w = watcher.MemoryWatcher(interval_s=0.05)
    w.start(lambda: db.connect(paths.DB_PATH))
    assert w.running
    try:
        time.sleep(0.15)
        f.write_text(HOUSING.replace("18000 INR", "24000 INR"), encoding="utf-8")
        _bump_mtime(f)
        f.write_text(HOUSING.replace("18000 INR", "24000 INR"), encoding="utf-8")
        deadline = time.time() + 5
        while time.time() < deadline:
            r = conn.execute(
                "SELECT object FROM facts WHERE predicate='lease_amount_monthly' "
                "AND retracted_at IS NULL").fetchone()
            if r and r["object"] == "24000 INR":
                break
            time.sleep(0.05)
        assert r and r["object"] == "24000 INR", "the poller must pick it up"
    finally:
        w.stop()
    assert not w.running


def test_stop_is_idempotent_and_a_daemon_thread_never_blocks_exit(store):
    conn, f = store
    from friday import paths
    w = watcher.MemoryWatcher(interval_s=0.05)
    w.start(lambda: db.connect(paths.DB_PATH))
    w.stop()
    w.stop()                       # must not raise
    assert not w.running
    import threading
    names = [t.name for t in threading.enumerate()]
    assert "friday-memory-watcher" not in names


def test_a_crashing_poll_pass_is_swallowed(store, monkeypatch):
    """A watcher that takes the REPL down with it is worse than no watcher."""
    conn, f = store
    from friday import paths

    w = watcher.MemoryWatcher(interval_s=0.05)

    def boom(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(watcher.compiler, "compile_file", boom)
    f.write_text(HOUSING.replace("18000 INR", "25000 INR"), encoding="utf-8")
    _bump_mtime(f)
    f.write_text(HOUSING.replace("18000 INR", "25000 INR"), encoding="utf-8")

    rep = w.sync(conn)
    assert rep.errors and "disk on fire" in rep.errors[0]
    assert _rent(conn) == "18000 INR", "the index keeps its last good state"


# ── module-level surface ──────────────────────────────────────────────────────


def test_stale_fact_files_uses_the_db_as_the_snapshot(store):
    """The snapshot is `facts.origin_hash`, not an in-memory cache — so it survives a
    restart, which a cache would not."""
    conn, f = store
    assert watcher.stale_fact_files(conn) == []
    f.write_text(HOUSING.replace("18000 INR", "27000 INR"), encoding="utf-8")
    assert watcher.stale_fact_files(conn) == [f]


def test_the_default_watcher_is_shared_per_process(store):
    """So the CLI and a REPL share one mtime cache instead of each hashing every file."""
    a = watcher.default_watcher()
    b = watcher.default_watcher()
    assert a is b


def test_sync_report_summary_is_readable(store):
    conn, f = store
    w = watcher.MemoryWatcher()
    assert w.sync(conn).summary() == "no memory changes"
    f.write_text(HOUSING.replace("18000 INR", "28000 INR"), encoding="utf-8")
    _bump_mtime(f)
    f.write_text(HOUSING.replace("18000 INR", "28000 INR"), encoding="utf-8")
    s = w.sync(conn).summary()
    assert "housing.md" in s and "recompiled" in s


def _soul(root):
    """A minimal soul/ so the watcher has something to baseline."""
    d = root / "soul"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SOUL.md"
    p.write_text("# SOUL\nDirect and warm.\n", encoding="utf-8")
    return p


def test_a_cold_watcher_does_not_invent_soul_edits(conn, root):
    """Every CLI invocation is a new process with an empty mtime cache.

    Without a baseline pass, all five soul files look "moved" on the first sync and
    the CLI prints "soul edited" on every single turn — noise that would bury the one
    time you actually did edit SOUL.md. Fact files are unaffected: their snapshot is
    the database, which outlives the process.
    """
    facts = root / "memory" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    (facts / "work.md").write_text("employer: Acme\n", encoding="utf-8")
    sp = _soul(root)

    w = watcher.MemoryWatcher()
    rep = w.sync(conn)
    assert rep.soul_changed == []                        # a cold cache is not an edit
    # The first pass legitimately compiles work.md and audits THAT. What must not
    # appear is a soul edit nobody made.
    assert _soul_audit_rows(conn) == []
    assert len(rep.recompiled) == 1                      # facts still reconciled

    # ...and a REAL edit on the second pass is still reported, so the baseline did not
    # simply disable soul watching.
    _bump_mtime(sp)
    sp.write_text("# SOUL\nDirect and warm.\nMore voice.\n", encoding="utf-8")
    assert w.sync(conn).soul_changed == [str(sp)]


def _soul_audit_rows(conn):
    return conn.execute(
        "SELECT target, action FROM audit WHERE action='memory.soul_edit'"
    ).fetchall()


def test_a_real_soul_edit_still_lands_in_the_audit_trail(conn, root):
    _soul(root)
    w = watcher.MemoryWatcher()
    w.sync(conn)                                         # baseline, silent
    assert _soul_audit_rows(conn) == []

    p = root / "soul" / "SOUL.md"
    _bump_mtime(p)
    p.write_text("# SOUL\nchanged by hand\n", encoding="utf-8")
    rep = w.sync(conn)
    assert rep.soul_changed == [str(p)]
    rows = _soul_audit_rows(conn)
    assert len(rows) == 1
    assert rows[0][1] == "memory.soul_edit"


def test_a_cold_watcher_still_reconciles_stale_fact_files(conn, root):
    """The baseline silence is scoped to soul/. Facts must not get the same courtesy.

    Their snapshot is `facts.origin_hash` in the database, so a new process can always
    tell a genuine edit from a cold cache — and if it stopped checking, a hand-edit
    made while FRIDAY was not running would never be compiled.
    """
    facts = root / "memory" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    (facts / "housing.md").write_text("lease_amount_monthly: 18000 INR\n", encoding="utf-8")
    _soul(root)

    w = watcher.MemoryWatcher()
    assert len(w.sync(conn).recompiled) == 1               # first pass compiles it
    assert len(w.sync(conn).recompiled) == 0               # second pass is a no-op


def test_a_zero_fact_file_is_not_recompiled_forever(conn, root):
    """⭐ The bug this table exists for.

    Staleness used to be `COUNT(*) FROM facts WHERE origin_file=? AND origin_hash=?`.
    A file that parses to no facts never gets a row, so the count stays 0 forever and
    the file is judged stale on EVERY turn: recompiled, audited as a user hand-edit,
    and reported as changed — permanently. Dropping a notes-only or half-written .md
    into memory/facts/ is an ordinary thing to do, and the result was an audit trail
    full of edits nobody made, which is worse than no audit trail.
    """
    facts = root / "memory" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    notes = facts / "notes.md"
    notes.write_text(
        "# Scratch\n\nNot a fact yet — just thinking out loud.\nTODO: lease renewal\n",
        encoding="utf-8")
    _soul(root)

    w = watcher.MemoryWatcher()
    first = w.sync(conn)
    assert first.recompiled == [str(notes)]              # noticed once
    assert first.facts_written == 0                      # and got nothing out of it

    assert w.sync(conn).recompiled == []                 # never again
    assert w.sync(conn).recompiled == []
    hand_edits = conn.execute(
        "SELECT count(*) FROM audit WHERE action='memory.hand_edit'").fetchone()[0]
    assert hand_edits == 1                               # one audit row, not one per turn

    # Editing it for real is still detected — the record is of the content, not a
    # blanket "stop watching this file".
    _bump_mtime(notes)
    notes.write_text(
        "---\ndomain: notes\n---\n\n# Notes\n\n## Facts\n"
        "- lease_amount_monthly: **24000 INR**\n", encoding="utf-8")
    rep = w.sync(conn)
    assert rep.recompiled == [str(notes)]
    assert rep.facts_written == 1
    assert w.sync(conn).recompiled == []


def test_a_file_recreated_at_a_deleted_path_is_not_judged_by_the_old_hash(conn, root):
    """Deleting a file must clear its compiled-state row, not just its facts."""
    facts = root / "memory" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    p = facts / "temp.md"
    p.write_text("# Temp\nnothing derivable here\n", encoding="utf-8")
    _soul(root)

    w = watcher.MemoryWatcher()
    w.sync(conn)
    assert conn.execute("SELECT count(*) FROM origin_state").fetchone()[0] == 1

    p.unlink()
    assert w.sync(conn).deleted == [str(p)]
    assert conn.execute("SELECT count(*) FROM origin_state").fetchone()[0] == 0

    # Same path, different content, and it must be treated as new rather than matched
    # against a hash recorded for the file that was deleted.
    p.write_text(
        "---\ndomain: work\n---\n\n# Work\n\n## Facts\n"
        "- employer: **Acme Corp**\n", encoding="utf-8")
    rep = w.sync(conn)
    assert rep.recompiled == [str(p)]
    assert rep.facts_written == 1
