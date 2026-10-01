"""The Markdown-as-truth property, and the rebuild property that proves it."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_seed_writes_and_compiles(seeded):
    conn, stats = seeded
    assert stats.files >= 3, stats.summary()
    assert stats.facts > 0, stats.summary()
    n = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
    assert n == stats.facts
    assert stats.fts_rows > 0


def test_the_rebuild_property(seeded, root):
    """⭐ Innovation #2's actual claim: rm the artifact, rebuild from Markdown,
    and the answer is unchanged. If this fails, Markdown is not the truth — the DB
    is, and you have built a system you cannot audit or undo."""
    from friday.memory import compiler
    from friday.retrieval.pipeline import search

    conn, _ = seeded
    before = search(conn, "what is my monthly rent")
    assert before, "seeded housing.md should answer a rent query"
    before_val = before.kept[0].row["object"]

    # ⭐ Close before deleting: a file that is still open CANNOT be unlinked on
    # Windows (WinError 32) — POSIX removes it silently, which is why this test was
    # green on the machine it was written on and red in the Windows CI job. The
    # seeded conn is the fixture's; giving it back before the file goes matters
    # only on Windows, and costs nothing anywhere else.
    conn.close()

    # delete every artifact
    for p in (root / "artifacts").rglob("*"):
        if p.is_file():
            p.unlink()
    assert not (root / "artifacts" / "friday.db").exists()

    from friday.store import db
    conn2 = db.connect(root / "artifacts" / "friday.db")
    try:
        stats = compiler.compile_all(conn2)
        assert stats.facts > 0

        after = search(conn2, "what is my monthly rent")
        assert after, "rebuild must restore the rent fact"
        assert after.kept[0].row["object"] == before_val
    finally:
        conn2.close()  # same rule: the temp root is removed on teardown, so this handle
                       # must not still be holding friday.db open when pytest cleans up


def test_incremental_recompile_skips_unchanged(seeded, root):
    from friday.memory import compiler

    conn, _ = seeded
    housing = root / "memory" / "facts" / "housing.md"
    # unchanged -> compile_file returns the -1 sentinel
    assert compiler.compile_file(housing, conn) == -1

    text = housing.read_text(encoding="utf-8")
    housing.write_text(text + "\n- parking: **one covered spot** [f_seed099 · stated · conf 0.9]\n",
                       encoding="utf-8")
    n = compiler.compile_file(housing, conn)
    assert n > 0
    row = conn.execute("SELECT object FROM facts WHERE predicate='parking'").fetchone()
    assert row and "covered" in row["object"]


def test_incremental_does_not_disturb_other_files(seeded, root):
    """A hand-edit to housing.md must never touch work.md."""
    from friday.memory import compiler

    conn, _ = seeded
    work_before = conn.execute(
        "SELECT COUNT(*) AS n FROM facts WHERE origin_file LIKE '%work.md'"
    ).fetchone()["n"]

    housing = root / "memory" / "facts" / "housing.md"
    housing.write_text(
        housing.read_text(encoding="utf-8") + "\n- pets: **none allowed** [f_seed100 · stated]\n",
        encoding="utf-8",
    )
    compiler.compile_file(housing, conn)

    work_after = conn.execute(
        "SELECT COUNT(*) AS n FROM facts WHERE origin_file LIKE '%work.md'"
    ).fetchone()["n"]
    assert work_after == work_before


def test_hand_edit_without_conf_gets_user_edit_tier(seeded, root):
    """Law 7 / the trust feature: a hand-written line with no `conf` is treated as
    authoritative, because a human opened a file and typed it."""
    from friday.memory import compiler

    conn, _ = seeded
    p = root / "memory" / "facts" / "preferences.md"
    p.write_text(
        p.read_text(encoding="utf-8") + "\n- prefers: **plain text over markdown tables**\n",
        encoding="utf-8",
    )
    compiler.compile_file(p, conn)
    row = conn.execute(
        "SELECT source_kind, confidence FROM facts WHERE object LIKE '%plain text%'"
    ).fetchone()
    assert row is not None
    assert row["confidence"] >= 0.9, "an unannotated hand-edit must not be low-confidence"


def test_retracted_facts_are_indexed_but_marked(seeded):
    """Struck-through lines are RETRACTED, never deleted — otherwise 'what did I
    used to think?' stops being answerable and the bi-temporal property is gone."""
    conn, _ = seeded
    row = conn.execute(
        "SELECT object, retracted_at, superseded_by FROM facts WHERE id='f_seed040'"
    ).fetchone()
    assert row is not None, "the struck-through ₹24,000 lease line must be parsed"
    assert row["retracted_at"], "must be marked retracted"
    assert row["superseded_by"] == "f_seed041"
    # still findable by explicit belief-time query
    from friday.memory.facts import history_of
    hist = history_of(conn, "lease_amount_monthly")
    assert len(hist) == 2
    assert {r["object"] for r in hist} == {"₹24,000", "₹28,000"}


def test_predicate_class_and_single_valued_inferred(seeded):
    from friday.config import is_single_valued, predicate_class_of

    assert predicate_class_of("lease_amount_monthly") == "environment"
    assert predicate_class_of("name") == "identity"
    assert predicate_class_of("prefers_dark_mode") == "preference"
    assert predicate_class_of("mood_today") == "mood"
    assert is_single_valued("lives_in")
    assert not is_single_valued("prefers")


def test_decay_semantics():
    """Half-life behaviour + the reinforcement term: recalling a fact makes it
    stickier, which is the importance x recency x relevance the ecosystem converged
    on, but with a tunable per-class half-life instead of one global number."""
    from datetime import datetime, timedelta

    from friday.memory.decay import decay

    now = datetime.now().astimezone()
    recent = now.isoformat(timespec="seconds")
    old = (now - timedelta(days=60)).isoformat(timespec="seconds")

    assert decay(predicate_class="identity", confidence=1.0, asserted_at=old).tier == "immortal"
    assert decay(predicate_class="mood", confidence=0.9, asserted_at=old, now=now).tier == "archived"
    fresh = decay(predicate_class="preference", confidence=0.9, asserted_at=recent, now=now)
    assert fresh.tier == "active" and fresh.confidence > 0.85

    # reinforcement: same age, more accesses -> higher retained confidence
    a = decay(predicate_class="location", confidence=0.9, asserted_at=old, now=now, access_count=0)
    b = decay(predicate_class="location", confidence=0.9, asserted_at=old, now=now, access_count=20)
    assert b.confidence > a.confidence


def test_compile_errors_do_not_abort_the_build(seeded, root):
    """One malformed hand-edited file must not take down the whole memory index."""
    from friday.memory import compiler

    conn, _ = seeded
    bad = root / "memory" / "facts" / "broken.md"
    bad.write_text("---\ndomain: broken\n---\n\n- this is not a fact line\n" * 40, encoding="utf-8")
    stats = compiler.compile_all(conn)
    assert stats.files >= 4
    assert stats.facts > 0


# ── provenance: every compiled fact must be citable ───────────────────────────


def test_every_compiled_markdown_fact_is_citable(seeded):
    """⭐ A fact that cannot say where it came from cannot be trusted, and `/why` is
    the single thing the whole trust model rests on.

    A facts file recorded a quote only in an explicit `<!-- src: … "the words" -->`
    annotation, so every plain bullet compiled to `source_quote = NULL` — **15 of the
    16 facts `friday seed` writes**. The first command a new user runs therefore built
    a memory that could not cite 94% of itself. It stayed invisible for a long time
    because the uncitable facts ranked below the retrieval floor; unifying the stopword
    lists lifted one of them to rank 1 and Phase 0 exit condition #2 ("tell it a fact
    Monday, ask a follow-up Friday") failed on *"recalled but WITHOUT a usable
    citation"*. The gate was right and the data path was wrong.

    The evidence was never missing — `FactLine.raw` already held the source line. This
    asserts the invariant, not the mechanism, so it holds for any future fact source.
    """
    conn, _ = seeded
    rows = conn.execute(
        "SELECT id, source_quote, origin_file, origin_line FROM facts").fetchall()
    assert rows, "the seed corpus must produce facts for this to mean anything"
    uncitable = [r["id"] for r in rows if not (r["source_quote"] and r["origin_file"])]
    assert not uncitable, (
        f"{len(uncitable)}/{len(rows)} compiled facts cannot be cited by /why: {uncitable[:6]}"
    )


def test_the_citation_is_the_verbatim_line_and_an_explicit_quote_still_wins(seeded, root):
    """Two properties of the fallback, both of which are easy to get subtly wrong.

    VERBATIM: a citation is only worth anything if the user can open the file at that
    line and see exactly those characters — including the `[id · kind · conf]` meta
    tail. Tidying it up would make it a paraphrase of the evidence rather than the
    evidence, which is the failure mode the whole provenance design exists to prevent.

    PRECEDENCE: an explicit annotation records what the user actually SAID, which is
    stronger evidence than the bullet someone later wrote about it. The fallback must
    fill a gap, never overwrite one.
    """
    from friday.memory import compiler

    conn, _ = seeded
    f = root / "memory" / "facts" / "citation.md"
    f.write_text(
        "---\ndomain: citation\nversion: 1\n---\n\n# Citation\n\n"
        "- plain: **taken from its own line** [f_cite001 · stated · conf 0.9]\n"
        "- annotated: **taken from what was said** [f_cite002 · stated · conf 0.9]\n"
        "  <!-- src: chat:2026-09-30 \"I said this in chat, not in this file\" -->\n",
        encoding="utf-8")
    compiler.compile_file(f, conn)
    conn.commit()

    by = {r["id"]: r["source_quote"] for r in conn.execute(
        "SELECT id, source_quote FROM facts WHERE id IN ('f_cite001','f_cite002')")}

    assert by["f_cite001"] == "- plain: **taken from its own line** [f_cite001 · stated · conf 0.9]", (
        "the fallback citation must be the raw line, character for character"
    )
    assert by["f_cite002"] == "I said this in chat, not in this file", (
        "an explicit quote is stronger evidence than the bullet and must not be replaced"
    )


def test_the_citation_fallback_does_not_leak_into_retrieval_or_supervision(seeded, root):
    """The fallback answers "where did this come from?" — it must not also change how
    well a fact is FOUND, or what the RSC write gate is trained on.

    Both would be silent, and both would be bad. Indexing the raw line would
    double-weight the predicate and object tokens that already matched, moving every
    retrieval score for a reason unrelated to ranking. And a supervision span asserts
    "this span of an UTTERANCE produced this fact"; a Markdown bullet is an assertion,
    not an utterance, so feeding synthetic bullets to w_t would teach it that bullet
    syntax is evidence a user said something.
    """
    from friday.memory import compiler, supervision

    conn, _ = seeded
    before_fts = conn.execute("SELECT COUNT(*) AS n FROM memory_fts").fetchone()["n"]
    before_spans = conn.execute(
        "SELECT COUNT(*) AS n FROM supervision_spans").fetchone()["n"]

    f = root / "memory" / "facts" / "leakcheck.md"
    f.write_text(
        "---\ndomain: leakcheck\nversion: 1\n---\n\n# Leakcheck\n\n"
        "- unquoted: **a fact with no annotation at all** [f_leak001 · stated · conf 0.9]\n",
        encoding="utf-8")
    compiler.compile_file(f, conn)
    conn.commit()

    assert conn.execute("SELECT source_quote FROM facts WHERE id='f_leak001'"
                        ).fetchone()["source_quote"], "the fact itself must be citable"
    # the raw line is NOT indexed as extra search text: one fact adds exactly one row
    after_fts = conn.execute("SELECT COUNT(*) AS n FROM memory_fts").fetchone()["n"]
    assert after_fts == before_fts + 1, (
        f"indexing the fallback citation would change retrieval scores: {before_fts} -> {after_fts}"
    )
    # and it is NOT recorded as an utterance span for the write gate
    after_spans = conn.execute(
        "SELECT COUNT(*) AS n FROM supervision_spans").fetchone()["n"]
    assert after_spans == before_spans, (
        "a Markdown bullet is an assertion, not an utterance; it must not supervise w_t"
    )
    assert supervision is not None


def test_rebuild_says_what_to_do_when_the_database_is_locked(seeded, root, monkeypatch):
    """⭐ Windows cannot delete a file another process holds open. `friday rebuild`
    deletes artifacts/friday.db, so on Windows it fails whenever a gateway or a chat
    session is running — and the raw `PermissionError: [WinError 32] The process cannot
    access the file because it is being used by another process` reads like a bug in
    FRIDAY rather than what it actually is: a second FRIDAY.

    Simulated here by making unlink raise exactly that error, so the behaviour is pinned
    on every platform rather than only being observable on a Windows runner. POSIX
    unlinks open files silently, which is why this class of bug is invisible on the
    machine it was written on.
    """
    import pathlib as _pl

    from friday.memory import compiler

    conn, _ = seeded
    real_unlink = _pl.Path.unlink

    def locked_unlink(self, *a, **kw):
        if self.name == "friday.db":
            raise PermissionError(13, "The process cannot access the file because it is "
                                      "being used by another process")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(_pl.Path, "unlink", locked_unlink)

    with pytest.raises(RuntimeError) as ei:
        compiler.rebuild(root / "artifacts" / "friday.db")
    msg = str(ei.value)
    assert "friday serve" in msg, "must name the process that is usually holding the file"
    assert "Markdown is untouched" in msg, "must say nothing was lost"
    assert "WinError" not in msg.split("Underlying:")[0], (
        "the guidance must come before the raw OS detail, not after it"
    )

    # and the promise in the message must be true: the truth survived
    facts = list((root / "memory" / "facts").rglob("*.md"))
    assert facts, "a failed rebuild must not have touched the Markdown"


def test_the_cli_closes_the_connections_it_opened(seeded, root, monkeypatch):
    """The leak behind the Windows rebuild failure, pinned at its source.

    `_conn()` opened a connection per command and `main()` never closed any of them. On
    POSIX that is invisible — the process exits, and an open file can still be unlinked.
    On Windows the handle survives for the life of the process, so `friday rebuild`
    after `friday search` in the same process fails to delete the database. Tests call
    `main()` repeatedly in one process, which is exactly the CLI's own `cli()` helper
    doing what a Windows user's shell would do across processes.
    """
    import sqlite3

    from friday import cli as cli_mod

    monkeypatch.setenv("FRIDAY_ROOT", str(root))

    # Count what main() OPENS, because _OPEN_CONNS is empty by the time it returns —
    # that emptiness is the fix, so it cannot also be the evidence that anything happened.
    opened: list = []
    real_connect = cli_mod.store_db.connect

    def counting_connect(*a, **kw):
        c = real_connect(*a, **kw)
        opened.append(c)
        return c

    monkeypatch.setattr(cli_mod.store_db, "connect", counting_connect)
    cli_mod._OPEN_CONNS.clear()

    assert cli_mod.main(["search", "rent"]) == 0
    assert opened, "search must have opened a connection for this test to mean anything"
    assert cli_mod._OPEN_CONNS == [], (
        f"{len(cli_mod._OPEN_CONNS)} connection(s) still tracked after main() returned — "
        f"on Windows these block `friday rebuild` from deleting artifacts/friday.db"
    )
    # Not merely dropped from the list: ACTUALLY closed. sqlite3 has no `.closed`, so
    # the proof is that using it raises. A list that was cleared without closing would
    # pass the assertion above and still leak the handle.
    for c in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            c.execute("SELECT 1")
