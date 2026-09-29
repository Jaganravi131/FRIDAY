"""The Markdown-as-truth property, and the rebuild property that proves it."""

from __future__ import annotations

from pathlib import Path


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

    # delete every artifact
    for p in (root / "artifacts").rglob("*"):
        if p.is_file():
            p.unlink()
    assert not (root / "artifacts" / "friday.db").exists()

    from friday.store import db
    conn2 = db.connect(root / "artifacts" / "friday.db")
    stats = compiler.compile_all(conn2)
    assert stats.facts > 0

    after = search(conn2, "what is my monthly rent")
    assert after, "rebuild must restore the rent fact"
    assert after.kept[0].row["object"] == before_val


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
