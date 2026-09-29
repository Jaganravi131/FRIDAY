"""Law 1. The two axes, and the retraction pairs that make Rung 5 trainable.

This is the highest-value test file in the repo, because these properties are the
ones every other design gives up for convenience — and they are precisely the
properties the RSC's independently-addressed erase gate needs as training signal.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from friday.config import predicate_class_of

import pytest


def _t(days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")


@pytest.fixture
def fact():
    from friday.memory.facts import Fact
    return Fact


def test_new_assert_wins_and_the_loser_is_retracted_not_deleted(conn, fact):
    from friday.memory.facts import assert_fact

    first = assert_fact(conn, fact(subject="user", predicate="lives_in", object="Bengaluru",
                                   valid_from=_t(-400), source_quote="i live in bengaluru"))
    assert first["action"] == "new"
    assert not first["retracted"]

    second = assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                                    valid_from=_t(-100), source_quote="i moved to chennai"))
    assert second["action"] == "reconciled"
    assert len(second["retracted"]) == 1
    assert second["retracted"][0]["object"] == "Bengaluru"

    # RETRACTED, not deleted — the row is still there
    rows = conn.execute("SELECT object, retracted_at, valid_to, superseded_by FROM facts") \
               .fetchall()
    assert len(rows) == 2
    old = next(r for r in rows if r["object"] == "Bengaluru")
    assert old["retracted_at"] is not None, "the retraction must be stamped"
    assert old["valid_to"] is not None, "the value must be time-boxed on the world axis"
    assert old["superseded_by"] == second["written"]["id"]


def test_lower_confidence_cannot_overwrite(conn, fact):
    """The guard that stops a hallucination from rewriting your address."""
    from friday.memory.facts import assert_fact

    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                           confidence=1.0, source_kind="user_edit", valid_from=_t(-30)))
    r = assert_fact(conn, fact(subject="user", predicate="lives_in", object="Mumbai",
                               confidence=0.5, valid_from=_t(-1)))
    assert r["action"] == "rejected"
    assert "hand-edited" in r["message"]
    live = conn.execute("SELECT object FROM facts WHERE predicate='lives_in' AND retracted_at IS NULL").fetchall()
    assert [x["object"] for x in live] == ["Chennai"]


def test_observed_beats_stated_but_not_user_edit(conn, fact):
    from friday.memory.facts import assert_fact

    assert_fact(conn, fact(subject="user", predicate="email", object="a@b.com",
                           source_kind="stated", confidence=0.8, valid_from=_t(-10)))
    r = assert_fact(conn, fact(subject="user", predicate="email", object="c@d.com",
                               source_kind="observed", confidence=0.8, valid_from=_t(-1)))
    assert r["action"] == "reconciled", "observed outranks stated"

    r2 = assert_fact(conn, fact(subject="user", predicate="email", object="e@f.com",
                                source_kind="stated", confidence=0.99, valid_from=_t(0)))
    assert r2["action"] == "rejected", "a confident assertion still cannot beat an observation"


def test_world_time_query_respects_valid_intervals(conn, fact):
    """Axis 1. `as_of` must not be satisfied by the current fact."""
    from friday.memory.facts import assert_fact, facts_at

    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Bengaluru",
                           valid_from=_t(-800), valid_to=_t(-400), confidence=0.9,
                           source_kind="stated"))
    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                           valid_from=_t(-100), confidence=0.9, source_kind="stated"))

    assert [r["object"] for r in facts_at(conn, _t(-500))] == ["Bengaluru"]
    assert [r["object"] for r in facts_at(conn, _t(-50))] == ["Chennai"]
    # the gap between the two intervals is genuinely unknown — [] is correct
    assert facts_at(conn, _t(-250)) == []


def test_belief_time_query_is_a_different_answer(conn, fact):
    """⭐ Axis 2. The query that makes retraction-never-deletion load-bearing:
    what did FRIDAY BELIEVE on a date, as opposed to what was true.

    The scenario has to be built carefully, because the two axes are independent and
    it is easy to write a "test" that accidentally asserts something impossible.
    Here: FRIDAY learned about Bengaluru ten days ago, and learned about the move to
    Chennai two days ago. So five days ago it believed Bengaluru — and it still
    believes Bengaluru at that instant even though the move may already have
    happened in the world. That is belief time, and it is not the same question as
    `facts_at`.
    """
    from friday.memory.facts import assert_fact, beliefs_at

    old_learned = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")
    new_learned = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(timespec="seconds")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Bengaluru",
                           valid_from=_t(-800), asserted_at=old_learned,
                           confidence=0.9, source_kind="stated"))
    r = assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                               valid_from=_t(-100), asserted_at=new_learned,
                               confidence=0.95, source_kind="stated"))
    assert r["action"] == "reconciled"

    # Today: believes Chennai. Bengaluru is retracted, so it must NOT surface —
    # a retracted fact is historical, never current.
    assert [x["object"] for x in beliefs_at(conn, now)] == ["Chennai"]

    # Five days ago: FRIDAY had not yet been told about the move, so it believed
    # Bengaluru. Note the retraction of Bengaluru is stamped `now`, which is AFTER
    # the query instant — that is exactly why the old belief is still recoverable.
    assert [x["object"] for x in beliefs_at(conn, five_days_ago)] == ["Bengaluru"]


def test_belief_time_recovers_a_fact_retracted_later_the_same_day(conn, fact):
    """The regression this guards: truncating belief time to whole days makes a fact
    learned and retracted on the same day vanish from EVERY belief-time query, so
    `beliefs_at` silently returns []. `asserted_at`/`retracted_at` come from a clock,
    so they must be compared as instants, not dates."""
    from friday.memory.facts import assert_fact, beliefs_at

    morning = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(timespec="seconds")
    midday = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(timespec="seconds")

    assert_fact(conn, fact(subject="user", predicate="mood", object="tired",
                           valid_from=morning, asserted_at=morning,
                           confidence=0.6, source_kind="observed"))
    assert_fact(conn, fact(subject="user", predicate="mood", object="fine",
                           valid_from=midday, asserted_at=midday,
                           confidence=0.7, source_kind="observed"))

    # Both events are on the same calendar day. At midday-minus-an-hour FRIDAY
    # believed "tired"; a day-granular comparison would return [] here.
    between = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(timespec="seconds")
    assert [x["object"] for x in beliefs_at(conn, between)] == ["tired"]
    # and now it believes "fine"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assert [x["object"] for x in beliefs_at(conn, now)] == ["fine"]


def test_world_and_belief_time_diverge(conn, fact):
    """The two axes must be able to disagree. If they can't, you have one axis and
    have silently given up the property."""
    from friday.memory.facts import assert_fact, beliefs_at, facts_at

    # TRUE since -100 days, but only BELIEVED since now
    assert_fact(conn, fact(subject="user", predicate="job", object="research engineer",
                           valid_from=_t(-100), confidence=0.9, source_kind="stated"))
    then = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
    assert [r["object"] for r in facts_at(conn, then)] == ["research engineer"], "true then"
    assert beliefs_at(conn, then) == [], "but not yet believed"


def test_a_retraction_pair_is_emitted(conn, fact):
    """⭐ Phase 1's free supervision, collected from day one because it cannot be
    backfilled. The (key_old, key_new, value_old, value_new, keys_distinct,
    asserted_at, retracted_at, trace_id, char_start, char_end) 10-tuple that makes
    GDN-2's b_t and EDA's e_t trainable."""
    from friday.memory.facts import assert_fact
    from friday.rsc.gate_supervision import pair_batches

    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Bengaluru",
                           valid_from=_t(-800), source_quote="i live in bengaluru"))
    r = assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                               valid_from=_t(-100), source_quote="i moved to chennai"))
    assert r["supervision"]["pair"], "a retraction must emit a supervision pair"
    assert r["supervision"]["pair"]["keys_distinct"] is True, \
        "different object strings must give different addresses — this is the case a " \
        "write-anchored erase cannot reach"

    batches = pair_batches(conn)
    assert len(batches) == 1
    b = batches[0]
    assert b["address_distinct"] == 1
    assert {b["value_old"], b["value_new"]} == {"Bengaluru", "Chennai"}
    assert b["key_old"] != b["key_new"]


def test_repeated_retraction_of_the_same_value_is_not_a_pair(conn, fact):
    """A user who says the same thing twice is not new supervision. Emitting a pair
    here would teach the erase gate to fire on repetition."""
    from friday.memory.facts import assert_fact

    assert_fact(conn, fact(subject="user", predicate="mood", object="tired",
                           valid_from=_t(-2), source_kind="observed", confidence=0.6))
    r = assert_fact(conn, fact(subject="user", predicate="mood", object="tired",
                               valid_from=_t(-1), source_kind="observed", confidence=0.6))
    assert r["action"] == "nochange"
    assert r["supervision"]["pair"] is None


def test_history_of_orders_by_world_time(conn, fact):
    from friday.memory.facts import assert_fact, history_of

    assert_fact(conn, fact(subject="user", predicate="lease_amount_monthly", object="₹24,000",
                           valid_from=_t(-700), valid_to=_t(-200), confidence=0.9,
                           source_kind="stated"))
    assert_fact(conn, fact(subject="user", predicate="lease_amount_monthly", object="₹28,000",
                           valid_from=_t(-200), confidence=0.95, source_kind="stated"))
    hist = history_of(conn, "lease_amount_monthly")
    assert [h["object"] for h in hist] == ["₹28,000", "₹24,000"], "newest first"


def test_span_is_created_from_a_source_quote(conn, fact):
    """Salience spans -> the RSC's WRITE gate w_t. The span carries (trace_id,
    char_start, char_end) so the trainer can recover the exact tokens."""
    from friday.memory.facts import assert_fact
    from friday.rsc.gate_supervision import span_batches

    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                           valid_from=_t(-100), source_quote="i moved to chennai in june 2023"))
    b = span_batches(conn)
    assert len(b) == 1
    assert b[0]["char_start"] >= 0 and b[0]["char_end"] > b[0]["char_start"]
    assert b[0]["label"] == 1
    # `domain` routes to a FILE (memory/facts/housing.md); `predicate_class` drives
    # the decay half-life. Two different taxonomies — don't conflate them.
    assert b[0]["domain"] == "housing"
    assert predicate_class_of("lives_in") == "location"
    assert b[0]["span_text"] == "i moved to chennai in june 2023", \
        "a quote-derived span must carry its text inline: there is no trace file to point at"


def test_alpha_targets_follow_the_laws():
    """The supervision targets must encode the LAWS, not a heuristic. Facts retain,
    episodes forget, corrections forget harder — because if the trained decay gate
    learns to retain transients, the entire premise of docs/architecture/13 §2 is
    wrong and the RSC will drown in noise."""
    from friday.rsc.gate_supervision import alpha_for_class

    # identity never decays -> retention exactly 1.0. Everything else is a
    # per-token retention derived from its half-life, so it sits just under 1.0
    # and must be compared with a tolerance.
    assert alpha_for_class("identity") == 1.0
    for klass in ("environment", "preference", "location", "relationship", "mood", "other"):
        assert alpha_for_class(klass) == pytest.approx(1.0, abs=1e-3)

    # The ordering is the real assertion: shorter half-life -> lower retention.
    # This is what L_decay regresses toward, so if the order is wrong the RSC
    # learns to retain transients and forget facts, which is exactly backwards.
    assert alpha_for_class("mood") < alpha_for_class("location")
    assert alpha_for_class("location") < alpha_for_class("relationship")
    assert alpha_for_class("relationship") < alpha_for_class("preference")
    assert alpha_for_class("preference") < alpha_for_class("identity")


def test_readiness_report_blocks_before_enough_pairs(conn):
    """The gate that stops you from burning Kaggle hours on 40 pairs. Phase 3.5
    needs >= 2000 retraction pairs (docs/architecture/13 §4.5)."""
    from friday.memory.supervision import ready_for_phase_3_5, supervision_report

    ok, msg = ready_for_phase_3_5(conn)
    assert not ok
    assert "pairs" in msg
    assert supervision_report(conn)


def test_export_training_set(conn, fact, tmp_path):
    from friday.memory.facts import assert_fact
    from friday.memory.supervision import export_training_set

    assert_fact(conn, fact(subject="user", predicate="lives_in", object="Chennai",
                           valid_from=_t(-100), source_quote="i moved to chennai"))
    out = export_training_set(conn, str(tmp_path / "sup.jsonl"))
    assert out["rows"] >= 1
    text = Path_read(tmp_path / "sup.jsonl")
    assert "alpha_target" in text


def Path_read(p):
    return p.read_text(encoding="utf-8")


def test_schema_is_idempotent(tmp_path):
    from friday.store import db

    p = tmp_path / "t.db"
    c1 = db.connect(p)
    c2 = db.connect(p)  # second apply must not raise
    assert c2.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    c1.close(); c2.close()
