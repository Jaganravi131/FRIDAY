"""Doc 08's heartbeat, Week-9 layer: tick-ack schema, HEARTBEAT_OK suppression,
quiet hours, the annoyance budget, dedup, and the ONE standing check.

The property under test is the one doc 08 calls decisive: A HEARTBEAT WITH
NOTHING TO SAY STAYS SILENT, and a silent tick is logged, never delivered.
Every rule in the module docstring has a test here; the seeded HEARTBEAT.md
mirrors them, and the two must not drift apart.
"""

from __future__ import annotations

import json
from datetime import datetime


from friday.agent.heartbeat import HEARTBEAT_OK, Tick, beat
from friday.memory.facts import Fact, assert_fact


def _at(y, m, d, hh=12, mm=0):
    return datetime(y, m, d, hh, mm)


def _fact(conn, predicate, obj, **kw) -> str:
    """Assert one fact and return its row id (assert_fact returns a report dict)."""
    res = assert_fact(conn, Fact(subject="user", predicate=predicate, object=obj,
                                 source_kind="stated", source_quote="test", **kw))
    return res["written"]["id"]


def _ticks(conn):
    return conn.execute(
        "SELECT decision, detail FROM audit WHERE action='heartbeat.tick'"
        " ORDER BY id").fetchall()


def _delivered(conn):
    return conn.execute(
        "SELECT target FROM audit WHERE action='heartbeat.notify'"
        " AND decision='delivered' ORDER BY id").fetchall()


# ── silence is the norm ────────────────────────────────────────────────────────


def test_an_empty_store_is_silent_and_logged(conn):
    tick = beat(conn, now=_at(2026, 10, 16))
    assert tick.is_ok and tick.action == "silent"
    rows = _ticks(conn)
    assert len(rows) == 1
    assert rows[0]["decision"] == "silent" and rows[0]["detail"] == HEARTBEAT_OK


def test_junk_objects_never_raise_and_stay_silent(conn):
    for obj in ("vitamin D3 weekly", "Tuesday 09:30 IST", "", "₹28,000", "O positive",
                "2026-13-99", "not a date at all"):
        _fact(conn, f"junk_{abs(hash(obj)) % 10000}", obj)
    tick = beat(conn, now=_at(2026, 10, 16))
    assert tick.is_ok


def test_far_future_and_long_past_dates_stay_silent(conn):
    _fact(conn, "passport_expiry", "2031-01-01")
    _fact(conn, "joined_company", "2023-01-15")
    assert beat(conn, now=_at(2026, 10, 16)).is_ok


def test_retracted_facts_do_not_warn(conn):
    f = _fact(conn, "lease_renewal_due", "2026-10-20")
    assert_fact(conn, Fact(subject="user", predicate="lease_renewal_due",
                           object="2027-04-20", source_kind="stated",
                           source_quote="renewed"))
    conn.execute("UPDATE facts SET retracted_at=? WHERE id=?",
                 ("2026-10-10T00:00:00", f))
    conn.commit()
    assert beat(conn, now=_at(2026, 10, 16)).is_ok


# ── the one standing check: the deadline watch ─────────────────────────────────


def test_a_deadline_inside_the_window_notifies_once(conn):
    f = _fact(conn, "lease_renewal_due", "2026-10-26")
    t1 = beat(conn, now=_at(2026, 10, 16))
    assert t1.action == "notify" and t1.urgency == "normal"
    assert "lease renewal" in t1.message.lower()
    assert "2026-10-26" in t1.message and "10 days" in t1.message
    assert t1.reason and t1.check == f
    assert [r["target"] for r in _delivered(conn)] == [f]
    # ⭐ dedup: the same deadline never speaks twice — the audit trail is the memory
    t2 = beat(conn, now=_at(2026, 10, 17))
    assert t2.is_ok, "a deadline must notify ONCE, not every tick"


def test_a_close_deadline_is_high_urgency_and_names_the_gap(conn):
    _fact(conn, "lease_renewal_due", "2026-10-18")
    t = beat(conn, now=_at(2026, 10, 16))
    assert t.urgency == "high" and "2 days" in t.message
    _fact(conn, "visa_interview", "2026-10-13")          # overdue, inside grace
    conn.execute("DELETE FROM audit")                    # reset the budget/dedup
    conn.commit()
    t2 = beat(conn, now=_at(2026, 10, 16))
    assert t2.urgency == "high" and "was due 2026-10-13" in t2.message


def test_one_message_per_tick_batches_the_rest(conn):
    a = _fact(conn, "lease_renewal_due", "2026-11-01")
    b = _fact(conn, "passport_expiry", "2026-11-10")
    t1 = beat(conn, now=_at(2026, 10, 16))
    assert t1.action == "notify" and t1.check == a, "nearest deadline first"
    t2 = beat(conn, now=_at(2026, 10, 16, 12, 1))
    assert t2.action == "notify" and t2.check == b, "batched, not burst"


def test_a_check_that_raises_is_silence_not_a_crash(conn):
    def broken(conn, now):
        raise RuntimeError("check bug")
    tick = beat(conn, now=_at(2026, 10, 16), checks=(broken,))
    assert tick.is_ok


def test_an_empty_checklist_stays_silent(conn):
    """Doc 08 Week 9: run with an EMPTY checklist and verify it stays silent.
    The engine level version of the 3-day empty-checklist soak."""
    assert beat(conn, now=_at(2026, 10, 16), checks=()).is_ok


# ── quiet hours and the annoyance budget ───────────────────────────────────────


def test_quiet_hours_suppress_all_but_high_urgency(conn):
    _fact(conn, "dentist_appointment", "2026-11-02")     # 17 days out: normal
    t = beat(conn, now=_at(2026, 10, 16, 23, 30))
    assert t.is_ok and "quiet hours" in t.reason
    assert not _delivered(conn), "a suppressed message must not be counted as delivered"
    conn.execute("DELETE FROM audit")
    conn.commit()
    _fact(conn, "lease_renewal_due", "2026-10-18")       # 2 days out: high
    t2 = beat(conn, now=_at(2026, 10, 16, 23, 30))
    assert t2.action == "notify" and t2.urgency == "high"


def test_the_annoyance_budget_caps_delivery(conn):
    for i, day in enumerate(("2026-11-01", "2026-11-02")):
        _fact(conn, f"watch_{i}", day)
    # two already delivered inside the last hour of the TICK's clock (not the
    # wall clock — the budget is measured against the tick's own `now`)
    for i in (1, 2):
        ts = _at(2026, 10, 16, 11, 30).astimezone().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO audit(ts, actor, action, target, sense_id, decision, detail)"
            " VALUES (?,?,?,?,?,?,?)",
            (ts, "heartbeat", "heartbeat.notify", f"f_x{i}", None, "delivered", "earlier"))
    conn.commit()
    t = beat(conn, now=_at(2026, 10, 16))
    assert t.is_ok and "annoyance budget" in t.reason
    # and it must NOT have been counted as delivered for the fact it held back
    assert t.next_check_hint


# ── the tick-ack schema ─────────────────────────────────────────────────────────


def test_the_tick_schema_is_doc_08s(conn):
    """action/urgency/message/surface/reason/open_loops_touched/next_check_hint
    — the fields the Presence Fabric's routing policy consumes. A missing field
    here is a silent contract break downstream."""
    t = Tick()
    d = t.as_dict()
    assert set(d) == {"action", "urgency", "message", "surface", "reason",
                      "open_loops_touched", "next_check_hint"}
    assert d["action"] == "silent" and t.is_ok


def test_the_seeded_store_beats_cleanly(seeded):
    """End to end against the real seeded corpus. The seed carries exactly one
    date-bearing live fact (lease_renewal_due 2026-10-15); the day after, it
    earns one high-urgency word, and exactly one."""
    conn, _stats = seeded
    t = beat(conn, now=_at(2026, 10, 16))
    assert t.action == "notify" and t.urgency == "high"
    assert "lease renewal" in t.message.lower()
    assert beat(conn, now=_at(2026, 10, 17)).is_ok


# ── CLI ────────────────────────────────────────────────────────────────────────


def test_the_cli_subcommand_is_wired(tmp_path, monkeypatch, capsys):
    from friday import cli, paths, seed
    from friday.memory import compiler
    from friday.store import db

    monkeypatch.setenv("FRIDAY_ROOT", str(tmp_path))
    saved_root = paths.ROOT
    try:
        paths.rebind(tmp_path)
        paths.ensure_layout()
        seed.write_all()
        conn = db.connect(tmp_path / "artifacts" / "friday.db")
        try:
            compiler.compile_all(conn)
        finally:
            conn.close()

        db_arg = str(tmp_path / "artifacts" / "friday.db")
        code = cli.main(["--db", db_arg, "heartbeat"])
        out = capsys.readouterr().out
        assert code == 0, "a silent tick is SUCCESS — that is the whole design"
        assert "heartbeat" in out.lower()

        code = cli.main(["--db", db_arg, "heartbeat", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["action"] in ("silent", "notify")
        assert set(data) >= {"action", "urgency", "message", "reason"}
    finally:
        paths.rebind(saved_root)
        monkeypatch.undo()
