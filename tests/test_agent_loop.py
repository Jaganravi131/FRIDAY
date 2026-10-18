"""The agent loop's contract with the outside world, past one process lifetime.

Covers the properties an external review (2026-10-16) found missing or broken:

  * PERSISTENCE — a conversation must survive a restart. Both halves of an
    exchange go to `turns`, ordered by a session-scoped sequence that is NOT the
    tool-loop counter (which resets every call). Without this, `_recent_transcript`
    replayed only assistant answers: the user's words — corrections, constraints,
    preferences — existed nowhere the next process could read them.
  * CONFIRMATION IS NOT PERMISSION — a human "yes" must be the final approval of
    an action that is otherwise permitted, never a substitute for the sense being
    granted/enabled/in-scope.
  * TOOL RESULTS ARE NOT USER TURNS — they reach the model under the tool role,
    so nothing a tool returned can be mistaken for something the human said.
  * OUTCOMES ARE EXPLICIT — `ok: "denied" not in output` mislabelled errors and
    cancellations as successes, and budget exhaustion had no label at all.
  * SPANS ANCHOR TO REAL FILES — the salience tap must record the trace file the
    writer actually returned, not a filename guessed from a trace id.
"""
from __future__ import annotations

import json

from friday.agent.loop import Agent
from friday.agent.tools import ToolContext, ToolRegistry, ToolSpec, build_registry
from friday.llm.client import MockClient, Response, ToolCall
from friday.memory.traces import TraceWriter


# ── helpers ────────────────────────────────────────────────────────────────────


class ScriptedClient:
    """A client that says exactly what the test needs, in order.

    MockClient is retrieval-driven and only ever calls memory_search/memory_write;
    the policy gates need a client willing to ask for anything.
    """

    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=768):
        self.calls.append(messages)
        if not self.responses:
            return Response(text="done", model=self.name)
        r = self.responses.pop(0)
        if isinstance(r, str):
            return Response(text=r, model=self.name)
        name, args = r
        return Response(text="", tool_calls=[ToolCall(name, args)], model=self.name)


class EndlessToolClient:
    """Never answers; always wants one more tool call. For exhaustion."""

    name = "endless"

    def __init__(self):
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=768):
        self.calls.append(messages)
        return Response(text="", tool_calls=[
            ToolCall("memory_search", {"query": "anything"})], model=self.name)


def stub_email_registry() -> ToolRegistry:
    """The real registry with `email_send` swapped for a recorder.

    The policy gates don't care what the handler does; the test needs to know
    exactly whether it ran.
    """
    reg = build_registry()
    fired: list[dict] = []

    def _record(ctx: ToolContext, **args):
        fired.append(args)
        return {"sent": True, "to": args.get("to")}

    reg.register(ToolSpec("email_send", "stub", {"type": "object", "properties": {}}, _record))
    return reg, fired


# ── confirmation is the last gate, never a substitute for permission ──────────


def test_a_confirmed_yes_cannot_bypass_an_ungranted_sense(root):
    """⭐ The regression the review found: check() answered `confirmation_required`
    before consulting the sense registry, and the loop treats that verdict as
    "checked, awaiting the human" — so confirm→invoke skipped the permission
    entirely. With the sense never granted, a "yes" must still do nothing."""
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        reg, fired = stub_email_registry()
        ag = Agent(conn, client=ScriptedClient([("email_send", {"to": ["x@y.com"]}), "sent"]),
                   registry=reg, confirm=lambda name, args: True)
        st = ag.run("send the report")
        assert st.tool_log and st.tool_log[0].get("denied"), "the call must be refused"
        assert fired == [], "a confirmed yes must not execute an ungranted sense"
        assert "never been granted" in st.tool_log[0].get("reason", "")


def test_a_confirmed_yes_cannot_bypass_a_disabled_sense(root):
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        from friday.agent.policy import grant

        grant(conn, "email.send")
        conn.execute("UPDATE senses SET enabled=0 WHERE id='email.send'")
        reg, fired = stub_email_registry()
        ag = Agent(conn, client=ScriptedClient([("email_send", {"to": ["x@y.com"]}), "sent"]),
                   registry=reg, confirm=lambda name, args: True)
        st = ag.run("send the report")
        assert st.tool_log[0].get("denied")
        assert fired == [], "a confirmed yes must not execute a switched-off sense"
        assert "switched off" in st.tool_log[0].get("reason", "")


def test_a_confirmed_yes_cannot_bypass_a_scoped_grant(root):
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        from friday.agent.policy import grant

        grant(conn, "email.send", scope={"allow": ["gmail.com"]})
        reg, fired = stub_email_registry()
        ag = Agent(conn, client=ScriptedClient([
            ("email_send", {"to": ["leak@evil.example"]}), "sent"]),
            registry=reg, confirm=lambda name, args: True)
        st = ag.run("send the report")
        assert st.tool_log[0].get("denied")
        assert fired == [], "a confirmed yes must not widen a scoped grant"
        assert "outside its scope" in st.tool_log[0].get("reason", "")


def test_a_granted_confirmed_call_does_execute(root):
    """The gate must not overcorrect: granted, in scope, confirmed — then it runs."""
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        from friday.agent.policy import grant

        grant(conn, "email.send", scope={"allow": ["gmail.com"]})
        reg, fired = stub_email_registry()
        ag = Agent(conn, client=ScriptedClient([
            ("email_send", {"to": ["mom@gmail.com"]}), "sent"]),
            registry=reg, confirm=lambda name, args: True)
        st = ag.run("send the report")
        assert fired and fired[0]["to"] == ["mom@gmail.com"]
        assert not st.tool_log[0].get("denied")


def test_a_declined_confirmation_is_recorded_as_cancelled(root):
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        from friday.agent.policy import grant

        grant(conn, "email.send")
        reg, fired = stub_email_registry()
        ag = Agent(conn, client=ScriptedClient([
            ("email_send", {"to": ["x@y.com"]}), "ok"]),
            registry=reg, confirm=lambda name, args: False)
        st = ag.run("send the report")
        assert fired == []
        assert st.tool_log[0].get("cancelled") is True


# ── persistence: the conversation outlives the process ─────────────────────────


def _rows(conn, sid):
    """(idx, role) in reading order: pairs share an exchange idx, user before
    assistant ('user' > 'assistant' alphabetically, so role DESC is chrono)."""
    return conn.execute(
        "SELECT idx, role FROM turns WHERE session_id=? ORDER BY idx, role DESC",
        (sid,)).fetchall()


def test_both_halves_of_every_exchange_are_persisted_in_order(root):
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        ag = Agent(conn, client=MockClient())
        sid = ag.run("my rent is 25000 INR").session_id
        ag.run("what is my monthly rent", session_id=sid)
        roles = [(r["idx"], r["role"]) for r in _rows(conn, sid)]
        # two exchanges, each a (user, assistant) pair sharing one sequence number
        assert roles == [(0, "user"), (0, "assistant"),
                         (1, "user"), (1, "assistant")], roles


def test_a_ten_turn_conversation_survives_a_restart(root):
    """⭐ The review's acceptance test: after a restart, a fresh Agent continuing
    the session must still see what the user actually said — corrections and
    constraints included — and answer from the facts those turns established."""
    from friday.store import db

    exchanges = [
        "my rent is 18000 INR",
        "my landlord is Kumar",
        "actually my rent went up to 19000 INR",
        "my manager is Priya Raman",
        "call me short names only",
        "I prefer quiet mornings",
        "my lease renews in October",
        "I work at Acme Systems",
        "I live in Chennai",
        "what is my monthly rent",
    ]
    with db.connect(root / "artifacts" / "friday.db") as conn:
        ag = Agent(conn, client=MockClient())
        sid = ag.run(exchanges[0]).session_id
        for turn in exchanges[1:]:
            ag.run(turn, session_id=sid)

        # RESTART: a brand-new Agent over the same database continues the session
        ag2 = Agent(conn, client=MockClient())
        st = ag2.run("what is my monthly rent", session_id=sid)

        # the replayed transcript contains the user's words, in order, not just
        # assistant answers
        replayed = [t for t in st.transcript if t["role"] == "user"]
        assert any("went up to 19000" in t["content"] for t in replayed), (
            "the correction must survive the restart")
        # and the answer comes from the corrected fact
        assert "19000" in st.final_text.replace(",", ""), st.final_text


def test_the_tool_loop_counter_does_not_drive_the_persistence_order(root):
    """`turn_idx` counts tool-loop iterations and resets every call; using it as
    the turns-table sequence made every run's rows collide (both runs wrote
    idx=1). The sequence must be allocated per session instead."""
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        ag = Agent(conn, client=MockClient())
        sid = ag.run("my rent is 25000 INR").session_id   # a tool call happens here
        ag.run("what is my monthly rent", session_id=sid)  # so does one here
        rows = _rows(conn, sid)
        idxs = [r["idx"] for r in rows]
        assert idxs == sorted(idxs), "sequence must be monotonic across runs"
        assert set(idxs) == {0, 1}, idxs   # two exchanges, distinct numbers
        assert [r["role"] for r in rows] == ["user", "assistant"] * 2, rows


# ── tool results are not user turns ────────────────────────────────────────────


def test_tool_results_reach_the_model_as_tool_role(seeded):
    conn, _ = seeded
    ag = Agent(conn, client=MockClient())
    st = ag.run("my rent is 25000 INR")
    tool_msgs = [m for m in (st.messages or []) if m.get("role") == "tool"]
    assert tool_msgs, "the tool result must be in the assembled messages"
    assert all(m["content"].startswith("[memory_write]") for m in tool_msgs)
    assert all(m.get("name") == "memory_write" for m in tool_msgs)
    # and nothing a tool said may pose as the user
    posing = [m for m in (st.messages or [])
              if m.get("role") == "user" and "[tool result]" in (m.get("content") or "")]
    assert posing == []


# ── outcomes are explicit, not inferred from a substring ───────────────────────


def _last_trace(st):
    """The assistant trace for a run, read back from the day files. The user
    trace carries st.trace_id; the assistant trace is the last 'turn' record for
    the same session."""
    from friday import paths

    found = None
    for p in sorted(paths.TRACES.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") == "turn" and rec.get("role") == "assistant" \
                        and rec.get("session_id") == st.session_id:
                    found = rec
    return found


def test_tool_calls_carry_explicit_outcome_labels(root):
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        from friday.agent.policy import grant

        grant(conn, "email.send")
        reg, fired = stub_email_registry()
        answers = iter([False, True])          # decline the first, confirm the second
        ag = Agent(conn, client=ScriptedClient([
            ("email_send", {"to": ["x@y.com"]}),      # declined -> cancelled
            ("email_send", {"to": ["x@y.com"]}),      # confirmed -> executed
            "all sent"]),
            registry=reg, confirm=lambda name, args: next(answers))
        st = ag.run("send the report twice", max_turns=6)
        assert len(fired) == 1, "only the confirmed call may execute"
        rec = _last_trace(st)
        outcomes = [(c["name"], c["outcome"], c["ok"]) for c in rec["tool_calls"]]
        assert outcomes[0] == ("email_send", "cancelled", False), outcomes
        assert outcomes[1] == ("email_send", "succeeded", True), outcomes
        assert rec["outcome"] == "succeeded"


def test_budget_exhaustion_is_an_explicit_outcome(root):
    """A loop that hits max_turns without answering is not a success with no
    tool calls — it is its own outcome, named in the trace."""
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        ag = Agent(conn, client=EndlessToolClient())
        st = ag.run("chase this forever", max_turns=2)
        assert st.exhausted
        assert "stopped" in st.final_text
        rec = _last_trace(st)
        assert rec["outcome"] == "budget_exhausted"
        # the tool calls that DID happen are still labelled individually
        assert all(c["outcome"] == "succeeded" for c in rec["tool_calls"])
        assert len(rec["tool_calls"]) == 2


# ── spans anchor to the file the writer returned ───────────────────────────────


def test_the_salience_span_points_at_the_real_trace_file(root):
    """⭐ The span used to be built from `{trace_id[:10]}.jsonl` — an id, not a
    date — so every trace-anchored span pointed at a file that has never existed
    and `span_text` resolved to "". The writer knows the real path; it must be
    the one recorded."""
    from pathlib import Path

    from friday import paths
    from friday.store import db

    with db.connect(root / "artifacts" / "friday.db") as conn:
        ag = Agent(conn, client=MockClient())
        ag.run("my rent is 25000 INR")   # memory_write -> fact -> span
        row = conn.execute(
            """SELECT trace_id, trace_file, char_start, char_end FROM supervision_spans
               ORDER BY rowid DESC LIMIT 1""").fetchone()
        assert row is not None, "a trace-anchored fact must produce a span"
        f = row["trace_file"]
        assert f and f.endswith(".jsonl"), f
        assert paths.TRACES in Path(f).parents, (
            "the span must point inside the traces directory, at the day file")
        assert f != str(paths.TRACES / f"{row['trace_id'][:10]}.jsonl"), (
            "the span must not use the id-as-date reconstruction")
        # and the span resolves back to real text from the actual user turn
        writer = TraceWriter()
        text = writer.span_text(f, row["char_start"], row["char_end"])
        assert "my rent is 25000 INR" in text, "span must resolve to this turn's line"
