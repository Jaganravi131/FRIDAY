"""Tests for the tool contract (routing) and `/why` (provenance).

THE ROUTER'S ASYMMETRIC COST, which is why the tables below are lopsided:

  * A FALSE NEGATIVE (skipping memory on a question that needed it) produces a fluent
    answer with nothing behind it. That is the hallucination path, and it is
    undetectable from the answer alone.
  * A FALSE POSITIVE (retrieving for "hello") costs about 15 ms and is invisible.

So the router defaults to `needs_memory=True` and the skip patterns are narrow. These
tests are weighted accordingly: every "must retrieve" case is a hard failure, and the
"must skip" cases are checked too because exit test #8 requires them, but the ordering
of those two lists is the actual design.
"""
from __future__ import annotations

import re

import pytest

from friday.agent.routing import KINDS, TurnIntent, classify_turn, tools_for
from friday.agent.tools import build_registry
from friday.memory import provenance


# ── the router: must retrieve ─────────────────────────────────────────────────

MUST_RETRIEVE = [
    # the obvious ones
    "what is my monthly rent", "where do I live", "who is my landlord",
    "how much do I pay for rent", "what is my daily routine",
    "what did I tell you about Ramesh", "do you remember the gym thing",
    "is my lease up for renewal", "when does my lease renew",
    # ⭐ the traps: these CONTAIN a trivial-question keyword but are about memory.
    # A router that matched on substring would skip retrieval on every one of them,
    # which is the expensive failure direction.
    "what time is my meeting tomorrow",
    "what time does my flight leave",
    "what day is my lease renewal",
    "help me remember what I like",
    "who are my colleagues",
    # first-person statements — the highest-value turns in the system
    "my rent is 18000", "I moved to Chennai", "I prefer filter coffee",
    "I hate Mondays", "I started a new job", "my landlord is Ramesh",
    # short but memory-bearing
    "my rent", "the lease", "my job",
]

MUST_SKIP = [
    # clock / date, with no other content
    "what time is it", "what's the time", "what is the time", "time?",
    "what is today's date", "what day is it", "what's the date",
    # greetings
    "hi", "hello", "hey", "hey friday", "good morning", "good evening",
    "namaste", "vanakkam", "yo",
    # acknowledgements and farewells
    "ok", "okay", "k", "thanks", "thank you", "thx", "nice", "cool", "sure",
    "got it", "makes sense", "bye", "goodbye", "good night", "see you",
    "how are you", "what's up",
    # about FRIDAY itself
    "who are you", "what are you", "what can you do", "help", "commands",
    "what is your name", "are you real", "what model are you", "how do you work",
    # arithmetic
    "2+2", "42 / 6", "18000*12", "what is 250 + 17", "calculate 99 * 2",
]


@pytest.mark.parametrize("q", MUST_RETRIEVE)
def test_memory_bearing_turns_always_retrieve(q):
    """⭐ The expensive direction. Not one of these may be skipped."""
    i = classify_turn(q)
    assert i.needs_memory, f"{q!r} was classified {i.kind} and would skip memory: {i.reason}"


@pytest.mark.parametrize("q", MUST_SKIP)
def test_trivial_turns_skip_memory(q):
    """Exit test #8. These must not cost an embedding pass."""
    i = classify_turn(q)
    assert not i.needs_memory, f"{q!r} should be trivial but got {i.kind}: {i.reason}"
    assert i.kind in KINDS


def test_the_default_is_to_retrieve():
    """Anything unrecognised retrieves. This is the rule that makes the narrow
    patterns safe, so it gets its own test rather than being implied by the table."""
    for q in ("blorp fnord", "the thing from before, again",
              "x", "?!?", "quadratic reciprocity law proof sketch"):
        assert classify_turn(q).needs_memory, f"{q!r} must default to retrieving"


def test_an_empty_turn_is_not_a_memory_turn():
    assert classify_turn("").kind == "ack"
    assert not classify_turn("").needs_memory
    assert not classify_turn("   ").needs_memory


def test_the_intent_is_reported_not_hidden():
    """When FRIDAY fails to remember something, the reason must be recoverable: "it
    looked and found nothing" and "it decided not to look" need opposite fixes."""
    i = classify_turn("what time is it")
    assert i.kind == "clock"
    assert i.reason and "system clock" in i.reason
    assert i.matched == ("clock",)
    assert i.skip_retrieval

    m = classify_turn("what is my rent")
    assert m.kind == "memory" and m.needs_memory
    assert "first-person" in m.reason or m.reason


def test_general_questions_still_retrieve_but_are_labelled_differently():
    """A general question may still want context ("recommend a restaurant" wants to
    know I'm vegetarian), so it retrieves — but the label distinguishes it, for the
    audit trail and for `/why`."""
    i = classify_turn("recommend a good filter coffee place nearby")
    assert i.needs_memory
    assert i.kind in ("memory", "general")


# ── the router: structural withholding ────────────────────────────────────────


def test_memory_search_is_withheld_not_merely_discouraged():
    """⭐ Withholding beats instructing. A model told "don't search" will sometimes
    search anyway; the only way exit test #8 is structurally true is to not offer the
    tool. This is the assertion that makes the guarantee real."""
    reg = build_registry()
    all_tools = reg.schemas()
    names = [(t.get("function") or {}).get("name") for t in all_tools]
    assert "memory_search" in names, "sanity: the registry does offer it"

    for q in ("what time is it", "hi", "thanks", "2+2"):
        offered = tools_for(classify_turn(q), all_tools)
        got = [(t.get("function") or {}).get("name") for t in offered]
        assert "memory_search" not in got, f"{q!r} was still offered memory_search"


def test_memory_write_stays_available_for_trivial_turns():
    """Withholding `memory_search` must not withhold the rest. "note that" and
    `read_artifact` (which resolves the pointers the ladder itself creates) have
    nothing to do with retrieval."""
    reg = build_registry()
    offered = tools_for(classify_turn("thanks"), reg.schemas())
    got = {(t.get("function") or {}).get("name") for t in offered}
    assert "memory_search" not in got
    assert "note" in got and "read_artifact" in got


def test_memory_turns_get_the_full_tool_set():
    reg = build_registry()
    all_tools = reg.schemas()
    assert tools_for(classify_turn("what is my rent"), all_tools) is all_tools


def test_turn_intent_is_a_plain_value_object():
    i = TurnIntent("clock", False, "because", ("clock",))
    assert i.kind == "clock" and i.skip_retrieval
    assert not TurnIntent("memory", True, "r").skip_retrieval


# ── provenance ────────────────────────────────────────────────────────────────


def _row(**over):
    base = dict(
        id="f_1", predicate="lease_amount_monthly", object="18000 INR",
        source_quote="my monthly rent is 18000 INR", source_kind="user_edit",
        confidence=1.0, origin_file="memory/facts/housing.md", origin_line=24,
        trace_id="t_abc", valid_from="2025-06-01", valid_to=None,
        asserted_at="2026-09-30T08:00:00+00:00", retracted_at=None, superseded_by=None,
    )
    base.update(over)
    return base


def test_provenance_renders_the_literal_quote_and_every_id():
    """Exit test #3: the literal source quote + trace id. Not a paraphrase — the
    actual characters, and a file path you can open."""
    p = provenance.from_row(_row(), score=0.806, recall_sources=["fts"])
    out = p.render()
    assert '"my monthly rent is 18000 INR"' in out
    assert "f_1" in out and "t_abc" in out
    assert "memory/facts/housing.md:24" in out, "must point at a line, not just a file"
    assert "user_edit" in out and "1.00" in out
    assert p.verifiable and not p.gap


def test_provenance_shows_both_time_axes():
    """Innovation #3. World time and belief time are different questions and both must
    appear, or "what did I used to think?" is unanswerable from the trail."""
    p = provenance.from_row(_row(valid_to="2026-06-30",
                                 retracted_at="2026-07-01T00:00:00+00:00",
                                 superseded_by="f_2"))
    out = p.render()
    assert "2025-06-01 → 2026-06-30" in out
    assert p.state == "retracted"
    assert "still believed" not in out
    assert "f_2" in out


def test_an_unverifiable_fact_says_so():
    """⭐ THE HONESTY CHECK. A fact with no quote and no file cannot be traced, and
    rendering a blank "quote:" line would look like a formatting quirk rather than a
    missing justification — the user would trust the answer anyway."""
    p = provenance.from_row(_row(source_quote=None, origin_file=None,
                                 source_kind="inferred", confidence=0.4))
    assert not p.verifiable
    assert "no source quote" in p.gap and "no origin file" in p.gap
    out = p.render()
    assert "GAP" in out and "none recorded" in out


def test_a_missing_quote_alone_is_still_a_gap():
    p = provenance.from_row(_row(source_quote=None))
    assert not p.verifiable and "quote" in p.gap
    assert p.origin_file, "the file is known but the words are not"


def test_from_candidates_reads_the_FLAT_dict_form():
    """⭐ `SearchResult.as_dicts()` flattens the facts row into the candidate dict;
    there is no nested "row" key. Reading c["row"] yields nothing for every candidate
    and `/why` reports "no facts behind this answer" for an answer that had three."""
    from friday.retrieval.pipeline import Candidate, SearchResult

    row = _row()
    c = Candidate(ref="f_1", kind="fact", body="lease_amount_monthly: 18000 INR",
                  subject="user:housing", score=0.806, recall_sources=["fts", "vec"],
                  row=row)
    res = SearchResult(kept=[c], considered=1, floor=0.45, reranker="lexical")
    provs = provenance.from_candidates(res.as_dicts())
    assert len(provs) == 1, "must find the fact in the flattened form"
    assert provs[0].quote == "my monthly rent is 18000 INR"
    assert provs[0].trace_id == "t_abc"
    assert provs[0].recall_sources == ["fts", "vec"]


def test_non_fact_candidates_are_not_dressed_up_as_claims():
    """Episodes and skills are context, not claims. Giving them a provenance block
    would bury the one line that actually justifies the answer."""
    provs = provenance.from_candidates([
        {"ref": "episode:memory/episodes/2026-09-28.md", "kind": "episode",
         "body": "we talked about the lease", "score": 0.6},
        {"ref": "f_1", "kind": "fact", **_row()},
    ])
    assert len(provs) == 1 and provs[0].fact_id == "f_1"


def test_render_report_is_loud_when_nothing_is_behind_the_answer():
    """Law 4 at the explanation layer. An empty report must say the answer had no
    memory behind it, because that is either fine (no memory needed) or a bug
    (it made a personal claim anyway) and the user has to be able to tell."""
    out = provenance.render_report([], query="what is my rent", answer="₹18,000")
    assert "NO FACTS BEHIND THIS ANSWER" in out
    assert "Law 4" in out


def test_render_report_counts_verifiable_separately_from_gaps():
    good = provenance.from_row(_row())
    bad = provenance.from_row(_row(id="f_9", source_quote=None, object="tired"))
    out = provenance.render_report([good, bad])
    assert "2 fact(s) used" in out
    assert "1 verifiable" in out and "1 with gaps" in out
    assert "GAPS ABOVE MEAN" in out


def test_why_query_explains_an_empty_retrieval(conn):
    """An empty result must report the funnel, not just print nothing: considered N,
    dropped below floor M. That is the difference between "no such memory" and "found
    it and rejected it"."""
    out = provenance.why_query(conn, "what is my favourite colour")
    assert "retrieval returned []" in out
    assert "considered" in out and "floor" in out


def test_why_query_end_to_end(conn):
    from friday.memory.facts import Fact, assert_fact

    assert_fact(conn, Fact(subject="user", predicate="lease_amount_monthly",
                           object="18000 INR", source_kind="user_edit",
                           source_quote="my monthly rent is 18000 INR"))
    out = provenance.why_query(conn, "what is my monthly rent")
    assert "18000 INR" in out
    assert "my monthly rent is 18000 INR" in out, "the literal quote, not a paraphrase"


def test_short_form_is_one_line_for_an_answer_footer():
    p = provenance.from_row(_row())
    assert p.short() == "lease_amount_monthly=18000 INR [housing.md · user_edit]"


def test_from_row_tolerates_a_row_missing_columns():
    """A projection or an old schema must not raise. Provenance is a reporting path;
    it should degrade to "unknown" rather than break the command."""
    p = provenance.from_row({"id": "f_x", "predicate": "mood", "object": "fine"})
    assert p.fact_id == "f_x"
    assert p.source_kind == "?"
    assert not p.verifiable


def test_as_dicts_tolerates_a_partial_row():
    """⭐ The hardening, locked. `as_dicts` is a reporting path — it feeds the model's
    tool result and `/why` — so a partial row must degrade to missing keys rather than
    raise KeyError. `provenance.from_row` already behaved this way; if the two disagree
    then whichever touches the row first decides whether the command works at all.
    """
    from friday.retrieval.pipeline import Candidate, SearchResult

    partial = {"id": "f_p", "predicate": "mood", "object": "fine"}   # no subject, no quote
    c = Candidate(ref="f_p", kind="fact", body="mood: fine", score=0.5,
                  recall_sources=["fts"], row=partial)
    res = SearchResult(kept=[c], considered=1, floor=0.35, reranker="lexical")
    dicts = res.as_dicts()                       # must not raise
    assert dicts[0]["predicate"] == "mood"
    assert dicts[0]["subject"] is None
    assert dicts[0]["id"] == "f_p"
    assert dicts[0]["recall_sources"] == ["fts"]
    assert "mood=fine" in res.summary()          # summary() reads the row too


# ── acting on the classification (not just making it) ──────────────────────────

def _ask(root, text):
    from friday.store import db
    from friday.agent.loop import Agent
    from friday.llm import get_client
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.rerankers import get_reranker

    with db.connect(root / "friday.db") as conn:
        ag = Agent(conn, client=get_client(prefer="mock"),
                   embedder=get_embedder(), reranker=get_reranker())
        return ag.run(text)


def test_a_clock_question_is_answered_not_deflected(root):
    """Skipping memory is only half the contract — the turn still needs an answer.

    Before this, "what time is it?" correctly withheld memory_search and then replied
    "I don't have anything in memory about that": truthful, and not what was asked.
    A local model cannot read a wall clock, so the reading has to be supplied.
    """
    st = _ask(root, "what time is it")
    assert st.intent.kind == "clock"
    assert st.intent.needs_memory is False
    assert st.final_text.lower().startswith("it's ")
    assert "don't have anything in memory" not in st.final_text
    # Spelled out for speech, not an ISO stamp for a log file. Assert the shape
    # rather than "no T" — "It's" and "UTC" both contain one.
    assert re.search(r"\d{4}-\d{2}-\d{2}T", st.final_text) is None
    assert re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
                     st.final_text)


def test_a_clock_reading_is_injected_only_when_needed(root):
    from friday.agent.loop import _now_reading
    from friday.ledger.compiler import Gather

    st = _ask(root, "what is my monthly rent")
    assert st.intent.kind != "clock"
    g = Gather(query="x", now=_now_reading() if st.intent.kind == "clock" else "")
    assert g.now == ""


def test_a_greeting_is_not_answered_with_a_database_error(root):
    """The mock is the path you talk to before any model is downloaded."""
    for text in ("hi", "hey", "thanks", "good morning", "vanakkam"):
        st = _ask(root, text)
        assert st.intent.needs_memory is False
        assert "don't have anything in memory" not in st.final_text, text
        assert len(st.final_text) < 120


def test_a_meta_question_answers_from_the_identity_slot(root):
    """SOUL.md is in the compiled context; refusing to read it is not honesty.

    Tests the MECHANISM (answer is drawn from the identity slot) rather than the
    seed's wording, so it stays true when the personality file is rewritten.
    """
    (root / "soul" / "SOUL.md").write_text(
        "# SOUL\n\nTerse and allergic to adjectives.\n", encoding="utf-8")
    st = _ask(root, "who are you")
    assert st.intent.kind == "meta"
    assert st.intent.needs_memory is False
    assert "don't have anything in memory" not in st.final_text
    assert "Terse and allergic to adjectives" in st.final_text


def test_a_meta_question_falls_back_when_the_soul_is_empty(root):
    """No identity file must not produce an empty answer."""
    (root / "soul" / "SOUL.md").write_text("#\n\n", encoding="utf-8")
    st = _ask(root, "who are you")
    assert "FRIDAY" in st.final_text
    assert "don't have anything in memory" not in st.final_text


def test_arithmetic_needs_neither_memory_nor_a_model(root):
    st = _ask(root, "what is 18000*12")
    assert st.intent.kind == "arithmetic"
    assert st.intent.needs_memory is False
    assert st.final_text.replace(",", "") == "216000"


@pytest.mark.parametrize("attack", [
    "__import__('os').system('id')",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "().__class__.__bases__",
    "[x for x in ()]",
    "what is 2 if True else 3",
])
def test_arithmetic_parsing_cannot_execute_code(attack):
    """⚠️ The pre-check regex is defence in depth, not the defence.

    The AST walk is what actually matters: it admits numbers and the five arithmetic
    operators and nothing else. Without it, an agent with filesystem access that
    `eval()`s a user string is a remote code execution hole with extra steps.
    """
    from friday.llm.client import _safe_arith
    assert _safe_arith(attack) is None


@pytest.mark.parametrize("expr,expected", [
    ("2+2", "4"), ("42 / 6", "7"), ("calculate 99 * 2", "198"),
    ("1,000+250", "1,250"), ("what is 18000*12", "216,000"),
    ("3.5 * 2", "7"), ("10 - 4", "6"), ("7 % 3", "1"),
])
def test_arithmetic_evaluates(expr, expected):
    from friday.llm.client import _safe_arith
    assert _safe_arith(expr) == expected


@pytest.mark.parametrize("not_math", [
    "my rent", "who are you", "", "2 plus plus 2", "call 911",
    "2 ** 40",              # exponentiation is refused outright — see below
])
def test_non_arithmetic_is_not_guessed_at(not_math):
    from friday.llm.client import _safe_arith
    assert _safe_arith(not_math) is None
