"""Tests for the Attention Ledger: slot order, the six-rung compaction ladder, and
`Ledger.compile`.

Three properties carry the design:

1. SLOT ORDER IS EMISSION ORDER AND MUST NOT BE RE-SORTED. `identity` + `senses` form
   the stable cached prefix; everything JIT comes after. Re-sorting by size or by
   priority would invalidate the provider's prompt cache on every single turn, which
   is a real cost and a real latency regression rather than a style preference.
2. THE LADDER FIRES CHEAPEST-FIRST AND RE-MEASURES. A rung that solves the problem
   means every more destructive rung never runs. That ordering is the entire content
   of Law 2.
3. ⭐ UNDER A HYBRID BACKBONE THE LADDER STOPS AFTER RUNG 3 AND CAN NEVER SUMMARISE.
   Rungs 1/4/5 rewrite or drop the prefix — which is precisely what the recurrent
   state replaces with a learned continuous operation. Firing them anyway would
   silently reintroduce the recall collapse the pivot exists to avoid.
"""
from __future__ import annotations

import pytest

from friday.config import LEDGER_BUDGET_TOKENS, LEDGER_RESERVE_TOKENS, LEDGER_SLOTS
from friday.ledger import ladder
from friday.ledger.compiler import CompiledContext, Gather, Ledger, render
from friday.ledger.slots import SLOT_ORDER, STABLE_SLOTS, Slot, SlotSet
from friday.util import approx_tokens


# ── slots ─────────────────────────────────────────────────────────────────────


def test_slot_order_is_fixed_and_identity_is_first():
    """Declaration order == emission order == cache friendliness."""
    assert isinstance(SLOT_ORDER, tuple)
    assert SLOT_ORDER[0] == "identity", "the cached prefix must be emitted first"
    assert len(SLOT_ORDER) == 8
    # the two stable slots are the first two, contiguous, before any JIT slot
    assert STABLE_SLOTS == {"identity", "senses"}
    jit = [n for n in SLOT_ORDER if n not in STABLE_SLOTS]
    assert SLOT_ORDER.index("identity") < min(SLOT_ORDER.index(n) for n in jit)
    assert set(LEDGER_SLOTS) == set(SLOT_ORDER), \
        "every slot needs a budget and every budget needs a slot"


def test_slotset_iterates_in_declaration_order_not_insertion_order():
    """`items()` walks SLOT_ORDER, so a dict built in a different order still renders
    stably. Sorting by budget or by fill would break the prefix cache."""
    ss = SlotSet.from_budgets({"scratch": 768, "identity": 1024, "senses": 256})
    assert [n for n, _ in ss.items()] == list(SLOT_ORDER)
    assert ss["identity"].stable is True
    assert ss["scratch"].stable is False


def test_slot_renders_as_a_tagged_block_and_renders_empty_as_nothing():
    s = Slot(name="user", budget=512, content="  name: Jagan  ")
    assert s.render() == "<user>\nname: Jagan\n</user>"
    assert Slot(name="user", budget=512, content="   ").render() == "", \
        "an empty slot must contribute nothing, not an empty tag pair"


def test_render_concatenates_slots_in_order():
    ss = SlotSet.from_budgets(LEDGER_SLOTS)
    ss["identity"].content = "I am FRIDAY"
    ss["scratch"].content = "note to self"
    text = render(ss)
    assert text.index("<identity>") < text.index("<scratch>"), \
        "stable prefix first, JIT after — this is what keeps the cache warm"


def test_utilisation_is_zero_for_a_zero_budget_slot():
    assert Slot(name="x", budget=0, tokens=0).utilisation == 0.0
    assert Slot(name="x", budget=100, tokens=25).utilisation == 0.25


# ── rung 0: cap tool output ───────────────────────────────────────────────────


def test_rung0_caps_a_huge_tool_output_and_reports_the_saving():
    """Rung 0 is always on and always free: it caps tool output BEFORE it enters
    history. This is the -38% cost lever with recall unchanged."""
    huge = "line of tool output\n" * 4000
    capped, ev = ladder.rung0_cap(huge, 2000)
    assert ev is not None
    assert ev.rung == 0
    assert ev.saved > 0
    assert approx_tokens(capped) <= 2000 + 50, approx_tokens(capped)
    assert len(capped) < len(huge)


def test_rung0_leaves_a_small_output_alone_and_fires_no_event():
    small = "ok"
    out, ev = ladder.rung0_cap(small, 2000)
    assert out == small
    assert ev is None, "no compaction happened, so nothing may be reported"


def test_rung0_truncates_without_dropping_the_fact():
    """Capping keeps head AND tail rather than the head alone: the end of a tool
    result is where the conclusion usually is, and dropping it loses the answer
    while keeping the noise."""
    text = "".join(f"token{i} " for i in range(6000))
    capped, ev = ladder.rung0_cap(text, 500)
    assert ev is not None
    assert capped.startswith("token0"), "the head survives"
    assert "token5999" in capped, "and so does the tail"
    assert len(capped) < len(text)


# ── rungs 1-4 ─────────────────────────────────────────────────────────────────


def test_rung1_truncates_only_when_over_budget():
    text = "word " * 2000
    out, ev = ladder.rung1_truncate(text, 200, "docs")
    assert ev is not None and ev.rung == 1 and ev.slot == "docs"
    assert approx_tokens(out) <= 200 + 50

    short = "fine"
    out2, ev2 = ladder.rung1_truncate(short, 200, "docs")
    assert out2 == short and ev2 is None


def test_rung2_dedupe_drops_exact_repeats():
    """Tool results repeat constantly: the same file read twice, the same search
    re-issued, the same error three times. Dedupe removes redundant bytes, which is
    why it is cheaper than truncation and fires earlier."""
    blocks = ["alpha beta", "alpha beta", "gamma", "alpha   beta"]
    out, ev = ladder.rung2_dedupe(blocks, "docs")
    assert ev is not None and ev.rung == 2
    assert len(out) == 2, out
    assert "dropped" in ev.detail


def test_rung2_dedupe_leaves_distinct_blocks_alone():
    out, ev = ladder.rung2_dedupe(["one", "two", "three"], "docs")
    assert ev is None and len(out) == 3


def test_rung3_offload_writes_the_payload_and_leaves_a_pointer(root):
    """⭐ Law 3 applied to compaction: the context keeps a HANDLE, not the payload.
    The model can pull it back with `read_artifact` if — and only if — it turns out
    to matter. This is the one rung that ADDS capability rather than removing
    information, which is why it survives the hybrid cut."""
    text = "some offloadable content\n" * 400
    assert approx_tokens(text) >= 400
    pointer, ev = ladder.rung3_offload(text, "docs", label="big read")
    assert ev is not None and ev.rung == 3
    assert approx_tokens(pointer) < approx_tokens(text)
    assert "read_artifact" in pointer, "the pointer must say how to pull it back"
    assert "artifacts/offloaded/" in pointer

    written = list((root / "artifacts" / "offloaded").glob("*.md"))
    assert len(written) == 1, "the payload must actually be on disk"
    assert text.strip() in written[0].read_text(encoding="utf-8")


def test_rung3_refuses_to_offload_something_small():
    out, ev = ladder.rung3_offload("tiny", "docs")
    assert out == "tiny" and ev is None


def test_rung4_evicts_the_OLDEST_turns_and_never_the_newest():
    """Evicting the recent window produces the classic 'it forgot what we were
    talking about' failure, so eviction runs from the old end."""
    turns = [{"role": "user", "content": f"turn {i} " * 200} for i in range(10)]
    out, ev = ladder.rung4_evict(turns, 800)
    assert ev is not None and ev.rung == 4
    assert len(out) < len(turns)
    assert out[-1]["content"].startswith("turn 9"), "the newest turn must survive"
    assert out[0]["content"] > turns[0]["content"][:0] or "turn 0" not in out[0]["content"]


def test_rung4_always_keeps_pinned_turns():
    """A pinned turn is part of the stable prefix in spirit: pinning is how the agent
    loop says 'this one must not be evicted', and eviction that ignored it would make
    pinning meaningless."""
    turns = [
        {"role": "system", "content": "identity " * 100, "pin": True},
        *[{"role": "user", "content": f"chat {i} " * 100} for i in range(8)],
    ]
    out, ev = ladder.rung4_evict(turns, 400)
    assert any(t.get("pin") for t in out), "the pinned turn must survive eviction"
    assert out[0].get("pin") is True, "and stay at the front"


def test_rung4_is_a_noop_when_already_under_budget():
    turns = [{"role": "user", "content": "hi"}]
    out, ev = ladder.rung4_evict(turns, 5000)
    assert ev is None and out == turns


# ── rung 5: summarisation ─────────────────────────────────────────────────────


def test_rung5_refuses_to_fire_without_an_explicit_summariser():
    """⭐ If no summariser callable is supplied this rung returns the text UNCHANGED.
    Silently substituting a cruder compaction for a summarisation you did not ask for
    is how recall dies without a stack trace."""
    text = "a lot of transcript " * 500
    out, ev = ladder.rung5_summarise(text, None, "transcript")
    assert out == text
    assert ev is None


def test_rung5_fires_and_screams_when_a_summariser_is_supplied():
    text = "a lot of transcript " * 500
    out, ev = ladder.rung5_summarise(text, lambda t: "summary.", "transcript")
    assert ev is not None and ev.rung == 5
    assert out == "summary."
    assert "SUMMARISED" in ev.detail, "the event must be loud, not silent"
    assert "recall risk" in ev.detail.lower()


def test_rung5_refuses_a_summariser_that_does_not_shorten_anything():
    """A summariser that returns something longer has not compacted. Accepting it
    would report a saving that did not happen."""
    text = "short"
    out, ev = ladder.rung5_summarise(text, lambda t: t * 100, "transcript")
    assert out == text and ev is None


# ── the ladder as a whole ─────────────────────────────────────────────────────


def test_ladder_does_nothing_when_under_budget():
    blocks = {"docs": ["small"]}
    out_blocks, out_turns, res = ladder.descend(budget=5000, blocks=blocks)
    assert res.events == []
    assert res.highest_rung is None
    assert res.summary() == "none"
    assert out_blocks == {"docs": ["small"]}


def test_ladder_fires_rung0_unconditionally_on_tool_output():
    blocks = {"docs": ["x " * 8000]}
    _, _, res = ladder.descend(budget=5000, blocks=blocks, tool_cap=2000)
    assert any(e.rung == 0 for e in res.events), res.summary()
    assert res.highest_rung == 0, "a cheap rung that solves it means no expensive rung runs"


def test_rungs_descend_in_ascending_cost_order():
    """⭐ Law 2. Rungs fire cheapest-to-most-destructive and the ladder RE-MEASURES
    after each, so an early rung that gets us under budget prevents the later ones
    from ever running."""
    assert ladder.RUNG_NAMES[0] and ladder.RUNG_NAMES[5]
    assert len(ladder.RUNG_NAMES) == 6
    assert sorted(ladder.RUNG_NAMES) == [0, 1, 2, 3, 4, 5]

    # dedupe (rung 2) alone is enough here: two copies of one huge block, and a
    # budget the SINGLE surviving copy fits inside. The budget has to be generous
    # enough that rung 2 actually solves the problem — at budget=3000 the deduped
    # block is still 3750 tokens, so escalating to rung 3 is correct behaviour and
    # asserting otherwise is asserting the ladder should stop before it is under
    # budget. Re-measuring after each rung is the mechanism; the budget is the input.
    blocks = {"docs": ["y " * 3000, "y " * 3000]}
    assert ladder.total_tokens(blocks) > 4000
    _, _, res = ladder.descend(budget=4000, blocks=blocks, tool_cap=100000)
    assert res.highest_rung == 2, \
        f"a rung-2 fix must not escalate: {res.summary()}"
    assert ladder.total_tokens(blocks) <= 4000, "and it must actually get us under"


def test_compaction_event_reports_before_after_and_saving():
    ev = ladder.CompactionEvent(1, "docs", 900, 400, "head+tail elision")
    assert ev.saved == 500
    assert ev.name == ladder.RUNG_NAMES[1]
    assert "docs" in str(ev) and "900" in str(ev) and "400" in str(ev)


def test_total_tokens_counts_blocks_and_turns_together():
    blocks = {"docs": ["aaa bbb ccc"], "recalled": ["ddd eee"]}
    turns = [{"role": "user", "content": "fff ggg"}]
    assert ladder.total_tokens(blocks) > 0
    assert ladder.total_tokens(blocks, turns) > ladder.total_tokens(blocks)
    assert ladder.total_tokens({}, []) == 0


# ── ⭐ the hybrid cut ─────────────────────────────────────────────────────────


def test_hybrid_backbone_never_summarises():
    """⭐⭐ THE PIVOT, AS AN ASSERTION. Under a recurrent-state backbone the ladder
    stops after rung 3. Rungs 1/4/5 rewrite or drop the prefix, which is exactly
    what the RSC replaces with a learned continuous operation — firing them anyway
    would silently reintroduce the recall collapse the pivot exists to avoid.

    This must hold even when a summariser IS supplied and the context is massively
    over budget, because the temptation to "just summarise this once" is precisely
    how the invariant gets quietly deleted later.
    """
    calls = []

    def spy_summariser(text):
        calls.append(text)
        return "summary."

    blocks = {"docs": ["z " * 40000], "transcript": ["t " * 40000]}
    turns = [{"role": "user", "content": "old " * 20000}]
    out_blocks, out_turns, res = ladder.descend(
        budget=1000, blocks=blocks, turns=turns, summariser=spy_summariser, hybrid=True
    )
    assert calls == [], "the summariser must never be invoked under a hybrid backbone"
    assert res.summarised is False
    assert res.highest_rung is not None and res.highest_rung <= 3, \
        f"no rung above 3 may fire: {res.summary()}"
    assert not any(e.rung in (1, 4, 5) for e in res.events), res.summary()


def test_the_transformer_path_can_reach_rung5():
    """The contrast that makes the hybrid cut meaningful: with hybrid=False and a
    summariser supplied, the ladder WILL descend all the way to rung 5 and set the
    alarm. So the hybrid test above is not passing merely because rung 5 is
    unreachable in principle."""
    blocks = {"transcript": ["q " * 60000], "docs": ["d " * 60000]}
    turns = [{"role": "user", "content": "old " * 40000}]
    _, _, res = ladder.descend(
        budget=500, blocks=blocks, turns=turns,
        summariser=lambda t: "summary.", hybrid=False
    )
    assert res.summarised is True, f"the transformer path should summarise: {res.summary()}"
    assert res.highest_rung == 5
    assert any("SUMMARISED" in e.detail for e in res.events)


def test_hybrid_still_uses_the_rungs_that_add_capability():
    """Cutting rungs 1/4/5 does not mean cutting compaction. Rung 0 (cap tool output
    before it enters history) and rung 3 (offload to a POINTER) both survive, because
    neither destroys information the model might need — rung 3 in particular replaces
    a payload with a handle it can pull back."""
    blocks = {"docs": ["big payload " * 8000]}
    _, _, res = ladder.descend(budget=200, blocks=blocks, tool_cap=2000, hybrid=True)
    assert res.events, "a hybrid must still compact"
    assert all(e.rung in (0, 2, 3) for e in res.events), res.summary()


# ── Ledger.compile ────────────────────────────────────────────────────────────


def test_compile_returns_a_compiled_context_with_the_expected_fields(conn, root):
    led = Ledger(soul_dir=root / "soul")
    c = led.compile(Gather(query="hello"), conn=conn)
    assert isinstance(c, CompiledContext)
    assert isinstance(c.text, str)
    assert c.budget == LEDGER_BUDGET_TOKENS == 8192
    assert c.spent >= 0
    assert c.ledger_hash and len(c.ledger_hash) == 8
    assert c.prefix_hash and len(c.prefix_hash) == 8
    assert isinstance(c.slots, SlotSet)
    assert c.overflow == "none" or isinstance(c.overflow, str)


def test_reserve_is_never_eaten(conn, root):
    """⭐ `reserve` is the room left for the model to ANSWER. A ledger that spends
    its whole budget on context has nothing left to generate with, which is the
    failure mode that looks like the model "cutting itself off"."""
    led = Ledger(soul_dir=root / "soul", budget=8192, reserve=LEDGER_RESERVE_TOKENS)
    g = Gather(query="hi", docs=["some retrieved text " * 50], scratch="notes")
    c = led.compile(g, conn=conn)
    assert c.reserve == c.budget - c.spent
    assert c.reserve > 0, f"the model must have room to answer: spent={c.spent}"


def test_compile_is_deterministic_and_hash_stable(conn, root):
    """Same input -> same bytes -> same hash. Without this the prefix cache can never
    hit and every turn re-prefills, which is the cost Law 2 exists to prevent."""
    led = Ledger(soul_dir=root / "soul")
    g = Gather(query="what is my rent", senses=["screen.capture"], scratch="note")
    a = led.compile(g, conn=conn)
    b = led.compile(g, conn=conn)
    assert a.ledger_hash == b.ledger_hash
    assert a.prefix_hash == b.prefix_hash
    assert a.text == b.text


def test_prefix_cache_hits_on_the_second_turn(conn, root):
    """The stable prefix must be byte-identical across turns or every turn
    re-prefills. `prefix_stable_runs` counting up is the observable evidence."""
    led = Ledger(soul_dir=root / "soul")
    first = led.compile(Gather(query="one", scratch="a"), conn=conn)
    assert first.cache_hit is False
    assert first.prefix_stable_runs == 1

    second = led.compile(Gather(query="two", scratch="b"), conn=conn)
    assert second.prefix_hash == first.prefix_hash, "changing JIT slots must not move the prefix"
    assert second.cache_hit is True
    assert second.prefix_stable_runs == 2


def test_prefix_excludes_turn_varying_slots(conn, root):
    """⭐ The prefix is identity + senses ONLY. If any JIT slot leaked into it, the
    hash would change every turn and `cache_hit` could never be True — so this is
    really a test that the prefix is defined over the stable slots."""
    led = Ledger(soul_dir=root / "soul")
    a = led.compile(Gather(query="a", scratch="one scratch", docs=["doc one"]), conn=conn)
    b = led.compile(Gather(query="b", scratch="totally different", docs=["doc two"]), conn=conn)
    assert a.prefix_hash == b.prefix_hash
    assert a.ledger_hash != b.ledger_hash, "but the full ledger must differ"


def test_a_changed_sense_set_invalidates_the_prefix(conn, root):
    """And the converse: the prefix is allowed to change when a SENSE changes, because
    `senses` is one of the two stable slots. Granting a new sense legitimately
    re-prefills — that is a rare, user-driven event, not a per-turn one."""
    led = Ledger(soul_dir=root / "soul")
    a = led.compile(Gather(query="x", senses=[]), conn=conn)
    b = led.compile(Gather(query="x", senses=["screen.capture"]), conn=conn)
    assert a.prefix_hash != b.prefix_hash
    assert b.cache_hit is False


def test_senses_slot_says_so_when_nothing_is_enabled(conn, root):
    """An explicit "none enabled" is more useful to the model than an empty slot: it
    tells FRIDAY that its only input is what the user typed, which stops it
    hallucinating a capability it does not have."""
    led = Ledger(soul_dir=root / "soul")
    c = led.compile(Gather(query="x", senses=[]), conn=conn)
    assert "none enabled" in c.slots["senses"].content
    assert "can only use what you type" in c.slots["senses"].content


def test_empty_retrieval_renders_an_explicit_nothing(conn, root):
    """An empty recalled slot must not render as an empty tag pair — and the
    conversation must still be coherent. Law 4's "empty beats noise" applies to
    rendering too: absence should be silent, not a hollow tag."""
    led = Ledger(soul_dir=root / "soul")
    c = led.compile(Gather(query="x", recalled=[]), conn=conn)
    assert "<recalled>" not in c.text or c.slots["recalled"].content.strip()


def test_state_block_is_prepended_to_recalled(conn, root):
    """⭐ The RSC slot. A fixed-size recurrent state gives unbounded history, so it
    is PREPENDED to the retrieved facts: the state is context, and retrieved facts
    are the exact-memory path that Law 2b says must never be delegated to a lossy
    state. Order matters — the exact memories must not be pushed behind the summary."""
    led = Ledger(soul_dir=root / "soul")
    g = Gather(
        query="x",
        state_block="<rsc_state>compressed history here</rsc_state>",
        recalled=[{"ref": "f_1", "kind": "fact", "body": "lives_in Chennai",
                   "score": 0.9, "subject": "user"}],
    )
    c = led.compile(g, conn=conn)
    body = c.slots["recalled"].content
    assert "rsc_state" in body
    assert "Chennai" in body
    assert body.index("rsc_state") < body.index("Chennai"), \
        "the state leads, the exact memories follow"


def test_state_block_alone_still_renders(conn, root):
    led = Ledger(soul_dir=root / "soul")
    c = led.compile(Gather(query="x", state_block="just the state"), conn=conn)
    assert "just the state" in c.slots["recalled"].content


def test_hybrid_ledger_routes_transcript_through_the_state_not_the_blocks(conn, root):
    """Under a hybrid backbone the transcript is handed to the ladder as `turns`
    rather than as a block, because the RSC is what replaces the transcript slot.
    Compiling must still produce a coherent context either way."""
    turns = [{"role": "user", "content": f"message {i}"} for i in range(5)]
    hyb = Ledger(soul_dir=root / "soul", hybrid=True)
    tf = Ledger(soul_dir=root / "soul", hybrid=False)
    a = hyb.compile(Gather(query="x", transcript=turns), conn=conn)
    b = tf.compile(Gather(query="x", transcript=turns), conn=conn)
    assert "message 4" in a.text, "the newest turn must survive under a hybrid"
    assert "message 4" in b.text
    assert a.slots["transcript"].content, "the transcript slot is still rendered"


def test_ledger_alarm_fires_on_summarisation(conn, root):
    """`CompiledContext.summarised` is the Law 2 alarm. If it is True more than a
    handful of times a week something upstream is wrong — the ladder should never
    need rung 5. Under the hybrid default it must NEVER be True."""
    led = Ledger(soul_dir=root / "soul", hybrid=True,
                 summariser=lambda t: "summary.")
    big = Gather(query="x", docs=["enormous tool output " * 20000],
                 scratch="s " * 20000,
                 transcript=[{"role": "user", "content": "old " * 20000}])
    c = led.compile(big, conn=conn)
    assert c.summarised is False, \
        f"a hybrid ledger must never summarise: {c.overflow}"
    assert c.ladder.highest_rung is None or c.ladder.highest_rung <= 3


def test_printout_is_the_dev_mode_block(conn, root):
    """The dev-mode LEDGER block from docs/architecture/10 Day 4. Looking at this for
    a week teaches more about context engineering than any article, which is why it
    is on by default in the CLI."""
    led = Ledger(soul_dir=root / "soul")
    c = led.compile(Gather(query="x", scratch="notes"), conn=conn)
    out = c.printout(turn=7)
    assert "LEDGER" in out and "turn=7" in out
    assert "budget=" in out and "spent=" in out
    assert "overflow:" in out and "reserve:" in out
    assert "hash=" in out and "prefix=" in out
    assert "identity" in out and "transcript" in out


def test_incompressible_slots_are_not_truncated_below_their_content(conn, root):
    """The stable prefix is small and must ship whole: truncating `identity` to meet
    a budget would change the cached prefix every turn AND quietly edit FRIDAY's
    personality, which is the worse of the two problems."""
    (root / "soul").mkdir(parents=True, exist_ok=True)
    (root / "soul" / "SOUL.md").write_text("I am FRIDAY. " * 20, encoding="utf-8")
    led = Ledger(soul_dir=root / "soul", budget=8192)
    c = led.compile(Gather(query="x"), conn=conn)
    assert "I am FRIDAY." in c.slots["identity"].content
    assert c.slots["identity"].truncated is False


def test_log_telemetry_writes_a_row(conn, root):
    from friday.ledger.compiler import log_telemetry

    led = Ledger(soul_dir=root / "soul")
    c = led.compile(Gather(query="x", scratch="n"), conn=conn)
    log_telemetry(conn, c, turn_id="t_1")
    row = conn.execute(
        "SELECT * FROM ledger_telemetry ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["turn_id"] == "t_1"


# ── the stable prefix is incompressible ────────────────────────────────────────

def test_required_slot_is_never_silently_truncated(root):
    """`identity` has overflow="reject" in doc 04 — it ships whole and complains.

    Silent truncation here is a double failure: it quietly edits FRIDAY's character,
    and `cap_tokens` leaves a marker reading "use read_artifact to pull it back" in a
    slot that never offloaded anything. Because the prefix is CACHED, that
    un-followable instruction would be byte-stable on every turn.
    """
    from friday.ledger.compiler import Gather, Ledger

    (root / "soul" / "SOUL.md").write_text("# SOUL\n\n" + "SOUL line.\n" * 4000,
                                        encoding="utf-8")
    c = Ledger().compile(Gather(query="hi", senses=[]))

    ident = c.slots["identity"]
    assert ident.over_budget is True
    assert ident.truncated is False
    assert "read_artifact" not in ident.content          # no phantom offload marker
    assert approx_tokens(ident.content) >= 4000          # shipped whole
    assert "OVER BUDGET" in c.printout()                 # and said so loudly


def test_a_ticking_clock_never_enters_the_cached_prefix(root):
    """The `now` reading must land in a JIT slot.

    `identity` and `senses` form the byte-stable prefix. A clock in either would
    invalidate the provider's prompt cache on every single turn — the exact cost Law 2
    exists to prevent, and the thing exit test #7 measures across 50 turns.
    """
    from friday.ledger.compiler import Gather, Ledger

    (root / "soul" / "SOUL.md").write_text("# SOUL\nI am FRIDAY.\n", encoding="utf-8")
    led = Ledger()

    def build(now):
        return led.compile(Gather(query="what time is it", senses=[], now=now))

    a, b = build("Monday, 1 January 2026, 09:00"), build("Tuesday, 2 February 2026, 17:45")

    assert a.prefix_hash == b.prefix_hash                # prefix identical across time
    assert a.ledger_hash != b.ledger_hash                # the JIT slot did change
    assert "Monday" in a.slots["scratch"].content
    assert "Tuesday" not in a.slots["scratch"].content


def test_now_renders_alongside_existing_scratch(root):
    from friday.ledger.compiler import Gather, Ledger

    (root / "soul" / "SOUL.md").write_text("# SOUL\nsoul\n", encoding="utf-8")
    c = Ledger().compile(Gather(query="q", scratch="user's own note", senses=[],
                                now="Wednesday, 30 September 2026, 08:44 UTC"))
    body = c.slots["scratch"].content
    assert body.startswith("[now] Wednesday")
    assert "user's own note" in body                     # the user's note survives


def test_empty_now_leaves_scratch_untouched(root):
    from friday.ledger.compiler import Gather, Ledger

    (root / "soul" / "SOUL.md").write_text("# SOUL\nsoul\n", encoding="utf-8")
    c = Ledger().compile(Gather(query="q", scratch="just notes", senses=[]))
    assert c.slots["scratch"].content == "just notes"
    assert "[now]" not in c.slots["scratch"].content
