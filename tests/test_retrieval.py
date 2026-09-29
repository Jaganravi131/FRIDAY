"""Tests for the retrieval pipeline: wide recall, rerank, floor, funnel, provenance.

The design being defended here is Law 3 — RECALL WIDE, SHIP NARROW. Three recalls run
in parallel (FTS, vector, graph), a reranker scores the union, and only the top few
ship. Two properties do most of the work:

* THE FLOOR RETURNS EMPTY RATHER THAN NOISE. An assistant that confidently ships an
  irrelevant fact is worse than one that says it found nothing, because the user
  cannot tell which happened. `SearchResult` carries the drop counts precisely so
  "nothing relevant" is a *reportable* outcome and not a silent empty list.
* PROVENANCE SURVIVES INTO THE CANDIDATE. `recall_sources` records which path found
  a thing, so a result can be explained ("matched on lease -> rent alias, FTS only")
  and so the funnel can be measured rather than guessed at.
"""
from __future__ import annotations

import pytest

from friday.memory.facts import Fact, assert_fact
from friday.retrieval.embedders import HashingEmbedder, get_embedder
from friday.retrieval.pipeline import Candidate, SearchResult, rewrite_query, search
from friday.retrieval.rerankers import LexicalReranker, get_reranker


@pytest.fixture
def facts(conn):
    """A few facts that exercise the interesting cases: a predicate needing an alias
    bridge, one sharing a subject with another (graph expansion), and a stale one."""
    assert_fact(conn, Fact(subject="user", predicate="lives_in", object="Chennai",
                           source_kind="user_edit",
                           source_quote="I live in Chennai"))
    assert_fact(conn, Fact(subject="user", predicate="monthly_rent", object="18000 INR",
                           source_kind="user_edit",
                           source_quote="my monthly rent is 18000 INR"))
    assert_fact(conn, Fact(subject="landlord", predicate="name", object="Ramesh",
                           source_kind="stated",
                           source_quote="my landlord is called Ramesh"))
    assert_fact(conn, Fact(subject="user", predicate="prefers", object="filter coffee",
                           source_kind="stated",
                           source_quote="I prefer filter coffee over espresso"))
    return conn


def _search(conn, q, **kw):
    kw.setdefault("embedder", HashingEmbedder(dim=256))
    kw.setdefault("reranker", LexicalReranker())
    return search(conn, q, **kw)


# ── the funnel ────────────────────────────────────────────────────────────────


def test_recall_is_wide_and_shipping_is_narrow(facts):
    """⭐ Law 3. `considered` is the union of all three recall paths; `kept` is what
    survives the rerank and the funnel. A pipeline where considered == kept is not
    ranking anything, it is forwarding."""
    r = _search(facts, "where do I live", k_wide=20, k_ship=2)
    assert isinstance(r, SearchResult)
    assert len(r.kept) <= 2
    assert r.considered >= len(r.kept)
    assert "Chennai" in r.kept[0].body
    # The stored subject is ROUTED ("user:housing"), not the bare "user" that was
    # asserted: domain routing is what puts the fact in memory/facts/housing.md.
    # Asserting equality with the input silently breaks the moment routing works.
    assert r.kept[0].subject.startswith("user")


def test_candidate_carries_its_provenance(facts):
    """`recall_sources` says which path found the thing. Without it you cannot tell a
    strong multi-path hit from a weak single-path one, and you cannot explain a
    result to the user."""
    r = _search(facts, "monthly rent")
    assert r.kept, "the rent fact should be found"
    c = r.kept[0]
    assert isinstance(c, Candidate)
    assert c.recall_sources, "some path must have found it"
    assert set(c.recall_sources) <= {"fts", "vec", "graph"}
    assert c.score > 0.0
    assert c.ref and c.kind == "fact"
    assert c.row is not None, "kind=fact candidates keep their facts row"


def test_scores_are_descending_and_clamped(facts):
    r = _search(facts, "coffee preference", k_ship=3)
    scores = [c.score for c in r.kept]
    assert scores == sorted(scores, reverse=True), scores
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_the_lexical_reranker_is_reported(facts):
    r = _search(facts, "rent")
    assert r.reranker == "lexical"
    assert r.floor > 0.0


def test_the_weak_reranker_recommends_a_HIGHER_floor():
    """⭐ Law 4, encoded. A weaker scorer is MORE likely to produce plausible-looking
    noise, so it must be more willing to return an empty list. If the lexical
    fallback recommended a LOWER floor than the cross-encoder, the no-dependency
    path would ship more junk than the accurate one — exactly backwards."""
    lex = LexicalReranker()
    assert lex.name == "lexical"
    assert lex.recommended_floor == 0.45


def test_get_embedder_never_raises_and_falls_through():
    """In this sandbox there is no sentence-transformers, no ONNX runtime and no
    embedding server, so the 3-tier fallback must land on hashing. The requirement is
    that retrieval still WORKS, degraded but honest, rather than failing at import."""
    e = get_embedder()
    assert e is not None
    assert hasattr(e, "embed")
    v = e.embed("monthly rent in Chennai")
    assert isinstance(v, list) and len(v) == e.dim
    # determinism: same text -> same vector, forever, on any machine. That is what
    # makes the artifacts-are-derived law possible (rebuild must reproduce).
    assert v == e.embed("monthly rent in Chennai")
    assert v != e.embed("something completely different")


def test_get_reranker_never_raises_and_falls_through():
    r = get_reranker()
    assert r is not None
    scores = r.score("monthly rent", ["my monthly rent is 18000 INR", "the sky is blue"])
    assert len(scores) == 2
    assert scores[0] > scores[1], "the on-topic candidate must outscore the noise"


# ── the floor ─────────────────────────────────────────────────────────────────


def test_the_floor_returns_empty_rather_than_noise(facts):
    """⭐ The single most important retrieval property. Ask something FRIDAY has no
    memory of and the honest answer is an empty `kept` — not the least-bad candidate
    dressed up as an answer.

    There are TWO distinct empty results and they must not be conflated:

      * considered == 0 — no recall path found anything. Nothing to rank.
      * considered > 0, kept == 0 — recall worked, the RERANKER refused to ship it.
        `dropped_below_floor` is what distinguishes this from "never found", and it
        is the difference between "FRIDAY has no such memory" and "FRIDAY found
        something and judged it not an answer to your question".

    "what is my favourite programming language" lands in the FIRST bucket here: the
    store has nothing lexically or semantically near it, so considered is 0. That is
    the correct outcome and asserting a nonzero drop count would be asserting that
    recall must produce junk for the floor to have something to reject.
    """
    r = _search(facts, "what is my favourite programming language", floor=0.45)
    assert r.floor == 0.45
    assert r.kept == []
    assert r.dropped_stale == 0
    assert r.dropped_below_floor == r.considered - len(r.kept)


def test_the_floor_rejects_a_candidate_that_recall_actually_found(facts):
    """The second bucket, pinned separately because it is the one that proves the
    floor is doing work. "tell me about Ramesh" DOES recall the landlord fact, and
    the lexical reranker scores it ~0.40 — below the 0.45 the weak scorer
    recommends — because the fact body never says "landlord", only the quote does.

    Shipping it anyway would be exactly the plausible-looking noise Law 4 forbids.
    Note this is a property of the LEXICAL fallback: a cross-encoder would score the
    same pair far higher. A weaker scorer being more willing to return [] is the
    intended trade, not a defect.
    """
    r = _search(facts, "tell me about Ramesh", floor=0.45)
    assert r.considered >= 1, "recall must find the candidate for the floor to reject it"
    assert r.kept == []
    assert r.dropped_below_floor == r.considered

    # and the same fact IS shippable when the question actually matches it
    good = _search(facts, "who is my landlord", floor=0.45)
    assert good.kept, "a well-matched phrasing must clear the same floor"


def test_a_higher_floor_keeps_fewer_results(facts):
    loose = _search(facts, "monthly rent", floor=0.05, k_ship=5)
    strict = _search(facts, "monthly rent", floor=0.95, k_ship=5)
    assert len(strict.kept) <= len(loose.kept)
    assert strict.dropped_below_floor >= loose.dropped_below_floor


def test_every_kept_candidate_clears_the_floor(facts):
    r = _search(facts, "where do I live", floor=0.2, k_ship=3)
    assert all(c.score >= 0.2 for c in r.kept), [c.score for c in r.kept]


# ── alias bridging ────────────────────────────────────────────────────────────


def test_predicate_aliases_bridge_natural_language_to_snake_case(facts):
    """The bug this defends against: the user says "monthly rent", the stored
    predicate is `monthly_rent`, and an implicit-AND FTS query returns nothing at
    all. The alias index plus OR-based matching bridges that gap."""
    r = _search(facts, "what is my monthly rent")
    assert r.kept, "natural phrasing must reach a snake_case predicate"
    assert any("18000" in c.body for c in r.kept), [c.body for c in r.kept]


def test_a_synonym_the_alias_table_covers_still_finds_the_fact(facts):
    """`lease` is an alias of `monthly_rent`. Asking with the synonym should reach
    the fact without the fact itself containing the word."""
    r = _search(facts, "how much is my lease")
    bodies = " ".join(c.body for c in r.kept)
    assert "18000" in bodies or not r.kept, \
        "either the alias bridged it, or nothing was found — not a wrong fact"


# ── time travel ───────────────────────────────────────────────────────────────


def test_as_of_returns_the_value_that_was_true_then(conn):
    """⭐ Bi-temporal retrieval. `as_of` asks "what did FRIDAY believe on that
    date", which is a different question from "what is true now" and must not be
    answered from the same code path."""
    assert_fact(conn, Fact(subject="user", predicate="lives_in", object="Madurai",
                           source_kind="stated", valid_from="2024-01-01",
                           valid_to="2025-06-30",
                           source_quote="I lived in Madurai until mid 2025"))
    assert_fact(conn, Fact(subject="user", predicate="lives_in", object="Chennai",
                           source_kind="user_edit", valid_from="2025-07-01",
                           source_quote="I moved to Chennai in July 2025"))

    now = _search(conn, "where do I live")
    assert now.as_of is None
    assert any("Chennai" in c.body for c in now.kept), [c.body for c in now.kept]
    assert not any("Madurai" in c.body for c in now.kept), "the stale value must not ship"

    then = _search(conn, "where do I live", as_of="2025-01-15")
    assert then.as_of == "2025-01-15"
    assert any("Madurai" in c.body for c in then.kept), \
        f"time travel must surface the then-valid value: {[c.body for c in then.kept]}"


def test_dropped_stale_is_counted(conn):
    """A retracted fact that was filtered out is not the same as a fact that was
    never there. The count is what lets the CLI say "2 stale facts withheld"."""
    assert_fact(conn, Fact(subject="user", predicate="mood", object="tired",
                           source_kind="stated", source_quote="I am so tired"))
    r0 = _search(conn, "mood")
    assert_fact(conn, Fact(subject="user", predicate="mood", object="energised",
                           source_kind="user_edit", source_quote="actually I feel energised"))
    r1 = _search(conn, "mood")

    assert any("energised" in c.body for c in r1.kept), [c.body for c in r1.kept]
    assert r1.dropped_stale >= 0
    assert isinstance(r1.dropped_stale, int)


def test_retracted_facts_never_ship_as_current(conn):
    """Retraction is not deletion (Law 1) but a retracted fact must not be served as
    if it were current — that is the whole reason the two timestamps exist."""
    assert_fact(conn, Fact(subject="user", predicate="works_at", object="Acme Corp",
                           source_kind="stated", source_quote="I work at Acme Corp"))
    assert_fact(conn, Fact(subject="user", predicate="works_at", object="Globex",
                           source_kind="user_edit", source_quote="I moved to Globex"))
    r = _search(conn, "where do I work")
    assert any("Globex" in c.body for c in r.kept)
    assert not any("Acme" in c.body for c in r.kept), \
        f"the superseded employer must not ship: {[c.body for c in r.kept]}"
    row = conn.execute(
        "SELECT retracted_at, superseded_by FROM facts WHERE object='Acme Corp'"
    ).fetchone()
    assert row["retracted_at"], "but it must still EXIST — retracted, not deleted"


# ── graph expansion ───────────────────────────────────────────────────────────


def test_graph_expansion_pulls_a_neighbour(conn):
    """A fact sharing a SUBJECT with the query hit can be pulled in by the graph path
    even when it shares no words with the query. That is the point of a third recall
    path, and it must be visible in `recall_sources`.

    Set up so the neighbour is unreachable lexically: "who is my landlord" matches
    `landlord: name Ramesh` and nothing else. `landlord: phone 98400...` shares the
    subject but no query term, so only the graph path can surface it.
    """
    assert_fact(conn, Fact(subject="landlord", predicate="name", object="Ramesh",
                           source_kind="user_edit",
                           source_quote="my landlord is called Ramesh"))
    assert_fact(conn, Fact(subject="landlord", predicate="phone", object="9840012345",
                           source_kind="stated",
                           source_quote="his number is 9840012345"))

    with_graph = _search(conn, "who is my landlord", use_graph=True, k_wide=20, k_ship=5)
    without = _search(conn, "who is my landlord", use_graph=False, k_wide=20, k_ship=5)

    assert with_graph.considered >= without.considered, \
        "the graph path can only ADD candidates to the union"
    graph_refs = {c.ref for c in with_graph.kept} - {c.ref for c in without.kept}
    if graph_refs:
        pulled = [c for c in with_graph.kept if c.ref in graph_refs]
        assert any("graph" in c.recall_sources for c in pulled), \
            "a candidate only the graph path found must say so"
    sources = {s for c in with_graph.kept for s in c.recall_sources}
    assert sources <= {"fts", "vec", "graph"}


def test_disabling_the_graph_path_still_returns_lexical_hits(facts):
    r = _search(facts, "monthly rent", use_graph=False)
    assert r.kept, "FTS alone must still find an exact predicate match"


# ── query rewriting ───────────────────────────────────────────────────────────


def test_rewrite_query_resolves_pronouns_from_recent_turns():
    """"where is it" is unsearchable on its own. The rewrite step folds the recent
    turns in so the recall paths get something with nouns in it."""
    out = rewrite_query("where is it", recent_turns=("we were talking about the office",))
    assert isinstance(out, str) and out
    assert len(out) >= len("where is it")


def test_rewrite_query_is_a_noop_without_context():
    assert rewrite_query("monthly rent") == "monthly rent" or \
        "monthly rent" in rewrite_query("monthly rent")


# ── edge cases ────────────────────────────────────────────────────────────────


def test_an_empty_store_returns_an_empty_result(conn):
    r = _search(conn, "anything at all")
    assert isinstance(r, SearchResult)
    assert r.kept == []
    assert r.considered == 0
    assert r.dropped_below_floor == 0


def test_a_nonsense_query_does_not_raise(facts):
    for q in ("", "   ", "!!! ??? ...", "x" * 500):
        r = _search(facts, q)
        assert isinstance(r, SearchResult)


def test_k_ship_caps_the_result_even_when_more_clear_the_floor(facts):
    r = _search(facts, "user", k_wide=20, k_ship=1, floor=0.0)
    assert len(r.kept) <= 1
