"""T-004 — the doc 20 §5 coverage gaps, each fix pinned next to its Law 4 negative.

The board's rule for this item: every word or cue gets a test FIRST, a benign
negative case, and a check that no existing bench number drops. The negatives are
not decoration — two of the four candidate fixes died on theirs:

* **`program` was rejected.** Mapping it to the editor cluster lifts
  "what program do I type my code in" onto its fact, but the same mapping makes
  "what is my training program" score 0.589 against `editor: neovim` — a wrong
  answer shipped for an unrelated question. A false positive is worse than a miss
  (Law 4), so the word stays out and the gap waits for the embedder tier (T-003).
* **A blind possessive strip was rejected.** Dropping the clitic everywhere
  scores "what is my sister name" 0.429 against the manager fact — under the
  floor by 0.021, and over it once the multi-path nudge (×1.06) applies. That is
  the 0.2 bench false-positive rate doc 20 §5a recorded. The strip below only
  fires for lexicon vocabulary, where the semantics are curated.

What ships here: the cluster-conditional possessive strip, the thing-scoped
"what time" cue, and `usually`/`usual` as filler. The scores quoted in the
docstrings were measured on this corpus before the change.
"""

from __future__ import annotations

import pytest

from friday.memory.facts import Fact, assert_fact
from friday.retrieval.concepts import strip_possessives
from friday.retrieval.embedders import HashingEmbedder
from friday.retrieval.pipeline import _content_words, search
from friday.retrieval.rerankers import LexicalReranker
from friday.store.db import CONTENT_STOPWORDS, fts_terms


@pytest.fixture
def gaps(conn):
    """The bench corpus's gap-relevant facts, in the same shape `search` scores."""
    for subject, predicate, obj, quote in [
        ("user", "landlord_name", "Suresh Kumar", "my landlord is Suresh Kumar"),
        ("user", "manager_name", "Priya Raman", "my manager is Priya Raman"),
        ("user", "coffee_order", "filter coffee, no sugar",
         "I take filter coffee with no sugar"),
        ("user", "standup_day", "Tuesday 09:30 IST",
         "standup is Tuesday at half past nine"),
        ("user", "editor", "neovim", "I use neovim as my editor"),
    ]:
        assert_fact(conn, Fact(subject=subject, predicate=predicate, object=obj,
                               source_kind="user_edit", source_quote=quote))
    return conn


def _search(conn, q, **kw):
    kw.setdefault("embedder", HashingEmbedder(dim=256))
    kw.setdefault("reranker", LexicalReranker())
    return search(conn, q, **kw)


# ── possessive 's (doc 20 §5: "who is my flat's owner") ───────────────────────


def test_possessive_of_cluster_vocabulary_strips_the_clitic():
    """"my landlord's name" is the same question as "my landlord name": the clitic
    is morphology, not content. Measured with the strip: 0.899, above the floor;
    without it the stray `s` cost a third of the coverage (0.35-class) and the
    question fell below the floor.

    Plural possessives strip the same way when the base stems cleanly
    ("landlords'" -> "landlord"). Irregular ones ("bosses'", where the stemmer
    reduces `bosses` to `boss` but `boss` itself to `bos`) keep the residue —
    conservative is the safe direction; see the negative test below."""
    assert strip_possessives("who is my landlord's name") == "who is my landlord name"
    assert strip_possessives("my landlords' names") == "my landlord names"
    assert strip_possessives("my bosses' names") == "my bosses' names"
    rr = LexicalReranker()
    (score,) = rr.score(
        "who is my landlord's name",
        ['landlord_name: Suresh Kumar (landlord name, landlord, name) '
         '"my landlord is Suresh Kumar"'],
    )
    assert score >= rr.recommended_floor


def test_possessive_of_unknown_vocabulary_keeps_its_residue():
    """THE LOAD-BEARING NEGATIVE. "what is my sister's name" must stay below the
    floor against the manager fact (measured 0.311). The stray `s` is what keeps
    it there: a blind strip scores the same query 0.429, and the multi-path nudge
    (×1.06) carries that over the floor — the 0.2 false-positive regression that
    made doc 20 §5a revert the first attempt. The strip is therefore allowed only
    where the lexicon already understands the word."""
    assert strip_possessives("what is my sister's name") == "what is my sister's name"
    assert strip_possessives("my parents' address") == "my parents' address"
    rr = LexicalReranker()
    (score,) = rr.score(
        "what is my sister's name",
        ['manager_name: Priya Raman (manager name, manager, name) '
         '"my manager is Priya Raman"'],
    )
    assert score < rr.recommended_floor


def test_recall_layers_already_drop_the_clitic():
    """Both recall layers (FTS terms and pipeline content words) drop the clitic
    unconditionally, which is safe there: recall only widens the candidate set
    and never decides what ships. The reranker is the precision layer, so it is
    the one that had to be careful (see the test above)."""
    assert "s" not in fts_terms("who is my flat's owner")
    assert _content_words("who is my flat's owner") == ["flat", "owner"]


@pytest.mark.xfail(strict=True, reason=(
    "doc 20 §5: 'flat' maps to the rent concept, which the landlord fact does "
    "not carry, so coverage is 1/2 and the score lands at ~0.43. Closing it "
    "needs the rent and landlord clusters merged or a 'flat owner' alias on "
    "landlord_name — both measured to ship the landlord fact for 'my rent' "
    "(0.739) — or the embedder tier (T-003). Remove this pin only with one of "
    "those and a bench run that does not regress."
))
def test_flat_owner_probe_reaches_its_fact(gaps):
    r = _search(gaps, "who is my flat's owner")
    assert any(c.row and c.row["predicate"] == "landlord_name" for c in r.kept)


def test_a_possessive_question_round_trips_end_to_end(gaps):
    r = _search(gaps, "who is my landlord's name")
    assert r.kept and r.kept[0].row["predicate"] == "landlord_name"


# ── "what time" as a temporal cue (doc 20 §5: "what time is the scrum") ───────


def test_a_thing_scoped_time_question_finds_the_schedule():
    """"my standup time" was the bench's one noun-class miss (0.380, below the
    0.45 floor): `time` matched nothing, because the fact stores "Tuesday 09:30"
    and never the word "time". Weekday names and clock times now carry the
    temporal concept, so the fact answers WHEN without containing the word."""
    rr = LexicalReranker()
    standup = ('standup_day: Tuesday 09:30 IST (standup day, standup, day) '
               '"standup is Tuesday at half past nine"')
    for q in ("my standup time", "what time is the scrum"):
        (score,) = rr.score(q, [standup])
        assert score >= rr.recommended_floor, q


def test_a_time_question_round_trips_end_to_end(gaps):
    r = _search(gaps, "what time is the scrum")
    assert r.kept and r.kept[0].row["predicate"] == "standup_day"


def test_bare_time_stays_empty():
    """THE NEGATIVE. "how much time do I have" uses `time` as a mass noun with no
    thing in scope; mapping it to the temporal concept would let any schedule
    fact answer it (coverage 1/1). The cue is therefore thing-scoped: `time`
    carries the temporal concept only when the query names something else."""
    rr = LexicalReranker()
    standup = ('standup_day: Tuesday 09:30 IST (standup day, standup, day) '
               '"standup is Tuesday at half past nine"')
    (score,) = rr.score("how much time do I have", [standup])
    assert score < rr.recommended_floor


def test_a_time_question_about_an_unknown_thing_stays_empty():
    """Nothing about the dentist appointment is in memory, and the temporal
    concept must not make the standup schedule a substitute answer for it."""
    rr = LexicalReranker()
    standup = ('standup_day: Tuesday 09:30 IST (standup day, standup, day) '
               '"standup is Tuesday at half past nine"')
    (score,) = rr.score("what time is my dentist appointment", [standup])
    assert score < rr.recommended_floor


def test_a_when_less_fact_cannot_satisfy_the_temporal_concept():
    """The temporal concept must come from the FACT carrying a when (a weekday or
    a clock time), never from the query wanting one. "what time is breakfast"
    against the coffee fact scores 0.000: no shared concept at all — the habit
    fact cannot impersonate a schedule. (A question like "what time do I drink
    coffee" can still reach the coffee fact on the drink/coffee concepts alone;
    whether that is an answer is the grounding layer's call, T-007's, not the
    reranker's.)"""
    rr = LexicalReranker()
    coffee = ('coffee_order: filter coffee, no sugar (coffee order, coffee, order) '
              '"I take filter coffee with no sugar"')
    (score,) = rr.score("what time is breakfast", [coffee])
    assert score == 0.0


# ── "usually" / "have" (doc 20 §5: "what do I usually have to drink") ─────────


def test_usually_is_filler_not_content():
    """"usually" says how often, never which fact — the same job "just" and
    "really" already do. Measured with it excluded: the coffee fact scores 0.713
    for "what do I usually have to drink"; with it counted as a content term the
    coverage was 1/2 and the question fell below the floor (0.290).

    BOTH surface forms are required: `fts_terms` compares raw words, so it needs
    "usually"; the reranker compares stems, and `stem("usually") == "usual"`, so
    it needs "usual". One entry alone leaves one layer counting the word."""
    assert {"usually", "usual"} <= CONTENT_STOPWORDS
    assert fts_terms("what do I usually have to drink") == ["drink"]
    rr = LexicalReranker()
    coffee = ('coffee_order: filter coffee, no sugar (coffee order, coffee, order) '
              '"I take filter coffee with no sugar"')
    (score,) = rr.score("what do I usually have to drink", [coffee])
    assert score >= rr.recommended_floor


def test_a_usually_question_about_something_else_stays_empty():
    """THE NEGATIVE. Filler removal must not make the coffee fact an answer to a
    habit question about food."""
    rr = LexicalReranker()
    coffee = ('coffee_order: filter coffee, no sugar (coffee order, coffee, order) '
              '"I take filter coffee with no sugar"')
    (score,) = rr.score("what do I usually eat for dinner", [coffee])
    assert score < rr.recommended_floor


def test_have_is_filler_in_both_layers_now():
    """Doc 20 §5 named `have` as a gap and assumed it was already filler. It was
    NOT: CONTENT_STOPWORDS holds surface forms, the reranker compares stems, and
    `stem("have") == "hav"` — so `have` reached the scorer as a content term, and
    so did `were`/`wer`, `some`/`som`, `really`/`real`, `please`/`pleas` and
    `actually`/`actual`. That is why "what do I usually have to drink" lost
    coverage to a word the list believed it had dropped. The scorer now subtracts
    a stem-normalised view of the one canonical list; pinned here as an equality
    so the two phrasings can never drift apart again."""
    from friday.retrieval.rerankers import _STOPWORD_STEMS
    from friday.util import stem

    assert {"have", "has", "had"} <= CONTENT_STOPWORDS      # raw layer (fts_terms)
    for w in ("have", "has", "had", "were", "some", "really", "please", "actually"):
        assert stem(w) in _STOPWORD_STEMS, w
    rr = LexicalReranker()
    coffee = ('coffee_order: filter coffee, no sugar (coffee order, coffee, order) '
              '"I take filter coffee with no sugar"')
    with_have = rr.score("what do I have to drink", [coffee])[0]
    without = rr.score("what do I drink", [coffee])[0]
    assert with_have == without, "a stopword's stem must not cost coverage"


def test_a_habit_drink_question_round_trips_end_to_end(gaps):
    r = _search(gaps, "what do I usually have to drink")
    assert r.kept and r.kept[0].row["predicate"] == "coffee_order"


# ── "program": tried, rejected by its own negative (doc 20 §5) ─────────────────


def test_program_stays_out_of_the_lexicon():
    """The benign negative that killed the word: with `program` in the editor
    cluster, "what is my training program" scores 0.589 against `editor: neovim`
    — an FP by construction, because `program` names TV schedules and training
    plans as readily as editors. The lexicon's rule is that ambiguous words stay
    out; this pins the decision so it cannot be re-made quietly. The gap it
    leaves ("what program do I type my code in", 0.326) is real and stays open
    until the embedder tier can judge meaning instead of vocabulary."""
    from friday.retrieval.concepts import concept_key
    from friday.util import stem

    assert concept_key("program") == stem("program")
    rr = LexicalReranker()
    (score,) = rr.score("what is my training program",
                        ['editor: neovim "I use neovim as my editor"'])
    assert score < rr.recommended_floor
