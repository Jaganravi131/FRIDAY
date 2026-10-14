"""The concept lexicon: structural invariants, the Law 4 negative cases, and the effect
on recall and reranking. Each invariant here guards a failure mode that would otherwise
be silent (a retrieval that quietly returns the wrong fact is not an exception)."""

from __future__ import annotations

import pytest

from friday.retrieval import concepts
from friday.retrieval.concepts import CLUSTERS, concept_key, expand_terms
from friday.retrieval.rerankers import LexicalReranker
from friday.util import stem


def test_every_surface_word_belongs_to_exactly_one_cluster():
    """A word in two clusters makes the mapping order-dependent. Import would already
    raise on a stem collision; this also checks the raw surface words are unique."""
    seen: dict[str, str] = {}
    for name, words in CLUSTERS.items():
        for w in words:
            assert w not in seen or seen[w] == name, f"{w!r} in {seen.get(w)} and {name}"
            seen[w] = name


def test_no_cluster_is_keyed_to_a_predicate_name():
    """The lexicon is general vocabulary. A cluster named after a fact predicate would
    mean the bench corpus is being encoded, not the language."""
    predicate_like = {"manager_name", "lease_amount_monthly", "coffee_order", "editor"}
    assert not predicate_like & set(CLUSTERS)


@pytest.mark.parametrize("word", ["pay", "group", "type", "take", "write", "dentist"])
def test_ambiguous_or_distinct_words_are_not_in_the_lexicon(word):
    """Law 4: a generic word mapped to a concept makes ordinary questions retrieve the
    wrong fact. `dentist` is excluded on purpose: a dentist is not a doctor_name."""
    assert concept_key(word) == stem(word)


def test_synonyms_share_a_concept_key():
    assert concept_key("boss") == concept_key("manager") == concept_key("reports")
    assert concept_key("tablets") == concept_key("medication")


def test_expansion_adds_cluster_mates_but_not_the_query_word_itself():
    extra = expand_terms(["report"])
    assert "manager" in extra and "boss" in extra
    assert "report" not in extra


def test_expansion_is_empty_for_words_with_no_concept():
    assert expand_terms(["sister", "favourite", "movie"]) == []


def test_reranker_scores_a_paraphrase_as_covered():
    """'report' (query) and 'manager' (fact) share no literal word. Before the lexicon,
    coverage was 0 and the reranker dropped the fact. Now it must score it."""
    rr = LexicalReranker()
    text = "manager_name Priya Raman manager name"
    (score,) = rr.score("who do I report to", [text])
    (unrelated,) = rr.score("who do I report to", ["coffee_order filter coffee no sugar"])
    assert score > unrelated
    assert score >= rr.recommended_floor


def test_reranker_does_not_reward_an_unrelated_fact_for_a_shared_stopword():
    rr = LexicalReranker()
    (score,) = rr.score("who is my dentist", ["manager_name Priya Raman"])
    assert score < rr.recommended_floor


def test_module_exports_are_stable():
    # pipeline.py and rerankers.py import these names; renaming one is a silent break.
    for name in ("concept_key", "concept_of_stem", "expand_terms", "CLUSTERS"):
        assert hasattr(concepts, name)


# ── T-004: the gaps in doc 20 §5 ─────────────────────────────────────────────────
# Each case below is a MECHANISM named in doc 20 §5, written as a unit test before the
# code changes. These are not held-out evidence: the cases came from probe 2's misses,
# so they can show the mechanism is fixed, not that retrieval generalises. That needs
# the blind set (T-002).

from friday.bench import CORPUS  # noqa: E402  (the committed bench corpus, read-only here)
from friday.util import strip_possessives  # noqa: E402

from friday.config import aliases_for  # noqa: E402


def _pipeline_text(predicate: str, value: str, quote: str) -> str:
    """The text the pipeline actually reranks (`pipeline._rerank_text`): predicate, value,
    the predicate's aliases in brackets, then the quote. The aliases carry the word `name`
    into every `*_name` fact, so a test that leaves them out tests a different scorer."""
    al = aliases_for(predicate)
    tail = f" ({', '.join(al)})" if al else ""
    return f'{predicate}: {value}{tail} "{quote}"'


_FACTS = [_pipeline_text(p, v, q) for dom in CORPUS.values() for (p, v, q) in dom]

#: The largest recall nudge the pipeline applies (three paths: 1 + 0.06 * 2). A candidate
#: whose reranker score times this is still under the floor cannot be lifted over it.
_MAX_RECALL_NUDGE = 1.12


def _ranked(query: str) -> list[str]:
    scores = LexicalReranker().score(query, _FACTS)
    return [f for _, f in sorted(zip(scores, _FACTS), key=lambda t: -t[0])]


def _score_of(query: str, fact_prefix: str) -> float:
    scores = LexicalReranker().score(query, _FACTS)
    (i,) = [k for k, f in enumerate(_FACTS) if f.startswith(fact_prefix)]
    return scores[i]


# §5 possessive `'s` ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("who is my flat's owner", "who is my flat owner"),
    ("what's my rent", "what my rent"),
    ("Suresh's flat", "Suresh flat"),
    ("rents are high", "rents are high"),            # plural, not possessive: untouched
    ("the boss' desk", "the boss' desk"),            # plural possessive: deliberately untouched
])
def test_strip_possessives_removes_only_the_clitic(text, expected):
    assert strip_possessives(text) == expected


def test_a_possessive_query_scores_the_same_as_the_plain_one():
    """The stray `s` counted as an uncovered query term. Stripping the clitic in the
    tokeniser makes the two phrasings equivalent, which is what they mean."""
    fact = "landlord_name: Suresh Kumar"
    assert _score_of("who is my flat's owner", fact) == _score_of("who is my flat owner", fact)


# §5 "usually" / "have" ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("word", ["usually", "normally", "generally", "often", "sometimes"])
def test_frequency_adverbs_are_filler_not_content(word):
    """Like `actually` and `just` already in the list: they say how often, not what. Left
    as content they count as uncovered terms and drag every frequency question down."""
    from friday.store.db import CONTENT_STOPWORDS
    assert word in CONTENT_STOPWORDS


def test_what_i_usually_drink_reaches_the_coffee_fact():
    assert _score_of("what do I usually have to drink", "coffee_order") >= LexicalReranker().recommended_floor


def test_usually_does_not_make_an_unrelated_question_match():
    """Benign negative: a frequency word alone carries no identifying content. The raw
    fallback keeps `usually` when nothing else is left, so this must still score low."""
    assert all(s < LexicalReranker().recommended_floor
               for s in LexicalReranker().score("how usually", _FACTS))


# §5 "program" ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("word", ["program", "programme"])
def test_program_is_the_code_editor_concept(word):
    """A coding program is the editor. Both spellings: the user writes British English."""
    assert concept_key(word) == concept_key("editor")


def test_what_program_do_i_type_my_code_in_clears_the_floor_for_the_editor_fact():
    """Top-ranked is not enough: the floor is what the funnel keeps, and the editor fact
    was top-ranked but below it (0.33 vs 0.45) before this change."""
    assert _score_of("what program do I type my code in", "editor") >= LexicalReranker().recommended_floor


def test_program_in_the_tv_sense_does_not_clear_the_floor_for_the_editor_fact():
    """Benign negative. `program` is ambiguous, so one shared word must not be enough to
    keep the editor fact. The floor is what decides; rank position is not."""
    assert _score_of("what program is on TV tonight", "editor") < LexicalReranker().recommended_floor


# §5 "what time" ──────────────────────────────────────────────────────────────────

def test_what_time_is_the_scrum_clears_the_floor_for_the_standup_fact():
    """Before this change: top-ranked, but 0.38, below the 0.45 floor."""
    assert _score_of("what time is the scrum", "standup_day") >= LexicalReranker().recommended_floor


def test_a_clock_time_in_the_fact_is_a_time_concept():
    from friday.retrieval.concepts import text_concepts
    assert "c:time" in text_concepts("standup_day: Tuesday 09:30 IST")
    assert "c:time" in text_concepts("standup_day: Tuesday at 10am")
    assert "c:time" not in text_concepts("lease_amount_monthly: 18000 INR")


def test_what_time_does_not_match_a_fact_without_a_clock_time():
    """Benign negative: a time question must not pull in facts that hold no time."""
    assert _score_of("what time is my flight to Delhi", "lease_amount_monthly") < \
        LexicalReranker().recommended_floor


# ── T-004 follow-up: the generic field word ────────────────────────────────────────
# Stripping the possessive exposed a leak the stray `s` had been hiding (doc 20 §5a).
# "what is my sister's name": `sister` matches nothing in the corpus, but `name` matches
# the `name` inside every `*_name` predicate. A field-kind word must not cover a fact by
# itself. These cases are a FAMILY of unanswerable questions, not the single bench query,
# so the fix is checked on the mechanism and not on one example.

@pytest.mark.parametrize("who", ["sister", "cousin", "father", "pet", "neighbour"])
@pytest.mark.parametrize("field", ["name", "number", "date"])
def test_an_unknown_subject_with_a_generic_field_word_clears_nothing(who, field):
    """Law 4. The subject is not in the corpus, so the only shared word is generic."""
    q = f"what is my {who}'s {field}"
    floor = LexicalReranker().recommended_floor
    assert all(s * _MAX_RECALL_NUDGE < floor for s in LexicalReranker().score(q, _FACTS)), q


def test_a_generic_word_still_counts_when_it_is_all_the_query_has():
    """Benign negative. \"what is my name\" has nothing else to match, so `name` stays."""
    assert _score_of("what is my name", "manager_name") > 0
