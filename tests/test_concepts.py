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
