"""A concept lexicon: the stdlib-only bridge between how a person *asks* and how a fact is *stored*.

Why this exists
---------------
A fact is stored as ``manager_name: Priya Raman``. A person asks "who do I report to".
The two share no word, so every lexical recall path misses, and the reranker (which
needs literal coverage of the query's content words) discards whatever recall found.
``bench`` measured this: paraphrase recall@5 was 0.00 on the committed set and 0.17 on
a held-out probe (see ``docs/architecture/20-concept-lexicon.md``).

The fix here is deliberately *not* a model. Retrieval must work when no model is
reachable (the Law 4 / ₹0 constraint), so the first lever is a hand-curated thesaurus
of personal-life vocabulary. A neural embedder is the better long-run answer and is
still the optional tier; this layer is the floor it has to beat.

Rules, each enforced by ``tests/test_concepts.py``:

* **Each surface word belongs to at most one cluster.** A word in two clusters would
  make the mapping order-dependent, and therefore non-reproducible.
* **Clusters are not predicate names.** Nothing here is keyed to a fact's predicate;
  the lexicon is general vocabulary, so it generalises past the bench corpus.
* **Ambiguous or generic words are left out.** ``pay``, ``group``, ``type``, ``take``
  and ``write`` would each map ordinary questions onto the wrong fact. A false positive
  is worse than a miss (Law 4), so when in doubt a word stays out.
* **Near-synonyms that name a different thing stay separate.** ``dentist`` is not a
  ``doctor`` here, so "who is my dentist" must still come back empty.
"""

from __future__ import annotations

import re

from ..util import stem

#: cluster name -> surface words. Keep this list short and reviewable; add a word only
#: with a reason, and add a benign case to the tests when you do.
CLUSTERS: dict[str, tuple[str, ...]] = {
    "manager": (
        "manager", "boss", "supervisor", "superior", "report", "reports", "reporting",
        "charge",
    ),
    "rent": (
        "rent", "rented", "lease", "flat", "apartment", "payment", "monthly", "month",
    ),
    "tenancy_end": (
        "tenancy", "expire", "expires", "expiry", "finish", "finishes", "end", "ends",
        "ending",
    ),
    "landlord": (
        "landlord", "landlady", "owner", "owns", "owned", "lessor", "landowner",
    ),
    "drink": (
        "coffee", "tea", "chai", "drink", "drinks", "beverage", "beverages", "caffeine",
    ),
    "code_editor": (
        "editor", "ide", "vim", "neovim", "emacs", "code", "coding", "programming",
        # "program" is ambiguous (a TV program). It is here because in a question about
        # where the user types code it means the editor. Both spellings: the user writes
        # British English. The benign negative test pins the TV reading.
        "program", "programme",
    ),
    "doctor": (
        "doctor", "doctors", "physician", "treat", "treats", "treating", "ill", "sick",
    ),
    "medication": (
        "medication", "medications", "medicine", "medicines", "pill", "pills", "tablet",
        "tablets", "drug", "drugs", "supplement", "supplements", "dose", "dosage",
    ),
    "language": (
        "language", "languages", "speak", "speaks", "speaking", "tongue",
    ),
    "standup": (
        "standup", "scrum", "sync",
    ),
    "review": (
        "review", "reviews", "evaluate", "evaluated", "evaluation", "appraisal",
        "appraisals", "appraised", "assessment",
    ),
    "team": (
        "team", "squad",
    ),
    # "what time" asks for a clock time. A fact that stores one ("Tuesday 09:30 IST")
    # carries the concept too; see `text_concepts`. Only the word is listed here.
    "time": (
        "time",
    ),
}

#: Words that name a KIND of field, not a subject. Every `*_name` predicate carries `name`,
#: so a query whose only shared word is `name` matches every name fact. Such a word counts
#: toward coverage only when nothing more specific is in the query (see the reranker).
GENERIC_FIELD_WORDS: frozenset = frozenset({"name", "number", "date"})

#: A clock time in running text: 09:30, 10:00, 10am, 7 pm. Decimals (1.50) and bare
#: numbers (18000) are deliberately not clock times.
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.IGNORECASE)
_TIME_CONCEPT = "c:time"

#: surface word -> cluster name, keyed by the STEMMED surface form so that the lookup
#: agrees with `util.stem`, which the reranker applies to both sides of every comparison.
_CLUSTER_OF: dict[str, str] = {}
for _name, _words in CLUSTERS.items():
    for _w in _words:
        _key = stem(_w.lower())
        if _key in _CLUSTER_OF and _CLUSTER_OF[_key] != _name:
            raise ValueError(f"concept lexicon: {_w!r} is in two clusters")
        _CLUSTER_OF[_key] = _name


def concept_of_stem(s: str) -> str:
    """Map an already-stemmed token to its concept key, or leave it as the stem.

    The ``c:`` prefix keeps concept keys from colliding with real stems, so a cluster
    named ``team`` cannot be confused with a word that happens to stem to ``team``.
    """
    name = _CLUSTER_OF.get(s)
    return f"c:{name}" if name else s


def text_concepts(text: str) -> set[str]:
    """Every concept key a candidate text carries, for the reranker's coverage check.

    Words map through the lexicon. A clock time also carries the `time` concept, so a
    question about "the time" can be covered by a fact that holds one, without the fact
    having to contain the word "time".
    """
    out = {concept_key(w) for w in re.findall(r"[^\W_]+", (text or "").lower())}
    if _CLOCK.search(text or ""):
        out.add(_TIME_CONCEPT)
    return out


def concept_key(word: str) -> str:
    """Map one raw token to its concept key (stems it first)."""
    return concept_of_stem(stem(word))


def expand_terms(words: list[str]) -> list[str]:
    """Surface words to ADD to a recall query for these query terms.

    For each query term that belongs to a cluster, return that cluster's other surface
    words. Used on the FTS path only: FTS is OR-matched, so extra terms add candidates
    without removing any. The reranker still judges against the user's own words.
    """
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        name = _CLUSTER_OF.get(stem(w))
        if not name:
            continue
        for surface in CLUSTERS[name]:
            if surface not in seen and surface != w:
                seen.add(surface)
                out.append(surface)
    return out
