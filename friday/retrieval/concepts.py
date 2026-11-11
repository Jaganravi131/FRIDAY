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
}

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


# ── possessives (doc 20 §5) ─────────────────────────────────────────────────────
#
# "who is my flat's owner" tokenised to `flat`, `s`, `owner`, and the stray `s`
# cost the question a third of its coverage. The recall layers already drop the
# clitic (their length filters skip it), so the fix belongs to the tokeniser the
# reranker shares with them — but a BLIND strip is measurably unsafe: it scores
# "what is my sister name" 0.429 against the manager fact, under the 0.45 floor
# by only 0.021, and the multi-path nudge (×1.06) carries it over — the 0.2 bench
# false-positive rate that made doc 20 §5a revert the first attempt.
#
# So the strip is CLUSTER-CONDITIONAL: it fires only on vocabulary the lexicon
# already understands, where stripping is pure morphology ("landlord's" is
# "landlord"). Unknown possessives ("sister's") keep their residue — for them the
# conservative term count is the thing still holding unrelated facts below the
# floor. A change may remove query terms that match nothing; it may not quietly
# widen what counts as a match for uncurated words.

_POSS_CLITIC = re.compile(r"([a-z]+)'s\b")
_POSS_PLURAL = re.compile(r"\b([a-z]+)s'")


def _clustered(word: str) -> bool:
    return _CLUSTER_OF.get(stem(word)) is not None


def strip_possessives(text: str) -> str:
    """Remove the possessive clitic from CLUSTER VOCABULARY only. Expects lowercase
    text (the reranker lowercases before tokenising). Unknown words are returned
    untouched, residue and all — that asymmetry is the false-positive guard.
    """
    def _clitic(m: re.Match[str]) -> str:
        base = m.group(1)
        return base if _clustered(base) else m.group(0)

    def _plural(m: re.Match[str]) -> str:
        base = m.group(1)
        # "bosses'" leaves "bosse", which `stem` reduces to "boss"; try the bare
        # base first, then the base plus the plural s it swallowed.
        if _clustered(base) or _clustered(base + "s"):
            return base
        return m.group(0)

    return _POSS_PLURAL.sub(_plural, _POSS_CLITIC.sub(_clitic, text))
