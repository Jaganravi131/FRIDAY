"""Query-side synonym bridges for the stdlib retrieval path.

WHY THIS EXISTS
---------------
`BENCHMARKS.md` records the known defect: paraphrase recall@5 is 0.0 on the
no-dependency path. "who do I report to" shares no word with the `manager_name`
fact, so FTS5's OR-match finds nothing and the lexical reranker's overlap is
exactly 0. A hashing embedder is a lexical index wearing a vector's clothes —
it cannot learn that "report to" names the manager relation — so the only
stdlib route to signal is a curated one: a human-written map from the words
people SAY to the words facts are STORED under.

This is the same instrument as `config.PREDICATE_ALIASES`, turned around.
Aliases map store-predicates -> human phrasings (index side); this maps human
words -> store vocabulary (query side). Both exist because the zero-dependency
path has no model to do it.

CURATION RULES (what keeps this honest rather than benchmark-shaped)
--------------------------------------------------------------------
1. Near-synonyms, nominalizations and spelling variants ONLY. Each entry must
   be defensible to a dictionary, not to the benchmark corpus.
2. NO co-hyponym crossings between answer-bearing categories. `dentist` maps
   to `dental`, never to `doctor` — answering a dentist question with the
   physician's name is exactly the confabulation Law 4 exists to prevent, and
   `friday bench`'s false-positive gate enforces the exclusion.
3. Ambiguous headwords get NO entry. `book` (read vs reserve), `order`
   (sequence vs purchase), `charge` (fee vs battery), `post` (mail vs job) —
   an expansion that fires in two domains is a false-positive machine. The
   mirror rule holds for TARGETS: `team` maps to `squad, crew` but NOT to
   `group`, because `group` sits inside unrelated predicates (`blood_group`)
   and would crown the blood fact for "my team". Prefer narrow targets.
4. Bare temporal coordinates are NEVER expanded. `day`, `date`, `time` all
   appear inside scheduled facts (`standup_day`, "Tuesday 09:30"), so mapping
   `time -> day` would promote every timetable fact for every clock question.
   "my standup time" scoring below the floor is the visible cost of this rule,
   and it is documented in the tests rather than fixed by weakening the rule.
5. Single-hop only. The pipeline expands once and never chains: `salary` maps
   to `wage, income` but NOT to `pay`, because `pay -> rent` is a separate
   bridge and chaining them would answer salary questions with rent facts.
6. Single lowercase words on both sides. The expansion is applied to stemmed
   content words (stopwords already removed), so multi-word phrases would
   never match the lookup anyway.

PRECISION ARGUMENT (why a synonym hit counts at full weight)
------------------------------------------------------------
A pure synonym-bridge match — zero incidental word overlap, the ONLY
connection a human-curated relation — is HIGHER-precision evidence than an
incidental exact hit, not lower. Accidental matches are exact ("time" appears
everywhere); a curated bridge fires only where a person wrote the relation
down. So the reranker counts bridged query words toward coverage at full
weight, plus a small need-scaled bridge bonus (see
`rerankers.LexicalReranker`: zero at full coverage, so it compensates for
circumstantial query words without ever crowning a bridged candidate over an
exact one). The false-positive gate, not a discount factor, is what keeps
this honest.
"""

from __future__ import annotations

import re

#: headword -> related store-vocabulary words. Read the curation rules above
#: before adding an entry; each one is a precision claim about the query side.
SYNONYMS: dict[str, tuple[str, ...]] = {
    # ── work: the reporting line and the workplace ──
    "report": ("manager", "supervisor", "boss"),
    "boss": ("manager", "supervisor"),
    "supervisor": ("manager", "boss"),
    "employer": ("company", "workplace"),
    "company": ("employer", "workplace", "firm"),
    "workplace": ("employer", "company", "office"),
    "office": ("workplace",),
    "job": ("employment", "role", "occupation"),
    "role": ("job", "position", "title"),
    "salary": ("wage", "income", "compensation"),
    "wage": ("salary", "income"),
    "income": ("salary", "wage"),
    "promotion": ("increment", "hike"),
    "leave": ("vacation", "holiday"),
    "vacation": ("leave", "holiday"),
    "team": ("squad", "crew"),
    "meeting": ("standup", "sync", "gathering"),
    "standup": ("meeting", "sync", "scrum"),
    "deadline": ("due", "duedate"),
    "task": ("todo",),
    "todo": ("task",),
    "review": ("feedback", "appraisal"),
    "colleague": ("coworker",),
    # ── housing: dwelling words and cost words ──
    "rent": ("lease", "rental"),
    "lease": ("rent", "rental", "agreement"),
    "flat": ("apartment", "house", "rent", "lease"),
    "apartment": ("flat", "house"),
    "house": ("home", "residence"),
    "home": ("house", "residence"),
    "landlord": ("owner",),
    "owner": ("landlord",),
    "deposit": ("advance",),
    "address": ("location", "area"),
    "area": ("locality", "neighbourhood"),
    "locality": ("area", "neighbourhood"),
    "street": ("road", "lane"),
    "city": ("town",),
    # ── money: paying and spending ──
    "pay": ("payment", "rent", "cost"),
    "payment": ("pay", "bill"),
    "bill": ("invoice", "payment"),
    "cost": ("price", "amount"),
    "price": ("cost", "rate"),
    "buy": ("purchase",),
    "purchase": ("buy",),
    "budget": ("spend", "spending"),
    "bank": ("account",),
    # ── food and drink ──
    "drink": ("beverage", "coffee", "tea"),
    "beverage": ("drink", "coffee", "tea"),
    "coffee": ("filter", "espresso"),
    "eat": ("food", "meal", "dine"),
    "food": ("meal", "eat", "cuisine"),
    "cook": ("recipe", "food"),
    "recipe": ("cook", "food"),
    "lunch": ("meal",),
    "dinner": ("meal", "supper"),
    "breakfast": ("meal",),
    # ── health: professionals stay in their own lane (rule 2) ──
    "doctor": ("physician",),
    "physician": ("doctor",),
    "dentist": ("dental",),
    "dental": ("dentist",),
    "medication": ("medicine", "prescription", "dose"),
    "medicine": ("medication", "dose"),
    "hospital": ("clinic",),
    "clinic": ("hospital",),
    "allergy": ("allergic",),
    # ── tech and the tools of the routine ──
    "code": ("program", "software", "editor", "ide"),
    "program": ("code", "software"),
    "software": ("code", "program"),
    "editor": ("ide",),
    "ide": ("editor",),
    "write": ("author", "compose", "editor"),
    "author": ("write", "writer"),
    "phone": ("mobile", "device"),
    "mobile": ("phone", "device"),
    "laptop": ("computer",),
    "computer": ("laptop", "machine"),
    "wifi": ("internet", "network"),
    "internet": ("wifi", "network"),
    "network": ("internet", "wifi"),
    "password": ("passcode", "credentials"),
    "app": ("application",),
    "email": ("mail", "inbox"),
    # ── getting around ──
    "car": ("vehicle", "automobile"),
    "vehicle": ("car", "automobile"),
    "drive": ("car", "vehicle"),
    "flight": ("trip", "travel", "journey"),
    "trip": ("travel", "flight", "journey"),
    "travel": ("trip", "journey"),
    # ── body, rest, people ──
    "exercise": ("workout", "fitness", "gym"),
    "workout": ("exercise", "gym", "fitness"),
    "gym": ("fitness", "workout"),
    "sleep": ("rest", "nap"),
    "mother": ("mom", "amma"),
    "father": ("dad", "appa"),
    "amma": ("mother",),
    "appa": ("father",),
    "wife": ("spouse",),
    "husband": ("spouse",),
    "spouse": ("wife", "husband"),
    "sister": ("sibling",),
    "brother": ("sibling",),
    "name": ("called",),
    # ── spelling variants: pure win, zero false-positive risk ──
    "color": ("colour",),
    "colour": ("color",),
    "favorite": ("favourite",),
    "favourite": ("favorite",),
    "center": ("centre",),
    "centre": ("center",),
    "organize": ("organise",),
    "organise": ("organize",),
}

_WORD = re.compile(r"[a-z]+", re.UNICODE)


def expansions_for(word: str) -> tuple[str, ...]:
    """Related store-vocabulary words for one query word, or () if unmapped.

    Lookup is on the raw lowercase word; the caller stems both sides, so
    "reports" reaches the "report" entry through the stemmer, not through here.
    """
    return SYNONYMS.get((word or "").lower(), ())


def expansion_terms(text: str) -> list[str]:
    """All expansion terms for the content words of `text`, order-stable.

    Deliberately returns plain words, not stems: the caller (FTS query builder,
    embedder input) tokenises them through the same path as user words, which
    is what keeps recall and rerank consistent. Query words themselves are
    never re-emitted, and neither are stopwords or duplicates.
    """
    from ..store.db import CONTENT_STOPWORDS

    seen: dict[str, None] = {}
    for w in _WORD.findall((text or "").lower()):
        if w in CONTENT_STOPWORDS:
            continue
        for e in expansions_for(w):
            if e not in CONTENT_STOPWORDS and e != w:
                seen.setdefault(e, None)
    return list(seen)


def expanded_query(text: str) -> str:
    """`text` plus its expansion terms, for the RECALL paths only.

    Recall (FTS OR-match, hashing embedder, graph walk) runs on this; the
    reranker scores the ORIGINAL text with synonym-aware logic, so coverage
    denominators and echo detection still see the user's words, not ours.
    With no mapped words this returns `text` unchanged.
    """
    extra = expansion_terms(text)
    if not extra:
        return text
    return (text + " " + " ".join(extra)).strip()
