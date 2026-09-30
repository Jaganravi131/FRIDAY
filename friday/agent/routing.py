"""The tool contract: decide whether a turn needs memory BEFORE spending on it.

Phase 0 exit test #8: "Ask the time / a trivial question -> `memory_search` NOT
called". That is a latency and a naturalness requirement, not a tidy-up. Embedding a
query, running three recall paths and reranking costs more wall-clock than a local
model takes to answer "what time is it?", and an assistant that pauses to consult its
memory before saying "hello" feels like a database, not a companion.

DESIGN RULES, AND WHY THEY ARE ORDERED THIS WAY
-----------------------------------------------
1. DEFAULT IS `needs_memory=True`. A false negative here is far worse than a false
   positive: skipping retrieval on a question that needed it produces a confident
   answer with no memory behind it, which is the hallucination path. Wasting 15 ms of
   retrieval on "hello" is merely slightly slow. So the patterns below are narrow on
   purpose and anything unrecognised is treated as memory-eligible.
2. NO MODEL CALL. This has to be cheaper than the thing it is deciding whether to
   skip. A classifier that needs an inference pass to save an inference pass is a
   joke, and a regex table is deterministic, testable, and free.
3. THE DECISION IS REPORTED, NOT HIDDEN. `TurnIntent.reason` and the audit row exist
   so that when FRIDAY fails to remember something you can tell "it looked and found
   nothing" from "it decided not to look". Those need completely different fixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Turn kinds. `memory` and `general` both retrieve; the rest do not.
KINDS = ("clock", "greeting", "arithmetic", "meta", "ack", "memory", "general")


@dataclass
class TurnIntent:
    kind: str
    needs_memory: bool
    reason: str
    #: Patterns that matched, for debugging and for the audit trail.
    matched: tuple[str, ...] = ()

    @property
    def skip_retrieval(self) -> bool:
        return not self.needs_memory


# ── patterns ──────────────────────────────────────────────────────────────────
# Anchored and narrow. `^...$` on the normalised query wherever possible, so that a
# question which merely CONTAINS "time" ("what time does my lease renew?") is not
# mistaken for a clock query. That false positive would be the expensive kind.

_CLOCK = re.compile(
    r"""^(?:
          (?:what(?:'s|\s+is)?|tell\s+me|do\s+you\s+know)\s+
          )?
        (?:
          the\s+)?
        (?:
          (?:current\s+)?(?:time|date|day)(?:\s+(?:is\s+it|now|today))?
          | what\s+time\s+is\s+it
          | today'?s\s+date
        )
        \??$""",
    re.IGNORECASE | re.VERBOSE,
)
_CLOCK_SIMPLE = re.compile(
    r"\b(?:what|whats|what's)\s+(?:the\s+)?(?:time|date|day)\b", re.IGNORECASE
)
#: A clock query is only clock-y if it has no other content. "what time is my flight"
#: is a memory question about a flight, not a request for the current time.
_CLOCK_DISQUALIFIER = re.compile(
    r"\b(?:my|our|lease|rent|flight|meeting|train|deadline|renew|anniversary|born|"
    r"started|moved|birthday|due)\b", re.IGNORECASE
)

_GREETING = re.compile(
    r"^(?:hi|hey|hello|yo|namaste|vanakkam|good\s+(?:morning|afternoon|evening|night)"
    r"|sup|howdy|hola)(?:[!.,\s]+(?:there|friday|you|buddy|da|mach))?[\s!.,]*$",
    re.IGNORECASE,
)

_ACK = re.compile(
    r"^(?:ok|okay|k|thanks|thank\s+you|thx|ta|nice|cool|great|good|sure|alright|"
    r"got\s+it|makes\s+sense|bye|goodbye|see\s+you|gn|good\s+night|"
    r"how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up)(?:[!.,\s]*(?:friday|da|mach|bro))?[\s!.,]*$",
    re.IGNORECASE,
)

_META = re.compile(
    r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do"
    r"|help|commands?|what\s+do\s+you\s+know\s+about\s+yourself|are\s+you\s+(?:real|human|there)"
    r"|what\s+model\s+are\s+you|how\s+do\s+you\s+work)(?:\??)[\s!.,]*$",
    re.IGNORECASE,
)

_ARITH = re.compile(
    r"^[\s(]*[-+]?[\d.,]+\s*(?:[\s]*[-+*/x×^%]\s*[-+]?[\d.,]+\s*)*[\s)]*[?=]*[\s]*$",
)
_ARITH_WORDY = re.compile(
    r"^(?:what(?:'s|\s+is)|calculate|compute|how\s+much\s+is)\s+"
    r"[-+]?[\d.,]+(?:\s*[-+*/x×^%]\s*[-+]?[\d.,]+)+[\s?=]*$",
    re.IGNORECASE,
)

#: First-person markers: if the user is telling FRIDAY something about themselves,
#: that is memory-eligible even if it is short enough to look like an ack.
_ABOUT_ME = re.compile(
    r"\b(?:i\s+(?:am|am|live|work|like|prefer|love|hate|need|want|have|has|think|feel|"
    r"moved|started|stopped|quit|got|bought|sold|rent|pay|owe)|"
    r"my\s+\w+|me\b|mine\b)\b",
    re.IGNORECASE,
)


def _normalise(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip())


def classify_turn(query: str, *, recent_turns: tuple[str, ...] = ()) -> TurnIntent:
    """Decide what a turn needs. Deterministic, no model, no I/O.

    `recent_turns` is accepted for future use (a bare "yes" after "should I look up
    your rent?" is a memory turn) but deliberately does not change the verdict yet:
    resolving anaphora with a regex is how you get confidently wrong, and the safe
    default already retrieves.
    """
    q = _normalise(query)
    if not q:
        return TurnIntent("ack", False, "empty turn", ())

    low = q.lower()

    # ⭐ Anything first-person is memory-eligible, and this check runs BEFORE the
    # short-utterance patterns. "my rent is 18000" is 4 words and would otherwise be
    # at risk from an aggressive ack/greeting matcher; "I moved to Chennai" is the
    # single most important kind of turn in the whole system. Guard it first.
    if _ABOUT_ME.search(q):
        return TurnIntent("memory", True,
                          "first-person content — this is what memory is for",
                          ("about_me",))

    if _ARITH.match(q) or _ARITH_WORDY.match(q):
        return TurnIntent("arithmetic", False, "pure arithmetic; no fact can help", ("arith",))

    if _GREETING.match(q):
        return TurnIntent("greeting", False, "greeting; nothing to recall", ("greeting",))

    if _META.match(q):
        return TurnIntent("meta", False, "question about FRIDAY itself", ("meta",))

    if _ACK.match(q):
        return TurnIntent("ack", False, "acknowledgement; no question asked", ("ack",))

    if _CLOCK.match(q) or (_CLOCK_SIMPLE.search(q) and not _CLOCK_DISQUALIFIER.search(q)):
        return TurnIntent("clock", False,
                          "current time/date is read from the system clock, not memory",
                          ("clock",))

    # Everything else retrieves. Deliberately the fallback: see rule 1.
    kind = "memory" if _looks_like_a_memory_question(low) else "general"
    return TurnIntent(kind, True,
                      "unrecognised as trivial, so retrieve — a false negative here "
                      "means answering with no memory behind it",
                      ())


_WH_WORDS = ("what", "where", "when", "who", "whom", "whose", "which", "why", "how much",
             "how many", "how often", "do i", "did i", "have i", "am i", "is my", "are my",
             "my ", "remember", "last time", "usually", "always", "prefer")


def _looks_like_a_memory_question(low: str) -> bool:
    """Distinguish "a question about ME" from "a general question".

    Both retrieve — a general question may still benefit from context ("recommend a
    restaurant" wants to know I'm vegetarian) — so this only labels the intent for
    the audit trail and for `/why`. It must never gate retrieval.
    """
    return any(w in low for w in _WH_WORDS)


def tools_for(intent: TurnIntent, all_tools: list[dict]) -> list[dict]:
    """Filter the tool schemas offered for this turn.

    `memory_search` is withheld when the turn cannot need it. Withholding beats
    instructing: a model told "don't search" will still sometimes search, and the only
    way to make exit test #8 structurally true is to not offer the tool. `memory_write`
    stays available — "note that I hate Mondays" is a first-person turn and is already
    caught by the `_ABOUT_ME` guard above, so it never reaches here with a write
    intent.
    """
    if intent.needs_memory:
        return all_tools
    return [t for t in all_tools
            if (t.get("function", {}) or {}).get("name") != "memory_search"]
