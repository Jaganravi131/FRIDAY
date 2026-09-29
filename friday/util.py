"""Small shared helpers. Stdlib only."""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable

# ── time ───────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    """Local-time ISO-8601 with offset. FRIDAY is single-user and single-tz, and
    you read these timestamps with your eyes, so local beats UTC here."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # tolerate bare dates and partial dates: 2026-09, 2026
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt


def partial_date(s: str | None) -> str | None:
    """Normalise `2023-06` / `2023` / `2023-06-01` -> a comparable `YYYY-MM-DD`."""
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return s + "-01"
    if re.fullmatch(r"\d{4}", s):
        return s + "-01-01"
    dt = parse_iso(s)
    return dt.strftime("%Y-%m-%d") if dt else None


def age_days(asserted_at: str | None, now: datetime | None = None) -> float:
    dt = parse_iso(asserted_at)
    if dt is None:
        return 0.0
    return max(0.0, ((now or datetime.now().astimezone()) - dt).total_seconds() / 86400.0)


# ── ids ────────────────────────────────────────────────────────────────────────

_COUNTER = 0


def new_id(prefix: str) -> str:
    """Time-sortable, collision-resistant without a ULID dependency.

    `<prefix>_<base32-millis>_<6 hex of randomness+counter>`
    """
    global _COUNTER
    _COUNTER += 1
    ms = int(time.time() * 1000)
    b32 = _b32(ms)
    tail = hashlib.sha1(os.urandom(8) + str(_COUNTER).encode()).hexdigest()[:6]
    return f"{prefix}_{b32}{tail}"


_B32 = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford-ish, no i/l/o/u


def _b32(n: int) -> str:
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 32)
        out.append(_B32[r])
    return "".join(reversed(out))


def short_hash(text: str, n: int = 6) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


# ── tokens (approximation, deliberately dependency-free) ───────────────────────

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_tokens(text: str) -> int:
    """Cheap, monotone, good enough for budgeting.

    Not a real BPE count. It is *stable* and *comparable*, which is what a budget
    needs — the Ledger's job is relative allocation, not exact billing. Calibrate
    against `llama-server`'s reported prompt tokens and adjust the divisor.
    """
    if not text:
        return 0
    toks = _WORD.findall(text)
    # ~1.3 subword pieces per whitespace/punct token for English; Tamil and code
    # run higher, so add a length term.
    chars = sum(len(t) for t in toks)
    return max(1, int(len(toks) * 1.0 + chars / 4.0))


def cap_tokens(text: str, budget: int, marker: str = "…") -> str:
    """Law B Rung 0/Rung 1: CAP, don't summarise. Deterministic head+tail truncation.

    Keeping head *and* tail matters: tool output usually buries the answer in the
    middle-to-end, and head-only truncation silently drops it.
    """
    if approx_tokens(text) <= budget:
        return text
    head_budget = int(budget * 0.7)
    tail_budget = budget - head_budget
    words = text.split()
    head, acc = [], 0
    for w in words:
        acc += approx_tokens(w) + 1
        if acc > head_budget:
            break
        head.append(w)
    tail, acc = [], 0
    for w in reversed(words):
        acc += approx_tokens(w) + 1
        if acc > tail_budget:
            break
        tail.insert(0, w)
    dropped = approx_tokens(text) - approx_tokens(" ".join(head)) - approx_tokens(" ".join(tail))
    return (
        " ".join(head)
        + f"\n{marker} [{dropped} tokens elided — full output offloaded, "
        f"use `read_artifact` to pull it] {marker}\n"
        + " ".join(tail)
    )


# ── misc ───────────────────────────────────────────────────────────────────────

def dedupe(items: Iterable[Any], key=lambda x: x) -> list[Any]:
    seen: set[Any] = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


#: Very small suffix stripper. Not a Porter stemmer, and deliberately not: this
#: exists so that "live" matches "lives_in", "work" matches "works_at" and "rent"
#: matches "rents". Full Porter over-stems in ways that hurt a fact index
#: ("universal" and "university" both -> "univers"), and FRIDAY's vocabulary is
#: short and concrete enough that a handful of suffix rules covers the real cases.
_STEM_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),        # cities -> city
    ("sses", "ss"),      # addresses -> address
    ("oes", "o"),        # potatoes -> potato
    ("ches", "ch"),
    ("shes", "sh"),
    ("xes", "x"),
    ("zes", "z"),
    ("ing", ""),         # living -> liv
    ("tion", "tion"),
    ("ly", ""),
    ("es", ""),          # addresses -> address, boxes -> box
    ("s", ""),           # lives -> live, rents -> rent
)
#: Words where naive suffix stripping produces something useless.
_STEM_KEEP: frozenset = {"is", "as", "us", "was", "has", "this", "its", "in", "on"}


def stem(word: str) -> str:
    """Reduce a single lowercase word to a crude stem.

    Applied to BOTH sides of a lexical comparison, so it only ever helps: if the
    query says "live" and the fact says "lives_in", both reduce far enough to meet.
    Deterministic and stdlib-only, which is what keeps the no-dependency retrieval
    path actually working rather than being a stub that returns zeros.
    """
    w = (word or "").lower()
    if len(w) <= 3 or w in _STEM_KEEP:
        return w
    for suf, rep in _STEM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) + len(rep) >= 3:
            w = w[: -len(suf)] + rep
            break
    # A trailing silent "e" is the last gap: "live" and "liv" (from "living") have
    # to meet for a verbal predicate to match its gerund. Only strip when the stem
    # still has a vowel, so "the"/"ate" are not mangled into "th"/"at".
    if len(w) > 3 and w.endswith("e") and any(ch in "aiouy" for ch in w[:-1]):
        w = w[:-1]
    return w
