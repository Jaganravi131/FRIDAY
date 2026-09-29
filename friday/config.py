"""Runtime configuration. Defaults live here; config/*.yaml overrides them.

Deliberately small. Every number in this file is one you will want to change
after a week of looking at the Ledger printout (docs/architecture/10 Day 4).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import paths


# ── decay: the forgetting function ─────────────────────────────────────────────
# Source: docs/architecture/02-INNOVATION-memory-compiler.md §4
# None means "never decays". This is also L_decay's supervision target
# (docs/architecture/13 §5.1): the RSC's channel-wise decay alpha_t is initialised
# toward this policy, then the loss weight anneals to 0 and it learns freely.
HALF_LIVES_DAYS: dict[str, float | None] = {
    "mood": 2,             # "stressed about X" — gone fast
    "current_task": 3,
    "project_state": 14,
    "location": 30,
    "environment": 90,     # "runs Windows 11 on a VivoBook"
    "relationship": 180,
    "preference": 365,     # "prefers TypeScript" — very sticky
    "identity": None,      # "name is Jagan" — never decays
}
DEFAULT_HALF_LIFE_DAYS = 90.0

#: below this, drop from the indices (keep in Markdown, struck through)
DECAY_ARCHIVE_BELOW = 0.15
#: below this, only surface on a direct query — never auto-inject
DECAY_DEMOTE_BELOW = 0.40
#: each access pushes effective age back by this fraction (reinforcement)
DECAY_REINFORCEMENT = 0.15

# ── source_kind -> base confidence ─────────────────────────────────────────────
SOURCE_CONFIDENCE = {
    "user_edit": 1.00,   # ⭐ the trust feature: hand-edits are never auto-overwritten
    "stated": 0.95,
    "imported": 0.80,
    "observed": 0.60,
    "inferred": 0.40,
}

#: Which source may overwrite which. A higher rank wins regardless of confidence —
#: a *seen* bank statement beats a *said* number, because people misremember their
#: own rent and statements do not misremember themselves.
#:
#: `user_edit` is at the top and is special: nothing automatic may overwrite it at
#: any confidence (Law 7). A hand-edit means a human opened a file and typed it.
SOURCE_RANK = {
    "user_edit": 4,
    "observed": 3,
    "imported": 2,
    "stated": 1,
    "inferred": 0,
}

#: Within a rank, the newcomer must beat the incumbent by this much. The margin
#: exists so a stream of marginally-confident chatter cannot slowly walk a fact
#: somewhere the user never said it — the failure mode you get with `>` alone.
CONFIDENCE_MARGIN = 0.05

# ── predicate taxonomy ─────────────────────────────────────────────────────────
#: only one of these may be true at a time -> writing a new one RETRACTS the old
#: Predicates where only ONE value can be true at a time, so a new assertion must
#: RETRACT the old one rather than sit alongside it.
#:
#: ⚠️ SYNONYMS MUST BE LISTED TOO, and this is the bug that makes the omission
#: expensive rather than merely untidy. `employer` was here but `works_at` was not,
#: so asserting "I work at Acme Corp" and then "I moved to Globex" left BOTH live.
#: Retrieval then returns two contradictory employers with no way to tell which is
#: current — the exact failure bi-temporal bookkeeping exists to prevent, and it
#: fails silently because nothing raises. When you add a predicate, add the ways a
#: person would naturally say it. PREDICATE_CLASS_RULES below already routes both
#: spellings to the same decay class, which is how the inconsistency slipped through.
SINGLE_VALUED_PREDICATES = {
    # place
    "lives_in", "address_locality", "home_address", "current_city", "city",
    # money / housing
    "lease_amount_monthly", "monthly_rent", "rent", "lease_renewal_due",
    "landlord", "landlord_name",
    # work
    "employer", "works_at", "role", "job_title", "salary",
    # state
    "current_task", "mood", "relationship_status",
    # single owned/contactable things
    "phone", "phone_model", "email", "bank_account", "vehicle", "timezone",
    "primary_language", "daily_routine",
}

#: predicate -> predicate_class (drives decay). Prefix rules, longest match wins.
PREDICATE_CLASS_RULES: list[tuple[str, str]] = [
    ("name", "identity"), ("dob", "identity"), ("birth", "identity"),
    ("gender", "identity"), ("nationality", "identity"), ("native_", "identity"),
    ("lease_", "environment"), ("rent", "environment"), ("landlord", "relationship"),
    ("address", "location"), ("lives_in", "location"), ("city", "location"),
    ("locality", "location"), ("timezone", "environment"),
    ("prefers", "preference"), ("preference", "preference"), ("likes", "preference"),
    ("dislikes", "preference"), ("diet", "preference"), ("allerg", "identity"),
    ("mood", "mood"), ("feeling", "mood"), ("stressed", "mood"),
    ("current_task", "current_task"), ("todo", "current_task"), ("working_on", "current_task"),
    ("project", "project_state"), ("deadline", "project_state"), ("due", "project_state"),
    ("employer", "environment"), ("works_at", "environment"), ("role", "environment"),
    ("salary", "environment"), ("device", "environment"), ("os", "environment"),
    ("hardware", "environment"), ("ram", "environment"),
    ("relation", "relationship"), ("friend", "relationship"), ("family", "relationship"),
    ("amma", "relationship"), ("appa", "relationship"), ("colleague", "relationship"),
]

# ── retrieval ──────────────────────────────────────────────────────────────────
# docs/architecture/04 Law D: wide -> rerank -> narrow -> HARD FLOOR -> []
RETRIEVAL_K_WIDE = 20
RETRIEVAL_K_SHIP = 3
#: ⭐ Law 4. An empty list beats noise. Ask FRIDAY something it has no memory of
#: and it MUST return [], not three plausible-looking facts. This is the line
#: between a memory system and a hallucination machine.
RETRIEVAL_SCORE_FLOOR = 0.35
RETRIEVAL_GRAPH_HOPS = 2
EMBED_DIM = 1024

# ── Attention Ledger ───────────────────────────────────────────────────────────
#: Total working context. Raise this if the backbone is a hybrid linear-attention
#: model (LFM2 / RWKV-7): a fixed-size state means 32K-128K is affordable on
#: 16 GB, whereas a Transformer's KV cache caps you at 8-12K.
#: See docs/architecture/12 §5.4.
LEDGER_BUDGET_TOKENS = int(os.environ.get("FRIDAY_CTX", "8192"))

#: Per-slot budgets. `identity` + `senses` form the STABLE CACHED PREFIX and must
#: not change between turns or every turn re-prefills (Law 2).
LEDGER_SLOTS: dict[str, int] = {
    "identity": 1024,     # SOUL.md + AGENTS.md — the cached prefix
    "user": 512,          # USER.md (hard budget 1,400 chars ~ 350 tokens)
    "core_mem": 640,      # MEMORY.md — the ~20 facts that matter most
    "senses": 256,        # which senses are enabled right now
    "recalled": 1200,     # JIT retrieval results (Law 3)
    "docs": 3000,         # offloaded tool output, pulled by pointer
    "transcript": 2560,   # recent turns; with an RSC this becomes a fixed state
    "scratch": 768,       # rolling notes, NOT full history
}
#: never let the model eat this — it's the room to answer
LEDGER_RESERVE_TOKENS = 1024
#: Rung 1: cap each tool output before it enters history. The -38% cost lever,
#: recall unchanged. Survives the linear-attention pivot untouched (Rung 0).
TOOL_OUTPUT_CAP_TOKENS = 2000
#: Rung 4: offload-to-file-with-pointer above this
OFFLOAD_ABOVE_TOKENS = 20000
#: compact when the window is this full (Transformer backbones only; a hybrid
#: never reaches it because the window does not fill)
COMPACT_AT_FRACTION = 0.75

# ── inference ──────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.environ.get("FRIDAY_LLM_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.environ.get("FRIDAY_LLM_MODEL", "friday-local")
LLM_API_KEY = os.environ.get("FRIDAY_LLM_KEY", "not-needed")
LLM_TIMEOUT_S = 120.0
MAX_AGENT_TURNS = 12

# ── L0 gate / escalation ladder ────────────────────────────────────────────────
#: ₹0 budget: L2..L5 are DISABLED (docs/architecture/12 §8). When the top rung is
#: missing, `escalate` must degrade to "ask the user", never "guess confidently".
CLOUD_TIERS_ENABLED = False
L0_MAX_CHARS_FOR_LOCAL = 400


@dataclass
class Settings:
    """Everything above, as one object you can pass around and dump."""

    half_lives: dict[str, Any] = field(default_factory=lambda: dict(HALF_LIVES_DAYS))
    source_confidence: dict[str, float] = field(default_factory=lambda: dict(SOURCE_CONFIDENCE))
    ledger_budget: int = LEDGER_BUDGET_TOKENS
    ledger_slots: dict[str, int] = field(default_factory=lambda: dict(LEDGER_SLOTS))
    reserve: int = LEDGER_RESERVE_TOKENS
    score_floor: float = RETRIEVAL_SCORE_FLOOR
    k_wide: int = RETRIEVAL_K_WIDE
    k_ship: int = RETRIEVAL_K_SHIP
    embed_dim: int = EMBED_DIM
    llm_base_url: str = LLM_BASE_URL
    llm_model: str = LLM_MODEL
    cloud_tiers_enabled: bool = CLOUD_TIERS_ENABLED

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """Defaults, then optional JSON overrides from config/settings.json."""
        s = cls()
        p = path or (paths.CONFIG / "settings.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        return s

    def dump(self) -> dict[str, Any]:
        return asdict(self)


SETTINGS = Settings.load()


#: How people SAY a predicate, as opposed to how it is spelled in the store.
#:
#: This is not a convenience. FTS5 tokenises `lease_amount_monthly` as ONE token, so
#: the query "what's my rent?" matches nothing — the user's word and the store's word
#: share no substring. Without this map, semantic retrieval is the only bridge, and
#: the zero-dependency hashing embedder is a lexical n-gram hasher with no idea that
#: "rent" and "lease" are related. So on a fresh machine with no model server, every
#: natural-language query about money would return [].
#:
#: Snake_case splitting (done automatically) covers "monthly" and "lease"; this map
#: covers the synonyms that no amount of splitting will produce.
#: predicate -> the ways a person would actually ASK about it.
#:
#: Two jobs, and they fail differently. At INDEX time these strings are folded into
#: the FTS body, so a natural-language query can reach a snake_case predicate at all.
#: At RERANK time `_rerank_text` appends them, which is what lets the stdlib lexical
#: fallback score "where do I work" against `works_at: Globex` above the floor
#: instead of returning 0.000. Without them the no-dependency retrieval path finds
#: the right candidate and then throws it away.
#:
#: ⚠️ Cover the SYNONYMS, not just the canonical name. `lease_amount_monthly` had
#: aliases and `monthly_rent` had none, so "what is my monthly rent" depended on
#: stemming to bridge a gap the alias table was supposed to close.
PREDICATE_ALIASES: dict[str, tuple[str, ...]] = {
    "lease_amount_monthly": ("rent", "monthly rent", "house rent", "rent amount"),
    "monthly_rent": ("rent", "monthly rent", "house rent", "rent amount", "lease"),
    "rent": ("monthly rent", "house rent", "rent amount", "lease"),
    "works_at": ("employer", "company", "workplace", "where do i work", "job"),
    "name": ("called", "who is", "full name"),
    "landlord_name": ("landlord", "house owner"),
    "lease_renewal_due": ("lease renewal", "renewal date", "lease expiry"),
    "address_locality": ("area", "neighbourhood", "neighborhood", "where do i live"),
    "lives_in": ("city", "where do i live", "current city"),
    "employer": ("company", "workplace", "where do i work", "job"),
    "current_task": ("working on", "current work", "in progress"),
    "phone_model": ("phone", "mobile", "handset"),
    "primary_language": ("language", "languages spoken"),
    "landlord": ("house owner", "landlord name"),
    "cloud_budget": ("budget", "api budget", "spend"),
    "training_compute": ("gpu", "compute", "kaggle"),
}


def aliases_for(predicate: str) -> tuple[str, ...]:
    """Human phrasings for a predicate, plus its snake_case parts split into words.

    The split is unconditional and cheap; the alias map covers what splitting cannot.
    """
    p = (predicate or "").strip().lower()
    out = list(PREDICATE_ALIASES.get(p, ()))
    parts = [w for w in re.split(r"[_\-\s]+", p) if w]
    if len(parts) > 1:
        out.append(" ".join(parts))       # "lease amount monthly" as a phrase
        out.extend(parts)                 # and each word on its own
    seen: dict[str, None] = {}
    for o in out:
        if o and o != p:
            seen.setdefault(o, None)
    return tuple(seen)


def predicate_class_of(predicate: str) -> str:
    """Longest-prefix rule match; falls back to 'environment' (90-day half-life)."""
    p = predicate.lower().strip()
    best = ("", "environment")
    for prefix, klass in PREDICATE_CLASS_RULES:
        if prefix in p and len(prefix) > len(best[0]):
            best = (prefix, klass)
    return best[1]


#: non-canonical predicate -> the ONE predicate FRIDAY stores it under.
#:
#: ⭐ WHY THIS EXISTS. Without it, "what is my monthly rent" has two right answers.
#: The seed data says `lease_amount_monthly: ₹28,000`; a user who types
#: `friday write monthly_rent 18000` creates a SECOND live fact, because
#: `_blocked_by` and `_reconcile` both match on exact (subject, predicate). Neither
#: is single-valued against the other, so both stay live and retrieval returns
#: whichever reranks higher. Observed: FRIDAY confidently answered ₹28,000 seconds
#: after being told 18000 — the single worst failure mode for a context-aware
#: assistant, because the answer is fluent, sourced, and wrong.
#:
#: DELIBERATELY CONSERVATIVE. Only predicates that mean the SAME THING are merged.
#: `lives_in` (city) and `address_locality` (neighbourhood) are NOT merged, and
#: neither are `landlord` / `landlord_name` with anything else — collapsing two
#: genuinely different facts is worse than duplicating one, because it destroys data
#: rather than merely confusing a query. Add to this map only when you are certain
#: the two predicates cannot both be true at once with different values.
CANONICAL_PREDICATES: dict[str, str] = {
    # rent: one monthly amount for the place you live
    "monthly_rent": "lease_amount_monthly",
    "rent": "lease_amount_monthly",
    "house_rent": "lease_amount_monthly",
    "rent_amount": "lease_amount_monthly",
    "rent_monthly": "lease_amount_monthly",
    # employer: one place you work
    "works_at": "employer",
    "company": "employer",
    "workplace": "employer",
    # the person's own name
    "full_name": "name",
    # how you get to work / where you are
    "current_city": "lives_in",
}


def canonical_predicate(predicate: str) -> str:
    """Map a predicate to the one FRIDAY stores it under.

    Idempotent, and identity for anything not in the map — so a canonical predicate
    passes through untouched and an unknown one is preserved rather than mangled.
    """
    p = (predicate or "").strip().lower()
    return CANONICAL_PREDICATES.get(p, predicate or "")


def is_single_valued(predicate: str) -> bool:
    # Normalise first: `monthly_rent` must be single-valued because the fact is
    # STORED as `lease_amount_monthly`, and checking the alias would miss it.
    return canonical_predicate(predicate).lower().strip() in SINGLE_VALUED_PREDICATES


def half_life_of(predicate_class: str) -> float | None:
    return SETTINGS.half_lives.get(predicate_class, DEFAULT_HALF_LIFE_DAYS)
