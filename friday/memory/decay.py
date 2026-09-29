"""The forgetting function. Source: docs/architecture/02-INNOVATION-memory-compiler.md §4

Two consumers, and that duality is the point (docs/architecture/13 §4.1):

  1. RETRIEVAL — a fact whose decayed confidence falls below DEMOTE stops being
     auto-injected; below ARCHIVE it leaves the indices entirely (but stays in the
     Markdown, struck through, because the Markdown is the truth).

  2. THE RSC — this function is the supervision target for the channel-wise decay
     gate alpha_t in `L_decay`. The RSC is initialised toward this policy, then the
     loss weight anneals to zero and alpha_t is learned freely. A hand-tuned prior
     beats a cold start, and annealing means the prior never becomes a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import (
    DECAY_ARCHIVE_BELOW,
    DECAY_DEMOTE_BELOW,
    DECAY_REINFORCEMENT,
    half_life_of,
)
from ..util import age_days


@dataclass
class DecayResult:
    confidence: float          # decayed confidence, 0..1
    half_life: float | None    # days; None = never decays
    effective_age: float       # days, after reinforcement
    tier: str                  # "active" | "demoted" | "archived" | "immortal"

    @property
    def auto_inject(self) -> bool:
        """May this fact be placed in context without being asked for?"""
        return self.tier in ("active", "immortal")

    @property
    def keep_indexed(self) -> bool:
        """Should it remain in FTS/vec? Archived facts answer time-travel queries
        ("where did I live last year?") but never surface unprompted."""
        return self.tier != "archived"


def decay(
    *,
    predicate_class: str,
    confidence: float,
    asserted_at: str | None,
    access_count: int = 0,
    now: datetime | None = None,
) -> DecayResult:
    """confidence * 0.5 ** (effective_age / half_life), with reinforcement.

    The reinforcement term is what makes this more than exponential rot: recalling
    a fact makes it stickier. That is importance x recency x relevance — the scoring
    the whole memory ecosystem converged on — but with an explicit, tunable
    half-life per *class* instead of one global number.
    """
    hl = half_life_of(predicate_class)
    if hl is None:
        return DecayResult(confidence, None, 0.0, "immortal")

    raw_age = age_days(asserted_at, now)
    effective_age = raw_age / (1.0 + DECAY_REINFORCEMENT * max(0, access_count))
    decayed = confidence * (0.5 ** (effective_age / hl))

    if decayed < DECAY_ARCHIVE_BELOW:
        tier = "archived"
    elif decayed < DECAY_DEMOTE_BELOW:
        tier = "demoted"
    else:
        tier = "active"
    return DecayResult(decayed, hl, effective_age, tier)


def decay_row(row, now: datetime | None = None) -> DecayResult:
    """Same, from a `facts` sqlite Row."""
    return decay(
        predicate_class=row["predicate_class"],
        confidence=float(row["confidence"]),
        asserted_at=row["asserted_at"],
        access_count=int(row["access_count"] or 0),
        now=now,
    )


def policy_alpha(predicate_class: str, asserted_at: str | None, now: datetime | None = None) -> float:
    """The per-step retention this policy implies, in (0,1].

    This is the number `L_decay` regresses the RSC's channel-wise alpha_t toward.
    We convert a half-life in days into a per-token retention using an assumed
    tokens-per-day of conversational density, so that alpha^T over a day's worth
    of tokens equals the policy's one-day retention.
    """
    hl = half_life_of(predicate_class)
    if hl is None:
        return 1.0
    # One day of interactive use ~ 4,000 tokens (conservative; calibrate from
    # your own traces). alpha_per_token = 0.5 ** (1 day / hl / tokens_per_day).
    tokens_per_day = 4000.0
    return 0.5 ** (1.0 / float(hl) / tokens_per_day)
