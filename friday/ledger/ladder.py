"""The compaction ladder — Law 2 in code: *shrink the prefix, don't rewrite it.*

Rungs are ordered by (cost to quality) ascending. You fire the cheapest rung that
brings you back under budget, and you never skip ahead to summarisation:

    Rung 0  CAP          truncate each tool output at a fixed size BEFORE it
                         enters history.  -38% cost/turn, won 14 of 15
                         trajectories, recall UNCHANGED. Always on. Free.
    Rung 1  TRUNCATE     head+tail elision of the largest single block
    Rung 2  DEDUPE       drop repeated tool results / repeated retrieval hits
    Rung 3  OFFLOAD      move a block to artifacts/ and leave a POINTER (Law 3:
                         keep pointers, pull on demand)
    Rung 4  EVICT        drop the oldest transcript turns (not the newest)
    Rung 5  SUMMARISE    ⚠️ last resort. In one production eval summarisation cut
                         in-session recall 92% -> ~33% while costing 2x more than
                         keeping everything. Only fire when you can NAME the
                         constraint.

⭐ THE PIVOT AMENDMENT (docs/architecture/12 §1.2, /13):

Law 2 is a property of KV-prefix caching in *attention* models. Under a recurrent
state there is no prefix to cache-break, so rungs 1-5 collapse into ONE learned
operation — the Retention State Compiler — and rung 0 stays because it is free
and it works regardless of architecture.

So `BACKBONE_IS_HYBRID` is a real switch, not a comment:

    hybrid      -> rungs 0 and 3 only; the RSC owns the transcript slot and the
                   window never fills, so rung 5 is unreachable by construction
    transformer -> the full ladder, with COMPACT_AT_FRACTION as the trigger

Whichever way it goes, rung 5 is instrumented to scream. If you see it firing more
than a handful of times a week, something upstream is wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from ..util import approx_tokens, cap_tokens, new_id, short_hash

#: Set from the Phase 0 bake-off result (docs/architecture/12 §5.3).
BACKBONE_IS_HYBRID = os.environ.get("FRIDAY_HYBRID_BACKBONE", "1") == "1"

RUNG_NAMES = {
    0: "cap",
    1: "truncate",
    2: "dedupe",
    3: "offload",
    4: "evict",
    5: "summarise",
}


@dataclass
class CompactionEvent:
    rung: int
    slot: str
    before_tokens: int
    after_tokens: int
    detail: str = ""

    @property
    def name(self) -> str:
        return RUNG_NAMES.get(self.rung, str(self.rung))

    @property
    def saved(self) -> int:
        return self.before_tokens - self.after_tokens

    def __str__(self) -> str:
        return f"rung{self.rung}:{self.name} {self.slot} {self.before_tokens}->{self.after_tokens}"


@dataclass
class LadderResult:
    events: list[CompactionEvent] = field(default_factory=list)
    summarised: bool = False
    #: ⭐ Protected slots that still do not fit after every rung has fired. The ladder
    #: refuses to touch them, so "still over" is a fact about the user's files, not
    #: something more compaction can fix. Reported rather than resolved.
    required_overflow: list[str] = field(default_factory=list)

    @property
    def highest_rung(self) -> int | None:
        return max((e.rung for e in self.events), default=None)

    def summary(self) -> str:
        parts = [str(e) for e in self.events]
        if self.required_overflow:
            parts.append(
                "REQUIRED OVER: " + ", ".join(self.required_overflow)
                + " (stable prefix is never compacted — shorten the file)")
        if not parts:
            return "none"
        return ", ".join(parts)


# ── rung 0 ─────────────────────────────────────────────────────────────────────

#: ⭐ The stable prefix is not ladder fuel.
#:
#: Rungs 1, 3, 4 and 5 each name the slots they may touch ("docs", "recalled",
#: "scratch", "transcript"), so they could never reach the prefix. Rungs 0 and 2,
#: however, iterate EVERY block — and rung 0 fires unconditionally on the first pass.
#: That meant a SOUL.md longer than TOOL_OUTPUT_CAP_TOKENS (2000) was silently capped
#: by a rung whose own docstring says it exists for tool output, and `cap_tokens`
#: stamped it with "full output offloaded, use read_artifact to pull it back" — an
#: instruction that could never be followed, because slot-level capping writes no
#: artifact. The lie then sat in the byte-stable CACHED prefix on every turn.
#:
#: docs/architecture/04 gives `identity` overflow="reject" and required=True. Reject
#: means "tell the human", not "quietly rewrite FRIDAY's character".
PROTECTED_SLOTS = frozenset({"identity", "senses"})


def rung0_cap(text: str, budget: int) -> tuple[str, CompactionEvent | None]:
    """CAP a tool output before it enters history. Always on, always free.

    Never call this on a PROTECTED_SLOTS block — see that constant for why.
    """
    before = approx_tokens(text)
    capped = cap_tokens(text, budget)
    after = approx_tokens(capped)
    if after < before:
        return capped, CompactionEvent(0, "tool_output", before, after, "capped")
    return text, None


# ── rungs 1-4 ──────────────────────────────────────────────────────────────────

def rung1_truncate(text: str, budget: int, slot: str) -> tuple[str, CompactionEvent | None]:
    before = approx_tokens(text)
    if before <= budget:
        return text, None
    return cap_tokens(text, budget), CompactionEvent(1, slot, before, budget, "head+tail elision")


def rung2_dedupe(blocks: list[str], slot: str) -> tuple[list[str], CompactionEvent | None]:
    """Drop exact and near-exact repeats. Tool results repeat constantly: the same
    file read twice, the same search re-issued, the same error three times."""
    before = sum(approx_tokens(b) for b in blocks)
    seen: set[str] = set()
    out: list[str] = []
    for b in blocks:
        key = short_hash(" ".join(b.split()), 16)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    after = sum(approx_tokens(b) for b in out)
    if len(out) < len(blocks):
        return out, CompactionEvent(2, slot, before, after, f"dropped {len(blocks)-len(out)} repeats")
    return blocks, None


def rung3_offload(text: str, slot: str, *, label: str = "") -> tuple[str, CompactionEvent | None]:
    """Move a block to artifacts/offloaded/ and leave a POINTER.

    This is Law 3 applied to compaction: the context keeps a handle, not the
    payload, and the model can pull the payload back with `read_artifact` if — and
    only if — it turns out to matter. Under a hybrid backbone this is one of only
    two rungs that survive, because it is the one that *adds* capability rather
    than removing information.
    """
    before = approx_tokens(text)
    if before < 400:
        return text, None
    d = paths.ARTIFACTS / "offloaded"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{new_id('o')}.md"
    p = d / name
    header = f"# offloaded {slot} {label}\n\n"
    p.write_text(header + text, encoding="utf-8")
    pointer = (
        f"[{before} tokens offloaded to artifacts/offloaded/{name} "
        f"— use read_artifact('{name}') to pull it back if needed]"
    )
    return pointer, CompactionEvent(3, slot, before, approx_tokens(pointer), str(p))


def rung4_evict(turns: list[dict], budget: int, slot: str = "transcript") -> tuple[list[dict], CompactionEvent | None]:
    """Drop the OLDEST turns. Never the newest: the recent window is what makes the
    next turn coherent, and evicting it produces the classic 'it forgot what we
    were talking about' failure.

    The system/identity turn is always kept — it is part of the stable prefix.
    """
    before = sum(approx_tokens(t.get("content", "")) for t in turns)
    if before <= budget:
        return turns, None
    keep_from_end: list[dict] = []
    acc = 0
    for t in reversed(turns):
        n = approx_tokens(t.get("content", ""))
        if acc + n > budget and keep_from_end:
            break
        keep_from_end.insert(0, t)
        acc += n
    pinned = [t for t in turns if t.get("pin")]
    out = pinned + [t for t in keep_from_end if t not in pinned]
    after = sum(approx_tokens(t.get("content", "")) for t in out)
    return out, CompactionEvent(4, slot, before, after, f"evicted {len(turns)-len(out)} oldest")


def rung5_summarise(text: str, summariser, slot: str) -> tuple[str, CompactionEvent | None]:
    """⚠️ LAST RESORT. Requires an explicit summariser callable.

    Signature is `summariser(text) -> str`. If you don't pass one, this rung
    refuses to fire and returns the text unchanged — which is the correct
    behaviour, because silently substituting a cruder compaction for a
    summarisation you didn't ask for is how recall dies without a stack trace.
    """
    before = approx_tokens(text)
    if summariser is None:
        return text, None
    out = summariser(text)
    after = approx_tokens(out)
    if after >= before:
        return text, None
    return out, CompactionEvent(5, slot, before, after, "SUMMARISED — recall risk, investigate")


# ── the ladder ─────────────────────────────────────────────────────────────────

def descend(
    *,
    budget: int,
    blocks: dict[str, list[str]],
    turns: list[dict] | None = None,
    tool_cap: int = 2000,
    offload_above: int = 20000,
    summariser=None,
    hybrid: bool | None = None,
) -> tuple[dict[str, list[str]], list[dict], LadderResult]:
    """Fire the cheapest rung that gets us back under `budget`.

    `blocks` maps slot -> list of text blocks (mutated in place and returned).
    `turns` is the transcript, handled separately because eviction is per-turn.

    Rungs fire in ascending cost-to-quality order and we re-measure after each one,
    so a cheap rung that solves the problem means an expensive rung never runs.
    That ordering is the entire content of Law 2.
    """
    is_hybrid = BACKBONE_IS_HYBRID if hybrid is None else hybrid
    turns = turns or []
    res = LadderResult()

    def over() -> int:
        return total_tokens(blocks, turns) - budget

    if over() <= 0:
        return blocks, turns, res

    # Rung 0 is unconditional: cap every tool-output-looking block.
    for slot, items in blocks.items():
        if slot in PROTECTED_SLOTS:      # the prefix is not tool output
            continue
        for i, b in enumerate(items):
            new, ev = rung0_cap(b, tool_cap)
            if ev:
                items[i] = new
                res.events.append(ev)
    if over() <= 0:
        return blocks, turns, res

    # Rung 2: dedupe. Cheaper than truncation because it removes *redundant* bytes.
    # Still skipped for the prefix: "redundant" is a judgement about prose, and a
    # personality file repeats itself on purpose.
    for slot, items in list(blocks.items()):
        if slot in PROTECTED_SLOTS:
            continue
        new, ev = rung2_dedupe(items, slot)
        if ev:
            blocks[slot] = new
            res.events.append(ev)
    if over() <= 0:
        return blocks, turns, res

    # Rung 3: offload the largest block in the largest non-stable slot.
    for slot in ("docs", "recalled", "scratch"):
        items = blocks.get(slot) or []
        if not items:
            continue
        biggest = max(range(len(items)), key=lambda i: approx_tokens(items[i]))
        if approx_tokens(items[biggest]) < max(600, offload_above // 30):
            continue
        new, ev = rung3_offload(items[biggest], slot)
        if ev:
            items[biggest] = new
            res.events.append(ev)
            if over() <= 0:
                return blocks, turns, res

    if is_hybrid:
        # ⭐ Under a recurrent-state backbone we STOP HERE. Rungs 1/4/5 rewrite or
        # drop the prefix, which is exactly what the RSC replaces with a learned,
        # continuous operation. Firing them anyway would silently reintroduce the
        # 92%->33% recall collapse the pivot exists to avoid.
        # If we're still over, the honest move is to widen the window (a hybrid
        # can — the state is fixed-size) or drop a low-value slot wholesale.
        # Never to summarise.
        res.required_overflow = [
            sl for sl in PROTECTED_SLOTS
            if approx_tokens("\n".join(blocks.get(sl) or [])) > budget
        ]
        return blocks, turns, res

    # Rung 1: truncate the largest remaining block.
    for slot in ("docs", "recalled", "scratch", "transcript"):
        items = blocks.get(slot) or []
        for i, b in enumerate(items):
            n = approx_tokens(b)
            if n > 800:
                new, ev = rung1_truncate(b, max(400, n // 2), slot)
                if ev:
                    items[i] = new
                    res.events.append(ev)
                    if over() <= 0:
                        return blocks, turns, res

    # Rung 4: evict oldest turns.
    if turns:
        turn_tokens = sum(approx_tokens(t.get("content", "")) for t in turns)
        target = max(400, turn_tokens - max(0, over()))
        new_turns, ev = rung4_evict(turns, target)
        if ev:
            turns = new_turns
            res.events.append(ev)
            if over() <= 0:
                return blocks, turns, res

    # Rung 5: summarise. Screams.
    for slot in ("transcript", "docs"):
        items = blocks.get(slot) or []
        if not items:
            continue
        joined = "\n".join(items)
        new, ev = rung5_summarise(joined, summariser, slot)
        if ev:
            blocks[slot] = [new]
            res.events.append(ev)
            res.summarised = True
            if over() <= 0:
                break

    # Every rung has fired. If the prefix ALONE still exceeds the budget there is
    # nothing left to compact, and saying so is the whole point of overflow="reject".
    if over() > 0:
        res.required_overflow = [
            sl for sl in ("identity", "senses")
            if approx_tokens("\n".join(blocks.get(sl) or [])) > 0
            and approx_tokens("\n".join(blocks.get(sl) or [])) > budget
        ]
    return blocks, turns, res


def total_tokens(blocks: dict[str, list[str]], turns: list[dict] | None = None) -> int:
    """Everything the ladder measures, in one place."""
    total = sum(approx_tokens(b) for items in blocks.values() for b in items)
    if turns:
        total += sum(approx_tokens(t.get("content", "")) for t in turns)
    return total
