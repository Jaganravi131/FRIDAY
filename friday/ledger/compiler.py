"""The Attention Ledger — deterministic context assembly (Innovation #4).

Compile is a pure function: no model call, no I/O beyond reading the soul files,
under ~5 ms. Every property that makes context engineering debuggable comes from
that: the same inputs give the same bytes, so `ledger_hash` is a fingerprint you
can diff, cache-hit rate becomes measurable, and a bad turn is reproducible.

    gather candidates -> allocate per slot -> descend the ladder -> render -> hash
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from ..config import (
    COMPACT_AT_FRACTION,
    LEDGER_BUDGET_TOKENS,
    LEDGER_RESERVE_TOKENS,
    LEDGER_SLOTS,
    OFFLOAD_ABOVE_TOKENS,
    TOOL_OUTPUT_CAP_TOKENS,
)
from ..memory.decay import decay_row
from ..util import approx_tokens, now_iso, short_hash
from . import ladder
from .slots import SLOT_ORDER, STABLE_SLOTS, Slot, SlotSet


@dataclass
class CompiledContext:
    slots: SlotSet
    budget: int
    spent: int
    reserve: int
    ladder: ladder.LadderResult
    text: str
    ledger_hash: str
    prefix_hash: str
    prefix_stable_runs: int = 0
    cache_hit: bool = False

    @property
    def overflow(self) -> str:
        return self.ladder.summary()

    @property
    def summarised(self) -> bool:
        """Law 2 alarm. If this is True more than a handful of times a week,
        something upstream is wrong — the ladder should never need rung 5."""
        return self.ladder.summarised

    def printout(self, turn: int | None = None) -> str:
        """The dev-mode block from docs/architecture/10 Day 4.

        Looking at this for one week teaches more about context engineering than
        any article. It is on by default in the CLI for exactly that reason.
        """
        head = (
            f"LEDGER turn={turn if turn is not None else '?'}  budget={self.budget}  "
            f"spent={self.spent}  cache_hit={str(self.cache_hit).lower()}  "
            f"prefix_stable={self.prefix_stable_runs}"
        )
        rows = []
        for name, sl in self.slots.items():
            # ⚠OVER is louder than * on purpose: a truncated JIT slot is the ladder
            # doing its job, while an over-budget REQUIRED slot means the stable
            # prefix could not fit and was shipped whole — that needs a human decision.
            mark = "\u26a0OVER" if sl.over_budget else ("*" if sl.truncated else "")
            rows.append(f"{name} {sl.tokens}/{sl.budget}{mark}")
        # Hoisted out of the f-string: a backslash escape is not permitted inside an
        # f-string expression part before Python 3.12, and this has to run on 3.11.
        reserve_mark = "\u2713" if self.reserve > 0 else "\u2717 OVER"
        over = [n for n, s in self.slots.items() if s.over_budget]
        over_line = ""
        if over:
            over_line = (
                f"\u2502 \u26a0 REQUIRED SLOT OVER BUDGET: {', '.join(over)} shipped "
                f"whole rather than truncated.\n"
                f"\u2502   Shorten the file or raise its budget in LEDGER_SLOTS. "
                f"The stable prefix is never rationed.\n"
            )
        return (
            "\u250c " + head + "\n"
            + over_line +
            "\u2502 " + " \u00b7 ".join(rows[:4]) + "\n"
            "\u2502 " + " \u00b7 ".join(rows[4:]) + "\n"
            f"\u2502 overflow: {self.overflow}   reserve: {self.reserve} "
            f"{reserve_mark}\n"
            f"\u2514 hash={self.ledger_hash} prefix={self.prefix_hash}"
        )


@dataclass
class Gather:
    """Everything the Ledger may spend. The agent loop fills this; the Ledger
    decides what actually ships."""

    query: str = ""
    recalled: list[dict] = field(default_factory=list)   # SearchResult.as_dicts()
    docs: list[str] = field(default_factory=list)        # tool output / offloaded pulls
    transcript: list[dict] = field(default_factory=list) # [{role, content, pin?}]
    scratch: str = ""
    senses: list[str] = field(default_factory=list)
    state_block: str | None = None                       # ⭐ RSC output, when present
    #: ⭐ The current date/time, when the turn needs it. A local model has no reliable
    #: clock, so "what time is it?" cannot be answered from weights — and the tool
    #: contract deliberately withholds memory for it, so it cannot be answered from
    #: facts either. Injecting the reading is what makes the contract produce an
    #: ANSWER rather than "I don't have that in memory".
    now: str = ""


class Ledger:
    def __init__(
        self,
        *,
        budget: int = LEDGER_BUDGET_TOKENS,
        slot_budgets: dict[str, int] | None = None,
        reserve: int = LEDGER_RESERVE_TOKENS,
        soul_dir: Path | None = None,
        hybrid: bool | None = None,
        summariser=None,
    ):
        self.budget = budget
        self.slot_budgets = dict(slot_budgets or LEDGER_SLOTS)
        self.reserve_target = reserve
        self.soul = soul_dir or paths.SOUL
        self.hybrid = ladder.BACKBONE_IS_HYBRID if hybrid is None else hybrid
        self.summariser = summariser
        self._last_prefix_hash: str | None = None
        self._prefix_runs = 0

    # ── stable prefix ──────────────────────────────────────────────────────────

    def identity_text(self) -> str:
        parts = []
        for name in ("SOUL.md", "AGENTS.md"):
            p = self.soul / name
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
        return "\n\n".join(parts)

    def senses_text(self, senses: list[str]) -> str:
        if not senses:
            return "senses: none enabled (FRIDAY can only use what you type or paste)"
        return "senses enabled: " + ", ".join(sorted(senses))

    # ── compile ────────────────────────────────────────────────────────────────

    def compile(self, g: Gather, *, conn: sqlite3.Connection | None = None) -> CompiledContext:
        slots = SlotSet.from_budgets(self.slot_budgets)

        # STABLE PREFIX — must be byte-identical across turns or every turn re-prefills.
        _fill(slots["identity"], self.identity_text())
        _fill(slots["senses"], self.senses_text(g.senses))

        # JIT ZONE — changes every turn, always after the stable prefix.
        _fill(slots["user"], _read(self.soul / "USER.md"))
        _fill(slots["core_mem"], self.core_memory(conn))

        recalled = _render_recalled(g.recalled)
        if g.state_block:
            # ⭐ The RSC slot. Fixed size, unbounded history: this is what replaces
            # the transcript slot under a recurrent-state backbone
            # (docs/architecture/12 §6.4). It is *prepended* to recalled because the
            # state is context, and retrieved facts are the exact-memory path that
            # Law 2b says must never be delegated to a lossy state.
            recalled = g.state_block + ("\n" + recalled if recalled else "")
        _fill(slots["recalled"], recalled)

        _fill(slots["docs"], "\n\n".join(g.docs))

        # ⚠️ `now` goes in a JIT SLOT, never in `identity` or `senses`. Those two form
        # the stable cached prefix and must be byte-identical across turns; a ticking
        # clock in either would invalidate the provider's prompt cache on every single
        # turn, which is the exact cost Law 2 exists to prevent (exit test #7 measures
        # cache_hit_rate >= 0.85 over 50 turns). Scratch is JIT and small, so it is the
        # cheap correct home. Phase 3 (perception) should give ambient readings their
        # own JIT slot rather than sharing this one.
        scratch = g.scratch
        if g.now:
            scratch = f"[now] {g.now}" + (f"\n{scratch}" if scratch else "")
        _fill(slots["scratch"], scratch)

        # transcript: per-turn, because eviction is per-turn
        blocks = {
            name: ([slots[name].content] if slots[name].content else [])
            for name in SLOT_ORDER
            if name != "transcript"
        }
        turns = list(g.transcript)
        if turns and not self.hybrid:
            _fill(slots["transcript"], _render_turns(turns))
            blocks["transcript"] = [slots["transcript"].content]

        # The ladder, only if we're actually over.
        available = self.budget - self.reserve_target
        total = ladder.total_tokens(blocks, turns if self.hybrid else None)
        if total > available:
            blocks, turns, res = ladder.descend(
                budget=available,
                blocks=blocks,
                turns=turns if self.hybrid else None,
                tool_cap=TOOL_OUTPUT_CAP_TOKENS,
                offload_above=OFFLOAD_ABOVE_TOKENS,
                summariser=self.summariser,
                hybrid=self.hybrid,
            )
        else:
            res = ladder.LadderResult()
        # A protected slot that the ladder could not fix is still over budget, and
        # must be reported even when the slot-level check in _fill passed (it can
        # pass: 1234 < 1536 slot budget, but > available once every other slot fills).
        for name in res.required_overflow:
            if name in slots.slots:
                slots[name].over_budget = True

        # write post-ladder content back into the slots
        for name, items in blocks.items():
            if name not in slots.slots:
                continue
            s = slots[name]
            before = s.content
            s.content = "\n\n".join(items)
            s.tokens = approx_tokens(s.content)
            if s.stable:
                # ⭐ Belt and braces: ladder.PROTECTED_SLOTS should mean the prefix
                # never shrinks here. If it ever did, that is a bug in a rung, and the
                # honest response is to flag it as over-budget rather than to record a
                # routine truncation that nobody would look twice at.
                if s.tokens < approx_tokens(before) - 1:
                    s.over_budget = True
                continue
            s.truncated = s.tokens < approx_tokens(before) - 1
        if self.hybrid and turns:
            _fill(slots["transcript"], _render_turns(turns))

        # compact-at-fraction is a Transformer-only concern: a hybrid's window
        # doesn't fill, so the trigger is unreachable by construction.
        if not self.hybrid:
            frac = ladder.total_tokens(blocks, None) / max(1, self.budget)
            if frac >= COMPACT_AT_FRACTION and self.summariser is not None:
                pass  # rung 5 already had its chance inside descend()

        text = render(slots)
        spent = approx_tokens(text)
        prefix = "\n\n".join(
            slots[n].render() for n in ("identity", "senses") if slots[n].content.strip()
        )
        prefix_hash = short_hash(prefix, 8)
        if prefix_hash == self._last_prefix_hash:
            self._prefix_runs += 1
            cache_hit = True
        else:
            self._prefix_runs = 1
            cache_hit = False
        self._last_prefix_hash = prefix_hash

        return CompiledContext(
            slots=slots,
            budget=self.budget,
            spent=spent,
            reserve=self.budget - spent,
            ladder=res,
            text=text,
            ledger_hash=short_hash(text, 8),
            prefix_hash=prefix_hash,
            prefix_stable_runs=self._prefix_runs,
            cache_hit=cache_hit,
        )

    # ── core memory: MEMORY.md + the decayed top-N ─────────────────────────────

    def core_memory(self, conn: sqlite3.Connection | None, limit: int = 24) -> str:
        """MEMORY.md is hand-written truth. The decayed top-N is appended so that
        newly-compiled facts reach context before anyone edits MEMORY.md.

        Facts below DEMOTE are excluded — that's what 'auto-inject' means, and it's
        the mechanism that stops a six-month-old mood from haunting every turn.
        """
        parts = []
        mem = _read(self.soul / "MEMORY.md")
        if mem:
            parts.append(mem)
        if conn is None:
            return "\n".join(parts)
        rows = conn.execute(
            """SELECT * FROM facts WHERE retracted_at IS NULL
               ORDER BY confidence DESC, access_count DESC, asserted_at DESC LIMIT ?""",
            (limit * 3,),
        ).fetchall()
        lines = []
        for r in rows:
            d = decay_row(r)
            if not d.auto_inject:
                continue
            vf = f" (since {r['valid_from']})" if r["valid_from"] else ""
            lines.append(f"- {r['predicate']}: {r['object']}{vf}")
            if len(lines) >= limit:
                break
        if lines:
            parts.append("## Live facts (auto-compiled, decayed)\n" + "\n".join(lines))
        return "\n\n".join(parts)


# ── helpers ────────────────────────────────────────────────────────────────────

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _fill(slot: Slot, text: str) -> None:
    """Fill a slot, rationing at its budget — EXCEPT the stable prefix.

    Truncation here is slot-level rationing (deterministic, cheap); the *ladder* is
    for whole-context pressure.

    ⭐ STABLE SLOTS ARE NEVER TRUNCATED. `identity` and `senses` are `required` and
    their overflow policy in docs/architecture/04 is "reject", not "truncate". Two
    reasons, and the second is the one that hides:

      1. Truncating `identity` silently edits FRIDAY's character. The seeded
         SOUL.md + AGENTS.md is 1234 tokens against a 1024 budget, so this was not
         hypothetical — the personality file was being capped on every install.
      2. `cap_tokens` inserts a marker reading "full output offloaded, use
         read_artifact to pull it back". Nothing was offloaded: slot-level rationing
         never writes an artifact. So the prefix contained an instruction that could
         not be followed, and because the prefix is CACHED, that lie was byte-stable
         across every turn.

    Instead the slot ships whole and sets `over_budget`, which the printout reports.
    The fix is to shorten the file or raise the budget — a decision for the user, not
    something to do to them silently.
    """
    text = (text or "").strip()
    n = approx_tokens(text)
    if n > slot.budget > 0:
        if slot.stable:
            slot.over_budget = True
        else:
            from ..util import cap_tokens

            text = cap_tokens(text, slot.budget)
            n = approx_tokens(text)
            slot.truncated = True
    slot.content = text
    slot.tokens = n


def _render_recalled(recalled: list[dict]) -> str:
    if not recalled:
        return ""
    out = []
    for r in recalled:
        if r.get("predicate"):
            prov = []
            if r.get("source_kind"):
                prov.append(r["source_kind"])
            if r.get("asserted_at"):
                prov.append(f"learned {r['asserted_at'][:10]}")
            if r.get("valid_from"):
                prov.append(f"valid from {r['valid_from']}")
            if r.get("retracted_at"):
                prov.append(f"RETRACTED {r['retracted_at'][:10]} (historical)")
            tail = f"  [{' · '.join(prov)}]" if prov else ""
            quote = f'\n  > "{r["source_quote"]}"' if r.get("source_quote") else ""
            out.append(f"- {r['predicate']}: {r['object']}{tail}{quote}")
        else:
            out.append(f"- [{r.get('kind','?')}] {str(r.get('body',''))[:300]}")
    return (
        "Retrieved from long-term memory. Provenance is part of the answer: if a "
        "fact is marked RETRACTED it is historical, not current.\n" + "\n".join(out)
    )


def _render_turns(turns: list[dict]) -> str:
    out = []
    for t in turns:
        role = t.get("role", "user")
        out.append(f"{role}: {t.get('content','')}")
    return "\n".join(out)


def render(slots: SlotSet) -> str:
    """Emit in declaration order: stable prefix first, JIT after."""
    chunks = []
    for name in SLOT_ORDER:
        s = slots.get(name)
        if s is None:
            continue
        r = s.render()
        if r:
            chunks.append(r)
    return "\n\n".join(chunks)


def log_telemetry(conn: sqlite3.Connection, c: CompiledContext, turn_id: str | None = None) -> None:
    import json

    conn.execute(
        """INSERT INTO ledger_telemetry
           (ts, turn_id, budget_tokens, spent_tokens, reserve_tokens, cache_hit,
            prefix_stable, overflow, slots, ledger_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            now_iso(), turn_id, c.budget, c.spent, c.reserve, int(c.cache_hit),
            c.prefix_stable_runs, c.overflow if c.ladder.events else None,
            json.dumps(c.slots.as_dict()), c.ledger_hash,
        ),
    )
    conn.commit()
