"""Ledger slots: the budgeted, deterministic context schedule (Innovation #4).

The split that matters is STABLE PREFIX vs JIT ZONE:

    STABLE (byte-identical across turns -> the KV prefix cache hits)
        identity   SOUL.md + AGENTS.md
        senses     which senses are enabled right now

    JIT (changes every turn, always AFTER the stable prefix)
        user       USER.md
        core_mem   MEMORY.md, the ~20 facts that matter most
        recalled   retrieval results for THIS query (Law 3: pull, don't pre-pack)
        docs       offloaded tool output pulled by pointer (Rung 4)
        transcript recent turns — with an RSC this becomes a fixed-size state
        scratch    rolling notes, NOT full history

Ordering is not cosmetic. Anything that changes must come after everything that
doesn't, or you invalidate the cache on every turn and Law 2's whole point is lost.
Under a recurrent-state backbone the constraint relaxes (there is no prefix to
cache-break) but the ordering still helps: it puts stable identity where the model
can rely on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Declaration order == emission order == cache friendliness. Do not re-sort.
SLOT_ORDER: tuple[str, ...] = (
    "identity",   # stable
    "senses",     # stable
    "user",       # jit
    "core_mem",   # jit
    "recalled",   # jit
    "docs",       # jit
    "transcript", # jit
    "scratch",    # jit
)

STABLE_SLOTS = frozenset({"identity", "senses"})


@dataclass
class Slot:
    name: str
    budget: int
    content: str = ""
    tokens: int = 0
    truncated: bool = False
    dropped_items: int = 0
    stable: bool = False
    #: ⭐ Set when a REQUIRED slot could not fit its content and was shipped whole
    #: anyway. docs/architecture/04 gives `identity` the overflow policy "reject" —
    #: it is `required=True`, so it is not rationed like tool output. Silent
    #: truncation of the stable prefix is the worse failure: it quietly edits
    #: FRIDAY's personality AND leaves an elision marker in the cached prefix that
    #: points at an artifact which was never written, so `read_artifact` cannot
    #: recover it. Better to ship the whole thing and say so loudly.
    over_budget: bool = False

    @property
    def utilisation(self) -> float:
        return self.tokens / self.budget if self.budget else 0.0

    def render(self) -> str:
        if not self.content.strip():
            return ""
        return f"<{self.name}>\n{self.content.strip()}\n</{self.name}>"


@dataclass
class SlotSet:
    slots: dict[str, Slot] = field(default_factory=dict)

    @classmethod
    def from_budgets(cls, budgets: dict[str, int]) -> "SlotSet":
        return cls(slots={
            name: Slot(name=name, budget=budgets.get(name, 0), stable=name in STABLE_SLOTS)
            for name in SLOT_ORDER
        })

    def __getitem__(self, name: str) -> Slot:
        return self.slots[name]

    def get(self, name: str) -> Slot | None:
        return self.slots.get(name)

    def items(self):
        for name in SLOT_ORDER:
            if name in self.slots:
                yield name, self.slots[name]

    def as_dict(self) -> dict[str, Any]:
        return {n: [s.tokens, s.budget] for n, s in self.items()}
