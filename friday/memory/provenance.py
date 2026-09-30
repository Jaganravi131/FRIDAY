"""`/why` — show the literal evidence behind a personal answer.

Phase 0 exit test #3: "`/why` after any personal answer -> shows the literal source
quote + trace ID".

WHY THIS IS A FEATURE AND NOT A DEBUG COMMAND
---------------------------------------------
A personal agent that says "your rent is ₹28,000" and cannot show you where it got
that is indistinguishable from one that made it up. Provenance is the only way to tell
"FRIDAY read this in your lease file" from "FRIDAY pattern-matched something it saw
once" — and once you cannot tell those apart, you stop trusting any of its answers,
including the correct ones.

So the rule is: EVERY personal claim must be traceable to a literal quote in a file
you can open, with the id and the timestamps. Not a summary. Not a paraphrase. The
actual characters, and the actual file.

This is Law 7 (memory is a file you can read) applied to answers rather than to
storage: if you can edit the memory, you must also be able to audit the reasoning that
came out of it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Provenance:
    """One fact's full paper trail."""

    fact_id: str
    predicate: str
    value: str
    quote: str | None
    source_kind: str
    confidence: float | None
    origin_file: str | None
    origin_line: int | None
    trace_id: str | None
    #: world time
    valid_from: str | None
    valid_to: str | None
    #: belief time
    asserted_at: str | None
    retracted_at: str | None
    superseded_by: str | None
    score: float = 0.0
    recall_sources: list[str] = field(default_factory=list)
    #: what makes this one answerable: it has a quote and a file you can open
    verifiable: bool = True
    gap: str = ""

    @property
    def state(self) -> str:
        return "retracted" if self.retracted_at else "live"

    def short(self) -> str:
        """One line, for the answer footer."""
        src = Path(self.origin_file).name if self.origin_file else "no file"
        return f"{self.predicate}={self.value} [{src} · {self.source_kind}]"

    def render(self) -> str:
        """The full block. This is what `/why` prints."""
        lines = [
            f"▸ {self.predicate} = {self.value}",
            f"   quote:      \"{self.quote}\"" if self.quote
            else "   quote:      — none recorded (see gap below)",
            f"   fact id:    {self.fact_id}",
        ]
        if self.origin_file:
            where = self.origin_file
            if self.origin_line:
                where += f":{self.origin_line}"
            lines.append(f"   source:     {where}")
        else:
            lines.append("   source:     — no origin file recorded")
        if self.trace_id:
            lines.append(f"   trace:      {self.trace_id}")
        conf = f"{self.confidence:.2f}" if self.confidence is not None else "?"
        lines.append(f"   provenance: {self.source_kind} (confidence {conf})")
        lines.append(
            f"   world time: {self.valid_from or '?'} → {self.valid_to or 'now'}"
            f"   [{self.state}]"
        )
        lines.append(
            f"   believed:   {(self.asserted_at or '?')[:19]} → "
            f"{self.retracted_at[:19] if self.retracted_at else 'still believed'}"
        )
        if self.superseded_by:
            lines.append(f"   superseded: by {self.superseded_by}")
        if self.score or self.recall_sources:
            lines.append(
                f"   retrieved:  score {self.score:.3f} via "
                f"{'+'.join(self.recall_sources) or '?'}"
            )
        if self.gap:
            lines.append(f"   ⚠️  GAP:      {self.gap}")
        return "\n".join(lines)


def from_row(row: Any, *, score: float = 0.0,
             recall_sources: Iterable[str] = ()) -> Provenance:
    """Build provenance from a `facts` row (sqlite3.Row or dict)."""
    def g(k, default=None):
        try:
            v = row[k]
            return default if v is None else v
        except (IndexError, KeyError, TypeError):
            return default

    quote = g("source_quote")
    origin = g("origin_file")
    p = Provenance(
        fact_id=g("id", "?"),
        predicate=g("predicate", "?"),
        value=str(g("object", "")),
        quote=quote,
        source_kind=g("source_kind", "?"),
        confidence=g("confidence"),
        origin_file=origin,
        origin_line=g("origin_line"),
        trace_id=g("trace_id"),
        valid_from=g("valid_from"),
        valid_to=g("valid_to"),
        asserted_at=g("asserted_at"),
        retracted_at=g("retracted_at"),
        superseded_by=g("superseded_by"),
        score=score,
        recall_sources=list(recall_sources),
    )
    # ⭐ HONESTY CHECK. A fact with no quote and no file is not verifiable, and
    # saying so is the whole point of the command. Silently rendering an empty
    # "quote:" line would look like a formatting quirk rather than a missing
    # justification, and the user would trust the answer anyway.
    gaps = []
    if not quote:
        gaps.append("no source quote — cannot show you the words this came from")
    if not origin:
        gaps.append("no origin file — cannot show you where it is stored")
    p.verifiable = not gaps
    p.gap = "; ".join(gaps)
    return p


def from_candidates(cands: Iterable[dict]) -> list[Provenance]:
    """Build provenance from `SearchResult.as_dicts()` / `State.retrieved`.

    Only facts get a trail. Episodes and skills are context, not claims, so they are
    listed by reference rather than justified — and pretending otherwise would bury
    the one line that actually matters.
    """
    out: list[Provenance] = []
    for c in cands or []:
        # `SearchResult.as_dicts()` FLATTENS the facts row into the candidate dict —
        # there is no nested "row" key. Reading c["row"] would silently yield nothing
        # for every candidate, and `/why` would report "no facts behind this answer"
        # for an answer that had three.
        if c.get("kind") != "fact":
            continue
        if not (c.get("predicate") or c.get("id") or c.get("ref")):
            continue
        row = dict(c)
        row.setdefault("id", c.get("ref"))
        out.append(from_row(
            row,
            score=float(c.get("score") or 0.0),
            recall_sources=c.get("recall_sources") or [],
        ))
    return out


def render_report(provs: Iterable[Provenance], *, query: str = "",
                  answer: str = "") -> str:
    """The full `/why` output."""
    provs = list(provs)
    head = ["why FRIDAY said that"]
    if query:
        head.append(f"  question: {query!r}")
    if answer:
        head.append(f"  answer:   {answer.strip()[:200]}")
    head.append("")
    if not provs:
        head.append("⚠️  NO FACTS BEHIND THIS ANSWER.")
        head.append("    Either the question needed no memory, or FRIDAY answered from")
        head.append("    the model alone. If it made a personal claim, that is a bug —")
        head.append("    Law 4 says an empty retrieval should produce \"I don't have")
        head.append("    that\", not a confident guess.")
        return "\n".join(head)
    unverifiable = [p for p in provs if not p.verifiable]
    head.append(f"  {len(provs)} fact(s) used; "
                f"{len(provs) - len(unverifiable)} verifiable, {len(unverifiable)} with gaps")
    head.append("")
    for p in provs:
        head.append(p.render())
        head.append("")
    if unverifiable:
        head.append("⚠️  GAPS ABOVE MEAN: those claims cannot be traced to your words.")
        head.append("    Treat them as unconfirmed and correct or delete them.")
    return "\n".join(head).rstrip() + "\n"


def why_query(conn: sqlite3.Connection, query: str, *, embedder=None,
              reranker=None, k: int = 3) -> str:
    """`friday why "<query>"` — show the trail for what a question WOULD retrieve.

    Useful before asking, and the only way to inspect provenance for a question you
    have not asked yet in this process.
    """
    from ..retrieval.pipeline import search

    res = search(conn, query, k_ship=k, embedder=embedder, reranker=reranker)
    provs = from_candidates(res.as_dicts())
    head = ""
    if not res.kept:
        head = (f"retrieval returned [] for {query!r} — considered {res.considered}, "
                f"dropped below floor {res.dropped_below_floor}, stale {res.dropped_stale}.\n"
                f"Law 4: an empty list beats noise, so there is nothing to justify.\n\n")
    return head + render_report(provs, query=query)
