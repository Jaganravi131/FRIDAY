"""The three Phase-0 tools. Resist adding more.

Progressive disclosure exists precisely because tool definitions are a fixed
per-turn tax, and on a 2.6B model with an 8K budget you cannot afford six. Three
tools is a deliberate ceiling, not a starting point (docs/architecture/10 Day 5).

⭐ `memory_search` is load-bearing in a way the others aren't. Under a linear-attention
backbone it is not a convenience — ICLR 2025 proves that giving an RNN a function-call
primitive for in-context retrieval lifts its representation power to *all
polynomial-time solvable problems*, and that in-context RAG brings every model to
near-perfect accuracy. A fixed-size recurrent state loses associative recall first,
so this tool is the only exact memory FRIDAY has (Law 2b).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .. import paths
from ..config import RETRIEVAL_K_SHIP
from ..memory.facts import Fact, assert_fact, domain_for
from ..memory.traces import TraceWriter
from ..retrieval.pipeline import search
from ..util import approx_tokens, new_id, now_iso


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def invoke(self, name: str, args: dict, ctx: "ToolContext") -> Any:
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"unknown tool '{name}'", "available": self.names()}
        try:
            return spec.handler(ctx=ctx, **(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}",
                    "expected": list(spec.parameters.get("properties", {}))}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


@dataclass
class ToolContext:
    """Everything a tool may touch. Passing this instead of globals is what makes
    the policy engine able to intercept (Law 5): enforcement happens on the way in,
    in the data layer, not by asking the model nicely in the prompt."""

    conn: sqlite3.Connection
    scope: str = "interactive"          # interactive|heartbeat|dreaming|eval
    session_id: str = ""
    trace_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    embedder: Any = None
    reranker: Any = None
    recent_turns: list[str] = None      # type: ignore[assignment]
    traces: TraceWriter | None = None

    def __post_init__(self):
        if self.recent_turns is None:
            self.recent_turns = []
        if self.traces is None:
            self.traces = TraceWriter()


# ── memory_search ──────────────────────────────────────────────────────────────

SEARCH_DOC = """Search FRIDAY's long-term memory about the user.

Use ONLY when the turn references something not present in the current context: a past conversation, a personal fact, a preference, a prior decision, or an entity mentioned without introduction.

Do NOT use for general knowledge, the current task's files, or anything already visible in the core_memory block.

Pass as_of="2026-03-01" ONLY for explicit time-travel questions ("where did I live last year?", "what did I used to think?"). Returns at most 3 facts with provenance. An empty list is valid and common; do not retry with rephrased queries more than once."""


def memory_search(ctx: ToolContext, query: str, as_of: str | None = None) -> dict:
    res = search(
        ctx.conn, query, as_of=as_of, k_ship=RETRIEVAL_K_SHIP,
        embedder=ctx.embedder, reranker=ctx.reranker,
        recent_turns=ctx.recent_turns,
    )
    return {
        "results": res.as_dicts(),
        "count": len(res),
        "floor": res.floor,
        "reranker": res.reranker,
        "considered": res.considered,
        # ⭐ An empty result is a first-class answer, and the model must be told so
        # explicitly. Law 4: [] beats noise. Without this line a small model reads
        # "count: 0" as a failure and starts inventing.
        "note": (
            "Nothing in memory matched above the score floor. Say you don't know — "
            "do not invent a fact and do not re-ask more than once."
            if not res else None
        ),
        "summary": res.summary(),
    }


# ── memory_write ───────────────────────────────────────────────────────────────

WRITE_DOC = """Persist a durable fact about the user, THE MOMENT it is established.

Use when the user states a preference, a fact about themselves, a decision they want enforced later, or corrects something you believed.

Do NOT use for: transient task state, things true only this session, general knowledge, or anything you are inferring without evidence.

Writes to memory/facts/<domain>.md with provenance linking this turn. Triggers bi-temporal reconciliation: a conflicting current fact is RETRACTED (never deleted) and the response tells you which one won, so you can tell the user."""


def memory_write(
    ctx: ToolContext,
    predicate: str,
    object: str,
    subject: str | None = None,
    source_quote: str | None = None,
    valid_from: str | None = None,
    domain: str | None = None,
    source_kind: str = "stated",
) -> dict:
    if not predicate or not object:
        return {"error": "predicate and object are both required"}
    fact = Fact(
        subject=subject or "user",
        predicate=predicate.strip(),
        object=object.strip(),
        domain=domain or domain_for(predicate, object),
        source_kind=source_kind,
        source_quote=source_quote,
        valid_from=valid_from,
        source_refs=[ctx.trace_id] if ctx.trace_id else [],
        trace_id=ctx.trace_id,
        char_start=ctx.char_start,
        char_end=ctx.char_end,
    )
    report = assert_fact(ctx.conn, fact)
    # A write is a salient event by definition: it produced a fact. The supervision
    # span was recorded inside assert_fact; surface it so the audit trail shows the
    # RSC label being minted at the same moment as the fact.
    report["supervision"] = {
        "span_id": report.get("span_id"),
        "why": "positive salience label for the RSC write gate (w_t)",
    }
    if report.get("retracted"):
        report["supervision"]["pair"] = (
            "retraction pair recorded: free supervision for the RSC erase address (e_t). "
            "key_old != key_new is the case GDN-2 cannot reach and EDA can."
        )
    return report


# ── note ───────────────────────────────────────────────────────────────────────

NOTE_DOC = """Append a line to today's ephemeral log (memory/daily/YYYY-MM-DD.md).

For observations not durable enough to be facts: a mood, a passing mention, a task-in-progress, something you noticed on screen. Cheap; use freely. These are indexed and searchable but are never auto-injected into context, so they cost nothing per turn."""


def note(ctx: ToolContext, text: str) -> dict:
    if not text or not text.strip():
        return {"error": "text is required"}
    day = now_iso()[:10]
    p = paths.DAILY / f"{day}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = now_iso()[11:16]
    line = f"- {stamp} {text.strip()}\n"
    with p.open("a", encoding="utf-8") as fh:
        if p.stat().st_size == 0:
            fh.write(f"# {day}\n\n")
        fh.write(line)
    from ..store import db

    db.fts_upsert(ctx.conn, f"daily:{day}:{new_id('n')}", "daily", text.strip(), subject=day)
    ctx.conn.commit()
    return {"written": str(p.relative_to(paths.ROOT)), "tokens": approx_tokens(line)}


# ── read_artifact (added by the ladder, not by ambition) ───────────────────────

READ_ARTIFACT_DOC = """Pull back a block the Attention Ledger offloaded to disk (Rung 3).

The context contains pointers like `[… offloaded to artifacts/offloaded/o_xxx.md — use read_artifact('o_xxx.md') …]`. Call this ONLY when the pointer turns out to matter for the current turn. That is Law 3: keep pointers in context, pull the payload on demand."""


def read_artifact(ctx: ToolContext, name: str, max_tokens: int = 2000) -> dict:
    """Pull an offloaded block back into context.

    `name` is model-supplied, so it is treated as hostile input in the strict sense —
    not "probably fine, it came from our own model". A model that has read a
    prompt-injected document is a model that can be told what filename to ask for.

    Three things are checked, and each was a real hole:
      * `Path(name).name` strips traversal — `../../.ssh/id_rsa` becomes `id_rsa`.
      * Empty and dot names resolve to the offload DIRECTORY, which passes an
        `exists()` check and then raises IsADirectoryError out of the tool call. A
        crash in a tool is a crashed turn.
      * A symlink planted inside artifacts/offloaded/ defeats `.name` entirely,
        because the name is safe and the target is not. `resolve()` then re-check
        containment is the only thing that catches it. In Phase 0 only FRIDAY writes
        there, but Phase 3 mounts MCP skills that can write files, and this function
        will still be the one reading them.

    The payload is neutralized before it goes back: offloaded content is whatever was
    in the slot, which for `docs` is by definition imported.
    """
    safe = Path(name or "").name                      # never trust a path from the model
    if not safe or safe in (".", ".."):
        return {"error": "artifact name must be a plain filename"}
    root = (paths.ARTIFACTS / "offloaded").resolve()
    p = root / safe
    try:
        real = p.resolve()
    except OSError as e:
        return {"error": f"unreadable artifact name: {e}"}
    # Containment AFTER resolution: catches both traversal that survived .name and a
    # symlink whose target lives outside the offload directory.
    if real != p or not str(real).startswith(str(root) + "/") and real != root:
        return {"error": "artifact path escapes the offload directory"}
    if not real.is_file():                            # not exists(): a directory is not a file
        return {"error": f"no offloaded artifact named '{safe}'"}

    from ..security import Verdict, detect, neutralize
    from ..util import cap_tokens

    try:
        text = real.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": f"could not read artifact: {e}"}

    rep = detect(text)
    if rep.verdict is Verdict.HOSTILE:
        # Withholding here is the containment layer doing its job: the model asked for
        # a payload that turned out to be an instruction block. It gets told, rather
        # than being handed the block and trusted to notice.
        try:
            from .policy import log as _log

            _log(ctx.conn, actor="agent", action="tool.read_artifact", target=safe,
                 sense_id=None, decision="denied",
                 detail=f"withheld: {rep.summary()}")
        except Exception:
            pass    # auditing must never be the reason a tool call fails
        return {"error": (f"withheld '{safe}': it reads like instructions rather than "
                          f"data ({rep.summary()}). Not injected into context."),
                "withheld": rep.as_dict()}

    return {"name": safe, "text": cap_tokens(neutralize(text), max_tokens),
            "tokens": approx_tokens(text),
            **({"suspicious": rep.as_dict()} if rep.verdict is Verdict.SUSPICIOUS else {})}


# ── registry ───────────────────────────────────────────────────────────────────

def build_registry(*, phase: int = 0) -> ToolRegistry:
    """Phase 0 = three tools. `read_artifact` is included because the ladder itself
    creates the pointers it resolves — it is part of compaction, not a new capability.
    """
    r = ToolRegistry()
    r.register(ToolSpec("memory_search", SEARCH_DOC, {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to look for"},
            "as_of": {"type": "string", "description": "YYYY-MM-DD, for time-travel queries only"},
        },
        "required": ["query"],
    }, memory_search))
    r.register(ToolSpec("memory_write", WRITE_DOC, {
        "type": "object",
        "properties": {
            "predicate": {"type": "string", "description": "snake_case relation, e.g. lease_amount_monthly"},
            "object": {"type": "string", "description": "the value"},
            "domain": {"type": "string", "description": "facts file; inferred if omitted"},
            "source_quote": {"type": "string", "description": "the user's own words"},
            "valid_from": {"type": "string", "description": "YYYY-MM-DD, if not now"},
        },
        "required": ["predicate", "object"],
    }, memory_write))
    r.register(ToolSpec("note", NOTE_DOC, {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }, note))
    r.register(ToolSpec("read_artifact", READ_ARTIFACT_DOC, {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "filename from an offload pointer"},
        },
        "required": ["name"],
    }, read_artifact))
    return r
