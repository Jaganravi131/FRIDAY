"""The agent loop — hand-rolled, synchronous, ~every branch understandable.

    compile -> route -> generate -> tool calls (through the policy engine) -> observe

Two deliberate choices worth defending:

  * **Hand-rolled, not LangGraph.** On a 2.6B local model you need to see every
    branch, and a framework's abstractions cost tokens you don't have. Law 8: one
    language for the brain, one process for the truth.
  * **The trace is written after the answer, never before.** S0 capture must not
    block the response — a 200 ms human turn-gap budget doesn't accommodate a disk
    write and a JSONL serialisation on the critical path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import MAX_AGENT_TURNS, TOOL_OUTPUT_CAP_TOKENS
from ..ledger.compiler import CompiledContext, Gather, Ledger, log_telemetry
from ..ledger.ladder import rung0_cap
from ..llm.client import Client, NoRouteError, Response, ToolCall
from ..memory.traces import Trace, TraceWriter
from ..retrieval.pipeline import search
from ..util import approx_tokens, new_id, now_iso
from .policy import check, refusal
from ..memory import watcher as memory_watcher
from .routing import classify_turn, tools_for
from .tools import ToolContext, build_registry


@dataclass
class State:
    session_id: str
    user_turn: str
    scope: str = "interactive"       # interactive|heartbeat|dreaming|eval
    goal: str | None = None          # pinned for the whole run
    compiled: CompiledContext | None = None
    scratchpad: str = ""             # rolling notes, NOT full history
    messages: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    activated_skills: set[str] = field(default_factory=set)
    tool_failures: int = 0
    turn_idx: int = 0
    max_turns: int = MAX_AGENT_TURNS
    done: bool = False
    final_text: str = ""
    #: ⭐ Trust-boundary events the user was told about this turn. Kept on State so a
    #: caller (CLI, API, test) can see that something was withheld without parsing
    #: prose out of final_text.
    security_notices: list[str] = field(default_factory=list)
    trace_id: str = ""
    char_span: tuple[int, int] = (0, 0)
    tool_log: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    #: The tool-contract decision for this turn. Recorded so "it forgot" can be
    #: distinguished from "it decided not to look" — those need different fixes.
    intent: Any = None
    #: What the memory watcher noticed before this turn (hand-edits, deletions).
    memory_sync: Any = None


class Agent:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        client: Client,
        ledger: Ledger | None = None,
        registry=None,
        embedder=None,
        reranker=None,
        traces: TraceWriter | None = None,
        confirm: Callable[[str, dict], bool] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        watcher: memory_watcher.MemoryWatcher | None = None,
    ):
        self.conn = conn
        self.client = client
        self.ledger = ledger or Ledger()
        self.registry = registry or build_registry()
        self.embedder = embedder
        self.reranker = reranker
        self.traces = traces or TraceWriter()
        #: Called when a tool needs a human "yes". Default denies — an unattended
        #: agent must never take an irreversible action by default.
        self.confirm = confirm or (lambda name, args: False)
        self.on_event = on_event or (lambda kind, payload: None)
        #: ⭐ Law 7: hand-edits to memory/*.md are picked up on the next turn, with no
        #: restart. Default is the process-wide watcher so the CLI and a REPL share one
        #: mtime cache; pass `False` to disable (evals that must not touch the tree).
        self.watcher = memory_watcher.default_watcher() if watcher is None else watcher

    # ── the loop ───────────────────────────────────────────────────────────────

    def run(self, user_turn: str, *, session_id: str = "", scope: str = "interactive",
            senses: list[str] | None = None) -> State:
        st = State(
            session_id=session_id or self._open_session(scope),
            user_turn=user_turn,
            scope=scope,
            trace_id=new_id("t"),
        )
        st.transcript = self._recent_transcript(st.session_id)
        st.transcript.append({"role": "user", "content": user_turn})

        # 0a. LAW 7 — pick up hand-edits to memory before answering, so that
        # "edit housing.md, save, ask in the same breath" works. Cost is a stat() per
        # memory file; the hash is only recomputed for files whose mtime moved.
        if self.watcher is not None and self.watcher is not False:
            try:
                st.memory_sync = self.watcher.sync(self.conn, embedder=self.embedder)
                if st.memory_sync.changed:
                    self.on_event("memory_sync", {"summary": st.memory_sync.summary()})
            except Exception as e:
                # A watcher failure must never take the turn down with it. Answering
                # from a slightly stale index beats not answering.
                self.on_event("memory_sync_error", {"error": str(e)})

        # 0b. THE TOOL CONTRACT — decide whether this turn needs memory at all, before
        # spending on retrieval. Exit test #8: a trivial question must not call
        # memory_search. Withholding the tool is structural; instructing the model is
        # not.
        st.intent = classify_turn(user_turn,
                                  recent_turns=tuple(t["content"] for t in st.transcript[-4:]))
        self.on_event("intent", {"kind": st.intent.kind,
                                 "needs_memory": st.intent.needs_memory,
                                 "reason": st.intent.reason})

        # S0: capture the user turn FIRST and keep its char span. The span is what
        # a salience label points at, and it cannot be recovered later.
        # scope is st.scope, not None: the policy engine already branched on it, so a
        # trace that drops it cannot be audited against the decision it produced.
        tr = Trace(kind="turn", role="user", content=user_turn, session_id=st.session_id,
                   turn_idx=st.turn_idx, scope=st.scope, surface="cli", id=st.trace_id)
        _, cs, ce = self.traces.append(tr)
        st.char_span = (cs, ce)

        # 1. RETRIEVE (Law 3: just-in-time, not pre-packed) — skipped entirely when the
        # turn cannot need it. "what time is it" must not cost an embedding pass.
        if st.intent.needs_memory:
            res = search(self.conn, user_turn, embedder=self.embedder,
                         reranker=self.reranker,
                         recent_turns=[t["content"] for t in st.transcript[-4:]])
            st.retrieved = res.as_dicts()
            self.on_event("retrieved", {"count": len(res), "summary": res.summary()})
        else:
            st.retrieved = []
            self.on_event("retrieved", {"count": 0,
                                        "summary": f"skipped: {st.intent.reason}"})

        while not st.done and st.turn_idx < st.max_turns:
            # 2. COMPILE — deterministic, <5ms, no model call
            g = Gather(
                query=user_turn,
                recalled=st.retrieved,
                docs=[],
                transcript=st.transcript[-24:],
                scratch=st.scratchpad,
                senses=senses or [],
                # ⭐ Act on the tool-contract classification. Skipping retrieval is
                # only half of it: a clock question still deserves an answer, and a
                # local model cannot read a wall clock. Supplying the reading is what
                # turns "memory=skipped" from a shrug into a fast, correct reply.
                now=_now_reading() if st.intent and st.intent.kind == "clock" else "",
            )
            st.compiled = self.ledger.compile(g, conn=self.conn)
            self.on_event("ledger", {"hash": st.compiled.ledger_hash,
                                     "spent": st.compiled.spent,
                                     "cache_hit": st.compiled.cache_hit})

            # 3. GENERATE
            msgs = self._messages(st)
            st.messages = msgs
            try:
                resp = self.client.chat(msgs, tools=tools_for(st.intent,
                                                                self.registry.schemas()))
            except RuntimeError as e:
                if "cannot reach" in str(e):
                    st.final_text = (
                        "I can't reach a local model right now. Start llama-server "
                        "(docs/architecture/12 §11), or run with the mock client to "
                        "exercise the memory path offline."
                    )
                    st.done = True
                    break
                raise

            # 4. TOOL CALLS — every one through the policy engine (Law 5)
            if resp.tool_calls:
                for call in resp.tool_calls:
                    out = self._do_tool(call, st, senses or [])
                    st.tool_log.append(out)
                    # Rung 0: cap BEFORE it enters history. The -38% cost lever,
                    # recall unchanged, architecture-independent.
                    text, _ = rung0_cap(_stringify(out), TOOL_OUTPUT_CAP_TOKENS)
                    st.transcript.append({"role": "tool", "content": f"[{call.name}] {text}",
                                          "pin": False})
                st.turn_idx += 1
                continue

            # 5. FINAL
            st.final_text = resp.text.strip()
            st.done = True
            st.turn_idx += 1
            self._record_usage(resp, st)

        # 5b. ⭐ Surface anything the trust boundary withheld.
        self._announce_withheld(st)

        # 6. Persist the turn + write the assistant trace AFTER answering.
        self._persist(st, senses or [])
        return st

    def _announce_withheld(self, st: State) -> None:
        """Tell the user what was withheld from the context, and why.

        Containment that only shows up in the dev-mode ledger block is containment
        nobody experiences: under `--quiet` the notice is suppressed, and the MODEL is
        not told either — the item is simply absent, which from its point of view is
        indistinguishable from a retrieval miss. So it answers the question and says
        nothing, and the user never learns that a poisoned fact is sitting in their
        memory being retrieved on every related turn.

        Appended to the answer rather than injected into the context on purpose. The
        model cannot be relied on to relay a security notice it was handed — that is
        the same "ask it nicely" failure mode doc 09 §1 warns about, applied to our own
        defence. This goes straight to the user's screen.
        """
        withheld = getattr(getattr(st, "compiled", None), "withheld", None) or []
        if not withheld:
            return
        kinds = ", ".join(str(w.get("predicate") or "?") for w in withheld)
        cats = sorted({c for w in withheld for c in (w.get("categories") or [])})
        where = ", ".join(
            str(w.get("origin_file") or w.get("predicate") or "?") for w in withheld)
        note = (
            f"\u26a0 I withheld {len(withheld)} retrieved item(s) that read like "
            f"instructions rather than memory: {kinds}\n"
            f"  signals: {', '.join(cats) or 'flagged'}\n"
            f"  source: {where}\n"
            f"  Memory is data \u2014 it cannot tell me what to do. If one of those is "
            f"legitimate, say so and I will treat your word as the authority; otherwise "
            f"consider deleting it, because it will keep being retrieved."
        )
        st.final_text = (st.final_text + "\n\n" + note).strip()
        st.security_notices.append(note)
        self.on_event("security", {"withheld": withheld})

    # ── tools ──────────────────────────────────────────────────────────────────

    def _do_tool(self, call: ToolCall, st: State, senses: list[str]) -> dict:
        # ⭐ Only USER-role turns count as authority. The current turn is already in
        # st.transcript (appended before tools run), so this is "everything the human
        # has actually said in this session" and nothing else — no tool output, no
        # assistant text, no retrieved content. That exclusion is the whole control:
        # a memory write claiming the user's authority has to be justified by words
        # that appear here, and an injection arriving through memory cannot add to it.
        authority = [t.get("content") or "" for t in st.transcript
                     if t.get("role") == "user"]
        decision = check(self.conn, call.name, call.arguments, scope=st.scope,
                         authority_turns=authority)
        self.on_event("policy", {"tool": call.name, "verdict": decision.verdict,
                                 "sense": decision.sense})

        if decision.verdict == "denied":
            st.tool_failures += 1
            return refusal(decision, call.name)

        if decision.verdict == "confirmation_required":
            if not self.confirm(call.name, call.arguments):
                st.tool_failures += 1
                return {"cancelled": True, "reason": "user declined",
                        "tool": call.name,
                        "note": "Don't retry. Ask what they'd prefer instead."}

        ctx = ToolContext(
            conn=self.conn, scope=st.scope, session_id=st.session_id,
            trace_id=st.trace_id, char_start=st.char_span[0], char_end=st.char_span[1],
            embedder=self.embedder, reranker=self.reranker,
            recent_turns=[t["content"] for t in st.transcript[-4:]],
            traces=self.traces,
        )
        out = self.registry.invoke(call.name, call.arguments, ctx)
        if isinstance(out, dict) and out.get("error"):
            st.tool_failures += 1
        return out if isinstance(out, dict) else {"result": out}

    # ── prompt assembly ────────────────────────────────────────────────────────

    def _messages(self, st: State) -> list[dict]:
        assert st.compiled is not None
        msgs = [{"role": "system", "content": st.compiled.text}]
        # Short window only. The scratchpad, not the full history, is what carries
        # older context — that's the Ledger's job, not the message list's.
        for t in st.transcript[-10:]:
            role = t["role"] if t["role"] in ("user", "assistant") else "user"
            content = t["content"]
            if t["role"] == "tool":
                # tool results go back as user-role text: portable across servers
                # that don't implement the tool message role
                role, content = "user", f"[tool result] {content}"
            msgs.append({"role": role, "content": content})
        return msgs

    # ── persistence ────────────────────────────────────────────────────────────

    def _persist(self, st: State, senses: list[str]) -> None:
        c = st.compiled
        tid = new_id("u")
        self.conn.execute(
            """INSERT OR REPLACE INTO turns(id, session_id, idx, ts, role, surface, mode,
                                            content, tier, tokens_in, tokens_out, ttft_ms,
                                            total_ms, cache_hit, ledger_hash, trace_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, st.session_id, st.turn_idx, now_iso(), "assistant", "cli", "text",
             st.final_text, "L1", approx_tokens(st.user_turn), approx_tokens(st.final_text),
             None, None, int(c.cache_hit) if c else None,
             c.ledger_hash if c else None, st.trace_id),
        )
        self.conn.execute(
            "UPDATE sessions SET turn_count=turn_count+1 WHERE id=?", (st.session_id,)
        )
        if c is not None:
            log_telemetry(self.conn, c, turn_id=tid)

        tr = Trace(
            kind="turn", role="assistant", content=st.final_text, session_id=st.session_id,
            turn_idx=st.turn_idx, surface="cli", mode="text", tier="L1",
            tool_calls=[{"name": t.get("tool") or "", "ok": not t.get("denied")}
                        for t in st.tool_log],
            tokens_in=approx_tokens(st.user_turn), tokens_out=approx_tokens(st.final_text),
            cache_hit=bool(c.cache_hit) if c else None,
            ledger_hash=c.ledger_hash if c else None,
            tool_failures=st.tool_failures,
            scope=st.scope,
            senses=list(senses or []),
        )
        self.traces.append(tr)
        self.conn.commit()

    def _record_usage(self, resp: Response, st: State) -> None:
        if resp.prompt_tokens and st.compiled is not None:
            self.on_event("usage", {"prompt_tokens": resp.prompt_tokens,
                                    "ledger_estimate": st.compiled.spent,
                                    "note": "calibrate util.approx_tokens against this"})

    def _recent_transcript(self, session_id: str, limit: int = 24) -> list[dict]:
        rows = self.conn.execute(
            """SELECT role, content FROM turns WHERE session_id=?
               ORDER BY idx DESC LIMIT ?""", (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def _open_session(self, scope: str) -> str:
        sid = new_id("s")
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions(id, started_at, scope, surfaces) VALUES (?,?,?,?)",
            (sid, now_iso(), scope, '["cli"]'),
        )
        self.conn.commit()
        return sid


def _now_reading() -> str:
    """The current date and time, in the user's own locale and zone.

    Spelled out rather than left as an ISO stamp because the model has to say it back
    naturally: "Tuesday, 30 September, 14:05" is an answer, "2026-09-30T14:05:11+05:30"
    is a log line.
    """
    from datetime import datetime

    d = datetime.now().astimezone()
    zone = d.tzname() or ""
    return (f"{d:%A, %d %B %Y}, {d:%H:%M} {zone}".replace("  ", " ").strip())


def _stringify(obj: Any) -> str:
    import json

    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def escalate_or_ask(*, tier: str, reason: str) -> str:
    """₹0 has no L2..L5. When the ladder wants to escalate and there is nowhere to
    go, the correct degradation is *surface the uncertainty*, never *guess
    confidently* (docs/architecture/12 §8.1)."""
    from ..config import CLOUD_TIERS_ENABLED

    if CLOUD_TIERS_ENABLED:
        return f"escalating to {tier}: {reason}"
    raise NoRouteError(
        f"Local tier could not handle this ({reason}) and cloud tiers are disabled "
        f"at a ₹0 budget. Tell the user you're not sure rather than guessing."
    )
