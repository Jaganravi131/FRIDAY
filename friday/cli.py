"""The CLI you'll actually live in. docs/architecture/10 Day 6.

Subcommands:

    build      Markdown truth -> SQLite indices (idempotent, incremental)
    rebuild    rm the artifact, rebuild from Markdown — verifies Law 2
    seed       write starter soul/ + memory/facts/ files
    ask        one question, print the Ledger block, get an answer
    chat       a REPL session
    write      record a fact from the shell
    search     query memory directly, showing scores and the floor
    why        ⭐ the literal source quotes + trace ids behind an answer
    history    the bi-temporal trail for one predicate
    status     RAM/DB/supervision readiness at a glance
    ablation   the RSC ladder
    needle     the decisive recall benchmark
    audit      who did what, allowed or denied
    doctor     ⭐ will FRIDAY run on THIS machine? Every way delivery fails is an
               environment difference; each is checkable in milliseconds, and a printed
               fix beats a README describing the author's computer.
    serve      ⭐ the HTTP gateway — reach FRIDAY from your phone. Authenticated,
               loopback by default, remote turns audited as `remote`.
    senses     ⭐ tiered consent — see, grant and revoke what FRIDAY may perceive
    bench      ⭐ the retrieval regression gate. Law 10 says every fine-tune must
               re-run the needle test; this makes "must" a nonzero exit code.

`rich` is used if installed and ignored if not. The Ledger printout is on by
default in ask/chat because looking at it for a week teaches more about context
engineering than any article.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from . import paths
from .util import now_iso
from .config import LEDGER_BUDGET_TOKENS, RETRIEVAL_K_SHIP, canonical_predicate
from .ledger.compiler import Gather, Ledger
from .llm.client import get_client
from .memory import compiler as mcompiler
from .memory import provenance, watcher as memory_watcher
from .memory.facts import Fact, assert_fact, beliefs_at, facts_at, history_of
from .memory.supervision import export_training_set, ready_for_phase_3_5, supervision_report
from .memory.traces import TraceWriter, behavioural_signals
from .retrieval.embedders import get_embedder
from .retrieval.pipeline import search
from .retrieval.rerankers import get_reranker
from .store import db as store_db
from .agent.loop import Agent
from .agent.tools import build_registry


# ── helpers ────────────────────────────────────────────────────────────────────

def _conn(args) -> sqlite3.Connection:
    p = Path(args.db) if getattr(args, "db", None) else paths.DB_PATH
    return store_db.connect(p)


def _say(msg: str = "") -> None:
    print(msg)


def _rich_table(rows: list[list[str]], headers: list[str]) -> str:
    try:
        from rich.console import Console
        from rich.table import Table

        t = Table(*headers, show_lines=False)
        for r in rows:
            t.add_row(*[str(c) for c in r])
        import io

        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=120).print(t)
        return buf.getvalue()
    except Exception:
        w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
             for i, h in enumerate(headers)]
        line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers))
        out = [line, "  ".join("-" * x for x in w)]
        for r in rows:
            out.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
        return "\n".join(out)


# ── build / rebuild ────────────────────────────────────────────────────────────

def cmd_build(args) -> int:
    paths.ensure_layout()
    conn = _conn(args)
    emb = None if args.no_embed else get_embedder()
    st = mcompiler.compile_all(conn, embedder=emb)
    _say(st.summary())
    if st.errors:
        for e in st.errors[:10]:
            _say(f"  ! {e}")
    _say(f"  embedder={getattr(emb, 'name', 'none')}  vec_extension={store_db.HAS_VEC_EXT}")
    _say("  " + supervision_report(conn))
    return 0


def cmd_rebuild(args) -> int:
    paths.ensure_layout()
    emb = None if args.no_embed else get_embedder()
    st = mcompiler.rebuild(paths.DB_PATH, embedder=emb)
    _say("deleted artifacts/friday.db and rebuilt from Markdown:")
    _say("  " + st.summary())
    _say("\nIf `friday ask \"what's my rent?\"` still works now, your entire memory is")
    _say("a git repo and `git checkout` is an undo button for your agent's beliefs.")
    _say("That is Innovation #2, and it is worth verifying today rather than in month 3.")
    return 0


# ── seed ───────────────────────────────────────────────────────────────────────

def cmd_seed(args) -> int:
    """Write starter soul/ and memory/facts/ files. Never overwrites."""
    from . import seed as seed_mod

    written, skipped = seed_mod.write_all(force=args.force)
    for p in written:
        _say(f"  + {p}")
    for p in skipped:
        _say(f"  = {p} (exists)")
    _say(f"\nseeded {len(written)} files. Now edit soul/USER.md — it's yours, not mine.")
    return 0


# ── ask / chat ─────────────────────────────────────────────────────────────────

def _agent(conn, args) -> Agent:
    client = get_client(prefer="mock" if args.mock else "auto",
                        base_url=args.url, model=args.model)
    ledger = Ledger(budget=args.ctx, hybrid=not args.transformer_backbone)
    registry = build_registry()

    def confirm(name: str, payload: dict) -> bool:
        _say(f"\n⚠️  FRIDAY wants to run `{name}` with:")
        _say("   " + json.dumps(payload, ensure_ascii=False)[:400])
        ans = input("   allow? [y/N] ").strip().lower()
        return ans in ("y", "yes")

    def on_event(kind: str, payload: dict) -> None:
        if args.verbose:
            _say(f"  · {kind}: {json.dumps(payload, ensure_ascii=False)[:200]}")

    return Agent(conn, client=client, ledger=ledger, registry=registry,
                 embedder=get_embedder(), reranker=get_reranker(),
                 traces=TraceWriter(), confirm=confirm, on_event=on_event)


def cmd_ask(args) -> int:
    paths.ensure_layout()
    conn = _conn(args)
    agent = _agent(conn, args)
    st = agent.run(args.question, senses=args.senses.split(",") if args.senses else [])

    if st.memory_sync is not None and st.memory_sync.changed:
        # ⭐ Law 7, made visible. You edited a file and FRIDAY noticed — saying so is
        # what turns "did it pick that up?" from a question into a fact.
        _say(f"⟳ memory: {st.memory_sync.summary()}")
    if not args.quiet and st.compiled is not None:
        _say(st.compiled.printout(turn=st.turn_idx))
        _say("")
    _say(st.final_text)
    if st.intent is not None and not args.quiet:
        _say(f"\n[turn: {st.intent.kind} · memory="
             f"{'used' if st.intent.needs_memory else 'skipped'} · {st.intent.reason}]")
    if st.retrieved and not args.quiet:
        _say("[type `friday why` or \\why for the source quotes behind this]")
    _announce_egress(st)
    if st.tool_log and args.verbose:
        _say("\ntool calls:")
        for t in st.tool_log:
            _say("  " + json.dumps(t, ensure_ascii=False)[:300])
    if st.compiled is not None and st.compiled.summarised:
        _say("\n⚠️  LAW 2 ALARM: the ladder reached rung 5 (summarise). Recall risk.")
    return 0


def cmd_chat(args) -> int:
    paths.ensure_layout()
    conn = _conn(args)
    agent = _agent(conn, args)
    senses = args.senses.split(",") if args.senses else []
    _say(f"FRIDAY — {agent.client.name} · ctx={args.ctx} · "
         f"{'hybrid' if not args.transformer_backbone else 'transformer'} backbone")
    _say("Type your message. Ctrl-D or 'exit' to quit.")
    _say("  \\why sources behind the last answer · \\ledger toggle printout · \\status\n")
    show_ledger = not args.quiet
    session = ""
    last_query = last_answer = None
    last_retrieved = None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _say("\nbye.")
            return 0
        if not line:
            continue
        if line in ("exit", "quit"):
            return 0
        if line == "\\ledger":
            show_ledger = not show_ledger
            _say(f"ledger printout: {'on' if show_ledger else 'off'}")
            continue
        if line == "\\status":
            _status(conn)
            continue
        if line in ("\\why", "/why"):
            # Exit test #3: the literal source quote + trace id behind the last answer.
            if last_retrieved is None:
                _say("nothing to explain yet — ask me something first.")
                continue
            _say(provenance.render_report(
                provenance.from_candidates(last_retrieved),
                query=last_query or "", answer=last_answer or ""))
            continue
        st = agent.run(line, session_id=session, senses=senses)
        session = st.session_id
        last_query, last_answer, last_retrieved = line, st.final_text, st.retrieved
        if st.memory_sync is not None and st.memory_sync.changed:
            _say(f"⟳ memory: {st.memory_sync.summary()}")
        if show_ledger and st.compiled is not None:
            _say(st.compiled.printout(turn=st.turn_idx))
        _announce_egress(st)
        _say(f"\nfriday> {st.final_text}")
        _say("[\\why for sources · \\ledger · \\status · exit]\n")


# ── memory commands ────────────────────────────────────────────────────────────

def cmd_write(args) -> int:
    paths.ensure_layout()
    conn = _conn(args)
    w = TraceWriter()
    tid, _, cs, ce = w.append_content(
        f"{args.predicate}={args.value}", kind="cli_write"
    )
    fact = Fact(subject="user", predicate=args.predicate, object=args.value,
                source_kind="user_edit" if args.user_edit else "stated",
                valid_from=args.valid_from, source_quote=args.quote,
                trace_id=tid, char_start=cs, char_end=ce)
    # `Fact.__post_init__` canonicalised the predicate, so what got stored may not be
    # what was typed. SAY SO. Silently renaming is how a user ends up unable to find
    # the fact they just wrote, and it would look like FRIDAY ignored them.
    if fact.predicate != args.predicate:
        _say(f"note: '{args.predicate}' is stored as '{fact.predicate}' "
             f"— one predicate per concept, so a correction supersedes instead of "
             f"creating a second live fact.")
    report = assert_fact(conn, fact)
    _say(report["message"])

    # ⚠️ `written` is None when the assertion was REJECTED — Law 7 protects a
    # hand-edited value from being silently overwritten, and the confidence
    # hierarchy protects a user_edit from a lower-ranked source. Dereferencing it
    # unconditionally raised TypeError, which printed a traceback BEFORE the
    # explanation and turned "FRIDAY declined, here's why" into "FRIDAY crashed".
    # A refusal is a normal, correct outcome and must read like one.
    written = report.get("written")
    if written and written.get("file"):
        # `file` is None on the `nochange` path: the value was reinforced rather than
        # rewritten, so there is no file to report. Printing "-> None" would look like
        # a bug to the person reading it.
        _say(f"  -> {written['file']}")
    for r in report.get("retracted") or []:
        _say(f"  retracted: {r['predicate']}={r['object']} (id {r['id']})")
    sup = report.get("supervision") or {}
    if sup.get("span_id"):
        _say(f"  ⭐ salience span {sup['span_id']} -> supervises the RSC write gate w_t")
    if sup.get("pair"):
        _say(f"  ⭐ retraction pair recorded -> supervises the RSC erase address e_t")

    # Non-zero on a refusal, so a script can tell "recorded" from "declined". A
    # `nochange` reinforcement is a success — the value was already right.
    return 0 if report["action"] != "rejected" else 1


def cmd_search(args) -> int:
    conn = _conn(args)
    res = search(conn, args.query, as_of=args.as_of, k_ship=args.k,
                 embedder=get_embedder(), reranker=get_reranker())
    _say(f"query={args.query!r}  as_of={args.as_of or 'now'}")
    _say(f"considered={res.considered}  floor={res.floor:.2f}  reranker={res.reranker}  "
         f"below_floor={res.dropped_below_floor}  stale={res.dropped_stale}")
    if not res:
        _say("\n[] — nothing above the floor. Law 4: an empty list beats noise.")
        return 0
    rows = []
    for c in res.kept:
        if c.is_fact and c.row is not None:
            r = c.row
            rows.append([round(c.score, 3), r["predicate"], r["object"],
                         r["valid_from"] or "-", "RETRACTED" if r["retracted_at"] else "live",
                         "+".join(c.recall_sources)])
        else:
            rows.append([round(c.score, 3), c.kind, c.body[:48], "-", "-",
                         "+".join(c.recall_sources)])
    _say("")
    _say(_rich_table(rows, ["score", "predicate/kind", "value", "valid_from", "state", "via"]))
    return 0


def cmd_why(args) -> int:
    """Exit test #3: show the literal source quote + trace id behind an answer.

    With a query, it shows what that question WOULD retrieve and why. With `--last`
    it explains the most recent turn in this DB. Without either, it explains the
    facts currently in context (the auto-injected core memory), which is the honest
    answer to "why do you believe what you believe right now?".
    """
    paths.ensure_layout()
    conn = _conn(args)
    if args.query:
        _say(provenance.why_query(conn, args.query, embedder=get_embedder(),
                                  reranker=get_reranker(), k=args.k))
        return 0

    if args.last:
        row = conn.execute(
            """SELECT t.content AS q, s.content AS a, s.session_id
               FROM turns t JOIN turns s
                 ON s.session_id = t.session_id AND s.turn_idx = t.turn_idx
               WHERE t.role='user' AND s.role='assistant'
               ORDER BY t.id DESC LIMIT 1"""
        ).fetchone()
        if row is None:
            _say("no recorded turn yet — ask something first, then run `friday why --last`.")
            return 0
        _say(f"last turn: {row['q']!r}")
        _say(f"answered:  {str(row['a'])[:200]}")
        _say("")
        _say(provenance.why_query(conn, row["q"], embedder=get_embedder(),
                                  reranker=get_reranker(), k=args.k))
        return 0

    # No query: explain what is auto-injected right now.
    rows = conn.execute(
        """SELECT * FROM facts WHERE retracted_at IS NULL
           ORDER BY confidence DESC, access_count DESC LIMIT ?""", (args.k,)
    ).fetchall()
    provs = [provenance.from_row(r) for r in rows]
    _say(provenance.render_report(provs, query="(core memory currently in context)"))
    return 0


def cmd_history(args) -> int:
    conn = _conn(args)
    rows = history_of(conn, args.predicate)
    if not rows:
        _say(f"no history for predicate '{args.predicate}'")
        return 0
    table = []
    for r in rows:
        table.append([
            r["id"][:14], r["object"][:40],
            (r["valid_from"] or "?") + " → " + (r["valid_to"] or "now"),
            (r["asserted_at"] or "?")[:10] + (" → " + r["retracted_at"][:10]
                                              if r["retracted_at"] else " → believed"),
            r["source_kind"],
        ])
    canon = canonical_predicate(args.predicate)
    shown = canon if canon == args.predicate else f"{args.predicate}' (stored as '{canon}"
    _say(f"history of '{shown}' — WORLD time and BELIEF time, side by side")
    _say("this is what makes 'what did I used to think?' answerable (Innovation #3)\n")
    _say(_rich_table(table, ["id", "value", "valid (world)", "asserted→retracted (belief)", "src"]))
    return 0


def cmd_timecheck(args) -> int:
    conn = _conn(args)
    when = args.date
    _say(f"what was TRUE on {when}:")
    for r in facts_at(conn, when):
        _say(f"  - {r['predicate']}: {r['object']}")
    _say(f"\nwhat FRIDAY BELIEVED on {when}:")
    for r in beliefs_at(conn, when):
        _say(f"  - {r['predicate']}: {r['object']}")
    return 0


# ── status / audit ─────────────────────────────────────────────────────────────

def _status(conn: sqlite3.Connection) -> None:
    def n(q: str, *a) -> int:
        return int(conn.execute(q, a).fetchone()[0])

    rows = [
        ["facts (live)", n("SELECT COUNT(*) FROM facts WHERE retracted_at IS NULL")],
        ["facts (retracted)", n("SELECT COUNT(*) FROM facts WHERE retracted_at IS NOT NULL")],
        ["fts rows", n("SELECT COUNT(*) FROM memory_fts")],
        ["vec rows", n("SELECT COUNT(*) FROM memory_vec")],
        ["turns", n("SELECT COUNT(*) FROM turns")],
        ["audit entries", n("SELECT COUNT(*) FROM audit")],
        ["ledger budget", LEDGER_BUDGET_TOKENS],
        ["vec extension", store_db.HAS_VEC_EXT],
    ]
    _say(_rich_table([[str(a), str(b)] for a, b in rows], ["store", "value"]))
    _say("")
    _say(supervision_report(conn))
    ok, msg = ready_for_phase_3_5(conn)
    _say(f"Phase 3.5 ready: {'YES' if ok else 'NO'} — {msg}")
    sig = behavioural_signals(conn)
    _say(f"behavioural signals (₹0 GEPA feedback): {json.dumps(sig)}")


def cmd_status(args) -> int:
    paths.ensure_layout()
    conn = _conn(args)
    _status(conn)
    return 0


def cmd_audit(args) -> int:
    conn = _conn(args)
    q = "SELECT ts, actor, action, target, sense_id, decision FROM audit"
    a: list = []
    if args.today:
        # Exit test #9: `friday audit --today` — every tool call, decision and denial
        # for the day, including your own hand-edits. Scoping by day is what makes the
        # trail readable; an unbounded dump is how an audit log stops being read.
        day = args.day or now_iso()[:10]
        q += " WHERE ts LIKE ?"
        a.append(day + "%")
    q += " ORDER BY id DESC LIMIT ?"
    a.append(args.limit)
    rows = conn.execute(q, a).fetchall()
    if not rows:
        scope = f" for {args.day or now_iso()[:10]}" if args.today else ""
        _say(f"audit is empty{scope} — nothing has been enforced yet.")
        return 0
    _say(_rich_table([[r["ts"][11:19], r["actor"], r["action"], (r["target"] or "")[:36],
                       r["sense_id"] or "-", r["decision"]] for r in rows],
                     ["time", "actor", "action", "target", "sense", "decision"]))
    return 0


def cmd_export(args) -> int:
    conn = _conn(args)
    out = export_training_set(conn, args.out)
    _say(f"wrote {out['rows']} rows to {out['path']}")
    _say(f"  spans: {out['spans']}")
    _say(f"  pairs: {out['pairs']}")
    _say("\nThis is the Kaggle notebook's input. Take it with the .jsonl traces")
    _say("(which are gitignored and must be copied manually — they never leave the machine).")
    return 0


# ── research commands ──────────────────────────────────────────────────────────

def cmd_ablation(args) -> int:
    from rsc_ablation import main as abl  # type: ignore

    passthrough: list[str] = []
    if args.quick:
        passthrough.append("--quick")
    if args.md:
        passthrough += ["--md", args.md]
    if args.json:
        passthrough += ["--json", args.json]
    if args.rungs:
        passthrough += ["--rungs", args.rungs]
    return abl(passthrough)


def cmd_needle(args) -> int:
    from needle_test import main as nt  # type: ignore

    passthrough = ["--length", str(args.length), "--trials", str(args.trials),
                   "--url", args.url, "--model", args.model]
    if args.depths:
        passthrough += ["--depths", args.depths]
    if args.md:
        passthrough += ["--md", args.md]
    if args.json:
        passthrough += ["--out", args.json]
    if args.dataset_only:
        passthrough.append("--dataset-only")
    return nt(passthrough)


# ── parser ─────────────────────────────────────────────────────────────────────

#: The senses a user can grant, with what each one actually permits. Tiered consent is
#: enforced in policy.check() — "sense has never been granted. Ask, don't assume." — but
#: until this command existed there was no way to ANSWER that ask short of writing
#: Python. A permission model nobody can operate is not a permission model.
SENSES: dict[str, tuple[str, str]] = {
    "web.read":      ("read public web pages (web_read, wiki). Read-only: cannot log "
                      "in, submit, or change anything. Still an outbound channel, so it "
                      "is denied in unattended scopes and every fetch is announced.",
                      "low"),
    "calendar.read": ("read your calendar", "low"),
    "calendar.write":("create or change calendar events — ALWAYS_CONFIRM", "medium"),
    "email.read":    ("read your email", "medium"),
    "email.send":    ("send email as you — ALWAYS_CONFIRM, denied unattended", "high"),
    "location.read": ("read your location", "medium"),
    "screen.capture":("read or search your screen — the most invasive sense here", "high"),
    "mic.listen":    ("listen to your microphone", "high"),
    "fs.write":      ("write files outside FRIDAY_ROOT — ALWAYS_CONFIRM", "high"),
    "shell.exec":    ("run commands — ALWAYS_CONFIRM, denied unattended", "extreme"),
    "network.egress":("POST to arbitrary endpoints (http_post) — ALWAYS_CONFIRM", "high"),
}


def cmd_senses(args) -> int:
    """`friday senses` — who is allowed to perceive what.

    Lists every sense, whether you have granted it, and what granting it permits. This
    is the door that tiered consent was missing: policy.check() denies anything ungranted
    and says "ask, don't assume", and before this there was no way to answer.
    """
    from .agent.policy import enabled_senses, grant, revoke
    from .store import db

    conn = _conn(args)
    granted = set(enabled_senses(conn))

    if args.grant or args.revoke:
        target = args.grant or args.revoke
        if target not in SENSES:
            _say(f"unknown sense '{target}'. Known: {', '.join(sorted(SENSES))}")
            return 2
        if args.grant:
            desc, risk = SENSES[target]
            grant(conn, target)
            conn.commit()
            _say(f"✅ granted '{target}' (risk: {risk})")
            _say(f"   {desc}")
            if risk in ("high", "extreme"):
                _say("   ⚠️  This is a high-risk sense. Every use is confirmation-gated "
                     "and audited — check `friday audit --today` after a session.")
        else:
            revoke(conn, target)
            conn.commit()
            _say(f"🚫 revoked '{target}' — it will be denied from the next turn.")
        granted = set(enabled_senses(conn))

    _say("sense              granted   risk       what it permits")
    _say("-" * 78)
    for sid, (desc, risk) in sorted(SENSES.items(), key=lambda kv: kv[1][1]):
        mark = "yes" if sid in granted else "—"
        _say(f"{sid:18} {mark:9} {risk:10} {desc.split('.')[0]}.")
    extra = sorted(granted - set(SENSES))
    if extra:
        _say(f"\ngranted but not documented here: {', '.join(extra)}")
    _say("\n  grant:   friday senses --grant web.read")
    _say("  revoke:  friday senses --revoke web.read")
    _say("  Nothing is enabled by default, and a fresh install denies every sense.")
    return 0


#: Tools whose invocation means a request LEFT this machine. Announced by default.
EGRESS_TOOLS = frozenset({"web_read", "wiki", "http_post"})


def _announce_egress(st) -> None:
    """⭐ Show every outbound request, whether or not --verbose was passed.

    This exists because of an invariant in the policy tests: everything denied
    unattended should also be confirmation-gated interactively, on the principle that
    "the gate must depend on what the action does, not on who is watching." The
    read-only web tools deliberately break it — confirming every page fetch would make
    them useless, and useless safety features get disabled.

    The exception is only honest if presence provides real observability, and it did
    not: `tool_log` was printed solely under `--verbose`, so by default an interactive
    user was present but BLIND. A URL is an outbound channel — `?d=<your data>` leaves
    the machine in the request and shows up in the attacker's access log — so an
    unnoticed fetch is an unnoticed exfiltration. Presence becomes a control only once
    the fetch is on screen.
    """
    for t in st.tool_log or []:
        if not isinstance(t, dict):
            continue
        name = t.get("origin") or t.get("tool") or ""
        if name not in EGRESS_TOOLS:
            continue
        target = t.get("target") or t.get("url") or ""
        if t.get("error"):
            _say(f"⤷ web: {name} → {target or '(refused)'}  ✗ {str(t['error'])[:90]}")
        else:
            _say(f"⤷ web: {name} → {target}  "
                 f"({t.get('injected_tokens', '?')} tokens into context, trust=external)")


def cmd_doctor(args) -> int:
    """`friday doctor` — will this run on THIS machine?

    The deployment target is a specific laptop and development happens elsewhere, so
    almost every way this fails to deliver is an environment difference nobody noticed.
    Each is checkable in milliseconds, and a printed fix beats a README that describes
    the author's machine.
    """
    from .doctor import run as doctor_run

    rep = doctor_run(root=Path(args.root) if getattr(args, "root", None) else None)
    if getattr(args, "json", False):
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        print(rep.render())
    return 1 if rep.failed else 0


def cmd_bench(args) -> int:
    """`friday bench` — did retrieval just get worse?

    A merely *worse* answer raises no exception, which is why Law 10's "re-run the
    needle test after every fine-tune" gets skipped. This makes it a gate. Model-free
    and about a second, so it can run on every commit; `scripts/needle_test.py` still
    covers a real model's long-range recall.
    """
    from .bench import main as bench_main

    argv = []
    if args.record:
        argv.append("--record")
    if args.gate:
        argv.append("--gate")
    if args.json:
        argv.append("--json")
    if args.verbose:
        argv.append("--verbose")
    if args.file:
        argv += ["--file", args.file]
    return bench_main(argv)


def cmd_serve(args) -> int:
    """`friday serve` — the gateway. Exit test #12, and the Presence Fabric seed.

    Binds loopback unless you ask otherwise; `--host tailscale` is the intended way to
    reach FRIDAY from your phone, because the tunnel is encrypted and the port never
    appears on the internet.
    """
    from .serve import serve

    return serve(host=args.host, port=args.port, token=args.token, quiet=args.quiet)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="friday", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="artifact DB path (default artifacts/friday.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--db", default=None)

    p = sub.add_parser("build", help="Markdown truth -> SQLite indices"); common(p)
    p.set_defaults(fn=cmd_build); p.add_argument("--no-embed", action="store_true")

    p = sub.add_parser("rebuild", help="delete the artifact and rebuild (verifies Law 2)"); common(p)
    p.set_defaults(fn=cmd_rebuild); p.add_argument("--no-embed", action="store_true")

    p = sub.add_parser("seed", help="write starter soul/ and memory/facts/ files"); common(p)
    p.set_defaults(fn=cmd_seed); p.add_argument("--force", action="store_true")

    for name, fn, helptext in (
        ("ask", cmd_ask, "one question"),
        ("chat", cmd_chat, "a REPL session"),
    ):
        p = sub.add_parser(name, help=helptext); common(p)
        p.set_defaults(fn=fn)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--ctx", type=int, default=LEDGER_BUDGET_TOKENS)
        p.add_argument("--url", default=None)
        p.add_argument("--model", default=None)
        p.add_argument("--mock", action="store_true", help="force the offline mock client")
        p.add_argument("--senses", default="")
        p.add_argument("--quiet", action="store_true", help="hide the Ledger printout")
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--transformer-backbone", action="store_true",
                       help="use the full 6-rung ladder instead of hybrid rungs 0+3")

    p = sub.add_parser("write", help="record a durable fact"); common(p)
    p.set_defaults(fn=cmd_write)
    p.add_argument("predicate"); p.add_argument("value")
    p.add_argument("--valid-from", default=None); p.add_argument("--quote", default=None)
    p.add_argument("--user-edit", action="store_true",
                   help="mark as a hand-edit: confidence 1.0, never auto-overwritten")

    p = sub.add_parser("search", help="query memory, showing scores and the floor"); common(p)
    p.set_defaults(fn=cmd_search)
    p.add_argument("query"); p.add_argument("--as-of", default=None)
    p.add_argument("--k", type=int, default=RETRIEVAL_K_SHIP)

    p = sub.add_parser("why", help="⭐ the literal sources behind an answer"); common(p)
    p.set_defaults(fn=cmd_why)
    p.add_argument("query", nargs="?", default=None,
                   help="explain what this question would retrieve")
    p.add_argument("--last", action="store_true", help="explain the most recent turn")
    p.add_argument("--k", type=int, default=RETRIEVAL_K_SHIP)

    p = sub.add_parser("history", help="bi-temporal trail for one predicate"); common(p)
    p.set_defaults(fn=cmd_history); p.add_argument("predicate")

    p = sub.add_parser("timecheck", help="what was true / believed on a date"); common(p)
    p.set_defaults(fn=cmd_timecheck); p.add_argument("date")

    p = sub.add_parser("status", help="store, supervision readiness, signals"); common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("audit", help="the enforcement trail"); common(p)
    p.set_defaults(fn=cmd_audit); p.add_argument("--limit", type=int, default=30)
    p.add_argument("--today", action="store_true", help="only today's entries")
    p.add_argument("--day", default=None, help="YYYY-MM-DD, implies --today")

    p = sub.add_parser("export", help="dump RSC supervision to JSONL for Kaggle"); common(p)
    p.set_defaults(fn=cmd_export); p.add_argument("--out", default="artifacts/rsc_supervision.jsonl")

    p = sub.add_parser("ablation", help="the RSC ladder (pure-Python operators)"); common(p)
    p.set_defaults(fn=cmd_ablation)
    p.add_argument("--quick", action="store_true"); p.add_argument("--rungs", default="")
    p.add_argument("--md", default=None); p.add_argument("--json", default=None)

    p = sub.add_parser("needle", help="⭐ the decisive recall benchmark"); common(p)
    p.set_defaults(fn=cmd_needle)
    p.add_argument("--length", type=int, default=8000); p.add_argument("--trials", type=int, default=20)
    p.add_argument("--depths", default=""); p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="friday-local")
    p.add_argument("--md", default=None); p.add_argument("--json", default=None)
    p.add_argument("--dataset-only", action="store_true")

    p = sub.add_parser("senses", help="⭐ tiered consent — grant/revoke what FRIDAY may perceive")
    p.set_defaults(fn=cmd_senses); common(p)
    p.add_argument("--grant", default=None, metavar="SENSE")
    p.add_argument("--revoke", default=None, metavar="SENSE")

    p = sub.add_parser("doctor", help="⭐ will FRIDAY run on THIS machine?")
    p.set_defaults(fn=cmd_doctor); common(p)
    p.add_argument("--root", default=None, help="FRIDAY_ROOT to inspect (default: current)")
    p.add_argument("--json", action="store_true", help="machine-readable report")

    p = sub.add_parser("bench", help="⭐ the retrieval regression gate (Law 10)")
    p.set_defaults(fn=cmd_bench); common(p)
    p.add_argument("--record", action="store_true", help="write BENCHMARKS.md baseline")
    p.add_argument("--gate", action="store_true", help="exit 1 if worse than baseline")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", action="store_true", help="per-query results")
    p.add_argument("--file", default=None, help="BENCHMARKS.md path")

    p = sub.add_parser("serve", help="⭐ the HTTP gateway — reach FRIDAY from your phone")
    p.set_defaults(fn=cmd_serve); common(p)
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 (default) | tailscale | 0.0.0.0 | an explicit address")
    p.add_argument("--port", type=int, default=8642)
    p.add_argument("--token", default=None,
                   help="bearer token; default reads/creates config/serve.token (0600)")
    p.add_argument("--quiet", action="store_true")

    return ap


def _scripts_dir() -> Path:
    """Where `rsc_ablation.py` and `needle_test.py` live.

    ⚠️ Relative to the PACKAGE, not to `$FRIDAY_ROOT`. `scripts/` is code that ships
    with FRIDAY; `$FRIDAY_ROOT` is the user's DATA root (soul/, memory/, artifacts/).
    Resolving scripts against ROOT meant `friday ablation` and `friday needle` only
    worked when FRIDAY_ROOT happened to be a checkout of this repository — so they
    died with ModuleNotFoundError for anyone who installed the package, and in every
    test, because tests point FRIDAY_ROOT at an empty tmp_path. Falling back to ROOT
    keeps a developer checkout working if the package is moved.
    """
    here = Path(__file__).resolve().parent.parent / "scripts"
    if (here / "rsc_ablation.py").exists():
        return here
    return paths.ROOT / "scripts"


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    # let `friday ablation` / `friday needle` import their scripts
    sys.path.insert(0, str(_scripts_dir()))
    kw = {k: v for k, v in vars(args).items() if k not in ("fn", "cmd")}
    return args.fn(argparse.Namespace(**kw))


if __name__ == "__main__":
    raise SystemExit(main())
