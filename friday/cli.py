"""The CLI you'll actually live in. docs/architecture/10 Day 6.

Subcommands:

    build      Markdown truth -> SQLite indices (idempotent, incremental)
    rebuild    rm the artifact, rebuild from Markdown — verifies Law 2
    seed       write starter soul/ + memory/facts/ files
    ask        one question, print the Ledger block, get an answer
    chat       a REPL session
    write      record a fact from the shell
    search     query memory directly, showing scores and the floor
    history    the bi-temporal trail for one predicate
    status     RAM/DB/supervision readiness at a glance
    ablation   the RSC ladder
    needle     the decisive recall benchmark
    audit      who did what, allowed or denied

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
from .config import LEDGER_BUDGET_TOKENS, RETRIEVAL_K_SHIP, canonical_predicate
from .ledger.compiler import Gather, Ledger
from .llm.client import get_client
from .memory import compiler as mcompiler
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

    if not args.quiet and st.compiled is not None:
        _say(st.compiled.printout(turn=st.turn_idx))
        _say("")
    _say(st.final_text)
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
    _say("Type your message. Ctrl-D or 'exit' to quit. '\\ledger' toggles the printout.\n")
    show_ledger = not args.quiet
    session = ""
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
        st = agent.run(line, session_id=session, senses=senses)
        session = st.session_id
        if show_ledger and st.compiled is not None:
            _say(st.compiled.printout(turn=st.turn_idx))
        _say(f"\nfriday> {st.final_text}\n")


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
    rows = conn.execute(
        "SELECT ts, actor, action, target, sense_id, decision FROM audit ORDER BY id DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    if not rows:
        _say("audit is empty — nothing has been enforced yet.")
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

    p = sub.add_parser("history", help="bi-temporal trail for one predicate"); common(p)
    p.set_defaults(fn=cmd_history); p.add_argument("predicate")

    p = sub.add_parser("timecheck", help="what was true / believed on a date"); common(p)
    p.set_defaults(fn=cmd_timecheck); p.add_argument("date")

    p = sub.add_parser("status", help="store, supervision readiness, signals"); common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("audit", help="the enforcement trail"); common(p)
    p.set_defaults(fn=cmd_audit); p.add_argument("--limit", type=int, default=30)

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
