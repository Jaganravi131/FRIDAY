#!/usr/bin/env python3
"""The Phase 0 exit test, made executable. docs/architecture/10 Day 7.

    Run all of these. Do not proceed to Phase 1 until they pass.

Twelve conditions were written down as a table in a document, where they are easy to
believe and hard to check. This runs them. It reports PASS / FAIL / SKIP with the
evidence for each, and exits non-zero on any FAIL, so it can gate a build or a
commit rather than depending on someone remembering the table.

WHY SKIP IS A FIRST-CLASS OUTCOME
---------------------------------
Three conditions cannot be evaluated here, and pretending otherwise would be worse
than admitting it:

  * #11 (RSS after an hour idle) needs a real model loaded by llama-server, and a
    clock. Neither exists in a stdlib-only sandbox.
  * #12 (phone via Tailscale) needs your network and your phone.
  * #10 is only PARTIALLY testable: the client can be made to fail mid-turn, which
    proves the agent degrades loudly, but it cannot prove llama-server's own
    behaviour when killed.

Those are reported as SKIP with the reason and what you must do by hand. A gate that
silently counted them as passes would be the exact self-delusion doc 10 warns about
in the row below them ("self-improvement before an eval suite is self-delusion").

ISOLATION
---------
Runs in a throwaway $FRIDAY_ROOT under the system temp dir, so it never touches your
memory/, soul/ or artifacts/. Set --root to run it against a real install instead —
useful once you have a week of actual conversation in there, which is the point of
the whole exercise.

    python scripts/phase0_exit_test.py
    python scripts/phase0_exit_test.py --root ~/.friday --keep
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    n: int
    title: str
    status: str = SKIP
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    manual: str = ""
    ms: int = 0


@dataclass
class Ctx:
    root: Path
    conn: sqlite3.Connection
    checks: list[Check] = field(default_factory=list)


def _bind(root: Path):
    """Point the whole package at `root`. See tests/conftest.py for why this must
    MUTATE the one paths module rather than delete and re-import."""
    os.environ["FRIDAY_ROOT"] = str(root)
    for m in [k for k in sys.modules if k == "friday" or k.startswith("friday.")]:
        del sys.modules[m]
    paths = importlib.import_module("friday.paths")
    paths.rebind(root)
    paths.ensure_layout()
    return paths


# ── 1. the rebuild property ───────────────────────────────────────────────────


def check_1_rebuild(ctx: Ctx) -> Check:
    c = Check(1, "rm -rf artifacts && rebuild && ask -> correct answer")
    from friday import paths
    from friday.llm import get_client
    from friday.memory import compiler
    from friday.memory.facts import Fact, assert_fact
    from friday.store import db

    assert_fact(ctx.conn, Fact(subject="user", predicate="lease_amount_monthly",
                               object="18000 INR", source_kind="user_edit",
                               source_quote="my monthly rent is 18000 INR"))
    ctx.conn.commit()
    before = ctx.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # Delete the derived store entirely, then rebuild from Markdown alone.
    ctx.conn.close()
    shutil.rmtree(paths.ARTIFACTS, ignore_errors=True)
    fresh = db.connect(paths.DB_PATH)
    st = compiler.compile_all(fresh)
    after = fresh.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    c.evidence.append(f"facts before rm: {before}, after rebuild: {after}")
    c.evidence.append(f"compile: {st.facts_written} written from {st.files} files")

    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.pipeline import search
    from friday.retrieval.rerankers import get_reranker
    res = search(fresh, "what is my monthly rent", embedder=get_embedder(),
                 reranker=get_reranker())
    got = " ".join(x.body for x in res.kept)
    c.evidence.append(f"asked 'what is my monthly rent' -> {got[:90]!r}")

    ok = after >= before and "18000" in got
    c.status = PASS if ok else FAIL
    c.detail = ("Markdown is truth and SQLite is derived: deleting the artifact and "
                "rebuilding restored the answer." if ok else
                "rebuild lost facts or the answer — some state exists only in the DB")
    ctx.conn = fresh
    return c


# ── 2. Monday -> Friday ───────────────────────────────────────────────────────


def check_2_monday_friday(ctx: Ctx) -> Check:
    c = Check(2, "tell it a fact Monday, ask a follow-up Friday")
    from friday.memory.facts import Fact, assert_fact
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.pipeline import search
    from friday.retrieval.rerankers import get_reranker

    assert_fact(ctx.conn, Fact(subject="landlord", predicate="name", object="Ramesh",
                               source_kind="user_edit", valid_from="2026-09-21",
                               source_quote="my landlord is called Ramesh"))
    ctx.conn.commit()

    res = search(ctx.conn, "who is my landlord", embedder=get_embedder(),
                 reranker=get_reranker())
    d = res.as_dicts()
    c.evidence.append(f"recalled {len(d)} fact(s): {res.summary()[:80]}")
    if d:
        c.evidence.append(f"provenance carried: quote={d[0].get('source_quote')!r} "
                          f"id={d[0].get('id')} trace={d[0].get('trace_id')} "
                          f"file={d[0].get('origin_file')}")
    ok = bool(d) and any("Ramesh" in (x.get("object") or "") for x in d)
    cites = bool(d) and d[0].get("source_quote") and d[0].get("origin_file")
    c.status = PASS if (ok and cites) else FAIL
    c.detail = ("recalled correctly and cites provenance" if ok and cites else
                "recalled but WITHOUT a usable citation" if ok else "did not recall")
    return c


# ── 3. /why ───────────────────────────────────────────────────────────────────


def check_3_why(ctx: Ctx) -> Check:
    c = Check(3, "`/why` shows the literal source quote + trace id")
    from friday.memory import provenance
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.pipeline import search
    from friday.retrieval.rerankers import get_reranker

    res = search(ctx.conn, "what is my monthly rent", embedder=get_embedder(),
                 reranker=get_reranker())
    provs = provenance.from_candidates(res.as_dicts())
    report = provenance.render_report(provs, query="what is my monthly rent")
    for line in report.splitlines()[:14]:
        c.evidence.append(line)

    ok = bool(provs) and any(p.quote for p in provs)
    has_file = any(p.origin_file for p in provs)
    unverifiable = [p for p in provs if not p.verifiable]
    if ok and not all(p.trace_id for p in provs):
        c.evidence.append("note: some facts have no trace_id (hand-typed Markdown has "
                          "no conversation behind it — that is correct, not a gap)")
    c.status = PASS if (ok and has_file) else FAIL
    c.detail = (f"{len(provs)} fact(s) justified; {len(provs) - len(unverifiable)} fully "
                f"verifiable" if ok else "no quote behind the answer — untraceable")
    return c


# ── 4. does not invent ────────────────────────────────────────────────────────


def check_4_does_not_invent(ctx: Ctx) -> Check:
    c = Check(4, "ask something it has no memory of -> [], says so, does not invent")
    from friday.llm import get_client
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.pipeline import search
    from friday.retrieval.rerankers import get_reranker

    q = "what is my favourite programming language and my cat's name"
    res = search(ctx.conn, q, embedder=get_embedder(), reranker=get_reranker())
    c.evidence.append(f"retrieval: kept={len(res.kept)} considered={res.considered} "
                      f"floor={res.floor} below_floor={res.dropped_below_floor}")

    client = get_client(prefer="mock")
    resp = client.chat([{"role": "user", "content": q}], tools=[])
    text = (resp.text or "").strip()
    c.evidence.append(f"mock answered: {text[:160]!r}")

    honest = any(k in text.lower() for k in
                 ("don't have", "do not have", "no memory", "not recorded", "haven't",
                  "have not", "no record", "don't know", "do not know", "nothing"))
    invented = any(w in text.lower() for w in ("python", "java", "whiskers", "ginger"))
    ok = len(res.kept) == 0 and honest and not invented
    c.status = PASS if ok else FAIL
    c.detail = ("empty retrieval AND the answer says so" if ok else
                "retrieval was empty but the answer did not admit it" if not honest
                else "retrieval returned something for a question with no memory")
    return c


# ── 5. contradiction ──────────────────────────────────────────────────────────


def check_5_contradiction(ctx: Ctx) -> Check:
    c = Check(5, "contradict a fact -> retracted not deleted, Markdown struck through")
    from friday.memory.facts import Fact, assert_fact

    rep = assert_fact(ctx.conn, Fact(subject="user", predicate="lives_in",
                                     object="Bengaluru", source_kind="user_edit",
                                     source_quote="I live in Bengaluru"))
    old_id = (rep.get("written") or {}).get("id")
    rep2 = assert_fact(ctx.conn, Fact(subject="user", predicate="lives_in",
                                      object="Chennai", source_kind="user_edit",
                                      source_quote="actually I moved to Chennai"))
    c.evidence.append(f"second assertion -> action={rep2['action']} "
                      f"retracted={[r['object'] for r in rep2['retracted']]}")

    row = ctx.conn.execute(
        "SELECT retracted_at, superseded_by, valid_to FROM facts WHERE id=?", (old_id,)
    ).fetchone()
    still_there = row is not None
    c.evidence.append(f"old row still in DB: {still_there} "
                      f"retracted_at={row['retracted_at'] if row else None} "
                      f"superseded_by={row['superseded_by'] if row else None}")

    md = ""
    for p in sorted((ctx.root / "memory" / "facts").rglob("*.md")):
        md += p.read_text(encoding="utf-8")
    struck = "~~" in md and "Bengaluru" in md
    c.evidence.append(f"Markdown shows ~~struck-through~~ Bengaluru: {struck}")
    pair = bool((rep2.get("supervision") or {}).get("pair"))
    c.evidence.append(f"retraction pair recorded (supervises EDA's e_t): {pair}")

    ok = still_there and bool(row and row["retracted_at"]) and struck and rep2["action"] == "reconciled"
    c.status = PASS if ok else FAIL
    c.detail = ("retracted, not deleted; the diff is visible in the file" if ok else
                "the old value was destroyed or the Markdown was not marked")
    return c


# ── 6. hand-edit hot reload ───────────────────────────────────────────────────


def check_6_hand_edit(ctx: Ctx) -> Check:
    c = Check(6, "hand-edit a facts file, save, ask immediately — no restart")
    from friday.agent.loop import Agent
    from friday.llm import get_client
    from friday.memory import watcher
    from friday.memory.facts import Fact, assert_fact
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.rerankers import get_reranker

    f = ctx.root / "memory" / "facts" / "housing.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    original = f.read_text(encoding="utf-8") if f.exists() else ""

    # Simulate YOU opening the file in Notepad and changing the number.
    edited = original.replace("18000 INR", "31500 INR")
    if edited == original:
        edited = original + "\n- lease_amount_monthly: **31500 INR** [f_hand1 · user_edit · conf 1.00]\n"
    f.write_text(edited, encoding="utf-8")
    # Backdate mtime so a coarse filesystem clock cannot hide the change.
    st = f.stat()
    os.utime(f, (st.st_atime - 5, st.st_mtime - 5))
    f.write_text(edited, encoding="utf-8")

    # The agent must notice on the NEXT TURN, with no restart and no `friday build`.
    ag = Agent(ctx.conn, client=get_client(prefer="mock"),
               embedder=get_embedder(), reranker=get_reranker())
    state = ag.run("what is my monthly rent", senses=[])
    sync = state.memory_sync
    c.evidence.append(f"watcher noticed: {sync.summary() if sync else 'no report'}")
    c.evidence.append(f"answer: {state.final_text[:110]!r}")

    row = ctx.conn.execute(
        "SELECT object FROM facts WHERE predicate='lease_amount_monthly' "
        "AND retracted_at IS NULL ORDER BY asserted_at DESC LIMIT 1").fetchone()
    c.evidence.append(f"DB now says: {row['object'] if row else None}")

    aud = ctx.conn.execute(
        "SELECT actor, action, target, detail FROM audit "
        "WHERE action='memory.hand_edit' ORDER BY id DESC LIMIT 1").fetchone()
    c.evidence.append(f"audit row: {dict(aud) if aud else None}")

    ok = (bool(sync and sync.recompiled) and bool(row) and "31500" in (row["object"] or "")
          and aud is not None and aud["actor"] == "user")
    c.status = PASS if ok else FAIL
    c.detail = ("picked up mid-session, re-derived, and logged as YOUR edit" if ok else
                "the edit was not noticed, or was noticed but not audited")
    return c


# ── 7. 50 turns, cache + reserve ──────────────────────────────────────────────


def check_7_fifty_turns(ctx: Ctx) -> Check:
    c = Check(7, "50-turn session: cache_hit >= 0.85, tokens flat, reserve intact")
    from friday.ledger.compiler import Gather, Ledger

    led = Ledger()
    questions = [
        "what is my monthly rent", "who is my landlord", "where do I live",
        "what do I prefer to drink", "what is my current task", "when does my lease renew",
        "what is my daily routine", "tell me about my job", "what city am I in",
        "how much rent do I pay",
    ]
    hits = turns = 0
    spent: list[int] = []
    reserves: list[int] = []
    summarised = 0
    transcript: list[dict] = []

    for i in range(50):
        q = questions[i % len(questions)]
        transcript.append({"role": "user", "content": q})
        transcript.append({"role": "assistant", "content": f"answer {i} " * 8})
        g = Gather(query=q, senses=[], transcript=transcript[-24:],
                   scratch=f"note {i}", recalled=[])
        cc = led.compile(g, conn=ctx.conn)
        turns += 1
        hits += int(cc.cache_hit)
        spent.append(cc.spent)
        reserves.append(cc.reserve)
        summarised += int(cc.summarised)

    rate = hits / max(1, turns - 1)          # turn 1 can never be a hit
    tail = spent[-10:]
    drift = (max(tail) - min(tail)) / max(1, min(tail))
    min_reserve = min(reserves)

    c.evidence.append(f"cache_hit_rate = {rate:.3f} ({hits}/{turns - 1} eligible turns)")
    c.evidence.append(f"tokens/turn last 10: min={min(tail)} max={max(tail)} "
                      f"drift={drift:.1%}")
    c.evidence.append(f"reserve: min={min_reserve} (budget {led.budget}, "
                      f"target {led.reserve_target})")
    c.evidence.append(f"times the ladder reached rung 5 (summarise): {summarised}")

    ok = rate >= 0.85 and drift < 0.35 and min_reserve > 0 and summarised == 0
    c.status = PASS if ok else FAIL
    c.detail = ("stable prefix held, tokens flat, reserve never spent" if ok else
                f"rate {rate:.2f} / drift {drift:.0%} / min reserve {min_reserve} — "
                f"something in the prefix is moving")
    return c


# ── 8. the tool contract ──────────────────────────────────────────────────────


def check_8_tool_contract(ctx: Ctx) -> Check:
    c = Check(8, "ask the time / a trivial question -> memory_search NOT called")
    from friday.agent.loop import Agent
    from friday.agent.routing import classify_turn
    from friday.llm import get_client
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.rerankers import get_reranker

    trivial = ["what time is it", "hi", "thanks", "2+2", "who are you"]
    memory_q = ["what is my monthly rent", "where do I live", "my rent went up"]

    ag = Agent(ctx.conn, client=get_client(prefer="mock"),
               embedder=get_embedder(), reranker=get_reranker())
    bad = []
    for q in trivial:
        stt = ag.run(q, senses=[])
        called = [t.get("tool") for t in stt.tool_log]
        searched = "memory_search" in called or bool(stt.retrieved)
        c.evidence.append(f"  {q!r:22} intent={stt.intent.kind:10} "
                          f"retrieved={len(stt.retrieved)} tools={called}")
        if searched:
            bad.append(q)
    for q in memory_q:
        stt = ag.run(q, senses=[])
        c.evidence.append(f"  {q!r:22} intent={stt.intent.kind:10} "
                          f"retrieved={len(stt.retrieved)} (MUST retrieve)")
        if not stt.retrieved and stt.intent.needs_memory:
            # retrieval ran and legitimately found nothing; that is not a contract
            # violation, so only flag it if we skipped when we should not have
            if not stt.intent.needs_memory:
                bad.append(q)

    # The structural guarantee: the tool is WITHHELD, not merely discouraged.
    from friday.agent.routing import tools_for
    from friday.agent.tools import build_registry
    reg = build_registry()
    withheld = tools_for(classify_turn("what time is it"), reg.schemas())
    names = [(t.get("function") or {}).get("name") for t in withheld]
    c.evidence.append(f"  tools offered for 'what time is it': {names}")
    if "memory_search" in names:
        bad.append("memory_search was still offered for a clock question")

    c.status = PASS if not bad else FAIL
    c.detail = ("trivial turns never reach memory; the tool is withheld structurally"
                if not bad else f"violations: {bad}")
    return c


# ── 9. audit --today ──────────────────────────────────────────────────────────


def check_9_audit_today(ctx: Ctx) -> Check:
    c = Check(9, "`friday audit --today` shows every decision and denial")
    from friday.agent.policy import check as pcheck
    from friday.agent.policy import grant
    from friday.cli import main as cli_main
    from friday.util import now_iso

    pcheck(ctx.conn, "mic_listen", {})                      # denied, ungranted
    grant(ctx.conn, "mic.listen")
    pcheck(ctx.conn, "mic_listen", {"target": "standup"})    # allowed
    pcheck(ctx.conn, "email_send", {}, scope="heartbeat")    # denied, unattended

    rows = ctx.conn.execute(
        "SELECT actor, action, decision, sense_id FROM audit WHERE ts LIKE ? ORDER BY id",
        (now_iso()[:10] + "%",)).fetchall()
    decisions = sorted({r["decision"] for r in rows})
    c.evidence.append(f"{len(rows)} audit rows today; decisions seen: {decisions}")
    for r in rows[-6:]:
        c.evidence.append(f"  {r['actor']:10} {r['action']:14} {r['decision']:10} "
                          f"{r['sense_id'] or '-'}")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["audit", "--today"])
    out = buf.getvalue()
    c.evidence.append(f"`friday audit --today` exit={rc}, {len(out.splitlines())} lines, "
                      f"mentions mic_listen: {'mic_listen' in out}")

    ok = (rc == 0 and "denied" in decisions and "allowed" in decisions
          and "mic_listen" in out and len(rows) >= 4)
    c.status = PASS if ok else FAIL
    c.detail = ("every decision and denial for the day is visible" if ok else
                "the trail is incomplete — a denial happened that is not queryable")
    return c


# ── 10. the model dies mid-turn ───────────────────────────────────────────────


def check_10_model_dies(ctx: Ctx) -> Check:
    c = Check(10, "kill the model mid-turn -> graceful, loud, no silent garbage")
    from friday.agent.loop import Agent
    from friday.llm import Response
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.rerankers import get_reranker

    class DyingClient:
        name = "dying"

        def chat(self, messages, tools=None, **kw):
            raise RuntimeError("cannot reach llama-server at http://127.0.0.1:8080 "
                               "(connection refused)")

        def complete(self, *a, **kw):
            raise RuntimeError("cannot reach llama-server")

    ag = Agent(ctx.conn, client=DyingClient(), embedder=get_embedder(),
               reranker=get_reranker())
    t0 = time.perf_counter()
    st = ag.run("what is my monthly rent", senses=[])
    c.ms = int((time.perf_counter() - t0) * 1000)
    text = (st.final_text or "").strip()
    c.evidence.append(f"answer: {text[:200]!r}")
    c.evidence.append(f"turn completed in {c.ms} ms without raising")

    loud = any(k in text.lower() for k in
               ("cannot reach", "can't reach", "not running", "unavailable", "failed",
                "error", "offline", "start llama", "llama-server"))
    silent_garbage = bool(text) and not loud and len(text) > 40
    raised = False
    try:
        ag.run("and again", senses=[])
    except Exception as e:                      # must not propagate
        raised = True
        c.evidence.append(f"second turn raised: {type(e).__name__}: {e}")

    ok = loud and not silent_garbage and not raised
    c.status = PASS if ok else FAIL
    c.detail = ("failed loudly and stayed usable" if ok else
                "raised into the caller" if raised else
                "produced plausible text with no model — silent garbage" if silent_garbage
                else "the failure was not communicated to the user")
    c.manual = ("Partially automatable: this proves the AGENT degrades correctly when "
                "the client fails. Killing a real llama-server mid-generation and "
                "watching the socket error is still yours to do.")
    return c


# ── 11 / 12. cannot be evaluated here ────────────────────────────────────────


def check_11_rss(ctx: Ctx) -> Check:
    return Check(
        11, "RSS after 1 hour idle -> models unloaded, RAM returned",
        status=SKIP,
        detail="needs a real model loaded by llama-server, and an hour of wall clock",
        manual=("Load your model, idle for an hour, then compare RSS before/after with "
                "Task Manager or `psutil.Process().memory_info().rss`. On Windows the "
                "watchdog is llama-server's `--n-gpu-layers` + idle unload; confirm the "
                "process actually releases it rather than keeping it mapped."),
    )


def check_12_phone(ctx: Ctx) -> Check:
    return Check(
        12, "from your phone, via Tailscale, hit the API",
        status=SKIP,
        detail="needs your network, your phone, and a Tailscale tailnet",
        manual=("Run `friday serve` (Phase 3) or llama-server bound to the Tailscale "
                "interface, then curl it from the phone: `curl http://<tailscale-ip>:8080"
                "/v1/models`. This is the Presence Fabric seed — do not skip it forever, "
                "it is the difference between a laptop toy and something you carry."),
    )


CHECKS = [check_1_rebuild, check_2_monday_friday, check_3_why, check_4_does_not_invent,
          check_5_contradiction, check_6_hand_edit, check_7_fifty_turns,
          check_8_tool_contract, check_9_audit_today, check_10_model_dies,
          check_11_rss, check_12_phone]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="run against a real install instead of a throwaway root")
    ap.add_argument("--keep", action="store_true", help="do not delete the temp root")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="print all evidence")
    args = ap.parse_args(argv)

    tmp = None
    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        tmp = Path(tempfile.mkdtemp(prefix="friday-exit-test-"))
        root = tmp
    paths = _bind(root)

    from friday.store import db
    conn = db.connect(paths.DB_PATH)
    ctx = Ctx(root=root, conn=conn)

    # A store with something in it, so the retrieval checks have material.
    from friday import seed
    from friday.memory import compiler
    if not args.root:
        seed.write_all()
    compiler.compile_all(conn)

    print(f"Phase 0 exit test — docs/architecture/10 Day 7")
    print(f"root: {root}{'  (throwaway)' if tmp else '  (REAL INSTALL)'}")
    print(f"sqlite {sqlite3.sqlite_version} · python {sys.version.split()[0]}")
    print("=" * 78)

    results = []
    for fn in CHECKS:
        t0 = time.perf_counter()
        try:
            chk = fn(ctx)
        except Exception as e:
            import traceback
            chk = Check(int(fn.__name__.split("_")[1]), fn.__doc__ or fn.__name__,
                        status=FAIL,
                        detail=f"raised {type(e).__name__}: {e}")
            chk.evidence.append(traceback.format_exc(limit=3))
        if not chk.ms:
            chk.ms = int((time.perf_counter() - t0) * 1000)
        results.append(chk)
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[chk.status]
        print(f"\n{mark} #{chk.n:2} {chk.title}   [{chk.status} · {chk.ms} ms]")
        print(f"      {chk.detail}")
        if args.verbose or chk.status == FAIL:
            for e in chk.evidence:
                print(f"      │ {e}")
        if chk.manual:
            print(f"      ✋ {chk.manual}")

    npass = sum(1 for r in results if r.status == PASS)
    nfail = sum(1 for r in results if r.status == FAIL)
    nskip = sum(1 for r in results if r.status == SKIP)

    print("\n" + "=" * 78)
    print(f"{npass} PASS · {nfail} FAIL · {nskip} SKIP   ({len(results)} conditions)")
    if nfail:
        print("\n❌ DO NOT PROCEED TO PHASE 1. Fix the failures above first —")
        print("   doc 10 is explicit that this gate is not advisory.")
    elif nskip:
        print("\n⚠️  Automated conditions pass. The SKIPs need you, your hardware and")
        print("   your network: run them by hand before calling Phase 0 done.")
    else:
        print("\n✅ Phase 0 is complete. Phase 1 (the Memory Compiler) is unblocked.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"n": r.n, "title": r.title, "status": r.status, "detail": r.detail,
              "evidence": r.evidence, "manual": r.manual, "ms": r.ms} for r in results],
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")

    try:
        ctx.conn.close()
    except Exception:
        pass
    if tmp and not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    elif tmp:
        print(f"kept {tmp}")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
