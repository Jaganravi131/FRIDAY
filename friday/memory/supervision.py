"""⭐ RSC supervision taps — docs/architecture/13 §4.1

GDN-2/EDA expose three gate branches. FRIDAY's Memory Compiler already computes
three matching signals, for unrelated reasons:

    channel-wise decay  alpha_t  <-  the decay function      (memory/decay.py)
    independent erase   e_t      <-  RETRACTION PAIRS        (this module)
    channel-wise write  w_t      <-  SALIENCE SPANS          (this module)

Retractions yield (key_old, key_new) with key_old != key_new. That inequality is
*exactly* why a write-anchored erase (GDN-2) cannot reach the stale association,
and exactly what `L_erase` trains.

⚠️ THESE CANNOT BE BACKFILLED. Every correction the user makes and never records
is training data lost forever. That is why this module is wired into Phase 1 and
not Phase 3.5: the labels are cheap to collect and impossible to recover.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ..util import new_id, now_iso


@dataclass
class SpanLabel:
    """A span of text, labelled by whether it produced an S2 fact.

    Two provenance modes, because the supervision has to survive both paths a fact
    can arrive by:

      * TRACE-ANCHORED (`trace_id` + `trace_file` + offsets). The normal case for a
        fact extracted from a conversation. The text is NOT copied into the DB —
        only the (file, start, end) triple — because traces are ambient capture and
        must stay gitignored. The trainer reads the substring back on demand.

      * INLINE (`text` set, `trace_file` empty). The case for a fact that arrived
        with a quote but no trace: a CLI `friday write --quote "…"`, a line
        hand-typed into `memory/facts/*.md`, or anything imported. There is no file
        to point at, so the quote itself is stored. Losing these would silently drop
        supervision for every hand-authored fact — which is the highest-confidence
        population in the store (Law 7) and therefore the most valuable to learn from.
    """

    trace_id: str
    trace_file: str
    char_start: int
    char_end: int
    label: int                 # 1 = produced a fact, 0 = sampled negative
    fact_id: str | None = None
    domain: str | None = None
    id: str | None = None
    text: str | None = None    # inline mode only; NULL for trace-anchored spans

    @property
    def text_len(self) -> int:
        return max(0, self.char_end - self.char_start)

    @property
    def inline(self) -> bool:
        return bool(self.text) and not self.trace_file


@dataclass
class RetractionPair:
    """A (superseded, superseding) fact pair — EDA's erase/write address training example."""

    fact_old_id: str
    fact_new_id: str
    key_old: str
    key_new: str
    value_old: str
    value_new: str
    trace_old_id: str | None = None
    trace_new_id: str | None = None
    span_old_id: str | None = None
    span_new_id: str | None = None
    id: str | None = None

    @property
    def address_distinct(self) -> bool:
        """The EDA case: erase must happen at an address other than the write address."""
        return self.key_old.strip().lower() != self.key_new.strip().lower()


# ── writes ─────────────────────────────────────────────────────────────────────

def record_span(conn: sqlite3.Connection, span: SpanLabel) -> str:
    """Record one salience label. Called by the compiler every time a fact is
    extracted (label=1) and by the negative sampler (label=0)."""
    sid = span.id or new_id("sp")
    conn.execute(
        """INSERT OR REPLACE INTO supervision_spans
           (id, trace_id, trace_file, char_start, char_end, label, fact_id, domain,
            span_text, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            sid, span.trace_id, span.trace_file or None, span.char_start, span.char_end,
            int(span.label), span.fact_id, span.domain, span.text, now_iso(),
        ),
    )
    return sid


def span_text(conn: sqlite3.Connection, span_id: str) -> str | None:
    """Resolve a span back to the exact characters it labels.

    Inline spans return their stored text. Trace-anchored spans read the substring
    out of the trace file — the text is never copied into the DB, because traces are
    ambient capture and stay gitignored (docs/architecture/09). A missing trace file
    returns None rather than raising: traces are allowed to be pruned, and a training
    run that loses a few spans is fine, whereas one that crashes on a pruned trace
    is not.
    """
    r = conn.execute(
        "SELECT trace_file, char_start, char_end, span_text FROM supervision_spans WHERE id=?",
        (span_id,),
    ).fetchone()
    if r is None:
        return None
    if r["span_text"]:
        return r["span_text"]
    if not r["trace_file"]:
        return None
    from pathlib import Path

    from .. import paths

    p = Path(r["trace_file"])
    if not p.is_absolute():
        p = paths.ROOT / p
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    return text[int(r["char_start"]):int(r["char_end"])] or None


def record_pair(conn: sqlite3.Connection, pair: RetractionPair) -> str:
    """Record one retraction pair. Called from the bi-temporal reconciliation in
    memory/facts.py — i.e. from the write you were already doing."""
    pid = pair.id or new_id("pr")
    conn.execute(
        """INSERT OR REPLACE INTO supervision_pairs
           (id, fact_old_id, fact_new_id, key_old, key_new, value_old, value_new,
            trace_old_id, trace_new_id, span_old_id, span_new_id,
            address_distinct, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            pid, pair.fact_old_id, pair.fact_new_id, pair.key_old, pair.key_new,
            pair.value_old, pair.value_new, pair.trace_old_id, pair.trace_new_id,
            pair.span_old_id, pair.span_new_id, int(pair.address_distinct), now_iso(),
        ),
    )
    return pid


# ── reads (training-set export) ────────────────────────────────────────────────

def span_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT label, COUNT(*) AS n FROM supervision_spans GROUP BY label"
    ).fetchall()
    out = {str(r["label"]): int(r["n"]) for r in rows}
    return {"positive": out.get("1", 0), "negative": out.get("0", 0), "total": sum(out.values())}


def pair_counts(conn: sqlite3.Connection) -> dict[str, int]:
    total = conn.execute("SELECT COUNT(*) AS n FROM supervision_pairs").fetchone()["n"]
    distinct = conn.execute(
        "SELECT COUNT(*) AS n FROM supervision_pairs WHERE address_distinct=1"
    ).fetchone()["n"]
    return {"total": int(total), "address_distinct": int(distinct)}


def export_training_set(conn: sqlite3.Connection, out_path) -> dict:
    """Dump both label sets to JSONL for the Kaggle notebook.

    Two record kinds in one file: `{"kind": "span"|"pair", …}`. The RSC trainer
    filters on `kind`.

    Each row is ENRICHED beyond the raw table, because the notebook must be
    self-contained — Kaggle has no access to this machine's traces, and a file of
    (char_start, char_end) offsets pointing at a gitignored JSONL is not a training
    set, it's a shopping list:

      * spans gain `alpha_target` (the decay-policy retention for the fact's
        predicate class — this is what `L_decay` regresses toward) and `text`
        (the resolved substring, inlined so the offsets become checkable).
      * pairs gain `keys_distinct`, the EDA discriminator, spelled out rather than
        left as a 0/1 column named `address_distinct`.

    Trace-anchored spans whose file has been pruned export with `text: null`. That
    is expected and survivable: the offsets remain, so the row is still usable for
    length statistics, and losing some spans to trace rotation is normal.
    """
    from pathlib import Path
    import json

    from ..memory.decay import policy_alpha
    from ..config import predicate_class_of

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = n_spans = n_pairs = 0
    with p.open("w", encoding="utf-8") as fh:
        for r in conn.execute("SELECT * FROM supervision_spans ORDER BY created_at"):
            row = dict(r)
            klass = None
            if row.get("fact_id"):
                f = conn.execute(
                    "SELECT predicate_class FROM facts WHERE id=?", (row["fact_id"],)
                ).fetchone()
                klass = f["predicate_class"] if f else None
            row["predicate_class"] = klass
            row["alpha_target"] = policy_alpha(klass, None) if klass else None
            row["text"] = span_text(conn, row["id"])
            fh.write(json.dumps({"kind": "span", **row}, ensure_ascii=False, default=str) + "\n")
            n += 1
            n_spans += 1
        for r in conn.execute("SELECT * FROM supervision_pairs ORDER BY created_at"):
            row = dict(r)
            row["keys_distinct"] = bool(row.get("address_distinct"))
            fh.write(json.dumps({"kind": "pair", **row}, ensure_ascii=False, default=str) + "\n")
            n += 1
            n_pairs += 1
    return {"path": str(p), "rows": n, "spans": n_spans, "pairs": n_pairs,
            "span_counts": span_counts(conn), "pair_counts": pair_counts(conn)}


def supervision_report(conn: sqlite3.Connection) -> str:
    """The line you want to see in `friday status` from Week 2 onward."""
    s, p = span_counts(conn), pair_counts(conn)
    return (
        f"RSC supervision: {s['positive']} positive / {s['negative']} negative spans "
        f"(w_t) · {p['total']} retraction pairs, {p['address_distinct']} address-distinct (e_t)"
    )


#: Phase 1 exit-test thresholds (docs/architecture/00 Phase 1).
SPAN_TARGET = 100
PAIR_TARGET = 20


def ready_for_phase_3_5(conn: sqlite3.Connection) -> tuple[bool, str]:
    s, p = span_counts(conn), pair_counts(conn)
    ok = s["positive"] >= SPAN_TARGET and p["total"] >= PAIR_TARGET
    msg = (
        f"spans {s['positive']}/{SPAN_TARGET}, pairs {p['total']}/{PAIR_TARGET}"
        + ("" if ok else "  ← keep collecting; you cannot backfill these")
    )
    return ok, msg
