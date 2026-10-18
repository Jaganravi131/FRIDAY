"""⭐ Innovation 9b — the Memory Compiler IS the gate supervision.

docs/architecture/13 §4.1. GDN-2/EDA expose three gate branches; FRIDAY's compiler
already computes three matching signals, for unrelated reasons, since Phase 1:

    channel-wise decay  alpha_t  <-  the DECAY FUNCTION     (memory/decay.policy_alpha)
    independent erase   e_t      <-  RETRACTION PAIRS       (memory/supervision)
    channel-wise write  w_t      <-  the SALIENCE HEAD      (memory/supervision)

Retractions yield (key_old, key_new) with key_old != key_new — free supervision for
exactly the mechanism EDA adds, from a schema designed in docs/architecture/03 for a
completely different purpose.

This module builds training batches out of the SQLite tables. It is the bridge
between the memory system (which decides) and the operator (which needs to be told
what deciding looked like).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

from ..memory.decay import policy_alpha
from ..memory.supervision import pair_counts, span_counts


@dataclass
class Batch:
    """One RSC training example, assembled from FRIDAY's own records."""

    history_text: str
    span_labels: list[int]              # per-token/per-span salience targets -> w_t
    span_bounds: list[tuple[int, int]]
    retraction_pairs: list[dict]        # (key_old, key_new, value_old, value_new) -> e_t
    alpha_targets: list[float]          # per predicate class -> alpha_t
    alpha_classes: list[str]
    needles: list[dict]                 # {"query","value","depth"} -> L_recall
    meta: dict[str, Any]

    @property
    def n_positive(self) -> int:
        return sum(1 for y in self.span_labels if y)

    @property
    def n_address_distinct(self) -> int:
        return sum(1 for p in self.retraction_pairs if p.get("address_distinct"))


# ── decay targets ──────────────────────────────────────────────────────────────

def alpha_targets(conn: sqlite3.Connection, limit: int = 64) -> tuple[list[str], list[float]]:
    """Per-fact channel-wise decay targets, straight from the hand-written policy.

    A fact's predicate_class determines its half-life; `policy_alpha` converts that
    into a per-token retention. This is what `L_decay` regresses alpha_t toward
    before the curriculum weight anneals to zero.
    """
    rows = conn.execute(
        """SELECT predicate_class, asserted_at FROM facts
           WHERE retracted_at IS NULL ORDER BY confidence DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    classes = [r["predicate_class"] for r in rows]
    alphas = [policy_alpha(c, None) for c in classes]
    return classes, alphas


def alpha_for_class(predicate_class: str) -> float:
    return policy_alpha(predicate_class, None)


# ── span labels (w_t) ──────────────────────────────────────────────────────────

def span_batches(conn: sqlite3.Connection, *, limit: int = 256,
                 negative_ratio: float = 3.0) -> list[dict]:
    """Positive spans (produced a fact) plus sampled negatives.

    Negatives are sampled, not exhaustively taken: almost every span in a
    conversation fails to produce a fact, so an unsampled negative set would be
    ~99% of the corpus and the head would learn "always 0". `negative_ratio` caps
    the imbalance, and `losses.l_salience` additionally class-balances the BCE.
    """
    pos = conn.execute(
        """SELECT * FROM supervision_spans WHERE label=1
           ORDER BY created_at DESC LIMIT ?""", (limit,),
    ).fetchall()
    neg = conn.execute(
        """SELECT * FROM supervision_spans WHERE label=0
           ORDER BY RANDOM() LIMIT ?""", (int(limit * negative_ratio),),
    ).fetchall()
    return [dict(r) for r in (*pos, *neg)]


def sample_negatives(conn: sqlite3.Connection, writer, *, per_positive: int = 3) -> int:
    """Mint negative labels from spans that produced NO fact.

    Called by the nightly Dreaming job. Without it `l_salience` has no negatives and
    the head is untrainable. Cheap: pick random char windows from trace files that
    don't intersect a positive span.
    """
    import random

    from ..memory.supervision import SpanLabel, record_span

    pos_rows = conn.execute(
        "SELECT trace_file, char_start, char_end FROM supervision_spans WHERE label=1"
    ).fetchall()
    if not pos_rows:
        return 0
    by_file: dict[str, list[tuple[int, int]]] = {}
    for r in pos_rows:
        by_file.setdefault(r["trace_file"], []).append((r["char_start"], r["char_end"]))

    rng = random.Random(0)
    made = 0
    for trace_file, spans in by_file.items():
        from pathlib import Path

        p = Path(trace_file)
        if not p.is_absolute():
            from .. import paths

            p = paths.ROOT / p
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        n = len(text)
        if n < 240:
            continue
        for _ in range(per_positive * len(spans)):
            width = rng.randint(80, 240)
            start = rng.randint(0, max(0, n - width))
            end = start + width
            if any(not (end <= s or start >= e) for s, e in spans):
                continue  # overlaps a positive; skip
            record_span(conn, SpanLabel(
                trace_id=f"file:{p.stem}", trace_file=str(trace_file),
                char_start=start, char_end=end, label=0,
            ))
            made += 1
    conn.commit()
    return made


# ── retraction pairs (e_t) ─────────────────────────────────────────────────────

def pair_batches(conn: sqlite3.Connection, *, limit: int = 256,
                 address_distinct_first: bool = True) -> list[dict]:
    """Retraction pairs, address-distinct first.

    Ordering matters for the ablation: rung 4 (GDN-2) and rung 5 (EDA) differ ONLY
    on address-distinct pairs. If a batch is mostly address-identical, the two rungs
    look the same and you learn nothing.
    """
    order = "address_distinct DESC, created_at DESC" if address_distinct_first else "created_at DESC"
    rows = conn.execute(
        f"SELECT * FROM supervision_pairs ORDER BY {order} LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── needles (L_recall) ─────────────────────────────────────────────────────────

def needles_from_facts(conn: sqlite3.Connection, *, limit: int = 32) -> list[dict]:
    """Turn real facts into recall probes.

    Better than synthetic needles for training, because the phrasing is yours:
    "what's my rent" against lease_amount_monthly is the query FRIDAY will actually
    get. `scripts/needle_test.py` still owns the *evaluation* needles (synthetic,
    controlled depth) — training and eval must not share examples.
    """
    rows = conn.execute(
        """SELECT predicate, object, valid_from, asserted_at FROM facts
           WHERE retracted_at IS NULL ORDER BY access_count DESC, confidence DESC
           LIMIT ?""", (limit,),
    ).fetchall()
    out = []
    for i, r in enumerate(rows):
        pred = r["predicate"].replace("_", " ")
        out.append({
            "query": f"what is my {pred}?",
            "value": r["object"],
            # synthetic depth: spread across the window so the loss can't be
            # minimised by only remembering recent tokens
            "depth": 500 + i * 250,
            "predicate": r["predicate"],
        })
    return out


# ── assembly ───────────────────────────────────────────────────────────────────

def build_batch(conn: sqlite3.Connection, writer=None, *, span_limit: int = 128,
                pair_limit: int = 64) -> Batch:
    """Everything the five-term loss needs, from one query set."""
    spans = span_batches(conn, limit=span_limit)
    pairs = pair_batches(conn, limit=pair_limit)
    classes, alphas = alpha_targets(conn)
    needles = needles_from_facts(conn)

    bounds = [(int(s["char_start"]), int(s["char_end"])) for s in spans]
    labels = [int(s["label"]) for s in spans]
    history = _join_span_text(conn, spans, writer)

    return Batch(
        history_text=history,
        span_labels=labels,
        span_bounds=bounds,
        retraction_pairs=pairs,
        alpha_targets=alphas,
        alpha_classes=classes,
        needles=needles,
        meta={"spans": span_counts(conn), "pairs": pair_counts(conn)},
    )


def _join_span_text(conn: sqlite3.Connection, spans: Sequence[dict], writer=None) -> str:
    """Resolve spans back to the exact characters they label.

    Two provenance modes, mirroring `supervision.SpanLabel`:

      * TRACE-ANCHORED — only the (file, start, end) triple is stored and the text is
        read back on demand. Traces are never committed and never copied into the DB,
        which keeps the git repo free of ambient capture while still giving the
        trainer exact substrings.
      * INLINE — `trace_file` is NULL and the text lives in `span_text`. These are
        facts that arrived with a quote but no trace: CLI writes, hand-typed Markdown
        lines, imports. Passing NULL to `Path()` raises TypeError, and since the
        compiler now records inline spans for every Markdown-sourced fact, that was
        not a hypothetical — it broke `build_batch` on any freshly seeded store.

    A missing or pruned trace contributes nothing rather than raising: losing a few
    spans to trace rotation is normal, whereas a trainer that crashes on a pruned
    trace is not.
    """
    if writer is None:
        from ..memory.traces import TraceWriter

        writer = TraceWriter()
    chunks = []
    for s in spans:
        inline = s.get("span_text")
        if inline:
            chunks.append(inline)
            continue
        tf = s.get("trace_file")
        if not tf:
            continue
        try:
            t = writer.span_text(tf, int(s["char_start"]), int(s["char_end"]))
        except (OSError, TypeError, ValueError):
            continue
        if t:
            chunks.append(t)
    return "\n".join(chunks)


def readiness_report(conn: sqlite3.Connection) -> str:
    """The line that tells you whether Phase 3.5 has fuel."""
    s, p = span_counts(conn), pair_counts(conn)
    classes, _ = alpha_targets(conn, limit=1)
    return (
        f"RSC supervision — w_t: {s['positive']}+/{s['negative']}- spans · "
        f"e_t: {p['total']} pairs ({p['address_distinct']} address-distinct) · "
        f"alpha_t: policy over {len(classes) or 'n/a'} classes · "
        f"{'READY' if s['positive'] >= 100 and p['total'] >= 20 else 'NOT READY — keep collecting'}"
    )
