"""Bi-temporal facts + reconciliation. Innovation #3.

Two independent time axes, and keeping them apart is the whole feature:

    WORLD time    valid_from / valid_to      when the fact was TRUE in the world
    BELIEF time   asserted_at / retracted_at when FRIDAY BELIEVED it

That separation is what makes both of these answerable with the same store:

    "What's my rent?"                  -> valid now, believed now
    "Where did I live last year?"      -> valid then, believed now
    "What did I used to think X was?"  -> valid then, RETRACTED now  (belief time)

Retraction is never deletion. A superseded fact keeps its row, gets
`retracted_at` set, and is struck through in the Markdown. Deleting it would
destroy the bi-temporal property and — separately — destroy the RSC's
`L_erase` training data (memory/supervision.py).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path

from .. import paths
from ..config import (
    CONFIDENCE_MARGIN,
    SOURCE_CONFIDENCE,
    SOURCE_RANK,
    canonical_predicate,
    is_single_valued,
    predicate_class_of,
)
from ..util import new_id, now_iso, parse_iso, partial_date
from . import mdfacts, supervision


@dataclass
class Fact:
    """An in-memory fact. `id` empty => not yet persisted."""

    subject: str
    predicate: str
    object: str
    #: Empty means "not yet routed". `assert_fact` resolves it via `domain_for()`,
    #: which picks the file this fact belongs in. Deliberately NOT defaulted to
    #: "general" — a non-empty default would make `fact.domain or domain_for(...)`
    #: always truthy, silently routing every fact into general.md and defeating the
    #: whole point of splitting memory/facts/ by domain.
    domain: str = ""
    source_kind: str = "stated"
    confidence: float | None = None
    salience: float = 0.5
    valid_from: str | None = None
    valid_to: str | None = None
    asserted_at: str | None = None
    retracted_at: str | None = None
    superseded_by: str | None = None
    source_refs: list[str] | None = None
    source_quote: str | None = None
    section: str = "Facts"
    id: str = ""
    # provenance for supervision
    trace_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        # ⭐ THE SINGLE CHOKE POINT for predicate canonicalisation. Every fact
        # arrives as a `Fact` — from the CLI, from a Markdown line, from an agent
        # tool call, from an import — so normalising here covers every path at once.
        # Doing it in `assert_fact` instead would miss facts that are only ever
        # compared rather than written, and doing it at each call site would miss
        # the next call site somebody adds.
        self.predicate = canonical_predicate(self.predicate)

    @property
    def predicate_class(self) -> str:
        return predicate_class_of(self.predicate)

    @property
    def single_valued(self) -> bool:
        return is_single_valued(self.predicate)

    @property
    def subject_for_query(self) -> str:
        """Facts files are domain-scoped, so the subject is `user:<domain>`
        (or plain `user` for the general domain)."""
        return f"user:{self.domain}" if self.domain and self.domain != "general" else "user"

    @property
    def effective_confidence(self) -> float:
        if self.confidence is not None:
            return self.confidence
        return SOURCE_CONFIDENCE.get(self.source_kind, 0.5)

    def to_line(self) -> mdfacts.FactLine:
        return mdfacts.FactLine(
            id=self.id or new_id("f"),
            predicate=self.predicate,
            object=self.object,
            domain=self.domain,
            source_kind=self.source_kind,
            confidence=self.effective_confidence,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            retracted=bool(self.retracted_at),
            retracted_at=self.retracted_at,
            superseded_by=self.superseded_by,
            section=self.section,
            source_quote=self.source_quote,
            source_refs=list(self.source_refs or []),
        )


# ── domain routing ─────────────────────────────────────────────────────────────

_DOMAIN_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("lease", "rent", "landlord", "address", "lives_in", "locality", "housing",
      "residence", "apartment", "flat"), "housing"),
    (("employer", "works_at", "role", "salary", "colleague", "manager", "job"), "work"),
    (("project", "deadline", "due", "current_task", "working_on", "todo"), "projects"),
    (("device", "os", "hardware", "ram", "laptop", "phone_model", "timezone"), "environment"),
    (("prefers", "preference", "likes", "dislikes", "diet", "allerg"), "preferences"),
    (("mood", "feeling", "stressed", "health", "sleep"), "state"),
    (("relation", "friend", "family", "amma", "appa", "brother", "sister"), "people"),
    (("name", "dob", "birth", "gender", "nationality", "native"), "identity"),
    (("bank", "account", "upi", "card", "tax", "pan"), "finance"),
]


def domain_for(predicate: str, text: str = "") -> str:
    blob = f"{predicate} {text}".lower()
    for keys, dom in _DOMAIN_HINTS:
        if any(k in blob for k in keys):
            return dom
    return "general"


def facts_path(domain: str) -> Path:
    return paths.FACTS / f"{domain}.md"


# ── the write path ─────────────────────────────────────────────────────────────

def assert_fact(
    conn: sqlite3.Connection,
    fact: Fact,
    *,
    reconcile: bool = True,
) -> dict:
    """Persist a fact to Markdown (truth) and SQLite (index), reconciling conflicts.

    Returns a report: what was written, what was retracted, which won, and what
    supervision it produced. The report matters — FRIDAY must be able to *tell the
    user* what it changed, because silent memory mutation is how you lose trust in a
    personal agent.

    `action` is one of:

        new         no live fact for this subject+predicate; written
        reconciled  a live fact was superseded and RETRACTED (not deleted)
        nochange    the same value was asserted again; reinforced, not rewritten
        rejected    the incumbent outranks this assertion; nothing was written

    Ordering matters here and is not arbitrary. The supervision pair references
    `span_new_id`, so the pair row cannot be inserted before the new fact's span
    exists — `PRAGMA foreign_keys=ON` will (correctly) refuse it. Hence: reconcile
    (compute + retract) -> write Markdown -> index the fact -> record the span ->
    THEN insert the pairs.
    """
    fact.domain = fact.domain or domain_for(fact.predicate, fact.object)
    if not fact.id:
        fact.id = new_id("f")
    fact.asserted_at = fact.asserted_at or now_iso()

    pending_pairs: list[supervision.RetractionPair] = []
    retracted: list[dict] = []
    action = "new"

    if reconcile:
        blocked = _blocked_by(conn, fact)
        if blocked is not None:
            return {
                "action": "rejected",
                "written": None,
                "retracted": [],
                "supervision": {"span_id": None, "pair": None},
                "conflict_resolved": False,
                "message": _rejection_message(fact, blocked),
                "blocked_by": dict(blocked),
            }
        retracted, pending_pairs, action = _reconcile(conn, fact)

    if action == "nochange":
        conn.commit()
        return {
            "action": "nochange",
            "written": {"id": fact.id, "predicate": fact.predicate, "object": fact.object,
                        "domain": fact.domain, "file": None},
            "retracted": [],
            "supervision": {"span_id": None, "pair": None},
            "conflict_resolved": False,
            "message": (f"{fact.predicate} = {fact.object} is already recorded; "
                        f"reinforced its access count instead of rewriting it."),
        }

    path = facts_path(fact.domain)
    _write_markdown(path, fact, retracted)
    _index_fact(conn, fact, path)
    _link_superseded(conn, fact.id, retracted)

    # ⭐ SALIENCE TAP — free supervision for the RSC write gate (w_t).
    #
    # Cannot be backfilled, so it fires on EVERY path a fact can arrive by:
    #
    #   1. trace-anchored — a fact extracted from a conversation. Offsets point into
    #      a trace being written right now; traces are gitignored, so this is the
    #      only chance to record where it came from.
    #   2. inline — a fact with a source_quote but no trace (CLI `friday write
    #      --quote`, a hand-typed Markdown line, an import). There is no file to
    #      anchor to, so the quote is stored on the span itself.
    #
    # Path 2 matters more than it looks: hand-authored facts are the
    # highest-confidence population in the store (Law 7), so they are the most
    # valuable examples for the write gate to learn from. Dropping them silently
    # would bias w_t toward conversational phrasing only.
    span_id = None
    if fact.trace_id and fact.char_start is not None and fact.char_end is not None:
        span_id = supervision.record_span(
            conn,
            supervision.SpanLabel(
                trace_id=fact.trace_id,
                trace_file=str(paths.TRACES / f"{fact.trace_id[:10]}.jsonl"),
                char_start=fact.char_start,
                char_end=fact.char_end,
                label=1,
                fact_id=fact.id,
                domain=fact.domain,
            ),
        )
    elif fact.source_quote:
        quote = fact.source_quote.strip()
        span_id = supervision.record_span(
            conn,
            supervision.SpanLabel(
                trace_id=fact.trace_id or f"inline:{fact.id}",
                trace_file="",
                char_start=0,
                char_end=len(quote),
                label=1,
                fact_id=fact.id,
                domain=fact.domain,
                text=quote,
            ),
        )

    # ⭐ RETRACTION TAP — free supervision for the EDA erase address (e_t).
    # Inserted LAST, because span_new_id must exist by then.
    pair_report = None
    for pair in pending_pairs:
        pair.span_new_id = span_id
        pid = supervision.record_pair(conn, pair)
        if pair_report is None:
            pair_report = {
                "id": pid,
                "predicate": fact.predicate,
                "value_old": pair.value_old,
                "value_new": pair.value_new,
                "key_old": pair.key_old,
                "key_new": pair.key_new,
                "keys_distinct": pair.key_old != pair.key_new,
            }
    conn.commit()

    return {
        "action": action,
        "written": {"id": fact.id, "predicate": fact.predicate, "object": fact.object,
                    "domain": fact.domain, "file": str(path.relative_to(paths.ROOT))},
        "retracted": retracted,
        "supervision": {"span_id": span_id, "pair": pair_report},
        "conflict_resolved": bool(retracted),
        "message": _report(fact, retracted),
    }


def _incumbents(conn: sqlite3.Connection, new: Fact) -> list:
    """Every LIVE fact competing with `new` for the same slot.

    ⚠️ Matches on the CANONICAL predicate, not on string equality. A store written
    before canonicalisation — or a hand-edited Markdown line that says
    `monthly_rent:` while the seed says `lease_amount_monthly:` — holds the older
    spelling, and an exact-match query would not see it. Missing it means both facts
    stay live and "what is my rent" has two answers. `new.predicate` is already
    canonical (`Fact.__post_init__`), so only the stored side needs mapping.
    """
    rows = conn.execute(
        """SELECT * FROM facts
           WHERE subject=? AND retracted_at IS NULL""",
        (new.subject_for_query,),
    ).fetchall()
    want = new.predicate
    return [r for r in rows if canonical_predicate(r["predicate"]) == want]


def _incumbent(conn: sqlite3.Connection, new: Fact):
    """The most recently asserted competing live fact, or None."""
    rows = _incumbents(conn, new)
    if not rows:
        return None
    return max(rows, key=lambda r: r["asserted_at"] or "")


def _blocked_by(conn: sqlite3.Connection, new: Fact) -> sqlite3.Row | None:
    """The guard that stops a hallucination from rewriting your address.

    Ranking, from docs/architecture/03 §3 — and `SOURCE_RANK` in config.py is the
    authority, this list is a mirror of it:

        user_edit (4)  >  observed (3)  >  imported (2)  >  stated (1)  >  inferred (0)

    ⚠️ `imported` outranks `stated`. An earlier draft of this docstring had those two
    the other way round, which reads plausible and is wrong: a bulk import of your own
    bank statement is better evidence about your rent than one passing remark in a
    conversation. If this list and SOURCE_RANK ever disagree, fix this list.

    A hand-edit is sacred (Law 7): nothing automatic may overwrite it, at any
    confidence, without asking. Below that, a higher-ranked source wins outright;
    within a rank, the higher confidence wins — but only by a margin, so that a
    stream of marginally-confident chatter cannot slowly walk a fact somewhere the
    user never said.
    """
    rank = SOURCE_RANK
    new_rank = rank.get(new.source_kind, 0)
    new_conf = new.effective_confidence

    row = _incumbent(conn, new)
    if row is None:
        return None
    if _same_value(row["object"], new.object) and not _is_negation(new.object):
        return None                     # reinforcement, handled by _reconcile

    old_rank = rank.get(row["source_kind"], 0)
    old_conf = row["confidence"] if row["confidence"] is not None else 0.5

    if old_rank == rank["user_edit"] and new_rank != rank["user_edit"]:
        # ⭐ Never overwrite a hand-edit AUTOMATICALLY — but an explicit user_edit
        # may replace one, because it IS you saying so explicitly.
        #
        # The unconditional version of this check made the refusal message a lie. It
        # says "Say so explicitly if you want it changed", and then `friday write
        # --user-edit` — which is exactly saying so explicitly — got refused too. So
        # did editing the file and recompiling. Nothing could ever change a
        # hand-edited fact short of hand-deleting the Markdown line, which silently
        # turned the highest-authority source into a permanent, uncorrectable value.
        # For a personal agent whose rent, address and employer all change, that is
        # not a safety property; it is a bug wearing one.
        #
        # Same-rank falls through to the recency rule below, which is the right
        # semantic: your latest explicit statement supersedes your earlier one, and
        # `_reconcile` records the retraction pair that supervises the RSC erase
        # address e_t. A correction you make by hand is the highest-value training
        # signal in the whole system, so blocking it would discard the best data.
        return row
    if new_rank < old_rank:
        return row
    if new_rank == old_rank:
        # Same rank, same authority: RECENCY wins. "I moved to Chennai" after
        # "I live in Bengaluru" is a genuine change in the world, not a conflict to
        # be adjudicated — rejecting it would freeze every single-valued fact at its
        # first value forever, which is the opposite of the point.
        #
        # The margin only guards a WEAKER source trying to beat a stronger one on
        # confidence alone, which is the actual hallucination-creep failure mode.
        if new_conf + CONFIDENCE_MARGIN < old_conf:
            return row
        return None
    return None


def _rejection_message(fact: Fact, incumbent: sqlite3.Row) -> str:
    if incumbent["source_kind"] == "user_edit":
        return (
            f"Not overwriting {fact.predicate}: the current value "
            f"'{incumbent['object']}' was hand-edited by you, and a hand-edit is "
            f"sacred (Law 7). Say so explicitly if you want it changed."
        )
    return (
        f"Not overwriting {fact.predicate} = '{incumbent['object']}': that came from "
        f"{incumbent['source_kind']} (confidence {incumbent['confidence']}), which "
        f"outranks this {fact.source_kind} assertion at {fact.effective_confidence:.2f}."
    )


def _report(fact: Fact, retracted: list[dict]) -> str:
    if not retracted:
        return f"Recorded {fact.predicate} = {fact.object}."
    old = retracted[0]
    return (
        f"Recorded {fact.predicate} = {fact.object}. "
        f"This supersedes {fact.predicate} = {old['object']} "
        f"(believed {old['asserted_at'][:10] if old.get('asserted_at') else '?'}); "
        f"the old value is retracted, not deleted — 'what did I used to think?' still works."
    )


def _reconcile(conn: sqlite3.Connection, new: Fact) -> tuple[list[dict], list, str]:
    """Bi-temporal reconciliation. Returns (retracted, pending_pairs, action).

    Two cases:

      * SINGLE-VALUED predicate (lives_in, lease_amount_monthly, employer, …):
        only one can be true at a time, so any live fact with the same
        subject+predicate is RETRACTED and superseded_by the new one.

      * MULTI-VALUED predicate (prefers, friend_of, project_uses):
        an exact object match is a no-op reinforcement (bump access_count — the
        decay function's reinforcement term does the rest). A *contradiction*
        ("I don't like X" after "likes X") retracts by negation detection.

    Pairs are RETURNED, not inserted. `supervision_pairs.span_new_id` is a foreign
    key into `supervision_spans`, and the new fact's span does not exist yet at this
    point — inserting here fails the FK constraint. The caller inserts them after
    the span is recorded.

    A repeated assertion of the same value is `nochange`, not a retraction. Emitting
    a pair for it would teach the erase gate to fire on repetition, which is exactly
    backwards: saying the same thing twice is evidence, not a correction.
    """
    retracted: list[dict] = []
    pairs: list = []
    reinforced = 0
    rows = _incumbents(conn, new)

    for r in rows:
        same_object = _same_value(r["object"], new.object)
        if same_object and not _is_negation(new.object):
            # reinforcement, not a conflict
            conn.execute(
                "UPDATE facts SET access_count=access_count+1, last_accessed=? WHERE id=?",
                (now_iso(), r["id"]),
            )
            reinforced += 1
            continue
        if new.single_valued or _is_negation(new.object) or same_object:
            when = new.valid_from or _today()
            # NOTE: `superseded_by` is NOT set here. It is a foreign key into
            # facts(id), and the new fact has not been inserted yet — setting it now
            # fails the FK constraint (correctly). The caller links the pointer after
            # _index_fact(); see _link_superseded().
            conn.execute(
                """UPDATE facts SET retracted_at=?, valid_to=COALESCE(valid_to, ?)
                   WHERE id=?""",
                (now_iso(), when, r["id"]),
            )
            retracted.append(
                {"id": r["id"], "object": r["object"], "asserted_at": r["asserted_at"],
                 "predicate": r["predicate"], "valid_from": r["valid_from"]}
            )
            _strike_through_markdown(r, when, new.id)
            _index_retraction(conn, r["id"])

            # ⭐ RETRACTION TAP — free supervision for the EDA erase address (e_t).
            pairs.append(
                supervision.RetractionPair(
                    fact_old_id=r["id"],
                    fact_new_id=new.id,
                    key_old=_address_key(r["predicate"], r["object"]),
                    key_new=_address_key(new.predicate, new.object),
                    value_old=r["object"],
                    value_new=new.object,
                    trace_old_id=_trace_of(conn, r["id"]),
                    trace_new_id=new.trace_id,
                )
            )

    # action semantics:
    #   reconciled — something was retracted, so this is a correction
    #   nochange   — every live row said the same thing; reinforced, nothing written
    #   new        — no live rows at all
    if retracted:
        action = "reconciled"
    elif rows and reinforced == len(rows):
        action = "nochange"
    else:
        action = "new"
    return retracted, pairs, action


def _address_key(predicate: str, value: str) -> str:
    """The 'address' a fact occupies in the recurrent state.

    Deliberately derived from predicate + a normalised hint of the value, so that
    "I live at 12 Marina Road" and "I moved to 48 Velachery Main" produce
    DIFFERENT keys — which is exactly the case GDN-2's write-anchored erase cannot
    reach and EDA's independent erase address can (docs/architecture/13 §2.1).
    """
    head = " ".join(value.lower().split()[:2])
    return f"{predicate.lower()}::{head}"


def _trace_of(conn: sqlite3.Connection, fact_id: str) -> str | None:
    row = conn.execute(
        "SELECT trace_id FROM supervision_spans WHERE fact_id=? LIMIT 1", (fact_id,)
    ).fetchone()
    return row["trace_id"] if row else None


def _same_value(a: str, b: str) -> bool:
    na = "".join(c for c in a.lower() if c.isalnum())
    nb = "".join(c for c in b.lower() if c.isalnum())
    return na == nb


_NEG = ("not ", "no ", "never ", "don't", "dont ", "isn't", "isnt ", "stopped ", "quit ", "hate ")


def _is_negation(value: str) -> bool:
    v = value.lower().strip()
    return any(v.startswith(n) or n in v[:24] for n in _NEG)


def _today() -> str:
    return date.today().isoformat()


# ── Markdown write (truth) ─────────────────────────────────────────────────────

def _write_markdown(path: Path, fact: Fact, retracted: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ff = mdfacts.parse_file(path) if path.exists() else mdfacts.FactsFile(path=path, domain=fact.domain)

    line = fact.to_line()
    line.section = fact.section or _section_for(fact.predicate)

    replaced = False
    for i, ex in enumerate(ff.facts):
        if ex.id == line.id or (
            ex.predicate == line.predicate and not ex.retracted and fact.single_valued
        ):
            ff.facts[i] = line
            replaced = True
            break
    if not replaced:
        ff.facts.append(line)

    # reflect retractions in the Markdown too — struck through, never removed
    for r in retracted:
        for i, ex in enumerate(ff.facts):
            if ex.id == r["id"] and not ex.retracted:
                ex.retracted = True
                ex.retracted_at = _today()
                ex.superseded_by = fact.id
                ex.valid_to = ex.valid_to or r.get("valid_from")
                ff.facts[i] = ex

    ff.last_compiled = now_iso()
    path.write_text(mdfacts.render_file(ff), encoding="utf-8")


def _strike_through_markdown(row: sqlite3.Row, when: str, superseded_by: str) -> None:
    """Mark a fact retracted in its origin Markdown file, in place."""
    origin = Path(row["origin_file"])
    if not origin.is_absolute():
        origin = paths.ROOT / origin
    if not origin.exists():
        return
    ff = mdfacts.parse_file(origin)
    changed = False
    for f in ff.facts:
        if f.id == row["id"] and not f.retracted:
            f.retracted = True
            f.retracted_at = when
            f.superseded_by = superseded_by
            f.valid_to = f.valid_to or partial_date(row["valid_from"])
            changed = True
    if changed:
        ff.last_compiled = now_iso()
        origin.write_text(mdfacts.render_file(ff), encoding="utf-8")


def _section_for(predicate: str) -> str:
    klass = predicate_class_of(predicate)
    return {
        "identity": "Identity",
        "location": "Where",
        "environment": "Environment",
        "preference": "Preferences",
        "relationship": "People",
        "mood": "Current state",
        "current_task": "Right now",
        "project_state": "Projects",
    }.get(klass, "Facts")


# ── SQLite index (artifact) ────────────────────────────────────────────────────

def _index_fact(conn: sqlite3.Connection, fact: Fact, path: Path) -> None:
    from ..store import db

    text = path.read_text(encoding="utf-8")
    fhash = __import__("hashlib").sha256(text.encode()).hexdigest()
    conn.execute(
        """INSERT OR REPLACE INTO facts(
             id, subject, predicate, object, object_type, predicate_class, single_valued,
             confidence, salience, valid_from, valid_to, asserted_at, retracted_at,
             superseded_by, source_kind, source_refs, source_quote, review_after,
             access_count, last_accessed, origin_file, origin_hash, origin_line)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            fact.id, fact.subject_for_query, fact.predicate, fact.object, "str",
            fact.predicate_class, int(fact.single_valued),
            fact.effective_confidence, fact.salience,
            fact.valid_from, fact.valid_to, fact.asserted_at, fact.retracted_at,
            fact.superseded_by, fact.source_kind,
            json.dumps(fact.source_refs or []), fact.source_quote, None,
            0, None, str(path), fhash, None,
        ),
    )
    # Aliases are indexed here too, not only in the compiler: a fact asserted live
    # from conversation must be as findable as one compiled from Markdown, or the
    # two write paths produce memories with different recall.
    from ..config import aliases_for

    body = f"{fact.predicate} {fact.object} {' '.join(aliases_for(fact.predicate))}".strip()
    if fact.source_quote:
        body += f" {fact.source_quote}"
    db.fts_upsert(conn, fact.id, "fact", body, subject=fact.subject_for_query)


def _link_superseded(conn: sqlite3.Connection, new_id_: str, retracted: list[dict]) -> None:
    """Point each retracted fact at its replacement.

    Split out from `_reconcile` because of ordering: `superseded_by` is a foreign
    key into `facts(id)`, so it can only be written once the new row exists.
    Retraction itself happens earlier (the Markdown strike-through must be part of
    the same write that records the new value), so the pointer is the last step.
    """
    if not retracted:
        return
    conn.executemany(
        "UPDATE facts SET superseded_by=? WHERE id=?",
        [(new_id_, r["id"]) for r in retracted],
    )


def _index_retraction(conn: sqlite3.Connection, fact_id: str) -> None:
    """A retracted fact leaves the auto-inject set but STAYS in FTS: it must remain
    answerable for 'what did I used to think?' — that is the bi-temporal promise."""
    row = conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
    if not row:
        return
    from ..store import db

    db.fts_upsert(
        conn, fact_id, "fact",
        f"{row['predicate']} {row['object']} (retracted, previously believed)",
        subject=row["subject"],
    )


# ── reads ──────────────────────────────────────────────────────────────────────

def current_facts(conn: sqlite3.Connection, subject: str | None = None, limit: int = 200) -> list[sqlite3.Row]:
    q = "SELECT * FROM facts WHERE retracted_at IS NULL"
    args: list = []
    if subject:
        q += " AND subject=?"
        args.append(subject)
    q += " ORDER BY confidence DESC, asserted_at DESC LIMIT ?"
    args.append(limit)
    return conn.execute(q, args).fetchall()


def _day(s: str | None) -> date | None:
    """Truncate any stored temporal value to a calendar day.

    Used for WORLD time only — see `_instant` for why BELIEF time must not be
    truncated.
    """
    dt = parse_iso(s)
    return dt.date() if dt else None


def _pad_partial(s: str | None, *, end: bool = False) -> str | None:
    """Pad a partial date to a full-day boundary so TEXT comparison is correct.

    ⚠️ This is not cosmetic. The store legitimately holds three granularities side
    by side:

        valid_from    "2023-06-01"                          (Markdown, day)
        asserted_at   "2026-09-29T14:03:11.482913+05:30"    (now_iso(), microseconds)
        review_after  "2026-10"                             (partial, month)

    Comparing those as TEXT is wrong in a way that fails silently: a bare date is a
    strict string prefix of a full timestamp on the same day, so
    `"2026-09-29" <= "2026-09-29T14:03:11+05:30"` is True but the reverse is False,
    and an interval query returns [] instead of the fact.

    A partial date is padded to the boundary that matches how people mean it:
    "valid from 2023-06" starts at the beginning of June, and "valid to 2025-05"
    runs to the END of May — nobody who writes that means "until midnight on the
    1st". Values that are already full timestamps pass through untouched, so
    belief time keeps its precision.
    """
    if not s:
        return None
    t = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t + ("T23:59:59.999999" if end else "T00:00:00")
    if re.fullmatch(r"\d{4}-\d{2}", t):
        y, m = int(t[:4]), int(t[5:7])
        if end:
            nm = date(y + (m == 12), (m % 12) + 1, 1)
            last = nm - timedelta(days=1)
            return last.isoformat() + "T23:59:59.999999"
        return f"{t}-01T00:00:00"
    if re.fullmatch(r"\d{4}", t):
        return (f"{t}-12-31T23:59:59.999999" if end else f"{t}-01-01T00:00:00")
    return t


def _instant(s: str | None) -> datetime | None:
    """Full-precision instant, for BELIEF time.

    `asserted_at` and `retracted_at` are always machine-generated by `now_iso()`,
    so they are always full timestamps — and truncating them to a day is actively
    wrong. If FRIDAY learns a fact at 14:00 and retracts it at 14:05, then at
    14:02 it believed that fact. Day granularity collapses the whole episode to
    "never believed", and `beliefs_at` silently returns [] — which breaks the one
    query that makes retraction-never-deletion worth having.

    World time gets day granularity (`_day`) because "where did I live in 2022" is
    never a sub-day question and the Markdown truth cannot express one. Belief time
    gets instants because it is generated by a clock.
    """
    return parse_iso(_pad_partial(s, end=True))


def _valid_on(row: sqlite3.Row, when: str) -> bool:
    """WORLD time: was this fact true at `when`? An open end means unbounded."""
    d = _day(when) or date.today()
    vf = _day(row["valid_from"])
    vt = _day(row["valid_to"])
    return (vf is None or vf <= d) and (vt is None or vt >= d)


def _believed_on(row: sqlite3.Row, when: str) -> bool:
    """BELIEF time: did FRIDAY hold this belief at instant `when`?

    Distinct from `_valid_on` and must stay distinct — the two axes are allowed to
    disagree, and that disagreement is the entire feature. You can be told today
    about a move that happened last year: true-then, believed-now.
    """
    w = _instant(when) or datetime.now().astimezone()
    a = _instant(row["asserted_at"])
    if a is not None and a > w:
        return False
    r = _instant(row["retracted_at"])
    return r is None or r > w


def facts_at(
    conn: sqlite3.Connection,
    when: str,
    subject: str | None = None,
) -> list[sqlite3.Row]:
    """Time travel in WORLD time: what was true on `when`?"""
    q = "SELECT * FROM facts"
    args: list = []
    if subject:
        q += " WHERE subject=?"
        args.append(subject)
    rows = conn.execute(q + " ORDER BY confidence DESC", args).fetchall()
    return [r for r in rows if _valid_on(r, when)]


def beliefs_at(
    conn: sqlite3.Connection,
    when: str,
    subject: str | None = None,
) -> list[sqlite3.Row]:
    """Time travel in BELIEF time: what did FRIDAY believe on `when`?

    This is the query that makes "what did I used to think?" answerable, and it is
    the reason retraction is never deletion. Note it does NOT filter on
    `retracted_at IS NULL` — a retracted fact was still believed up to that moment,
    which is exactly what a belief-time query has to surface.
    """
    q = "SELECT * FROM facts"
    args: list = []
    if subject:
        q += " WHERE subject=?"
        args.append(subject)
    rows = conn.execute(q + " ORDER BY confidence DESC", args).fetchall()
    return [r for r in rows if _believed_on(r, when)]


def history_of(conn: sqlite3.Connection, predicate: str, subject: str | None = None) -> list[sqlite3.Row]:
    """Every value this predicate has ever held, NEWEST first, with both time axes.

    Newest-first because the two questions this serves are "what's current?" and
    "what did I used to think?" — in both, the most recent value is the answer and
    the older ones are the trail. Sorting on `valid_from` alone would be wrong for a
    fact learned late about an old state of the world, so fall back to belief time.
    """
    # ⭐ Match on the CANONICAL predicate, on BOTH sides. You can write with an alias
    # (`friday write monthly_rent …`) because `Fact.__post_init__` canonicalises it,
    # so looking the trail up with that same alias has to work too — otherwise the
    # CLI tells you it renamed the predicate and then claims no history exists for
    # either name. The stored side needs mapping because rows written before
    # canonicalisation, or hand-edited Markdown lines, keep the older spelling.
    want = canonical_predicate(predicate)
    q = "SELECT * FROM facts"
    args: list = []
    if subject:
        q += " WHERE subject=?"
        args.append(subject)
    rows = [r for r in conn.execute(q, args).fetchall()
            if canonical_predicate(r["predicate"]) == want]
    return sorted(rows, key=lambda r: (_day(r["valid_from"]) or _day(r["asserted_at"])
                                       or date.min), reverse=True)



