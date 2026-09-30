"""Markdown truth -> SQLite indices. Idempotent. Incremental.

"The single most important piece of code in Phase 0" — docs/architecture/10 Day 3.

The rebuild property is the whole point of Innovation #2, and it must be verified
on Day 3, not in month 3:

    rm -rf artifacts && python -m friday build && python -m friday ask "what's my rent?"

If that works, your entire memory is a git repo and `git checkout` is an undo
button for your agent's beliefs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..config import SOURCE_CONFIDENCE, aliases_for, is_single_valued, predicate_class_of
from ..store import db
from . import mdfacts, supervision, traces


@dataclass
class CompileStats:
    files: int = 0
    facts: int = 0
    skipped_unchanged: int = 0
    retracted: int = 0
    episodes: int = 0
    skills: int = 0
    daily: int = 0
    traces: int = 0
    fts_rows: int = 0
    vec_rows: int = 0
    #: facts written by THIS run, as opposed to `facts` (the store total). The
    #: distinction matters because `compile_all` is incremental: on a no-op run
    #: every file is skipped, so "wrote 0 facts" and "the store has 840 facts" are
    #: both true and reporting only the first one looks like a failed build.
    facts_written: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def summary(self) -> str:
        wrote = (f"wrote {self.facts_written} new" if self.facts_written
                 else "nothing to write")
        return (
            f"{self.facts} facts in store from {self.files} files "
            f"({self.skipped_unchanged} unchanged skipped, {wrote}, "
            f"{self.retracted} retracted); "
            f"{self.episodes} episodes, {self.skills} skills, {self.daily} daily, "
            f"{self.traces} traces; "
            f"index: {self.fts_rows} fts, {self.vec_rows} vec"
            + (f"; {len(self.errors)} errors" if self.errors else "")
        )


# ── facts ──────────────────────────────────────────────────────────────────────

def _origin_hash_matches(conn: sqlite3.Connection, path: Path, fhash: str) -> bool:
    """Has this exact content already been accounted for, by either record?

    ⚠️ BOTH records are consulted and EITHER one matching means "not stale". This is
    not belt-and-braces redundancy — the two are maintained by different writers:

      * `facts.origin_hash` is updated by `assert_fact` every time FRIDAY writes a
        fact. That is the entire loop-safety mechanism: without it, FRIDAY's own
        Markdown write would be detected as a hand-edit, recompiled, re-audited as
        the user, and could feed itself. Checking only origin_state regressed exactly
        this, and test_fridays_own_write_is_not_reported_as_a_hand_edit caught it.
      * `origin_state.origin_hash` is updated by `compile_file`, and is the only
        record that exists for a file that yields ZERO facts.

    A file FRIDAY just wrote matches on `facts`; a notes-only file matches on
    `origin_state`; a genuine hand-edit matches on neither.
    """
    try:
        row = conn.execute(
            "SELECT origin_hash FROM origin_state WHERE origin_file=?", (str(path),)
        ).fetchone()
        if row is not None and row["origin_hash"] == fhash:
            return True
    except sqlite3.OperationalError:            # table not created yet
        pass
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM facts WHERE origin_file=? AND origin_hash=?",
        (str(path), fhash),
    ).fetchone()
    return bool(row and row["n"] > 0)


def _record_origin(conn: sqlite3.Connection, path: Path, fhash: str, n: int) -> None:
    """Remember that this content was compiled, even if it produced no facts.

    Called on the zero-fact path too — that is the entire reason the table exists.
    """
    from datetime import datetime, timezone

    try:
        conn.execute(
            """INSERT INTO origin_state(origin_file, origin_hash, facts_derived, compiled_at)
               VALUES (?,?,?,?)
               ON CONFLICT(origin_file) DO UPDATE SET
                 origin_hash=excluded.origin_hash,
                 facts_derived=excluded.facts_derived,
                 compiled_at=excluded.compiled_at""",
            (str(path), fhash, n,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    except sqlite3.OperationalError:
        # An old database without the table must still compile. It loses incrementality
        # for zero-fact files only, which is the pre-fix behaviour, not a new failure.
        pass


def compile_file(path: Path, conn: sqlite3.Connection, *, embedder=None) -> int:
    """Parse one Markdown facts file -> upsert rows. Returns count written.

    Incremental: if (origin_file, origin_hash) is unchanged we skip the file
    entirely. A hand-edit changes the hash and re-derives *only that file*.

    ⭐ The unchanged check reads `origin_state`, NOT `facts`. Counting fact rows looks
    equivalent until a file yields none: a notes-only .md in memory/facts/ would then
    have COUNT(*)==0 forever, be treated as stale on every turn, and write an audit row
    for a hand-edit that never happened. `origin_state` records the hash of what was
    last compiled regardless of how many facts came out of it. The `facts` query is
    kept as a fallback so a database written before that table existed still upgrades
    by recompiling once, rather than by losing its incremental behaviour.
    """
    text = path.read_text(encoding="utf-8")
    fhash = hashlib.sha256(text.encode()).hexdigest()

    if _origin_hash_matches(conn, path, fhash):
        return -1  # sentinel: unchanged

    ff = mdfacts.parse_facts_text(text, path)
    seen_ids = {f.id for f in ff.facts if f.id}

    # this file changed -> drop its derived rows and re-derive. Note we do NOT
    # touch facts whose origin_file is a different path: a hand-edit to
    # housing.md must never disturb work.md.
    old = conn.execute("SELECT id FROM facts WHERE origin_file=?", (str(path),)).fetchall()
    for r in old:
        db.fts_delete(conn, r["id"], "fact")
        db.vec_delete(conn, r["id"])
    conn.execute("DELETE FROM facts WHERE origin_file=?", (str(path),))

    n = 0
    for fl in ff.facts:
        fid = fl.id or _stable_id(path, fl)
        if not fid:
            continue
        # ⭐ Law 7, and the single behaviour worth more for long-term trust than any
        # amount of retrieval accuracy: a line a human typed by hand is SACRED.
        #
        # The tell is the absence of a `[f_… · stated · conf 0.95]` meta block.
        # FRIDAY writes that block itself on every fact it records, so a line
        # without one was authored in a text editor by the user. Those get
        # source_kind=user_edit and confidence 1.0, which means `_blocked_by` will
        # refuse to let any automatic assertion overwrite them — FRIDAY has to ask.
        hand_typed = not fl.id and fl.confidence is None and not fl.status
        source_kind = "user_edit" if hand_typed else fl.source_kind
        conf = fl.confidence
        if conf is None:
            conf = SOURCE_CONFIDENCE.get(source_kind, 0.5)
        klass = predicate_class_of(fl.predicate)
        asserted = fl.retracted_at or _extract_asserted(fl.status) or _file_mtime_iso(path)

        conn.execute(
            """INSERT OR REPLACE INTO facts(
                 id, subject, predicate, object, object_type, predicate_class, single_valued,
                 confidence, salience, valid_from, valid_to, asserted_at, retracted_at,
                 superseded_by, source_kind, source_refs, source_quote, review_after,
                 access_count, last_accessed, origin_file, origin_hash, origin_line)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fid, fl.subject, fl.predicate, fl.object, "str", klass,
                int(is_single_valued(fl.predicate)),
                float(conf), 0.5, fl.valid_from, fl.valid_to, asserted,
                _iso(fl.retracted_at) if fl.retracted else None,
                fl.superseded_by, source_kind,
                json.dumps(fl.source_refs or []), fl.source_quote, None,
                0, None, str(path), fhash, fl.lineno,
            ),
        )
        body = _fact_search_body(fl.predicate, fl.object)
        if fl.note:
            # The human annotation is real information — "landlord: Ramesh
            # (WhatsApp only, does not answer calls)" should be findable by
            # searching "whatsapp". It is indexed but never part of the value.
            body += f" {fl.note}"
        if fl.source_quote:
            body += f" {fl.source_quote}"
        if fl.retracted:
            body += " (retracted, previously believed)"
        db.fts_upsert(conn, fid, "fact", body, subject=fl.subject)
        if embedder is not None:
            db.vec_upsert(conn, fid, "fact", embedder.embed(body))
        n += 1

        # ⭐ SALIENCE TAP for compiled facts. Without this, every fact that arrives
        # via Markdown — hand-typed lines, imports, anything seeded — contributes
        # ZERO supervision for the RSC write gate w_t, and the only spans in the
        # store come from live conversation. That silently biases the head toward
        # conversational phrasing and starves Phase 3.5 of its highest-confidence
        # examples. Inline mode: the quote is stored on the span, because there is
        # no trace file to point at.
        if fl.source_quote and fl.source_quote.strip():
            quote = fl.source_quote.strip()
            supervision.record_span(
                conn,
                supervision.SpanLabel(
                    trace_id=f"md:{fid}",
                    trace_file="",
                    char_start=0,
                    char_end=len(quote),
                    label=0 if fl.retracted else 1,
                    fact_id=fid,
                    domain=fl.domain,
                    text=quote,
                ),
            )

        # ⭐ Re-derive retraction pairs for facts that are marked superseded in the
        # Markdown itself (i.e. hand-edited or previously compiled). This is the one
        # supervision signal that CAN be rebuilt, because it is implied by
        # `superseded_by`. Pairs from live conversation cannot be.
        if fl.retracted and fl.superseded_by and fl.superseded_by in seen_ids:
            _record_pair_from_markdown(conn, path, fl, ff)
    _record_origin(conn, path, fhash, n)
    return n


def _record_pair_from_markdown(
    conn: sqlite3.Connection, origin: Path, old: mdfacts.FactLine, ff: mdfacts.FactsFile
) -> None:
    from . import supervision

    new = next((f for f in ff.facts if f.id == old.superseded_by), None)
    if new is None:
        return
    from .facts import _address_key  # same derivation as the live path

    supervision.record_pair(
        conn,
        supervision.RetractionPair(
            fact_old_id=old.id,
            fact_new_id=new.id,
            key_old=_address_key(old.predicate, old.object),
            key_new=_address_key(new.predicate, new.object),
            value_old=old.object,
            value_new=new.object,
        ),
    )


def _stable_id(path: Path, fl: mdfacts.FactLine) -> str:
    """A hand-written fact line with no `[f_xxx]` id still needs a stable primary
    key — otherwise every recompile churns the row and breaks provenance."""
    h = hashlib.sha256(
        f"{path.name}|{fl.section}|{fl.predicate}|{fl.object}|{fl.valid_from}".encode()
    ).hexdigest()[:12]
    return f"f_{h}"


def _extract_asserted(status: str | None) -> str | None:
    """Pull `asserted 2026-09-29` out of the meta bracket if a human wrote it."""
    if not status:
        return None
    import re

    m = re.search(r"asserted\s+(\d{4}-\d{2}(?:-\d{2})?)", status, re.IGNORECASE)
    return _iso(m.group(1)) if m else None


def _iso(s: str | None) -> str | None:
    if not s:
        return None
    from ..util import parse_iso

    dt = parse_iso(s)
    return dt.isoformat(timespec="seconds") if dt else s


def _file_mtime_iso(path: Path) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )


# ── episodes / daily / skills ──────────────────────────────────────────────────

def compile_markdown_dir(
    root: Path, conn: sqlite3.Connection, kind: str, *, embedder=None, limit_chars: int = 8000
) -> int:
    """Index a directory of narrative Markdown (episodes, daily, skills) into FTS."""
    if not root.exists():
        return 0
    n = 0
    for p in sorted(root.rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        rel = str(p.relative_to(paths.ROOT))
        ref = f"{kind}:{rel}"
        db.fts_upsert(conn, ref, kind, text[:limit_chars], subject=p.stem)
        if embedder is not None:
            db.vec_upsert(conn, ref, kind, embedder.embed(_chunk_for_embed(text)))
        n += 1
    return n


def _fact_search_body(predicate: str, object_value: str) -> str:
    """The searchable text for one fact: predicate, its aliases, and the value.

    Aliases are indexed because the user's word and the store's word rarely share a
    substring. FTS5 tokenises `lease_amount_monthly` as a single token, so without
    this the query "what's my rent?" matches nothing at all — and on a machine with
    no embedding server there is no semantic path to fall back on. See
    `config.PREDICATE_ALIASES`.

    The value comes FIRST and the aliases last, so prefix-weighted scoring still
    ranks a genuine predicate match above a mere alias match.
    """
    alias = " ".join(aliases_for(predicate))
    return f"{predicate} {object_value} {alias}".strip()


def _chunk_for_embed(text: str) -> str:
    """Embed the first ~500 chars: enough to place a document semantically without
    letting one long tail dominate the vector."""
    return " ".join(text.split())[:500]


def compile_traces(conn: sqlite3.Connection) -> int:
    return traces.index_traces(conn)


# ── the whole build ────────────────────────────────────────────────────────────

def compile_all(
    conn: sqlite3.Connection | None = None,
    *,
    embedder=None,
    db_path: Path | str | None = None,
) -> CompileStats:
    """Full rebuild or incremental update of every derived index.

    Order matters: facts before episodes, because an episode may reference a fact
    id and we want the FTS rows for facts to exist first.
    """
    own_conn = conn is None
    if own_conn:
        conn = db.connect(db_path)
    paths.ensure_layout()
    st = CompileStats()

    try:
        if paths.FACTS.exists():
            for p in sorted(paths.FACTS.rglob("*.md")):
                st.files += 1
                try:
                    n = compile_file(p, conn, embedder=embedder)
                except Exception as e:  # one bad file must not abort the build
                    st.errors.append(f"{p.name}: {e}")
                    continue
                if n < 0:
                    st.skipped_unchanged += 1
                else:
                    st.facts_written += n
        # Totals come from the store, not from this run's counters — an incremental
        # build that skips every file still has a full index, and the summary should
        # say so.
        st.facts = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        st.retracted = conn.execute(
            "SELECT COUNT(*) AS n FROM facts WHERE retracted_at IS NOT NULL"
        ).fetchone()["n"]

        st.episodes = compile_markdown_dir(paths.EPISODES, conn, "episode", embedder=embedder)
        st.daily = compile_markdown_dir(paths.DAILY, conn, "daily", embedder=embedder)
        st.skills = compile_markdown_dir(paths.SKILLS, conn, "skill", embedder=embedder)
        st.traces = compile_traces(conn)

        st.fts_rows = conn.execute("SELECT COUNT(*) AS n FROM memory_fts").fetchone()["n"]
        st.vec_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_vec"
        ).fetchone()["n"]
        conn.commit()
    finally:
        if own_conn:
            conn.close()
    return st


def rebuild(db_path: Path | str | None = None, *, embedder=None) -> CompileStats:
    """The Law-2 verification path: delete the artifact, rebuild from Markdown."""
    p = Path(db_path) if db_path else paths.DB_PATH
    for suffix in ("", "-wal", "-shm"):
        q = Path(str(p) + suffix)
        if q.exists():
            q.unlink()
    return compile_all(db_path=p, embedder=embedder)
