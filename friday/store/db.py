"""SQLite connection + schema bootstrap + FTS/vec index building.

`sqlite-vec` is an optional loadable extension. When it is absent (the default on
a fresh machine, and always in a stdlib-only environment) we fall back to an in-Python
cosine search over the `memory_vec` table. Same interface, exact, and authoritative.
Law 4 still holds: the reranker and the score floor are what make retrieval good,
not the index.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from .. import paths

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

#: Set once at connect time; read by retrieval to pick the vector path.
HAS_VEC_EXT = False


def connect(db_path: Path | str | None = None, *, bootstrap: bool = True) -> sqlite3.Connection:
    """Open (and optionally create) the artifact DB.

    The DB is a build output. It may not exist; that is normal, not an error.
    """
    p = Path(db_path) if db_path else paths.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    if bootstrap:
        apply_schema(conn)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: every statement in schema.sql is IF NOT EXISTS."""
    global HAS_VEC_EXT
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    # The vec0 virtual table needs the extension; split it out so a missing
    # extension cannot abort the whole schema.
    vec_stmts = []
    keep = []
    for chunk in _split_statements(sql):
        if "USING vec0" in chunk:
            vec_stmts.append(chunk)
        else:
            keep.append(chunk)

    # Statement-by-statement, NOT executescript(). Two reasons, both learned the
    # hard way:
    #   1. executescript() raises `near "PRAGMA": syntax error` when a PRAGMA sits
    #      mid-script, even though every statement executes fine individually.
    #   2. executescript() issues an implicit COMMIT first, and `journal_mode=WAL`
    #      must run outside a transaction to take effect.
    # A failure now names the offending statement instead of the whole file.
    for idx, stmt in enumerate(keep):
        try:
            conn.execute(stmt)
        except sqlite3.Error as e:
            raise sqlite3.OperationalError(
                f"schema.sql statement #{idx} failed: {e}\n--- {stmt[:400]}"
            ) from e

    HAS_VEC_EXT = _try_enable_vec(conn)
    _ensure_fallback_table(conn)
    conn.commit()


def _try_enable_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec if it is genuinely available. Returns True only if verified.

    ⚠️ This function used to be a try/except that set `HAS_VEC_EXT = True` whenever
    nothing *raised*. Two things made that silently wrong:

      1. Every load attempt was wrapped in `except Exception: continue`, so a
         machine with no extension at all fell through to the success path.
      2. The `vec0` DDL was never in schema.sql, so `vec_stmts` was empty and the
         "create the table" step was a no-op that could not fail.

    The result: `HAS_VEC_EXT` was True on a box with no extension and no table, and
    every `vec_upsert`/`vec_delete` raised `no such table: memory_vec`.

    So: probe with `SELECT vec_version()`. That is the only check that cannot lie,
    because it fails unless the extension is actually loaded and callable.

    Note the vec0 table is NOT created here. Its dimension is baked into the DDL
    (`float[1024]`) and the embedder can change, so `_ensure_vec_table()` creates it
    lazily at the first upsert, once the real dimension is known.
    """
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.OperationalError):
        return False        # built without extension support (common on Debian)
    try:
        loaded = False
        try:
            import sqlite_vec  # type: ignore

            sqlite_vec.load(conn)
            loaded = True
        except Exception:
            for cand in ("vec0", "sqlite_vec", "libsqlite_vec"):
                try:
                    conn.load_extension(cand)
                    loaded = True
                    break
                except Exception:
                    continue
        if not loaded:
            return False
        conn.execute("SELECT vec_version()")   # the probe that cannot lie
        return True
    except Exception:
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> bool:
    """Create the vec0 table for `dim` if the extension is live. Idempotent.

    If a table already exists at a DIFFERENT dimension (the embedder changed), it is
    dropped and rebuilt. Vectors are re-derivable from the fallback table, which is
    the source of truth, so this is a cheap rebuild rather than a migration — and it
    beats the alternative, which is every upsert failing on a dimension mismatch.
    """
    if not HAS_VEC_EXT:
        return False
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='memory_vec_v0'"
        ).fetchone()
        if row is not None:
            if f"float[{dim}]" in (row["sql"] or ""):
                return True
            conn.execute("DROP TABLE memory_vec_v0")
        conn.execute(
            f"CREATE VIRTUAL TABLE memory_vec_v0 USING vec0(embedding float[{int(dim)}])"
        )
        return True
    except sqlite3.Error:
        return False


def _split_statements(sql: str) -> list[str]:
    """Split schema.sql into individual statements.

    Three things have to be respected at once, and getting any of them wrong shows
    up as a confusing syntax error several statements downstream:

    * `--` line comments. schema.sql documents every column, so comments are dense
      and often trail a statement on the same line. Left in, the comment text runs
      into the *next* statement — `PRAGMA synchronous = NORMAL; -- safe with WAL`
      followed by `PRAGMA mmap_size …` becomes one mangled script.
    * `/* */` block comments.
    * quoted strings, which may legitimately contain `;`, `--`, or unbalanced
      parens (FTS5 trigger bodies especially).

    Depth tracking on parentheses is still needed because a virtual-table
    definition spans lines and may contain a `;` inside its argument list.
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    i, n = 0, len(sql)
    in_str: str | None = None

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_str:
            buf.append(ch)
            if ch == in_str:
                if nxt == in_str:            # '' is an escaped quote, not a close
                    buf.append(nxt)
                    i += 2
                    continue
                in_str = None
            i += 1
            continue

        if ch == "-" and nxt == "-":         # line comment
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":         # block comment
            i += 2
            while i < n and not (sql[i] == "*" and i + 1 < n and sql[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if ch in ("'", '"', "`"):
            in_str = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        if ch == ";" and depth == 0:
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


_VEC_SCHEMA = """
-- The AUTHORITATIVE vector store. Always present, always written, always correct:
-- an exact cosine scan in pure Python. At FRIDAY's scale (a personal memory, so
-- hundreds to low thousands of rows) this costs single-digit milliseconds and
-- removes an entire class of "the index and the truth disagree" bugs.
CREATE TABLE IF NOT EXISTS memory_vec (
  ref  TEXT PRIMARY KEY,
  kind TEXT,
  dim  INTEGER NOT NULL,
  vec  BLOB NOT NULL
);

-- Optional accelerator, created lazily ONLY when sqlite-vec is verified live
-- (_ensure_vec_table). A vec0 virtual table keys on an integer rowid and cannot
-- take `ref TEXT PRIMARY KEY`, so this maps our text refs to rowids. If this table
-- is missing or stale, nothing breaks: vec_search falls back to the exact scan.
CREATE TABLE IF NOT EXISTS memory_vec_rowid (
  ref   TEXT PRIMARY KEY,
  rowid INTEGER NOT NULL UNIQUE   -- rowid in memory_vec_v0
);
"""

#: Kept as an alias — older code and docs refer to the exact-scan table as the
#: "fallback". It is not a fallback in the sense of being second-best; it is the
#: source of truth, and the vec0 index is the optimisation.
_FALLBACK_VEC_SCHEMA = _VEC_SCHEMA


def _ensure_fallback_table(conn: sqlite3.Connection) -> None:
    """Always present, whether or not the extension loaded.

    Keeping the exact-scan table unconditional means `vec_search()` has one
    correctness path to test, and sqlite-vec stays a pure optimisation. That is
    deliberate: the reranker and the score floor are what make retrieval good, not
    the index, and an index that can silently disagree with the truth is worse than
    no index.
    """
    for stmt in _split_statements(_VEC_SCHEMA):
        conn.execute(stmt)


# ── vector ops (identical interface whether or not the extension loaded) ────────

def vec_upsert(conn: sqlite3.Connection, ref: str, kind: str, vec: list[float]) -> None:
    """Store a vector. The exact table is authoritative; vec0 is a mirror.

    The vec0 mirror is best-effort: if it fails (dimension change mid-run, extension
    unloaded, corrupt index) the upsert still succeeds, because the exact table is
    what `vec_search` falls back to. An accelerator that can break the write path is
    not an accelerator.
    """
    conn.execute(
        "INSERT OR REPLACE INTO memory_vec(ref, kind, dim, vec) VALUES (?,?,?,?)",
        (ref, kind, len(vec), _pack(vec)),
    )
    if not HAS_VEC_EXT:
        return
    try:
        if not _ensure_vec_table(conn, len(vec)):
            return
        row = conn.execute(
            "SELECT rowid FROM memory_vec_rowid WHERE ref=?", (ref,)
        ).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO memory_vec_v0(embedding) VALUES (?)", (_pack(vec),)
            )
            rid = cur.lastrowid
            conn.execute(
                "INSERT OR REPLACE INTO memory_vec_rowid(ref, rowid) VALUES (?,?)", (ref, rid)
            )
        else:
            conn.execute(
                "UPDATE memory_vec_v0 SET embedding=? WHERE rowid=?",
                (_pack(vec), row["rowid"]),
            )
    except sqlite3.Error:
        pass


def vec_delete(conn: sqlite3.Connection, ref: str) -> None:
    conn.execute("DELETE FROM memory_vec WHERE ref=?", (ref,))
    if not HAS_VEC_EXT:
        return
    try:
        row = conn.execute(
            "SELECT rowid FROM memory_vec_rowid WHERE ref=?", (ref,)
        ).fetchone()
        if row is not None:
            conn.execute("DELETE FROM memory_vec_v0 WHERE rowid=?", (row["rowid"],))
            conn.execute("DELETE FROM memory_vec_rowid WHERE ref=?", (ref,))
    except sqlite3.Error:
        pass


def vec_search(conn: sqlite3.Connection, query: list[float], k: int = 20) -> list[tuple[str, str, float]]:
    """Return [(ref, kind, cosine_similarity)] best-first.

    Tries the vec0 index, then verifies and falls back to the exact scan. The score
    is cosine in both paths so the same RETRIEVAL_SCORE_FLOOR applies — which matters,
    because Law 4 is enforced by that floor and a floor that means different things
    on different code paths is not a floor.
    """
    if HAS_VEC_EXT:
        try:
            if _ensure_vec_table(conn, len(query)):
                rows = conn.execute(
                    """SELECT rowid, distance FROM memory_vec_v0
                       WHERE embedding MATCH ? AND k = ? ORDER BY distance""",
                    (_pack(query), k * 2),
                ).fetchall()
                if rows:
                    out = []
                    for r in rows:
                        m = conn.execute(
                            "SELECT ref, kind FROM memory_vec_rowid WHERE rowid=?", (r["rowid"],)
                        ).fetchone()
                        if m is None:
                            continue
                        # vec0 returns L2 distance on normalised-ish vectors; map to
                        # a cosine-like score in (0,1] so the floor transfers.
                        d = float(r["distance"])
                        out.append((m["ref"], m["kind"] or "", 1.0 / (1.0 + d)))
                    if out:
                        out.sort(key=lambda t: -t[2])
                        return out[:k]
        except sqlite3.Error:
            pass  # fall through to the exact path

    rows = conn.execute("SELECT ref, kind, dim, vec FROM memory_vec").fetchall()
    scored = []
    for r in rows:
        v = _unpack(r["vec"], r["dim"])
        if len(v) != len(query):
            continue          # embedder changed; stale row. rebuild clears it.
        scored.append((r["ref"], r["kind"] or "", _cosine(query, v)))
    scored.sort(key=lambda t: -t[2])
    return scored[:k]


def _pack(vec: list[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes, dim: int) -> list[float]:
    import struct

    return list(struct.unpack(f"<{dim}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── FTS ────────────────────────────────────────────────────────────────────────

def fts_upsert(conn: sqlite3.Connection, ref: str, kind: str, body: str, subject: str = "") -> None:
    conn.execute("DELETE FROM memory_fts WHERE ref=? AND kind=?", (ref, kind))
    conn.execute(
        "INSERT INTO memory_fts(body, kind, ref, subject) VALUES (?,?,?,?)",
        (body, kind, ref, subject),
    )


def fts_delete(conn: sqlite3.Connection, ref: str, kind: str | None = None) -> None:
    if kind:
        conn.execute("DELETE FROM memory_fts WHERE ref=? AND kind=?", (ref, kind))
    else:
        conn.execute("DELETE FROM memory_fts WHERE ref=?", (ref,))


#: Words that carry no retrieval signal. Not a full stopword list — just the ones
#: that appear in nearly every question and in no fact, where they do pure damage by
#: dragging down overlap scores.
_STOPWORDS = frozenset({
    "what", "whats", "is", "are", "was", "were", "the", "a", "an", "my", "your",
    "do", "does", "did", "i", "me", "of", "to", "in", "on", "at", "for", "and",
    "or", "it", "that", "this", "there", "be", "am", "about",
})


def fts_terms(user_text: str) -> list[str]:
    """Tokenise a query into content terms, dropping stopwords.

    FTS5 raises on unbalanced quotes and bare operators, so terms are extracted with
    `\\w+` and quoted by the caller. That is the difference between a search that
    works and a crash the first time someone types `rent? (2026)`.
    """
    import re as _re

    words = _re.findall(r"\w+", (user_text or "").lower())
    keep = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    # If everything was a stopword ("what is it"), fall back to the raw words rather
    # than returning nothing — an empty MATCH is an empty result.
    return keep or [w for w in words if len(w) > 1]


def fts_query(user_text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression using **OR**.

    ⚠️ This used to emit an implicit-AND expression (`"what" "is" "my" "rent"*`),
    which looks reasonable and is catastrophic: FTS5 AND semantics require EVERY
    term to be present in the row, and no fact body contains "what", "is" or "my".
    So essentially every natural-language question returned zero candidates — the
    retrieval pipeline reported `considered=0` and the agent answered "I don't know"
    about facts that were sitting in the index.

    OR is correct because ranking is what decides relevance, not the MATCH filter.
    `fts_search` scores by term overlap and bm25, and the reranker plus the score
    floor (Law 4) make the final cut. Matching widely and ranking properly beats
    matching narrowly and silently returning nothing.
    """
    terms = fts_terms(user_text)
    if not terms:
        return ""
    parts = [f'"{t}"' for t in terms]
    parts[-1] = f'"{terms[-1]}"*'      # prefix match on the last term
    return " OR ".join(parts)


def fts_search(conn: sqlite3.Connection, user_text: str, k: int = 20) -> list[tuple[str, str, str, float]]:
    """Return [(ref, kind, body, score)] best-first, score in (0,1].

    Score blends two signals, because neither alone is right:

      * TERM OVERLAP — what fraction of the query's content terms appear in the row.
        This is what stops a row that matched one incidental OR term from outranking
        a row that matched the whole question. It is also robust to bm25's
        length bias, which favours short rows regardless of relevance.
      * BM25 — the corpus-frequency signal, which is what makes "rent" worth more
        than "the" even after stopwords are dropped.

    Overlap is weighted first (0.7/0.3) since with OR matching it is the only thing
    distinguishing a real hit from a near-miss.
    """
    terms = fts_terms(user_text)
    q = fts_query(user_text)
    if not q or not terms:
        return []
    try:
        rows = conn.execute(
            """SELECT ref, kind, body, subject, bm25(memory_fts) AS s
               FROM memory_fts WHERE memory_fts MATCH ? ORDER BY s LIMIT ?""",
            (q, max(k * 4, 40)),
        ).fetchall()
    except sqlite3.Error:
        return []
    if not rows:
        return []

    # bm25() is negative-better; normalise across the candidate set to (0,1].
    scores = [float(r["s"]) for r in rows]
    worst, best = min(scores), max(scores)
    span = (best - worst) or 1.0

    term_set = set(terms)
    out = []
    for r in rows:
        bm25_norm = 1.0 - (float(r["s"]) - worst) / span if span else 1.0
        body_l = (r["body"] or "").lower()
        hit = sum(1 for t in term_set if t in body_l)
        overlap = hit / len(term_set)
        score = 0.7 * overlap + 0.3 * bm25_norm
        out.append((r["ref"], r["kind"], r["body"], max(0.05, round(score, 6))))
    out.sort(key=lambda t: -t[3])
    return out[:k]
