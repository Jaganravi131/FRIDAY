"""The retrieval path — all seven steps of docs/architecture/04 Law D.

    rewrite -> {fts5, vector, graph} -> dedupe -> rerank -> HARD FLOOR -> fresh -> narrow

Two rules in here are non-negotiable and both exist because of measured failures:

  * **The hard floor.** `[]` beats noise. Ask FRIDAY something it has no memory of
    and it must return an empty list, not three plausible-looking facts. This is
    the difference between a memory system and a hallucination machine.
  * **Freshness + retraction gating.** A retracted fact never auto-surfaces for a
    present-tense query, but it MUST still surface for a belief-time query
    ("what did I used to think?"). That asymmetry is the bi-temporal promise.

And one that the linear-attention pivot makes load-bearing (Law 2b): retrieval is
not an optimisation here. A fixed-size recurrent state is lossy by construction
and loses associative recall first, so **the external store is the only exact
memory FRIDAY has.** ICLR 2025 proves an in-context retrieval primitive is
*sufficient* to close the RNN representation gap — `memory_search` is that
primitive, and this file is its implementation.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from ..config import (
    aliases_for,
    RETRIEVAL_GRAPH_HOPS,
    RETRIEVAL_K_SHIP,
    RETRIEVAL_K_WIDE,
    RETRIEVAL_SCORE_FLOOR,
)
from ..memory.decay import decay_row
from ..store import db
from ..util import dedupe, jaccard, now_iso, partial_date
from .embedders import Embedder, get_embedder
from .rerankers import Reranker, get_reranker


@dataclass
class Candidate:
    ref: str                 # fact id | "episode:memory/episodes/2026-09-28.md"
    kind: str                # fact|episode|skill|daily
    body: str
    subject: str = ""
    score: float = 0.0       # final reranked score
    recall_score: float = 0.0
    recall_sources: list[str] = field(default_factory=list)  # fts|vec|graph
    row: Any = None          # the facts row, when kind == "fact"

    @property
    def is_fact(self) -> bool:
        return self.kind == "fact"


@dataclass
class SearchResult:
    kept: list[Candidate]
    considered: int
    floor: float
    reranker: str
    as_of: str | None = None
    dropped_below_floor: int = 0
    dropped_stale: int = 0

    def __len__(self) -> int:
        return len(self.kept)

    def as_dicts(self) -> list[dict]:
        """What the `memory_search` tool returns to the model. Provenance included
        on purpose: FRIDAY must be able to say *why* it believes something."""
        out = []
        for c in self.kept:
            d: dict[str, Any] = {"ref": c.ref, "kind": c.kind, "score": round(c.score, 3)}
            if c.is_fact and c.row is not None:
                r = c.row
                d.update(
                    {
                        "subject": _cell(r, "subject"),
                        "predicate": _cell(r, "predicate"),
                        "object": _cell(r, "object"),
                        "valid_from": _cell(r, "valid_from"),
                        "valid_to": _cell(r, "valid_to"),
                        "asserted_at": _cell(r, "asserted_at"),
                        "retracted_at": _cell(r, "retracted_at"),
                        "confidence": round(float(_cell(r, "confidence", 0.5) or 0.5), 3),
                        "source_kind": _cell(r, "source_kind"),
                        "source_quote": _cell(r, "source_quote"),
                        "origin_file": _cell(r, "origin_file"),
                        # ⭐ The rest of the paper trail. Exit test #3 requires the
                        # literal quote AND the trace id, and `/why` needs the line
                        # number to point at a place in a file you can open. These
                        # were missing, so a personal answer could show its quote but
                        # not the conversation it came from — enough to look
                        # justified without being traceable.
                        "id": _cell(r, "id", c.ref),
                        "trace_id": _cell(r, "trace_id"),
                        "origin_line": _cell(r, "origin_line"),
                        "superseded_by": _cell(r, "superseded_by"),
                        "recall_sources": list(c.recall_sources),
                    }
                )
            else:
                d["body"] = c.body[:400]
            out.append(d)
        return out

    def summary(self) -> str:
        if not self.kept:
            return (
                f"no memory matched (considered {self.considered}, "
                f"{self.dropped_below_floor} below floor {self.floor:.2f}, "
                f"{self.dropped_stale} stale/retracted)"
            )
        return ", ".join(
            f"{c.body[:48]} ({c.score:.2f})" if not c.is_fact
            else f"{c.row['predicate']}={c.row['object']} ({c.score:.2f})"
            for c in self.kept
        )


# ── step 1: query rewrite ──────────────────────────────────────────────────────

_PRONOUN = re.compile(r"\b(it|that|this|those|these|they|them|he|she|his|her)\b", re.I)


def rewrite_query(query: str, recent_turns: Sequence[str] = ()) -> str:
    """L0-cheap rewrite: resolve bare pronouns from the last few turns, expand
    contractions, keep the user's words.

    Deliberately NOT an LLM call in the fallback path. A 0.6B rewrite costs ~15 ms
    and helps, but retrieval must work when no model is reachable — so the
    deterministic version carries the load and the model is an enhancement.
    """
    q = query.strip()
    if not q:
        return q
    q = (
        q.replace("what's", "what is").replace("whats", "what is")
         .replace("don't", "do not").replace("can't", "cannot")
         .replace("i'm", "i am").replace("where's", "where is")
    )
    # If the query is mostly pronouns, pull nouns from recent turns.
    if _PRONOUN.search(q) and len(_content_words(q)) <= 2 and recent_turns:
        ctx = " ".join(list(recent_turns)[-3:])
        extra = [w for w in _content_words(ctx) if w not in _content_words(q)][:6]
        if extra:
            q = q + " " + " ".join(extra)
    return q


#: ⭐ NOT a second list. This used to be one, and it disagreed with `db._STOPWORDS` —
#: which is what the LexicalReranker scores against. Recall finding a fact and the
#: reranker discarding it is the failure mode the reranker's own docstring warns about,
#: and two stopword lists is how you get it by accident.
from ..store.db import CONTENT_STOPWORDS as _STOP


def _cell(row: Any, key: str, default: Any = None) -> Any:
    """Read one column from a facts row, tolerating a row that lacks it.

    Handles both `sqlite3.Row` and a plain dict. `Candidate.row` is typed `Any` and
    the pipeline always fills it from `SELECT *`, so in practice every column is
    present — but `as_dicts` is a REPORTING path (it feeds the model's tool result and
    `/why`), and a reporting path that raises KeyError on a partial row turns "show me
    what you found" into a traceback. `provenance.from_row` already degrades to
    "unknown" for exactly this reason; the two must agree, or the first one to touch a
    projection decides whether the command works.
    """
    try:
        v = row[key]
        return default if v is None else v
    except (KeyError, IndexError, TypeError):
        return default


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z\u0b80-\u0bff]{3,}", text.lower()) if w not in _STOP]


# ── steps 2-4: the three recall paths ──────────────────────────────────────────

def _fts_path(conn: sqlite3.Connection, q: str, k: int) -> list[Candidate]:
    out = []
    for ref, kind, body, score in db.fts_search(conn, q, k=k):
        out.append(Candidate(ref=ref, kind=kind, body=body, score=score,
                             recall_score=score, recall_sources=["fts"]))
    return out


def _vec_path(conn: sqlite3.Connection, q: str, k: int, embedder: Embedder) -> list[Candidate]:
    try:
        qv = embedder.embed(q)
    except Exception:
        return []
    out = []
    for ref, kind, score in db.vec_search(conn, qv, k=k):
        body = _body_for_ref(conn, ref, kind)
        out.append(Candidate(ref=ref, kind=kind, body=body, score=score,
                             recall_score=score, recall_sources=["vec"]))
    return out


def _graph_path(conn: sqlite3.Connection, q: str, k: int, hops: int = RETRIEVAL_GRAPH_HOPS) -> list[Candidate]:
    """Entity-graph walk. Cheap, and it catches what keywords miss: asking about
    "Amma" should surface facts about "Mrs. Lakshmi" because the entities table
    carries the alias."""
    terms = _content_words(q)
    if not terms:
        return []
    seeds: list[str] = []
    for t in terms[:6]:
        rows = conn.execute(
            "SELECT id, name, aliases FROM entities WHERE lower(name) LIKE ?", (f"%{t}%",)
        ).fetchall()
        for r in rows:
            seeds.append(r["id"])
            try:
                import json
                for a in json.loads(r["aliases"] or "[]"):
                    if t in str(a).lower():
                        seeds.append(r["id"])
            except Exception:
                pass
    if not seeds:
        return []

    reached: set[str] = set(seeds)
    frontier = list(set(seeds))
    for _ in range(max(0, hops)):
        if not frontier:
            break
        marks = ",".join("?" * len(frontier))
        rows = conn.execute(
            f"SELECT src, dst, predicate FROM relations WHERE src IN ({marks}) OR dst IN ({marks})",
            frontier + frontier,
        ).fetchall()
        nxt = []
        for r in rows:
            for e in (r["src"], r["dst"]):
                if e and e not in reached:
                    reached.add(e)
                    nxt.append(e)
        frontier = nxt

    out = []
    for eid in list(reached)[: k * 2]:
        rows = conn.execute(
            """SELECT id, predicate, object, subject FROM facts
               WHERE retracted_at IS NULL AND (subject=? OR object LIKE ? OR predicate LIKE ?)
               LIMIT ?""",
            (eid, f"%{eid}%", f"%{eid}%", 5),
        ).fetchall()
        for r in rows:
            out.append(Candidate(
                ref=r["id"], kind="fact", body=f"{r['predicate']} {r['object']}",
                subject=r["subject"], score=0.42, recall_score=0.42,
                recall_sources=["graph"],
            ))
    return out[:k]


def _body_for_ref(conn: sqlite3.Connection, ref: str, kind: str) -> str:
    if kind == "fact" and not ref.startswith(("episode:", "skill:", "daily:", "trace:")):
        r = conn.execute("SELECT predicate, object FROM facts WHERE id=?", (ref,)).fetchone()
        if r:
            return f"{r['predicate']} {r['object']}"
    r = conn.execute(
        "SELECT body FROM memory_fts WHERE ref=? AND kind=?", (ref, kind)
    ).fetchone()
    return r["body"] if r else ""


# ── steps 5-7: dedupe, rerank, floor, freshness, narrow ────────────────────────

def _merge(cands: Sequence[Candidate]) -> list[Candidate]:
    """Dedupe by ref, unioning recall sources and keeping the best recall score.

    A candidate found by two paths is more likely relevant, so we record that —
    but the *reranker* decides the final score, not the recall paths. Mixing them
    is how you get an ensemble that no longer respects the floor.
    """
    by_ref: dict[str, Candidate] = {}
    for c in cands:
        ex = by_ref.get(c.ref)
        if ex is None:
            by_ref[c.ref] = Candidate(**{**c.__dict__})
            continue
        ex.recall_score = max(ex.recall_score, c.recall_score)
        ex.recall_sources = sorted(set(ex.recall_sources) | set(c.recall_sources))
        if not ex.body and c.body:
            ex.body = c.body
    return list(by_ref.values())


def is_fresh(c: Candidate, as_of: str | None, *, include_retracted_for_time_travel: bool = False) -> bool:
    """Freshness + retraction gate.

    * No `as_of` (the normal case): drop retracted facts, drop facts whose world
      validity has ended, drop facts the decay function has archived.
    * With `as_of`: this is a time-travel query, so a retracted fact is EXACTLY
      what the user wants — keep it if it was valid/believed then.
    """
    if not c.is_fact or c.row is None:
        return True
    r = c.row
    d = decay_row(r)
    when = partial_date(as_of) if as_of else None

    if when:
        vf, vt = partial_date(r["valid_from"]), partial_date(r["valid_to"])
        world_ok = (vf is None or vf <= when) and (vt is None or vt >= when)
        belief_ok = r["asserted_at"][:10] <= when and (
            not r["retracted_at"] or r["retracted_at"][:10] > when
        )
        return world_ok or belief_ok

    if r["retracted_at"]:
        return include_retracted_for_time_travel
    if not d.keep_indexed:
        return False
    vt = partial_date(r["valid_to"])
    if vt and vt < partial_date(now_iso()):
        return False
    return True


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    as_of: str | None = None,
    k_wide: int = RETRIEVAL_K_WIDE,
    k_ship: int = RETRIEVAL_K_SHIP,
    floor: float | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    recent_turns: Sequence[str] = (),
    use_graph: bool = True,
) -> SearchResult:
    """The whole pipeline. Returns a SearchResult, which may legitimately be empty."""
    emb = embedder or get_embedder()
    rr = reranker or get_reranker()
    effective_floor = floor if floor is not None else max(RETRIEVAL_SCORE_FLOOR, rr.recommended_floor)

    q = rewrite_query(query, recent_turns)
    if not q.strip():
        return SearchResult([], 0, effective_floor, rr.name, as_of)

    cands: list[Candidate] = []
    cands += _fts_path(conn, q, k_wide)
    cands += _vec_path(conn, q, k_wide, emb)
    if use_graph:
        cands += _graph_path(conn, q, k_wide)
    merged = _merge(cands)

    # attach the facts rows so freshness/decay/provenance can be evaluated
    for c in merged:
        if c.is_fact and not c.ref.startswith(("episode:", "skill:", "daily:", "trace:")):
            c.row = conn.execute("SELECT * FROM facts WHERE id=?", (c.ref,)).fetchone()
            if c.row is not None:
                c.subject = c.row["subject"]
                c.body = f"{c.row['predicate']} {c.row['object']}"

    # rerank on real text
    texts = [_rerank_text(c) for c in merged]
    scores = rr.score(q, texts) if texts else []
    q_words = set(_content_words(q))
    for c, s, text in zip(merged, scores, texts):
        # a multi-path recall is weak corroborating evidence; nudge, don't dominate
        c.score = min(1.0, s * (1.0 + 0.06 * (len(c.recall_sources) - 1)))
        c.score *= KIND_PRIOR.get(c.kind, DEFAULT_KIND_PRIOR)
        if _is_echo(text, q_words):
            c.score *= ECHO_PENALTY

    _suppress_redundant(merged, texts)
    merged.sort(key=lambda c: -c.score)

    kept: list[Candidate] = []
    below = stale = 0
    for c in merged:
        if c.score < effective_floor:
            below += 1
            continue
        if not is_fresh(c, as_of):
            stale += 1
            continue
        kept.append(c)
        if len(kept) >= k_ship:
            break

    # record the access: the decay function's reinforcement term reads it, so
    # retrieving a fact makes it stickier. Recall is a self-strengthening signal.
    if kept:
        ids = [c.ref for c in kept if c.is_fact]
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE facts SET access_count=access_count+1, last_accessed=? WHERE id IN ({marks})",
                [now_iso(), *ids],
            )
            conn.commit()

    return SearchResult(
        kept=kept, considered=len(merged), floor=effective_floor, reranker=rr.name,
        as_of=as_of, dropped_below_floor=below, dropped_stale=stale,
    )


#: ⭐ How hard to demote a candidate that merely echoes the query. 0.25 puts a
#: perfect lexical match (score ~0.98) below the 0.45 floor, so an echo is not shipped
#: but is still counted in `considered` — it was recalled, and pretending otherwise
#: would make the funnel unreadable.
ECHO_PENALTY = 0.25

#: ⭐ Kind prior: THE CURRENT BELIEF OUTRANKS THE LOG OF HOW WE CAME TO BELIEVE IT.
#:
#: This corrects a structural bias, not a tuned preference. A trace records a write, so
#: its body is a short restatement of the fact it wrote — `standup_day=Tuesday 10:00 IST`
#: for the fact `standup_day Tuesday 10:00 IST`. The reranker's overlap term divides by
#: the union of query and candidate words, so the SHORTER body always scores higher on
#: the same match. Measured live: trace 0.759 against fact 0.746 for "when is my
#: standup". Every fact therefore has a shadow that beats it by construction, and the
#: agent — which ships only facts, because only facts are bi-temporal, carry provenance
#: and can be justified by `/why` — saw its top slots filled with logs and answered
#: *"I don't have anything in memory about that"* for a fact it had.
#:
#: The ordering is by what each kind can WARRANT, not by how well it matches:
#:   fact    1.00  the answer-bearing kind; `/why` can show its source quote
#:   skill   0.95  a procedure — the answer to "how do I …"
#:   episode 0.90  narrative context; supports "what happened", not "what is true"
#:   daily   0.90  a log of activity, including of questions asked
#: These priors are GENTLE on purpose, and they are not what fixes the duplicate-trace
#: problem. Calibrating them to do that needs daily ~0.72, because a fact's rerank text
#: carries its aliases and its own source quote — `name: Ramesh (called, who is, full
#: name) "my landlord is called Ramesh"` scores 0.725 against a bare `landlord=Ramesh`
#: trace scoring a capped 1.000. A fact is penalised for carrying MORE evidence, and the
#: only way to out-prior that is to push logs below the floor, which would make "what
#: did I ask yesterday" unanswerable. `_suppress_redundant` below is the fix that does
#: not require choosing between those two.
KIND_PRIOR = {"fact": 1.0, "skill": 0.95, "episode": 0.92, "daily": 0.90}
DEFAULT_KIND_PRIOR = 0.90


#: A non-fact that says nothing a fact in the same result set does not already say.
#: 0.5 takes a capped 1.000 trace to 0.45 after the 0.90 kind prior — i.e. off the
#: shipping list — while leaving a trace that carries unique information untouched.
REDUNDANT_PENALTY = 0.5


def _suppress_redundant(merged: list, texts: list[str]) -> None:
    """⭐ Demote a candidate that duplicates a fact already in the result set.

    A trace records a write, so its body is a restatement of the fact it wrote:
    `standup_day=Tuesday 10:00 IST` for the fact `standup_day Tuesday 10:00 IST`. Two
    consequences follow, and the second is the one that breaks things:

      * the trace's body is SHORTER, and the reranker's overlap term divides by the
        union of query and candidate words, so the trace scores higher on the same
        match — a fact is penalised for carrying aliases and a source quote;
      * the agent ships only facts, so a top-k filled with a fact's own shadows leaves
        nothing shippable and the answer becomes *"I don't have anything in memory"*.

    So the rule is not "logs are worse than beliefs". It is: DO NOT SHIP THE SAME
    INFORMATION TWICE, and when a choice is unavoidable, ship the copy that carries
    provenance — because that is the one `/why` can justify.

    Subset-of-words is deliberately a weak, conservative test. It only fires when a fact
    in THIS result set already says everything the candidate says, so a log with any
    unique content ("asked about the notice period yesterday") is untouched and history
    stays answerable.
    """
    fact_words = [set(_content_words(t)) for c, t in zip(merged, texts) if c.is_fact]
    if not fact_words:
        return
    for c, t in zip(merged, texts):
        if c.is_fact:
            continue
        w = set(_content_words(t))
        if w and any(w <= fw for fw in fact_words):
            c.score *= REDUNDANT_PENALTY


def _is_echo(text: str, q_words: set[str]) -> bool:
    """⭐ True when `text` contains no content word the query did not already contain.

    AN ANSWER MUST ADD INFORMATION THE QUESTION DID NOT HAVE. A record whose entire
    content is the question is not a candidate answer, it is a recording of someone
    asking — and it matches that question perfectly, which is exactly why it is
    dangerous.

    Found live: FRIDAY indexes conversation traces so "what did I ask yesterday" is
    answerable. Asking "when is my standup" a second time retrieved the trace of the
    FIRST time asking it — body `'when is my standup'`, coverage 1.0, score 0.979 —
    above the `standup_day` fact, whose body is longer and therefore scores lower on
    overlap. The agent ships only facts, so it saw the top slots filled with logs of
    the user's own question, found no fact among them, and answered *"I don't have
    anything in memory about that"* for a fact it had. A perfect score, earned by
    containing nothing.

    This is scorer-independent and so lives in the pipeline rather than in
    `LexicalReranker`: a cross-encoder would make the same mistake more confidently,
    because an embedding of a question is very close to an embedding of itself.

    Facts are structurally immune, which is why this does not need a per-kind prior:
    a fact's text is `predicate object`, so it always contributes at least the
    predicate. What gets caught is logs, echoes and headings. Note the trace
    `'standup day: Tuesday 10:00 IST'` is NOT an echo — it answers the question — and
    keeps its score.
    """
    if not q_words:
        return False
    body = set(_content_words(text))
    return bool(body) and body <= q_words


def _rerank_text(c: Candidate) -> str:
    """Give the reranker something worth reading: predicate + object + quote + aliases.

    The ALIASES are the load-bearing part for the lexical fallback. `aliases_for`
    already maps `monthly_rent` -> "lease amount monthly" and splits snake_case into
    words; appending them means a human phrasing can match a machine predicate
    without the fact text containing it. The ONNX cross-encoder ignores the extra
    words because it scores meaning, not overlap, so this costs nothing there and
    rescues the no-dependency path — which is the one that has to work on a laptop
    with no model downloaded.
    """
    if c.is_fact and c.row is not None:
        r = c.row
        parts = [f"{r['predicate']}: {r['object']}"]
        al = aliases_for(r["predicate"] or "")
        if al:
            parts.append("(" + ", ".join(al) + ")")
        if r["source_quote"]:
            parts.append(f'"{r["source_quote"]}"')
        if r["valid_from"]:
            parts.append(f"(since {r['valid_from']})")
        return " ".join(parts)
    return c.body


def find_duplicates(conn: sqlite3.Connection, threshold: float = 0.86) -> list[tuple[str, str, float]]:
    """Near-duplicate live facts, for the nightly Dreaming job to merge."""
    rows = conn.execute(
        "SELECT id, predicate, object FROM facts WHERE retracted_at IS NULL"
    ).fetchall()
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["predicate"] != b["predicate"]:
                continue
            sim = jaccard(a["object"], b["object"])
            if sim >= threshold:
                out.append((a["id"], b["id"], round(sim, 3)))
    return out
