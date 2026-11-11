"""Dreaming, seed 1 — the memory review turn.

Replicates the second half of Hermes Agent's learning loop (their background
memory review, every N turns: "Review the conversation above and consider saving
to memory if appropriate... If nothing is worth saving, just say 'Nothing to
save.'") — with one structural difference that matters more than the prompt:

HERMES prevents a bad review with PROMPT TEXT (routing rules, size limits,
"don't capture environment failures"). FRIDAY prevents it with the POLICY
ENGINE: a review runs in the `dreaming` scope, where the provenance gate denies
any `stated` memory_write whose quote does not appear in what the user actually
said — and the turns being reviewed ARE that authority, replayed from the turns
table (T-019). A review that fabricates "the user told me X" is denied by the
data layer, not by the model's good intentions. The prompt text still carries
the anti-hoarding discipline (Hermes's hardest-won lessons, adapted):

  * save ONCE, in the right store, only durable facts about the user;
  * "Nothing to save" is a valid, expected outcome;
  * do NOT capture: transient task state, environment-dependent failures,
    unresolved failures dressed up as working methods, or anything the user
    would retract.

This is the feature HEARTBEAT_MD lists under "Check (pending the model tier /
Dreaming)". It stays a manual command until the model tier exists (Q2): the
heartbeat runs deterministic rules only, and a review turn needs a model.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_REVIEW_INSTRUCTION = """Review our recent conversation for anything durable that was NOT already saved to memory.

- Save ONLY durable facts about the user (preferences, corrections, decisions they want enforced later). Quote their actual words in source_quote.
- If a fact contradicts something already stored, write the new fact — bi-temporal reconciliation retracts the old one and tells the user.
- "Nothing to save" is a valid and expected answer. Do not manufacture memories to seem useful.
- Do NOT save: transient task state, session-specific details, facts about other people, or anything you are merely inferring.

Recent conversation (user turns only, newest last):"""


@dataclass
class ReviewTurns:
    session_id: str
    turns: list[str]          # the user's words, in order


def review_turns(conn: sqlite3.Connection, session_id: str | None = None,
                 n: int = 20) -> ReviewTurns:
    """The last N user turns of a session — the review's scope AND its authority.

    Same session, not a copy: `_recent_transcript` replays these exact turns into
    the next run's transcript, and `_do_tool` builds `authority_turns` from
    user-role entries. That is what makes the provenance gate check the review's
    quotes against the conversation being reviewed — no separate plumbing, the
    persistence layer IS the authority source.
    """
    sid = session_id
    if sid is None:
        row = conn.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        if row is None:
            return ReviewTurns(session_id="", turns=[])
        sid = row["id"]
    rows = conn.execute(
        """SELECT content FROM turns WHERE session_id=? AND role='user'
           ORDER BY idx DESC, role ASC LIMIT ?""", (sid, n),
    ).fetchall()
    return ReviewTurns(session_id=sid, turns=[r["content"] for r in reversed(rows)])


def review_prompt(conn: sqlite3.Connection, session_id: str | None = None,
                  n: int = 20) -> str:
    rt = review_turns(conn, session_id, n)
    return _prompt_for(rt)


def _prompt_for(rt: ReviewTurns) -> str:
    if not rt.turns:
        return ""
    quoted = "\n\n".join(f"User: {t}" for t in rt.turns)
    return f"{_REVIEW_INSTRUCTION}\n\n{quoted}"


def run_review(conn: sqlite3.Connection, *, client=None, session_id: str | None = None,
               n: int = 20):
    """Run one review turn in the dreaming scope.

    Returns the Agent State (final_text, tool_log). Every memory_write it
    attempts goes through `check()` with the reviewed turns as authority: an
    honest quote passes, a fabricated one is DENIED — the dreaming scope has no
    user present to confirm, so unauthorised provenance is a denial, not a
    question. That is the guarantee Hermes's prompt-only review cannot give.
    """
    from ..agent.loop import Agent
    from ..llm import get_client

    rt = review_turns(conn, session_id, n)
    prompt = _prompt_for(rt)
    if not prompt:
        return None
    ag = Agent(conn, client=client or get_client(prefer="mock"))
    st = ag.run(prompt, session_id=rt.session_id, scope="dreaming")
    return st
