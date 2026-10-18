"""Dreaming, seed 2 — the skill review: distill the class-level lesson.

Completes the Hermes Agent replication (doc 21). Their self-improvement loop
has two halves; seed 1 (review.py) took the memory half. This is the skill
half — their `_SKILL_REVIEW_PROMPT`, which is the best writing discipline in
their repo: "Be ACTIVE", user corrections are FIRST-CLASS skill signals, a
consulted-but-wrong skill gets patched NOW, procedure first, a pitfall is a
generalizable imperative with one clause of WHY (never an incident narrative),
the same lesson learned twice is ONE rule, fix in place never "UPDATE:", and
the do-not-capture list (environment failures, negative tool claims, transient
errors, one-off narratives, unresolved failures dressed as workflow).
Prompt fragments MIT-adapted with attribution.

Where FRIDAY departs — three gates Hermes does not have, all in the data layer:

  1. PROVENANCE. A proposal must carry source_quotes, and every one of them is
     checked with `policy.quote_is_user_authorised` against the turns being
     reviewed — the same function that gates memory_write. A distilled "the
     user prefers X" that the user never said is DENIED, not discouraged.
  2. READ-BEFORE-WRITE, ENFORCED BY CONSTRUCTION. Hermes refuses a patch
     unless the review called skill_view first (a session flag). Here the
     runner only embeds the N most-used skill bodies in the prompt and only
     accepts a patch for a skill whose body WAS embedded. The model cannot
     patch what it did not read; there is no flag to forget.
  3. THE STABLE-PREFIX LAW (doc 21 §3, the one direct-write rejection). A
     dream NEVER touches `skills/`. It writes a DRAFT under `dreams/`
     (git-tracked, user-visible diff), and the draft reaches `skills/` only
     when the user runs `friday skills --promote <name>` — hands on the diff,
     the doc 06 governance ladder's manual rung. Until then the draft ships
     nowhere: it is not in the ledger, not in FTS, not in the prompt.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from .. import paths
from ..util import now_iso

log = logging.getLogger(__name__)

#: How many current skill bodies ride the distill prompt, most-used first.
#: Patchable == embedded: the budget IS the read-before-write boundary.
_PATCH_WINDOW = 3
#: Total chars of skill bodies the prompt will carry. Beyond this the
#: least-used skills are simply not patchable this pass (they are the ones
#: the pruning order would retire first anyway — doc 06 G-gates).
_BODY_BUDGET_CHARS = 6000

_DISTILL_INSTRUCTION = """Review our recent conversation and the skill library below, and distill any CLASS-LEVEL lesson.

What a skill IS: instructions for doing a class of task the most efficient and correct way, to THIS user's specifications — the procedure, the order, the user's preferences for how the result should look, and the pitfalls that cost time. A future session should be able to follow it and produce what the user wants on the first try.

Signals that warrant action (any one is enough):
- The user corrected your style, format, verbosity, workflow, or approach. Frustration ("stop doing X", "too verbose", "why are you explaining") is a FIRST-CLASS skill signal, not just a memory signal.
- A non-trivial technique, fix, or debugging path emerged that a future session would benefit from.
- A skill below turned out wrong, missing a step, or outdated — patch it NOW.

Preference order — pick the earliest that fits:
1. PATCH an existing skill below (only skills whose full text is shown). Send the COMPLETE new file text with the fix made IN PLACE — edit the sentence that misled; never append "UPDATE:" underneath it.
2. CREATE a new class-level skill, only when nothing below covers the class. The name must name the class of task, never an incident ("lease-renewal", not "fix-october-rent-error").

Writing rules:
- Procedure first: steps in the order they are done, with the concrete commands and decision points.
- A pitfall is a generalizable imperative plus one clause of WHY — "check the notice period before drafting — the Chennai lease auto-renews 30 days out". Not a narrative of what happened.
- No dates, ticket IDs, or quoted user text as content — the rule must stand without the incident behind it.
- The same lesson learned twice is ONE rule: if it is already in the skill, strengthen that step instead of adding a second copy.

Do NOT capture (these become self-imposed constraints that bite later): environment-dependent failures (missing binaries, "command not found", unconfigured credentials), negative claims about tools ("X is broken"), transient errors that resolved, one-off task narratives, or an unresolved failure dressed up as a "reliable workflow" — if nothing actually worked, say nothing.

Reply with ONE JSON object and nothing else:
{{"action": "create" | "patch" | "none",
 "name": "<class-level-name>",            // patch: the EXISTING skill's name
 "description": "<=120 chars, what it does",
 "when_to_use": "<=200 chars, when this skill applies",
 "body": "<full markdown body; for a patch, the COMPLETE new body>",
 "source_quotes": ["the user's actual words that justify this"],
 "rationale": "one line"}}
"none" is a valid and expected answer. Do not manufacture a lesson to seem useful.

Skill library (most-used first; only skills with full text below may be patched):"""


@dataclass
class DistillReport:
    """What one distill pass did. `denied`/`rejected` never write anything."""

    action: str = "none"            # none|create|patch|denied|invalid
    name: str = ""
    draft_path: str = ""            # set when a draft was written
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.action == "none":
            return "no lesson to save"
        if self.action == "invalid":
            return "proposal unparseable — nothing written"
        verb = {"create": "drafted new skill", "patch": "drafted patch for skill",
                "denied": "denied"}[self.action]
        out = f"{verb} {self.name!r}"
        if self.draft_path:
            out += f" → {self.draft_path}"
        if self.reasons:
            out += " (" + "; ".join(self.reasons) + ")"
        return out


# ── prompt ─────────────────────────────────────────────────────────────────────


def _embedded_skills(conn: sqlite3.Connection) -> list[dict]:
    """The most-used ACTIVE skills, body included, under the char budget.

    The tail beyond the budget is intentionally unpatchable this pass — see
    module docstring, gate 2.
    """
    rows = conn.execute(
        "SELECT name, path FROM skills WHERE status='active' "
        "ORDER BY uses DESC, name ASC"
    ).fetchall()
    out: list[dict] = []
    spent = 0
    for r in rows:
        p = paths.ROOT / r["path"]
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        if not text:
            continue
        if len(out) >= _PATCH_WINDOW or spent + len(text) > _BODY_BUDGET_CHARS:
            out.append({"name": r["name"], "text": None})   # listed, not patchable
            continue
        spent += len(text)
        out.append({"name": r["name"], "text": text})
    return out


def distill_prompt(conn: sqlite3.Connection, session_id: str | None = None,
                   n: int = 20) -> str:
    """Instruction + current library (search-before-append context) + the turns."""
    from .review import review_turns

    rt = review_turns(conn, session_id, n)
    if not rt.turns:
        return ""
    parts = [_DISTILL_INSTRUCTION]
    embedded = _embedded_skills(conn)
    if embedded:
        for s in embedded:
            if s["text"] is None:
                parts.append(f"--- skill: {s['name']} (index only — not patchable "
                             "this pass, body over budget)")
            else:
                parts.append(f"--- skill: {s['name']} ---\n{s['text']}")
    else:
        parts.append("(library is empty — every proposal is a create)")
    parts.append("Recent conversation (user turns only, newest last):\n\n"
                 + "\n\n".join(f"User: {t}" for t in rt.turns))
    return "\n\n".join(parts)


# ── proposal parsing + the gates ───────────────────────────────────────────────


def _parse_proposal(text: str) -> dict | None:
    """First balanced JSON object in the reply, tolerant of surrounding prose."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _assemble(prop: dict) -> str:
    """The draft SKILL.md text for a create (frontmatter + body)."""
    fm = [
        "---",
        f"name: {prop.get('name', '')}",
        f"description: {prop.get('description', '')}",
        "version: 1",
        f"when_to_use: {prop.get('when_to_use', '')}",
        "---",
        "",
        str(prop.get("body", "")).strip(),
        "",
    ]
    return "\n".join(fm)


def _validate_draft(conn: sqlite3.Connection, text: str, *,
                    expect_existing: bool) -> tuple[str, list[str]]:
    """`parse_skill_text` — the SAME validation contract registration uses.

    No fallback_name on purpose: a draft must declare its own frontmatter
    name; inheriting a directory name would be validation by coincidence.
    Returns (parsed_name, problems); problems non-empty means reject.
    """
    from .skills import parse_skill_text

    skill, problems = parse_skill_text(text)
    if not problems and expect_existing:
        row = conn.execute("SELECT 1 FROM skills WHERE name=?", (skill.name,)).fetchone()
        if row is None:
            problems = [f"patch target {skill.name!r} is not a registered skill"]
    return skill.name, problems


def propose_draft(conn: sqlite3.Connection, *, client=None,
                  session_id: str | None = None, n: int = 20) -> DistillReport | None:
    """Run one skill-review turn in the dreaming scope; gate; write a DRAFT.

    Returns None when there is nothing to review. Every other failure mode is
    a DistillReport with reasons — a denied or invalid proposal writes
    NOTHING, by construction: the dream only ever writes under `dreams/`.
    """
    from ..agent import policy
    from ..agent.loop import Agent
    from ..llm import get_client
    from ..security import Verdict, detect
    from .review import review_turns

    rt = review_turns(conn, session_id, n)
    prompt = distill_prompt(conn, rt.session_id, n)
    if not prompt:
        return None
    ag = Agent(conn, client=client or get_client(prefer="mock"))
    st = ag.run(prompt, session_id=rt.session_id, scope="dreaming")

    prop = _parse_proposal(st.final_text or "")
    if prop is None:
        return DistillReport(action="invalid",
                             reasons=["reply was not a JSON object"])
    action = str(prop.get("action", "")).strip().lower()
    if action == "none":
        return DistillReport(action="none")

    name = str(prop.get("name", "")).strip()
    quotes = [str(q) for q in prop.get("source_quotes", []) if str(q).strip()]
    reasons: list[str] = []

    # ── gate 1: provenance — the same check that gates memory_write ──────────
    if not quotes:
        reasons.append("no source_quotes — a lesson must cite the user's words")
    for q in quotes:
        if not policy.quote_is_user_authorised(q, rt.turns):
            reasons.append(f"quote not found in the reviewed turns: {q[:60]!r}")

    if action not in ("create", "patch"):
        reasons.append(f"unknown action {action!r}")
        return DistillReport(action="invalid", name=name, reasons=reasons)

    if not reasons:
        if action == "create":
            text = _assemble(prop)
        else:  # patch: the proposal carries the COMPLETE new file text
            text = str(prop.get("body", "")).strip()
            if not text.startswith("---"):
                reasons.append("patch body must be the complete SKILL.md file, "
                               "frontmatter included")
        if not reasons:
            parsed_name, problems = _validate_draft(
                conn, text, expect_existing=(action == "patch"))
            reasons.extend(problems)
            if not problems:
                # the draft directory name must be the skill the draft declares
                if parsed_name != name:
                    reasons.append(
                        f"name mismatch: frontmatter says {parsed_name!r}, "
                        f"proposal says {name!r}")
                elif action == "create":
                    # a create must not shadow an existing skill — that is a patch
                    if conn.execute("SELECT 1 FROM skills WHERE name=?",
                                    (name,)).fetchone():
                        reasons.append(f"a skill named {name!r} already exists — "
                                       "patch it instead")
                else:
                    # gate 2, enforcement side: only skills whose body was
                    # embedded in THIS prompt may be patched
                    if name not in {s["name"] for s in _embedded_skills(conn)
                                    if s["text"] is not None}:
                        reasons.append(
                            f"{name!r} was not readable in this pass (over the "
                            "body budget) — run again or patch by hand")
            hostile = detect(f"{prop.get('when_to_use', '')} {prop.get('description', '')}")
            if hostile.verdict is Verdict.HOSTILE:
                reasons.append("routing line scored HOSTILE by the injection "
                               "detector — withheld")
    if reasons:
        rep = DistillReport(action="denied", name=name, reasons=reasons)
        policy.log(conn, actor="dreaming", action="dream.skill_denied",
                   target=name or None, decision="denied",
                   detail="; ".join(reasons)[:500])
        return rep

    # ── gate 3: drafts live under dreams/, never skills/ ─────────────────────
    sub = "patches" if action == "patch" else "skills"
    d = paths.ROOT / "dreams" / sub / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    (d / "PROPOSAL.json").write_text(json.dumps({
        "kind": action, "name": name,
        "quotes": quotes,
        "rationale": str(prop.get("rationale", ""))[:500],
        "session_id": rt.session_id, "created": now_iso(),
    }, indent=2), encoding="utf-8")
    policy.log(conn, actor="dreaming", action="dream.skill_proposed", target=name,
               decision="allowed",
               detail=f"draft written under dreams/{sub}/{name}; promotion is "
                      "the user's (`friday skills --promote)")
    return DistillReport(action=action, name=name,
                         draft_path=str(d / "SKILL.md"))
