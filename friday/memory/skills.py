"""S3 — Skills. `skills/<name>/SKILL.md`, agentskills.io format.

The layer that turns repeated experience into a reusable procedure
(docs/architecture/03 §skills, docs/architecture/06). The shape is deliberately
the same as the agentskills.io open standard (and Hermes Agent's skills, which
validated it at scale): one directory per skill, a SKILL.md with frontmatter
(`name`, `description`, `version`, `when_to_use`, `triggers`) and a body that is
the procedure itself.

Progressive disclosure is the whole design (seed.py's README says why): tool and
skill definitions are a fixed per-turn tax, and on a small model with an 8K
budget you cannot afford six. So:

  * `when_to_use` is the ONLY part visible before activation — it rides the
    ledger's stable `skills` slot as a one-line index entry.
  * The body is indexed for retrieval (`kind='skill'` FTS rows) and reaches the
    model only when a query actually recalls it.

Format rules are enforced at registration, not hoped for at write time. The
description rule comes from Hermes's hard-won lesson: their system-prompt skill
index truncates the description, so anything past the cut "is silently cut and
never routes" — a limit is not cosmetic when an index truncates. FRIDAY's index
line is `when_to_use`-based, so the *routable* text is what gets the length
discipline here.

Not replicated from Hermes: their skills are trusted files. FRIDAY's index ships
on EVERY turn, so it gets the same hostile-text boundary as `core_memory`
(neutralize + drop HOSTILE lines) — a poisoned SKILL.md must not become a
persistent prompt injection.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from ..store import db
from ..util import approx_tokens

#: agentskills.io name grammar: lowercase-hyphenated, ≤64 chars, no spaces.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: The routing surface that rides the stable `skills` slot. A skill whose
#: when_to_use cannot say what it is for in this many characters will never be
#: activated by retrieval either — length discipline on the routable text is the
#: whole game (see module docstring).
_MAX_WHEN_TO_USE = 200

#: One sentence, states the capability not the implementation.
_MAX_DESCRIPTION = 120

#: Same cap compile_markdown_dir used for narrative Markdown: the FTS body is
#: for recall matching, and the recalled render caps it again downstream.
_BODY_LIMIT = 8000

_FRONTMATTER = re.compile(r"\A---\s*\n(?P<fm>.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class Skill:
    """One registered skill. The `skills` table row is this, persisted."""

    name: str
    description: str = ""
    when_to_use: str = ""
    version: int = 1
    triggers: list[str] = field(default_factory=list)
    path: str = ""
    status: str = "active"        # hand-authored files enter active; the
    uses: int = 0                 # draft→shadow pipeline is for distilled skills


@dataclass
class SkillsReport:
    scanned: int = 0
    registered: int = 0
    removed: int = 0
    skipped: list[str] = field(default_factory=list)   # (name, why) as strings

    def summary(self) -> str:
        bits = [f"scanned {self.scanned}", f"registered {self.registered}"]
        if self.removed:
            bits.append(f"removed {self.removed}")
        if self.skipped:
            bits.append(f"skipped {len(self.skipped)}")
        return ", ".join(bits)


def _parse_scalar(fm_text: str, key: str) -> str:
    """First `key: value` line in the frontmatter block, None-safe."""
    for line in fm_text.splitlines():
        m = re.match(rf"^{re.escape(key)}\s*:\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'").strip()
    return ""


def _parse_list(fm_text: str, key: str) -> list[str]:
    """`key: [a, b]` (inline) or YAML block form. Tolerant on purpose."""
    inline = _parse_scalar(fm_text, key)
    if inline.startswith("[") and inline.endswith("]"):
        return [t.strip().strip('"\'') for t in inline[1:-1].split(",") if t.strip()]
    out: list[str] = []
    in_block = False
    for line in fm_text.splitlines():
        if re.match(rf"^{re.escape(key)}\s*:\s*$", line):
            in_block = True
            continue
        if in_block:
            m = re.match(r"^\s*-\s+(.*)$", line)
            if not m:
                break
            out.append(m.group(1).strip().strip('"\''))
    return out


def parse_skill(path: Path) -> tuple[Skill, list[str]]:
    """Parse one SKILL.md. Returns (skill, problems); problems empty == valid.

    A file with problems is reported and skipped, never half-registered — a
    skill whose routing line is missing cannot be activated by design, and
    registering it anyway would put a dead entry in every turn's index.
    """
    text = path.read_text(encoding="utf-8")
    skill, problems = parse_skill_text(
        text, fallback_name=path.parent.name,
        path=str(path.relative_to(paths.ROOT)))
    return skill, problems


def parse_skill_text(text: str, *, fallback_name: str = "",
                     path: str = "") -> tuple[Skill, list[str]]:
    """Pure parser — the validation contract, independent of any file location.

    `fallback_name` is the directory convention for on-disk skills; a DRAFT
    (distill.py) passes none, so a missing frontmatter `name:` fails the
    grammar instead of inheriting whatever directory it happens to sit in.
    """
    problems: list[str] = []
    skill = Skill(name=fallback_name, path=path)
    m = _FRONTMATTER.match(text)
    if not m:
        return skill, [("no --- frontmatter block (agentskills.io format: "
                        "name, description, version, when_to_use)")]
    fm = m.group("fm")
    skill.name = _parse_scalar(fm, "name") or fallback_name
    skill.description = _parse_scalar(fm, "description")
    skill.when_to_use = _parse_scalar(fm, "when_to_use")
    skill.triggers = _parse_list(fm, "triggers")
    v = _parse_scalar(fm, "version")
    try:
        skill.version = int(v.split(".")[0]) if v else 1
    except ValueError:
        problems.append(f"version {v!r} is not numeric")

    if not _NAME_RE.match(skill.name):
        problems.append(f"name {skill.name!r} must be lowercase-hyphenated, "
                        "<=64 chars, no spaces")
    if not skill.description:
        problems.append("description is required (one sentence: what it does)")
    elif len(skill.description) > _MAX_DESCRIPTION:
        problems.append(f"description is {len(skill.description)} chars "
                        f"(max {_MAX_DESCRIPTION}) — state the capability, not the implementation")
    if not skill.when_to_use:
        problems.append("when_to_use is required — it is the ONLY text visible "
                        "before activation; without it the skill can never be routed to")
    elif len(skill.when_to_use) > _MAX_WHEN_TO_USE:
        problems.append(f"when_to_use is {len(skill.when_to_use)} chars "
                        f"(max {_MAX_WHEN_TO_USE}) — the index line is the routing surface")
    return skill, problems


def _routing_body(skill: Skill, text: str) -> str:
    """The FTS body: routing surface first (so a match on when_to_use ranks),
    triggers, then the procedure itself for recall matching. Value-first mirrors
    `_fact_search_body`: what routes it comes before what it contains."""
    fm_end = text.find("\n---", 4)
    body = text[fm_end + 4:].strip() if fm_end > 0 else text
    head = f"{skill.when_to_use} {skill.description} {' '.join(skill.triggers)}".strip()
    return f"{head}\n{body}"[:_BODY_LIMIT]


def register_skills(conn: sqlite3.Connection, *, embedder=None,
                    root: Path | None = None) -> SkillsReport:
    """Scan `skills/*/SKILL.md`, register valid ones, drop stale rows.

    Replaces compile_markdown_dir for the skills tree (which indexed every .md —
    including the seeded README — as an undifferentiated 'skill' row and never
    touched the `skills` table). Idempotent: a full re-register is a few
    DELETE+INSERT pairs per skill, cheap next to the fact compile.
    """
    from ..memory.compiler import _chunk_for_embed

    rep = SkillsReport()
    root = root or paths.SKILLS
    seen: set[str] = set()
    names: set[str] = set()
    if root.exists():
        for p in sorted(root.glob("*/SKILL.md")):
            rep.scanned += 1
            try:
                skill, problems = parse_skill(p)
                text = p.read_text(encoding="utf-8")
            except OSError as e:
                rep.skipped.append(f"{p.parent.name}: unreadable ({e})")
                continue
            if problems:
                rep.skipped.append(f"{skill.name}: " + "; ".join(problems))
                continue
            conn.execute(
                """INSERT OR REPLACE INTO skills
                   (name, version, when_to_use, triggers, path, status, shadow_pass,
                    uses, last_used, provenance)
                   VALUES (?,?,?,?,?,'active',NULL,
                           COALESCE((SELECT uses FROM skills WHERE name=?), 0),
                           (SELECT last_used FROM skills WHERE name=?), NULL)""",
                (skill.name, skill.version, skill.when_to_use,
                 json.dumps(skill.triggers), skill.path,
                 skill.name, skill.name),
            )
            body = _routing_body(skill, text)
            db.fts_upsert(conn, f"skill:{skill.name}", "skill", body,
                          subject=skill.name)
            if embedder is not None:
                db.vec_upsert(conn, f"skill:{skill.name}", "skill",
                              embedder.embed(_chunk_for_embed(body)))
            seen.add(f"skill:{skill.name}")
            names.add(skill.name)
            rep.registered += 1

    # Drop rows for skills that no longer exist on disk (deleted or renamed),
    # plus any legacy compile_markdown_dir rows under skills/.
    for r in conn.execute(
            "SELECT ref FROM memory_fts WHERE kind='skill'").fetchall():
        if r["ref"] not in seen:
            db.fts_delete(conn, r["ref"], "skill")
            rep.removed += 1
    for r in conn.execute("SELECT name FROM skills").fetchall():
        if r["name"] not in names:
            conn.execute("DELETE FROM skills WHERE name=?", (r["name"],))
    conn.commit()
    return rep


def skills_index(conn: sqlite3.Connection | None, *, budget_tokens: int = 192) -> str:
    """The compact, always-on directory of ACTIVE skills (doc 03: ~30 tok/skill).

    One line per skill: `- name — when_to_use`. Most-used first so the budget
    spends itself on what has actually earned activation. Empty string when
    there are no skills — an empty slot renders nothing, so a skill-less install
    compiles the byte-identical prompt it always did.

    ⚠️ SECURITY: this ships on EVERY turn (that is what 'always-on' means), so it
    gets the core_memory boundary — neutralize structurally, and a line that
    still reads as HOSTILE is dropped, not shipped. A poisoned SKILL.md must not
    become a persistent prompt injection riding the stable prefix.
    """
    from ..security import Verdict, detect, neutralize

    if conn is None:
        return ""
    rows = conn.execute(
        """SELECT name, when_to_use, uses FROM skills
           WHERE status='active' ORDER BY uses DESC, name ASC""").fetchall()
    if not rows:
        return ""
    lines: list[str] = []
    dropped = 0
    used = 0
    for r in rows:
        line = f"- {r['name']} — {neutralize(str(r['when_to_use'] or ''))}"
        if detect(line).verdict is Verdict.HOSTILE:
            dropped += 1
            continue
        if used and used + approx_tokens(line) > budget_tokens:
            continue
        lines.append(line)
        used += approx_tokens(line)
    out = "\n".join(lines)
    more = len(rows) - len(lines) - dropped
    if more > 0:
        out += f"\n(+{more} more — memory_search by name to activate)"
    if dropped:
        out += f"\n[{dropped} skill(s) withheld: read like instructions, not skills]"
    return out


# ── dreaming: activation telemetry + the promotion gate (T-024) ────────────────


def record_activations(conn: sqlite3.Connection, names: list[str]) -> int:
    """Bump `uses`/`last_used` for skills whose body actually shipped this turn.

    Hermes keeps this in a sidecar `.usage.json` "so telemetry stays out of
    user-authored SKILL.md" (tools/skill_usage.py); FRIDAY already has the
    columns, so the telemetry lives where the index ordering reads it. Best
    effort by contract (callers wrap it): telemetry must never fail a user
    turn. Only ACTIVE skills count — a retired skill gaining uses would be
    evidence against retiring it, collected after the decision.
    """
    from ..util import now_iso

    if not names:
        return 0
    now = now_iso()
    bumped = 0
    for n in sorted(set(names)):
        cur = conn.execute(
            "UPDATE skills SET uses = uses + 1, last_used = ? "
            "WHERE name = ? AND status = 'active'", (now, n))
        if cur.rowcount and cur.rowcount > 0:
            bumped += cur.rowcount
    conn.commit()
    return bumped


def list_drafts() -> list[dict]:
    """Pending dream drafts, newest last. Ships nowhere — this is the queue
    `friday skills` shows so the user remembers to promote or discard."""
    import json as _json

    out: list[dict] = []
    for kind, sub in (("create", "skills"), ("patch", "patches")):
        base = paths.DREAMS / sub
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            prop = {}
            pj = d / "PROPOSAL.json"
            if pj.exists():
                try:
                    prop = _json.loads(pj.read_text(encoding="utf-8"))
                except ValueError:
                    prop = {}
            out.append({
                "kind": kind, "name": d.name, "dir": str(d),
                "quotes": prop.get("quotes", []),
                "rationale": prop.get("rationale", ""),
            })
    return out


def promote_draft(conn: sqlite3.Connection, name: str) -> tuple[bool, str]:
    """THE promotion gate: move a dream draft into `skills/` and re-register.

    This is the user's hands-on-the-diff moment (doc 06 governance ladder,
    manual rung). A dream can propose; only this command — run by a human —
    makes a draft live. Re-validates the draft with the same parser
    registration uses, keeps the PROPOSAL.json beside the skill as the durable
    record of why it exists, and removes the spent dream.
    """
    import shutil

    from ..agent import policy

    for kind, sub in (("patch", "patches"), ("create", "skills")):
        d = paths.DREAMS / sub / name
        if not (d / "SKILL.md").exists():
            continue
        skill, problems = parse_skill(d / "SKILL.md")
        if problems:
            return False, f"draft {name!r} is not valid: {'; '.join(problems)}"
        dest_dir = paths.SKILLS / skill.name
        dest = dest_dir / "SKILL.md"
        if kind == "create" and dest.exists():
            return False, (f"skills/{skill.name}/SKILL.md already exists — "
                           "hand-edit it instead, or remove it first")
        if kind == "patch" and not dest.exists():
            return False, (f"patch target skills/{skill.name}/SKILL.md is gone — "
                           "`friday skills --scan` first")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text((d / "SKILL.md").read_text(encoding="utf-8"),
                        encoding="utf-8")
        kept = ""
        if (d / "PROPOSAL.json").exists():
            (dest_dir / "PROPOSAL.json").write_text(
                (d / "PROPOSAL.json").read_text(encoding="utf-8"),
                encoding="utf-8")
            kept = f"; proposal kept at skills/{skill.name}/PROPOSAL.json"
        register_skills(conn)
        shutil.rmtree(d)
        policy.log(conn, actor="user", action="memory.skill_promoted",
                   target=skill.name, decision="allowed",
                   detail=f"{kind} promoted from dreams/{sub}/{name}{kept}")
        return True, (f"promoted {skill.name} ({kind}) and registered it"
                      + kept)
    return False, f"no draft named {name!r} under dreams/"


def discard_draft(conn: sqlite3.Connection | None, name: str) -> tuple[bool, str]:
    """Throw a dream away. Also a user action, also audited — a declined
    lesson is a decision worth a row, not litter."""
    import shutil

    for sub in ("skills", "patches"):
        d = paths.DREAMS / sub / name
        if d.exists():
            shutil.rmtree(d)
            if conn is not None:
                from ..agent import policy
                policy.log(conn, actor="user", action="dream.discarded",
                           target=name, decision="allowed", detail=None)
            return True, f"discarded draft {name!r}"
    return False, f"no draft named {name!r} under dreams/"
