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
    problems: list[str] = []
    skill = Skill(name=path.parent.name, path=str(path.relative_to(paths.ROOT)))
    m = _FRONTMATTER.match(text)
    if not m:
        return skill, [("no --- frontmatter block (agentskills.io format: "
                        "name, description, version, when_to_use)")]
    fm = m.group("fm")
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
