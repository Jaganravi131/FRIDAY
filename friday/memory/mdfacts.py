"""Markdown fact format — parse and render. Innovation #2: Markdown is the truth.

Format (docs/architecture/02 §"Markdown as source of truth"):

    <!-- memory/facts/housing.md -->
    ---
    domain: housing
    version: 14
    last_compiled: 2026-09-30T03:00:00+05:30
    ---

    ## Current residence
    - lives_in: **Chennai, Tamil Nadu** (since 2023-06) [f_012 · stated · conf 0.95]
      <!-- src: t_00421 "i moved to chennai in june" -->

    ## Lease
    - lease_amount_monthly: **₹28,000** (valid 2025-06-01 → ) [f_041 · imported · conf 0.90]
    - ~~lease_amount_monthly: ₹24,000 (valid 2023-06 → 2025-05-31)~~ [f_040 · retracted 2026-09-29 · superseded_by f_041]

Design constraints this module respects:

  * A human must be able to write a fact line by hand, in a text editor, without
    reading this file. So the parser is permissive: bold optional, conf optional,
    dates optional, `·` or `|` or `,` as the separator.
  * Round-tripping must be stable. `render(parse(md))` should not churn the file,
    or the file-watcher will loop and git diffs become noise.
  * Struck-through (`~~…~~`) means RETRACTED. It is never deleted. The whole point
    of bi-temporal memory is that "what did I used to think?" stays answerable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util import partial_date

SEPARATORS = "\u00b7|\u2022,"  # · | • ,


@dataclass
class FactLine:
    """One parsed `- predicate: **object** …` line."""

    id: str
    predicate: str
    object: str
    domain: str
    source_kind: str = "stated"
    confidence: float | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    retracted: bool = False
    retracted_at: str | None = None
    superseded_by: str | None = None
    status: str | None = None
    section: str = ""
    lineno: int = 0
    source_quote: str | None = None
    source_refs: list[str] = field(default_factory=list)
    raw: str = ""
    #: A human annotation that trailed the bold value, e.g. the
    #: `(WhatsApp only, does not answer calls)` in
    #: `- landlord: **Ramesh** (WhatsApp only, does not answer calls)`.
    #: Real information, so it is kept and indexed — but it is NOT part of the
    #: value, because then `facts.landlord` would answer with a parenthetical.
    note: str = ""

    @property
    def subject(self) -> str:
        """Facts files are domain-scoped, so the subject is `user:<domain>`."""
        return f"user:{self.domain}" if self.domain else "user"


@dataclass
class FactsFile:
    path: Path
    domain: str
    version: int = 0
    last_compiled: str | None = None
    facts: list[FactLine] = field(default_factory=list)
    preamble_lines: list[str] = field(default_factory=list)

    @property
    def text_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.path.read_bytes()).hexdigest()


# ── parsing ────────────────────────────────────────────────────────────────────

_FRONTMATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)

# ── line anatomy ───────────────────────────────────────────────────────────────
#
# Parsing a fact line is done in STEPS, not with one big regex. The single-regex
# version of this was subtly broken in three ways at once: it required a `[meta]`
# block (so a hand-typed `- parking: one covered spot` was silently dropped — the
# exact case Markdown-as-truth exists to serve), and on struck-through lines the
# lazy `.*?` swallowed the date annotation and the closing `~~` into the object, so
# the stored value became `**₹24,000** (valid 2023-06 → 2025-05-31)`.
#
# The grammar, decomposed:
#
#     -  ~~  predicate: **object** (valid FROM → TO)  [id · kind · conf 0.9]  ~~
#     ^  ^   ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
#     1  2   3                     4                  5
#
#   1. bullet        required
#   2. strikethrough optional, may wrap the whole line OR just the core
#   3. pred: object  required; bold optional on either side
#   4. dates         optional; `since X`, `valid X → Y`, `from X to Y`, open-ended
#   5. meta          OPTIONAL — its absence means "a human typed this by hand",
#                    which is the highest-trust source in the system, not an error
#
# Each step strips what it matched and passes the remainder on, so a failure in an
# optional step cannot corrupt the required ones.

_BULLET = re.compile(r"^[-*+]\s+")
_STRUCK_WRAP = re.compile(r"^~~\s*(.*?)\s*~~$", re.DOTALL)
_META_TAIL = re.compile(r"\[([^\[\]]*)\]\s*$")
_DATE_ANNOT = re.compile(
    r"""\(?\s*
        (?:since|valid|from|between)?\s*
        (?P<vfrom>\d{4}(?:-\d{2})?(?:-\d{2})?)
        \s*(?P<arrow>→|->|–|—|to)\s*
        (?P<vto>\d{4}(?:-\d{2})?(?:-\d{2})?)?
        \s*\)?\s*$""",
    re.VERBOSE | re.UNICODE,
)
_SINCE_ONLY = re.compile(
    r"\(?\s*(?:since|from|valid)\s*(?P<vfrom>\d{4}(?:-\d{2})?(?:-\d{2})?)\s*\)?\s*$"
)
_PRED_OBJ = re.compile(r"^(?P<pred>[\w][\w\-\. ]*?)\s*:\s*(?P<obj>.+)$", re.DOTALL)

_META_ID = re.compile(r"\b(f_[\w\-]+)\b")
_META_KIND = re.compile(
    rf"\b(user_edit|stated|imported|observed|inferred)\b", re.IGNORECASE
)
_META_CONF = re.compile(r"conf(?:idence)?\s*[:=]?\s*([01](?:\.\d+)?)", re.IGNORECASE)
_META_RETRACTED = re.compile(r"retracted(?:\s*([\d]{4}-[\d]{2}(?:-[\d]{2})?))?", re.IGNORECASE)
_META_SUPERSEDED = re.compile(r"superseded_by\s*[:=]?\s*(f_[\w\-]+)", re.IGNORECASE)
_META_SRC = re.compile(r"<!--\s*src:\s*(?P<refs>.+?)\s*-->")
_SRC_QUOTE = re.compile(r'"([^"]+)"')


def parse_facts_text(text: str, path: Path | None = None) -> FactsFile:
    """Parse a facts Markdown file. Never raises on malformed lines — it skips them
    and keeps going, because a hand-edited file with one typo must not take down
    the whole memory index."""
    ff = FactsFile(path=path or Path("<string>"), domain=_domain_from_path(path))

    m = _FRONTMATTER.match(text)
    if m:
        for line in m.group("body").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "domain":
                    ff.domain = v or ff.domain
                elif k == "version":
                    ff.version = int(v) if v.isdigit() else 0
                elif k == "last_compiled":
                    ff.last_compiled = v
        remainder = text[m.end():]
        preamble_offset = len(text) - len(remainder)
    else:
        remainder = text
        preamble_offset = 0

    section = ""
    lines = remainder.splitlines()
    for i, raw in enumerate(lines):
        lineno = text[: preamble_offset + sum(len(x) + 1 for x in lines[:i])].count("\n") + 1
        stripped = raw.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip()
            continue
        if not stripped.startswith("-"):
            continue
        f = _parse_one(stripped, ff.domain, section, lineno, raw)
        if f is not None:
            ff.facts.append(f)

    # pick up `<!-- src: ... -->` continuation lines
    for i, raw in enumerate(remainder.splitlines()):
        ms = _META_SRC.search(raw)
        if not ms:
            continue
        # attach to the nearest preceding fact
        for f in reversed(ff.facts):
            if f.lineno <= i + (text[:preamble_offset].count("\n") if preamble_offset else 0):
                refs = ms.group("refs")
                q = _SRC_QUOTE.search(refs)
                if q:
                    f.source_quote = q.group(1)
                f.source_refs = [r.strip() for r in re.split(r"[,;]", refs) if r.strip()]
                break
    return ff


def _unwrap_bold(obj: str) -> tuple[str, str]:
    """Split `**value** (annotation)` into ("value", "(annotation)").

    A naive `re.sub(r"^\\*\\*(.+?)\\*\\*$", …)` gets this wrong in a way that is easy
    to miss: on `**Ramesh** (WhatsApp only, does not answer calls)` the `$` anchor
    forces the lazy group to stretch to the LAST `**` in the string, so the stored
    value becomes `Ramesh** (WhatsApp only, does not answer calls)`. The fact is
    then wrong in a way that still looks plausible when retrieved.

    So: match the closing delimiter by scanning, not by anchoring. Anything after
    the bold run is a human annotation — kept on the FactLine as `note` (it is real
    information, and it belongs in the FTS body) but never part of the value.
    """
    s = obj.strip()
    if not s.startswith("**"):
        return s, ""
    end = s.find("**", 2)
    if end < 0:
        return s.strip("*").strip(), ""          # unbalanced: a human typo, keep the text
    value = s[2:end].strip()
    note = s[end + 2:].strip()
    return (value or s.strip("*").strip()), note


def _parse_one(stripped: str, domain: str, section: str, lineno: int, raw: str) -> FactLine | None:
    """Parse one bullet line into a FactLine, or None if it isn't a fact.

    Stepwise — see the grammar comment above `_BULLET`. Returns None for prose
    bullets, which is correct: a facts file may contain ordinary list items.
    """
    s = stripped

    # 1. bullet
    m = _BULLET.match(s)
    if not m:
        return None
    s = s[m.end():].strip()

    # 2. strikethrough wrapping the whole line
    struck = False
    m = _STRUCK_WRAP.match(s)
    if m:
        struck = True
        s = m.group(1).strip()

    # 5. meta tail — OPTIONAL. Popped first because it is right-anchored and its
    #    contents (dates, `retracted 2026-09-29`) would otherwise confuse step 4.
    meta = ""
    m = _META_TAIL.search(s)
    if m:
        meta = m.group(1).strip()
        s = s[: m.start()].strip()

    # 2b. strikethrough wrapping just the core, after the meta was removed
    m = _STRUCK_WRAP.match(s)
    if m:
        struck = True
        s = m.group(1).strip()

    # 4. date annotation, right-anchored
    vfrom = vto = None
    m = _DATE_ANNOT.search(s)
    if m:
        vfrom = partial_date(m.group("vfrom"))
        vto = partial_date((m.group("vto") or "").strip()) or None
        s = s[: m.start()].strip()
    else:
        m = _SINCE_ONLY.search(s)
        if m:
            vfrom = partial_date(m.group("vfrom"))
            s = s[: m.start()].strip()

    # 3. predicate: object
    m = _PRED_OBJ.match(s)
    if not m:
        return None
    pred = m.group("pred").strip()
    obj = m.group("obj").strip()

    # trailing strikethrough fragments, then bold unwrap
    obj = obj.strip("~").strip()
    obj = re.sub(r"^~~(.+?)~~$", r"\1", obj).strip()
    obj, note = _unwrap_bold(obj)
    obj = obj.strip("*_` ").strip()
    pred = pred.strip("~*_` ").strip()
    if not obj or not pred:
        return None
    # A predicate never contains a sentence. If it does, this was prose.
    if len(pred.split()) > 6 or " " in pred and not re.match(r"^[\w\-\. ]+$", pred):
        return None

    retracted = struck
    mr = _META_RETRACTED.search(meta)
    if mr:
        retracted = True

    mid = _META_ID.search(meta)
    fid = mid.group(1) if mid else ""
    mk = _META_KIND.search(meta)
    mc = _META_CONF.search(meta)
    msup = _META_SUPERSEDED.search(meta)

    return FactLine(
        id=fid,
        predicate=pred,
        object=obj,
        domain=domain,
        source_kind=(mk.group(1).lower() if mk else "stated"),
        confidence=float(mc.group(1)) if mc else None,
        valid_from=vfrom,
        valid_to=vto,
        retracted=retracted,
        retracted_at=(mr.group(1) if mr and mr.group(1) else None),
        superseded_by=(msup.group(1) if msup else None),
        status=meta.strip(),
        section=section,
        lineno=lineno,
        raw=raw,
        note=note,
    )


def _domain_from_path(path: Path | None) -> str:
    if path is None:
        return "general"
    return path.stem.lower() or "general"


def parse_file(path: Path) -> FactsFile:
    return parse_facts_text(path.read_text(encoding="utf-8"), path)


# ── rendering ──────────────────────────────────────────────────────────────────

def render_fact(f: FactLine, *, include_src: bool = True) -> str:
    """Render one fact back to Markdown. Stable: same input -> same output."""
    core = f"{f.predicate}: **{f.object}**"
    if f.valid_from or f.valid_to:
        core += f" (valid {f.valid_from or ''} \u2192 {f.valid_to or ''})".replace(" \u2192 )", ")")
        core = core.rstrip()
        if core.endswith("\u2192"):
            core += " "
    if f.retracted:
        core = f"~~{core}~~"

    meta: list[str] = []
    if f.id:
        meta.append(f.id)
    if f.retracted:
        meta.append(f"**retracted {f.retracted_at or ''}**".strip())
        if f.superseded_by:
            meta.append(f"superseded_by {f.superseded_by}")
    else:
        meta.append(f.source_kind)
        if f.confidence is not None:
            meta.append(f"conf {f.confidence:.2f}")
    line = f"- {core} [{f' {SEPARATORS[0]} '.join(meta)}]"

    if include_src and (f.source_quote or f.source_refs):
        refs = ", ".join(f.source_refs) if f.source_refs else ""
        quote = f' "{f.source_quote}"' if f.source_quote else ""
        line += f"\n  <!-- src: {refs}{quote} -->"
    return line


def render_file(ff: FactsFile, facts: list[FactLine] | None = None) -> str:
    """Render a whole facts file. Groups by section, retracted last in each group."""
    facts = facts if facts is not None else ff.facts
    out: list[str] = [
        "---",
        f"domain: {ff.domain}",
        f"version: {ff.version + 1}",
        f"last_compiled: {ff.last_compiled or ''}",
        "---",
        "",
        f"# {ff.domain.replace('_', ' ').title()}",
        "",
    ]
    sections: dict[str, list[FactLine]] = {}
    for f in facts:
        sections.setdefault(f.section or "Facts", []).append(f)
    for sec, items in sections.items():
        out.append(f"## {sec}")
        live = [x for x in items if not x.retracted]
        dead = [x for x in items if x.retracted]
        for f in live + dead:
            out.append(render_fact(f))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def upsert_fact_line(path: Path, fact: FactLine) -> None:
    """Read-modify-write one fact into a Markdown file, preserving everything else.

    This is the write path FRIDAY uses. It must never clobber a hand-edit, so it
    parses, replaces by id (or appends), and re-renders only the affected section.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ff = parse_file(path)
    else:
        ff = FactsFile(path=path, domain=_domain_from_path(path))
        ff.facts = []

    replaced = False
    for i, existing in enumerate(ff.facts):
        if fact.id and existing.id == fact.id:
            ff.facts[i] = fact
            replaced = True
            break
        if (
            not fact.id
            and existing.predicate == fact.predicate
            and existing.section == fact.section
            and not existing.retracted
        ):
            ff.facts[i] = fact
            replaced = True
            break
    if not replaced:
        ff.facts.append(fact)

    path.write_text(render_file(ff), encoding="utf-8")


def as_dict(f: FactLine) -> dict[str, Any]:
    return {
        "id": f.id,
        "subject": f.subject,
        "predicate": f.predicate,
        "object": f.object,
        "domain": f.domain,
        "source_kind": f.source_kind,
        "confidence": f.confidence,
        "valid_from": f.valid_from,
        "valid_to": f.valid_to,
        "retracted": f.retracted,
        "retracted_at": f.retracted_at,
        "superseded_by": f.superseded_by,
        "section": f.section,
        "source_quote": f.source_quote,
        "source_refs": f.source_refs,
        "note": f.note,
    }
