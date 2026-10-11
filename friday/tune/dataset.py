"""Export the bi-temporal memory as an instruction dataset (docs/architecture/18 §4.1).

The point of a personal agent fine-tune is not to teach the model new knowledge —
RAG does that better and cheaper. It is to teach the model the PERSON: their
register, the shape of their days, the facts that are so assumed they never get
said twice. So the export has three strata, and which stratum a fact lands in is
decided by its provenance, not by heuristics:

  persona  — facts with source_kind stating / user_edit authority ("the user told
             me this") become Q→A pairs in the user's own voice context. These are
             ground truth about the person.
  routines — temporal/persona-adjacent facts with validity metadata become
             schedule-flavoured examples, so the model learns "standup is Tuesday"
             rather than hallucinating a plausible day.
  dialogue — compacted real conversations, where available, become multi-turn
             examples with the assistant's actual tone.

Every row passes through security.redact() BEFORE it is written: the dataset file
leaves the machine (it goes to Colab), so a leaked secret in memory must not ride
along. Law 10's gate applies afterwards: bench --gate decides whether the
fine-tuned model may replace the base, never vibes.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from ..security import redact
from ..security.redact import RedactionReport

USER_AUTHORITY_KINDS = ("stated", "user_edit")
_SYSTEM_PERSONA = (
    "You are FRIDAY, {name}'s personal assistant. You know their life because they "
    "told you, you say so plainly, and you never invent a detail to sound helpful."
)

# Question templates keep the pairs natural rather than parroting the store schema.
_FACT_TEMPLATES: dict[str, str] = {
    "schedule": "What does my {label} look like?",
    "housing": "What do you know about my living situation — anything about {label}?",
    "work": "About my work — what's my {label}?",
    "people": "Who is my {label}?",
    "preference": "What's my preference on {label}?",
    "identity": "What do you know about me — {label}?",
    "general": "What do you know about my {label}?",
}


@dataclass
class ExportStats:
    facts_seen: int = 0
    persona_rows: int = 0
    routine_rows: int = 0
    redacted_rows: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


def _label(predicate: str) -> str:
    return predicate.replace("_", " ").strip()


# Schedule-ish predicates get the routines stratum regardless of their decay class:
# the point is that the model learns "standup is Tuesday" as a TEMPORAL fact, not a
# static preference. (The facts table has no `domain` column — the markdown file a
# fact lives in is not its schema — so classification keys off predicate_class,
# which IS stored, plus this hint set.)
_SCHEDULE_HINTS = ("day", "time", "schedule", "routine", "standup", "meeting",
                   "gym", "prayer", "class", "wake", "sleep", "lunch", "dinner")


def _class_of(predicate_class: str | None, predicate: str) -> str:
    p = predicate.lower()
    if any(h in p for h in _SCHEDULE_HINTS):
        return "schedule"
    pc = (predicate_class or "").lower()
    if pc == "identity":
        return "identity"
    if pc == "preference":
        return "preference"
    if pc == "relationship":
        return "people"
    if pc == "location":
        return "housing"
    return "general"


def _ask_name(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT object FROM facts WHERE predicate IN ('name','full_name','called') "
        "AND retracted_at IS NULL ORDER BY asserted_at DESC LIMIT 1").fetchone()
    return row["object"] if row else "the user"


def export_dataset(conn: sqlite3.Connection, *, assistant_name: str = "FRIDAY",
                   max_rows: int = 4000) -> tuple[list[dict], ExportStats]:
    """Build instruction rows from live facts. Returns (rows, stats).

    Rows are chat-format dicts: {"messages": [{role, content}, ...]}. The file they
    land in is deliberately simple JSONL — every trainer from unsloth to axolotl
    reads it, and a format that outlives the training tool is a format you can
    re-train from in two years.
    """
    stats = ExportStats()
    rows: list[dict] = []
    system = _SYSTEM_PERSONA.replace("{name}", _ask_name(conn))

    seen: set[tuple[str, str]] = set()
    for r in conn.execute(
            "SELECT predicate, object, predicate_class, source_kind, source_quote, "
            "valid_from, valid_to, confidence FROM facts "
            "WHERE retracted_at IS NULL ORDER BY confidence DESC, asserted_at DESC"):
        stats.facts_seen += 1
        kind = (r["source_kind"] or "").lower()
        if kind not in USER_AUTHORITY_KINDS:
            # Inferred facts stay OUT of ground truth. Fine-tuning on your own
            # guesses is how the model's confidence in its own errors gets
            # parboiled into the weights — the dataset would teach the model to
            # believe what it already wrongly believes.
            stats.skipped["inferred_or_external"] = stats.skipped.get(
                "inferred_or_external", 0) + 1
            continue
        pred, obj = r["predicate"], (r["object"] or "").strip()
        if not pred or not obj:
            stats.skipped["empty"] = stats.skipped.get("empty", 0) + 1
            continue
        key = (pred, obj.lower())
        if key in seen:
            stats.skipped["duplicate"] = stats.skipped.get("duplicate", 0) + 1
            continue
        seen.add(key)

        cls = _class_of(r["predicate_class"], pred)
        question = _FACT_TEMPLATES[cls].format(label=_label(pred))
        when = ""
        if r["valid_from"]:
            when = f" (since {r['valid_from']})" if not r["valid_to"] else \
                f" (valid {r['valid_from']} → {r['valid_to']})"
        answer = f"Your {_label(pred)} is {obj}{when}."

        rep = RedactionReport()
        q_red = redact(question, report=rep)
        a_red = redact(answer, report=rep)
        if rep.count:
            stats.redacted_rows += 1
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": q_red},
            {"role": "assistant", "content": a_red},
        ], "_stratum": "routines" if cls == "schedule" else "persona"})
        if cls == "schedule":
            stats.routine_rows += 1
        else:
            stats.persona_rows += 1
        if len(rows) >= max_rows:
            break

    return rows, stats


def write_jsonl(rows: list[dict], path) -> int:
    """Write chat-format JSONL. Returns bytes written. The `_stratum` field is
    metadata for analysis; unsloth ignores unknown keys, and keeping it costs
    nothing while telling you what the run actually trained on."""
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))
