"""S0 — the Trace. Append-only raw capture, one JSONL file per day.

Everything downstream is derived from this. It is the only stage that is
*irreducible*: you can re-derive episodes, facts and skills from traces, but you
can never re-derive a trace you didn't write.

Two properties are load-bearing:

  * APPEND-ONLY, and never blocks the response. The agent loop writes the trace
    after it has answered, not before.
  * NEVER COMMITTED. `.gitignore` excludes `memory/traces/*.jsonl`. Ambient
    capture does not leave the machine. Facts (S2) are the git-pushed layer, and
    they carry a `source_quote` + `source_refs` pointer back into the trace.

Char offsets are recorded on every span because the RSC's salience labels are
*character ranges* (`supervision_spans.char_start/char_end`), and you cannot
recover them after the fact.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .. import paths
from ..util import new_id, now_iso


@dataclass
class Trace:
    """One interaction, however small."""

    kind: str                       # turn|heartbeat|dreaming|tool|eval|capture
    content: str = ""
    role: str = "user"              # user|assistant|system|tool
    session_id: str | None = None
    turn_idx: int | None = None
    surface: str = "cli"            # cli|desktop|web|mobile|voice-node
    mode: str = "text"              # text|voice|image
    #: The policy scope this interaction ran under. `policy.check` denies the
    #: outside world in unattended scopes, so a trace without its scope cannot be
    #: audited afterwards: you could not tell whether a denied `email_send` was
    #: FRIDAY being careful at 3am or a bug in an attended turn.
    scope: str | None = None        # interactive|heartbeat|dreaming|eval
    #: Which senses were enabled for this turn. Recorded so the tiered-consent story
    #: is reconstructable from the trace alone, and so the behavioural-signal
    #: analysis can tell a capability the user granted from one it assumed.
    senses: list[str] = field(default_factory=list)
    tier: str | None = None         # L0..L5 actually used
    sense_id: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int | None = None
    tokens_out: int | None = None
    ttft_ms: int | None = None
    total_ms: int | None = None
    cache_hit: bool | None = None
    ledger_hash: str | None = None
    # ⭐ behavioural signals — the free GEPA feedback function at ₹0
    # (docs/architecture/12 §8.2). These are a *better* reflection signal than a
    # model's opinion about itself, and they cost nothing to record.
    barge_in: bool = False          # user interrupted mid-speech
    rephrased: bool = False         # user immediately restated the same ask
    corrected: bool = False         # user said "no, I meant…"
    unnecessary_escalation: bool = False
    tool_failures: int = 0
    user_rating: int | None = None  # explicit thumbs, if offered
    id: str = field(default_factory=lambda: new_id("t"))
    ts: str = field(default_factory=now_iso)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class TraceWriter:
    """Append-only JSONL writer, partitioned by day."""

    def __init__(self, root: Path | None = None):
        self.root = root or paths.TRACES
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, ts: str | None = None) -> Path:
        stem = (ts or now_iso())[:10]
        return self.root / f"{stem}.jsonl"

    def append(self, trace: Trace) -> tuple[Path, int, int]:
        """Write one trace. Returns (path, char_start, char_end).

        The char offsets are what `supervision_spans` stores, so the salience
        label points at an exact substring of an exact file — reproducible
        training data with no copy of the content in the DB.
        """
        p = self.path_for(trace.ts)
        line = trace.to_json() + "\n"
        existed = p.exists()
        start = p.stat().st_size if existed else 0
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return p, start, start + len(line)

    def append_content(self, text: str, kind: str = "capture", **kw: Any) -> tuple[str, Path, int, int]:
        """Convenience: build a Trace from raw text and append it."""
        t = Trace(kind=kind, content=text, **kw)
        p, s, e = self.append(t)
        return t.id, p, s, e

    def read_day(self, day: str) -> list[dict]:
        p = self.root / f"{day}.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line must not break the day
        return out

    def span_text(self, trace_file: str | Path, char_start: int, char_end: int) -> str:
        """Resolve a supervision span back to its text. Used to build RSC batches."""
        p = Path(trace_file)
        if not p.is_absolute():
            p = paths.ROOT / p
        if not p.exists():
            return ""
        data = p.read_text(encoding="utf-8")
        return data[char_start:char_end]

    def all_days(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))


def index_traces(conn: sqlite3.Connection, writer: TraceWriter | None = None) -> int:
    """Make traces searchable without copying them into git-tracked Markdown.

    Traces stay on disk and out of the repo; only their index lives in SQLite,
    and SQLite lives in artifacts/ which is gitignored too. Nothing sensitive is
    ever pushed.
    """
    from ..store import db

    w = writer or TraceWriter()
    n = 0
    for day in w.all_days():
        for rec in w.read_day(day):
            body = (rec.get("content") or "").strip()
            if not body:
                continue
            ref = f"trace:{rec.get('id', '')}"
            db.fts_upsert(conn, ref, "daily", body[:4000], subject=rec.get("session_id") or "")
            n += 1
    conn.commit()
    return n


def behavioural_signals(conn: sqlite3.Connection, days: int = 30) -> dict[str, Any]:
    """Aggregate the free reflection signals (docs/architecture/12 §8.2).

    This is what replaces a cloud GEPA teacher at ₹0: instead of asking a frontier
    model why a prompt failed, count the ways the *user* told you it failed.
    """
    w = TraceWriter()
    counts = {"barge_in": 0, "rephrased": 0, "corrected": 0,
              "unnecessary_escalation": 0, "tool_failures": 0, "traces": 0}
    for day in w.all_days()[-days:]:
        for rec in w.read_day(day):
            counts["traces"] += 1
            for k in ("barge_in", "rephrased", "corrected", "unnecessary_escalation"):
                if rec.get(k):
                    counts[k] += 1
            counts["tool_failures"] += int(rec.get("tool_failures") or 0)
    counts["days"] = min(days, len(w.all_days()))
    return counts
