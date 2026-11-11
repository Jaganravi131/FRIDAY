"""The heartbeat — doc 08's proactivity engine, deterministic first.

    Law 1: proactivity is a HEARTBEAT, not a cron job. A scheduler that runs a
    script is automation; a tick that DECIDES whether to speak is a companion.

What exists here is the Week-9 layer of doc 08 §2: the tick loop, the tick-ack
schema, HEARTBEAT_OK suppression, the quiet-hours and annoyance-budget rules,
and audit logging — plus ONE standing check (the deadline watch), because doc
08's build order says add checks one at a time and measure. The remaining
checks in `soul/HEARTBEAT.md` (compilation, decay, retraction pairs) are
Dreaming-phase work and stay unimplemented on purpose: a heartbeat that speaks
when it has nothing measured to say is the failure mode the whole design
exists to prevent.

THE SUPPRESSION CONTRACT. A tick with nothing to say returns ``HEARTBEAT_OK``
and action ``silent``. The gateway (and any surface) must never deliver a
silent tick — it is logged to ``audit`` and dropped. "It should be *boring*
almost always. That's what makes the 20% land."

Deterministic ON PURPOSE. Every check here is read-only, stdlib-only, and
needs no model — so the heartbeat works when no model is reachable (Law 4),
and every notification can be explained by a query rather than a vibe. When a
model is attached, the tick can become an agentic turn with ``scope=
"heartbeat"`` (the loop and the unattended-scope policy denies already exist);
the schema and the suppression contract stay the same. The model EARNS its way
into the tick; it does not replace the floor.

Rules implemented (each pinned by a test, mirrored in `soul/HEARTBEAT.md`):
* default silent — speaking is the exception and must carry a reason
* at most ONE message per tick; more than one is batched to the next tick
* never between 22:30 and 08:00 unless urgency == "high"
* never more than 2 delivered per hour or 8 per day (the annoyance budget)
* a deadline notifies ONCE per fact — dedup lives in the audit trail, so
  "what did FRIDAY say and why" is one query
* no irreversible action: unattended scopes are denied by the policy engine,
  and this module never calls a tool that writes
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Callable

from ..config import aliases_for
from ..util import parse_iso, swallowed

#: The most important token in the system (doc 08 §2): nothing to say, said
#: out loud, so the suppressing side can find it.
HEARTBEAT_OK = "HEARTBEAT_OK"

#: Doc 08's annoyance budget, verbatim: "NEVER send more than 2 proactive
#: messages per hour, or 8 per day. If you have more than that, you are being
#: annoying. Batch them."
MAX_PER_HOUR = 2
MAX_PER_DAY = 8

#: Quiet hours, from soul/HEARTBEAT.md: no initiation between 22:30 and 08:00
#: unless the urgency is high.
QUIET_START = time(22, 30)
QUIET_END = time(8, 0)

#: The deadline watch window: notify when a stored date is this close (or up
#: to this long past — a just-missed deadline is still worth one word).
DEADLINE_WINDOW_DAYS = 30
DEADLINE_PAST_GRACE_DAYS = 7
#: Inside this many days (or overdue) a deadline is high urgency: it may speak
#: in the quiet window. Far-out dates may not.
DEADLINE_HIGH_DAYS = 3

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@dataclass
class Tick:
    """The tick-ack, doc 08 §2. `action='silent'` is the default and the norm;
    a tick that speaks must fill `reason`. `open_loops_touched` is part of the
    schema so the field exists when open loops do — it is [] today, honestly."""

    action: str = "silent"                 # silent | notify | act
    urgency: str = "low"                   # low | normal | high
    message: str = ""
    surface: str = "mobile"
    reason: str = ""
    open_loops_touched: list[str] = field(default_factory=list)
    next_check_hint: str = ""
    #: Which check spoke (a fact id for the deadline watch). The audit `target`
    #: and the dedup key.
    check: str = ""

    @property
    def is_ok(self) -> bool:
        return self.action == "silent"

    def as_dict(self) -> dict:
        return {
            "action": self.action, "urgency": self.urgency, "message": self.message,
            "surface": self.surface, "reason": self.reason,
            "open_loops_touched": list(self.open_loops_touched),
            "next_check_hint": self.next_check_hint,
        }


def _quiet_hours(now: datetime) -> bool:
    t = now.time()
    return t >= QUIET_START or t < QUIET_END


def _delivered_count(conn: sqlite3.Connection, *, hour_ago: datetime,
                     today: str) -> tuple[int, int]:
    """(delivered in the last hour, delivered today). The audit trail is the
    budget's source of truth: the count is what FRIDAY actually said, not what
    it meant to say."""
    rows = conn.execute(
        "SELECT ts FROM audit WHERE action='heartbeat.notify' AND decision='delivered'"
    ).fetchall()
    hour = day = 0
    for r in rows:
        ts = parse_iso(r["ts"])
        if ts is None:
            continue
        if ts >= hour_ago:
            hour += 1
        if r["ts"][:10] == today:
            day += 1
    return hour, day


def _already_notified(conn: sqlite3.Connection, fact_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM audit WHERE action='heartbeat.notify' AND decision='delivered'"
        " AND target=? LIMIT 1",
        (fact_id,),
    ).fetchone()
    return row is not None


def _humanise(predicate: str) -> str:
    """`lease_renewal_due` -> 'Lease renewal due'. Uses the alias table when it
    has the predicate (the same human phrasings retrieval scores against) and
    falls back to splitting snake_case."""
    aliases = aliases_for(predicate)
    words = (aliases[0] if aliases else predicate.replace("_", " ")).strip()
    return (words[:1].upper() + words[1:]) if words else "Fact"


def deadline_watch(conn: sqlite3.Connection, now: datetime) -> Tick | None:
    """THE one standing check: a date stored in a live fact, close enough to
    matter. Read-only, deterministic, notifies once per fact.

    Deliberately NOT a general "temporal reasoning" pass: it matches ISO dates
    in the object only, so `standup_day: Tuesday 09:30` (a schedule, not a
    deadline) and `lives_in: Chennai (since 2023-06)` (a validity range, not an
    object) stay quiet. A check that invents interpretations is a check that
    gets switched off.
    """
    best: tuple[int, sqlite3.Row] | None = None
    rows = conn.execute(
        "SELECT id, predicate, object FROM facts WHERE retracted_at IS NULL"
    ).fetchall()
    for r in rows:
        if _already_notified(conn, r["id"]):
            continue
        m = _ISO_DATE.search(r["object"] or "")
        if not m:
            continue
        try:
            due = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        days = (due - now.date()).days
        if not (-DEADLINE_PAST_GRACE_DAYS <= days <= DEADLINE_WINDOW_DAYS):
            continue
        if best is None or days < best[0]:
            best = (days, r)
    if best is None:
        return None
    days, r = best
    m = _ISO_DATE.search(r["object"] or "")
    due_str = m.group(1) if m else (r["object"] or "").strip()
    what = _humanise(r["predicate"])
    if days > 0:
        when = f"is on {due_str}"
        note = f"in {days} day{'s' if days != 1 else ''}"
    elif days == 0:
        when = f"is today ({due_str})"
        note = "today"
    else:
        when = f"was due {due_str}"
        note = f"{-days} day{'s' if -days != 1 else ''} ago"
    urgency = "high" if days <= DEADLINE_HIGH_DAYS else "normal"
    return Tick(
        action="notify",
        urgency=urgency,
        message=f"{what} {when} — {note}.",
        reason=f"deadline watch: within the {DEADLINE_WINDOW_DAYS}-day window",
        check=r["id"],
    )


#: Checks run in order; the first to speak wins the tick. A check that raises
#: is swallowed and skipped — a broken check must not break the heartbeat, and
#: must not become a silent dependency on exception order either.
Check = Callable[[sqlite3.Connection, datetime], "Tick | None"]
DEFAULT_CHECKS: tuple[Check, ...] = (deadline_watch,)

_URGENCY_ORDER = {"high": 0, "normal": 1, "low": 2}


def beat(conn: sqlite3.Connection, *, now: datetime | None = None,
         checks: tuple[Check, ...] | None = None) -> Tick:
    """One heartbeat tick. Never raises for data reasons; a check's failure is
    that check's silence. Writes the tick to the audit trail either way — a
    silent tick is LOGGED, never delivered."""
    from .policy import log

    now = now or datetime.now().astimezone()
    if now.tzinfo is None:                       # a naive `now` means local time
        now = now.astimezone()
    candidates: list[Tick] = []
    for check in (DEFAULT_CHECKS if checks is None else checks):
        try:
            tick = check(conn, now)
        except Exception as exc:  # noqa: BLE001 — see module docstring
            swallowed("friday/agent/heartbeat.py:beat", exc)
            continue
        if tick is not None and tick.action == "notify":
            candidates.append(tick)

    tick = Tick()
    if candidates:
        # ONE message per tick (doc 08: batch, don't burst). Highest urgency
        # first; ties break by check order, which is registry order — stable.
        candidates.sort(key=lambda t: _URGENCY_ORDER.get(t.urgency, 3))
        chosen = candidates[0]
        hour, day = _delivered_count(
            conn, hour_ago=now - timedelta(hours=1), today=now.date().isoformat())
        if hour >= MAX_PER_HOUR or day >= MAX_PER_DAY:
            tick = Tick(reason=(f"annoyance budget: {hour}/{MAX_PER_HOUR} this hour, "
                                f"{day}/{MAX_PER_DAY} today — holding the message"),
                        next_check_hint=chosen.check)
        elif _quiet_hours(now) and chosen.urgency != "high":
            tick = Tick(reason=f"quiet hours (until 08:00): {chosen.message}",
                        next_check_hint=chosen.check)
        else:
            tick = chosen
            log(conn, actor="heartbeat", action="heartbeat.notify",
                target=chosen.check, decision="delivered", detail=chosen.message)
    log(conn, actor="heartbeat", action="heartbeat.tick",
        decision="silent" if tick.is_ok else "delivered",
        detail=HEARTBEAT_OK if tick.is_ok else tick.message)
    return tick
