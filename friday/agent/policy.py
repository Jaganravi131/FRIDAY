"""Law 5 — enforce permissions in the DATA LAYER, never in the prompt.

Telling a model "please don't read the screen unless enabled" is not enforcement;
it is a suggestion that survives exactly until the model has a reason to ignore it.
Every tool call therefore passes through `check()`, which consults the Sense
Registry in SQLite and returns a verdict the *caller* acts on. The model never gets
to decide, and never learns whether a sense exists that it wasn't told about.

Three layers, of which this file is layer 2 (execution interception); layer 1 is
capability gating in the registry itself and layer 3 is data-layer middleware that
filters rows before they can reach a prompt (docs/architecture/09).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..util import now_iso


@dataclass
class Decision:
    verdict: str                     # allowed | denied | confirmation_required | redacted
    sense: str | None = None
    reason: str = ""
    redactions: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.verdict in ("allowed", "redacted")


#: tool -> the sense it needs. Absent means "no sense required" (pure cognition).
TOOL_SENSE: dict[str, str] = {
    "screen_read": "screen.capture",
    "screen_search": "screen.capture",
    "mic_listen": "mic.listen",
    "calendar_read": "calendar.read",
    "calendar_write": "calendar.write",
    "email_read": "email.read",
    "email_send": "email.send",
    "location_read": "location.read",
    "shell_sandboxed": "shell.exec",
    "shell": "shell.exec",
    "file_write": "fs.write",
    "http_post": "network.egress",
}

#: Irreversible or externally-visible actions always require a human "yes", even
#: with the sense enabled. This is the confirmation gate from
#: docs/architecture/08, enforced here rather than hoped for.
ALWAYS_CONFIRM: set[str] = {
    "email_send", "shell", "file_write", "http_post", "calendar_write",
    "delete_fact", "purge_memory",
}

#: Scope-sensitive: heartbeat and dreaming run unattended, so anything that
#: touches the outside world is denied outright rather than queued for a
#: confirmation nobody is there to give.
UNATTENDED_DENY: set[str] = {
    "email_send", "shell", "http_post", "file_write", "calendar_write",
}


def check(
    conn: sqlite3.Connection,
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    scope: str = "interactive",
) -> Decision:
    """The single choke point. Deterministic, table-driven, no model involved."""
    args = args or {}
    sense = TOOL_SENSE.get(tool_name)

    if tool_name in ALWAYS_CONFIRM and scope == "interactive":
        return Decision("confirmation_required", sense,
                        "irreversible or externally visible — show the exact payload first")

    if scope in ("heartbeat", "dreaming", "eval") and tool_name in UNATTENDED_DENY:
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
            decision="denied", detail="unattended scope may not touch the outside world")
        return Decision("denied", sense, f"{scope} runs unattended; this needs you present")

    if sense is None:
        return Decision("allowed", None, "no sense required")

    row = conn.execute("SELECT * FROM senses WHERE id=?", (sense,)).fetchone()
    if row is None:
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
            decision="denied", detail="sense not registered")
        return Decision("denied", sense,
                        f"sense '{sense}' has never been granted. Ask, don't assume.")
    if not int(row["enabled"]):
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
            decision="denied", detail="sense registered but disabled")
        return Decision("denied", sense, f"sense '{sense}' is registered but switched off")

    if not _scope_allows(row["scope"], args):
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
            decision="denied", detail="outside the granted scope")
        return Decision("denied", sense,
                        f"'{sense}' is granted but this target is outside its scope")

    log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
        decision="allowed")
    return Decision("allowed", sense)


def _scope_allows(scope_json: str | None, args: dict) -> bool:
    """Deny-list wins over allow-list. An absent scope means 'all of this sense'."""
    if not scope_json:
        return True
    try:
        scope = json.loads(scope_json)
    except (json.JSONDecodeError, TypeError):
        return False           # a corrupt grant is a denied grant
    target = str(args.get("target") or args.get("app") or args.get("path") or "")
    for bad in scope.get("deny", []) or []:
        if bad and bad.lower() in target.lower():
            return False
    allow = scope.get("allow") or []
    if allow and target:
        return any(a.lower() in target.lower() for a in allow)
    return True


def _target(args: dict) -> str:
    for k in ("target", "path", "app", "query", "predicate"):
        if k in args:
            return str(args[k])[:200]
    return json.dumps(args, ensure_ascii=False)[:200]


def log(conn: sqlite3.Connection, *, actor: str, action: str, target: str | None = None,
        sense_id: str | None = None, decision: str = "allowed", detail: str | None = None) -> None:
    """The audit trail. Every access, allowed or not — including YOUR hand-edits
    (docs/architecture/10 Day 3), because being able to see what the agent did is
    the precondition for trusting it."""
    conn.execute(
        """INSERT INTO audit(ts, actor, action, target, sense_id, decision, detail)
           VALUES (?,?,?,?,?,?,?)""",
        (now_iso(), actor, action, target, sense_id, decision, detail),
    )
    conn.commit()


# ── Sense Registry management ──────────────────────────────────────────────────

def grant(conn: sqlite3.Connection, sense_id: str, *, scope: dict | None = None,
          retention_d: int | None = None, granted_by: str = "user",
          redact_rules: list | None = None) -> None:
    """Tiered consent: each new sense is approved individually, by you, explicitly.
    Nothing is enabled by default — a fresh install has an empty registry, so every
    check() above denies until you grant."""
    conn.execute(
        """INSERT OR REPLACE INTO senses(id, enabled, scope, retention_d, redact_rules,
                                          granted_at, granted_by)
           VALUES (?,?,?,?,?,?,?)""",
        (sense_id, 1, json.dumps(scope or {}), retention_d,
         json.dumps(redact_rules or []), now_iso(), granted_by),
    )
    log(conn, actor="user", action="sense.grant", target=sense_id, sense_id=sense_id,
        decision="allowed", detail=json.dumps(scope or {}))


def revoke(conn: sqlite3.Connection, sense_id: str) -> None:
    conn.execute("UPDATE senses SET enabled=0 WHERE id=?", (sense_id,))
    log(conn, actor="user", action="sense.revoke", target=sense_id, sense_id=sense_id,
        decision="denied")


def enabled_senses(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT id FROM senses WHERE enabled=1 ORDER BY id").fetchall()
    return [r["id"] for r in rows]


def refusal(d: Decision, tool_name: str) -> dict:
    """A structured refusal the model can read and act on — not an exception.

    Returning readable text matters: a small model that gets a stack trace retries
    the same call. A model that gets "sense 'screen.capture' has never been granted,
    ask the user" asks the user.
    """
    return {
        "denied": True,
        "tool": tool_name,
        "verdict": d.verdict,
        "sense": d.sense,
        "reason": d.reason,
        "what_to_do": (
            "Tell the user this needs a permission they haven't granted, and stop. "
            "Do not retry, and do not work around it with another tool."
        ),
    }
