"""Connectors — FRIDAY's doorways to your other accounts (docs/architecture/18).

A connector is a small driver with a hard contract, because the failure mode it
lives near is not "API error" but "prompt-injected content from your inbox walked
into a tool call". Every connector MUST therefore:

  1. Declare which SENSE it consumes (email.read etc.). The tool that fronts it is
     registered under that sense, so tiered consent applies unchanged — you approve
     gmail.read separately from calendar.read, and `friday senses` stays the one
     place that shows the whole picture.
  2. Read credentials from config/connectors.json and NOWHERE ELSE. Not from
     memory/ (a fact is quotable into a prompt), not from env vars left lying in
     process dumps, and never logged — the audit trail records the ACCESS, never
     the SECRET. Redaction patterns in security/ already treat app-password-shaped
     strings as sensitive.
  3. Treat every byte it returns as IMPORTED content: launcher tools pass it
     through security.detect()/neutralize() before it can enter a prompt, and
     hostile payloads are withheld and audited rather than summarised.

Connectors here are READ-first. Reading your mail is reversible and boring;
sending mail as you is neither, which is why email_send is in policy.ALWAYS_CONFIRM
and UNATTENDED_DENY regardless of which connector implements it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .. import paths

CONFIG_NAME = "connectors.json"

# Fields in a connector's config section that must never be echoed back to a
# prompt, a log line, or an error message. Checked by name, not value.
SECRET_FIELDS = frozenset({"app_password", "password", "token", "api_key", "secret"})


def load_config(connector_id: str) -> dict:
    """One connector's section of config/connectors.json, with secrets marked.

    Missing file or missing section returns {} — callers treat that as
    "not configured" and say so, which is a better failure than a crash in a
    tool call (a crash is a crashed turn). Corrupt JSON also returns {} but the
    connector reports itself as misconfigured rather than absent.
    """
    p = paths.CONFIG / CONFIG_NAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"__corrupt__": str(p)}
    return data.get(connector_id) or {}


def config_help(connector_id: str, example: dict) -> str:
    """The exact stanza to paste, shown when a connector is asked before it is
    configured. example must contain PLACEHOLDERS, never a real shape resembling a
    secret — this string is designed to be safe to print into a chat."""
    pretty = json.dumps({connector_id: example}, indent=2)
    return (f"not configured. Add this to config/connectors.json in your FRIDAY root "
            f"(the file is never committed and never read into memory):\n{pretty}")


@dataclass
class Item:
    """One thing a connector fetched, normalised so the agent and the ledger see
    one shape regardless of source."""
    connector: str                  # e.g. "gmail"
    id: str                         # opaque per-connector identifier
    title: str                      # subject / first line — preview-safe
    body: str                       # full text; tools cap this before shipping
    author: str = ""                # from/envelope — shown, never trusted
    when: str = ""                  # ISO-ish date string if known
    extra: dict = field(default_factory=dict)


class Connector:
    """The contract. Drivers subclass; the launcher tools (agent/connectors_tools)
    do policy + containment and are the ONLY things the agent loop talks to."""

    id: str = "abstract"
    sense: str = ""                 # the policy sense this connector consumes

    def status(self) -> dict:
        raise NotImplementedError

    def fetch(self, **kw) -> list[Item]:
        raise NotImplementedError


_REGISTRY: dict[str, type[Connector]] = {}


def register(cls: type[Connector]) -> type[Connector]:
    _REGISTRY[cls.id] = cls
    return cls


def get(connector_id: str) -> Connector | None:
    cls = _REGISTRY.get(connector_id)
    return cls() if cls else None


def known() -> list[str]:
    # Importing the drivers is what populates the registry; keep the import list
    # next to the registry so a new driver can't forget either step.
    from . import caldav, gmail  # noqa: F401
    return sorted(_REGISTRY)
