"""Tests for the agent policy surface and the tool registry.

The two properties that matter here are both about *not assuming*:

1. NOTHING IS ENABLED BY DEFAULT. A fresh install has an empty `senses` registry, so
   every check() that needs a sense denies until you grant it. Tiered consent is the
   privacy requirement, and "deny until granted" is what makes it real rather than a
   settings page nobody reads.
2. IRREVERSIBLE OR EXTERNALLY VISIBLE ACTIONS NEED A HUMAN "YES" EVEN WHEN THE SENSE
   IS ENABLED. Granting `email.send` authorises FRIDAY to read and compose; it does
   not authorise it to send on its own initiative. `ALWAYS_CONFIRM` enforces that.

Plus the unattended rule: heartbeat and dreaming run with nobody watching, so a
confirmation prompt would go unanswered. Those scopes deny outright instead of
queueing.
"""
from __future__ import annotations

from friday.agent.policy import (
    ALWAYS_CONFIRM,
    Decision,
    TOOL_SENSE,
    UNATTENDED_DENY,
    check,
    enabled_senses,
    grant,
    log,
    refusal,
    revoke,
)


# ── tiered consent ─────────────────────────────────────────────────────────────


def test_nothing_is_enabled_on_a_fresh_install(conn):
    """The privacy default. No senses granted, so nothing that needs one may run.

    Note that ALWAYS_CONFIRM tools report `confirmation_required` rather than
    `denied`: check() tests the confirmation gate BEFORE consulting the sense
    registry. That is deliberate — the gate that a human must say yes is more
    specific than the gate that a sense must be granted, and reporting the more
    specific reason is more useful. Neither verdict allows the action, which is
    what this test actually cares about.
    """
    assert enabled_senses(conn) == []
    for tool, sense in TOOL_SENSE.items():
        d = check(conn, tool, {})
        assert not d.allowed, f"{tool} needs '{sense}' and must not run ungranted"
        assert d.verdict in ("denied", "confirmation_required"), (tool, d.verdict)
        assert d.sense == sense


def test_tools_without_a_sense_are_pure_cognition_and_always_allowed(conn):
    """A tool absent from TOOL_SENSE requires no sense: it only touches FRIDAY's own
    memory and reasoning, which is the whole point of running it."""
    for tool in ("memory_search", "memory_write", "think", "some_unlisted_tool"):
        assert tool not in TOOL_SENSE
        d = check(conn, tool, {})
        assert d.verdict == "allowed", f"{tool} should need no sense"
        assert d.sense is None
        assert "no sense required" in d.reason


def test_grant_then_check_allows(conn):
    grant(conn, "screen.capture")
    assert enabled_senses(conn) == ["screen.capture"]
    d = check(conn, "screen_read", {"target": "VS Code"})
    assert d.verdict == "allowed" and d.allowed
    assert d.sense == "screen.capture"


def test_grant_is_idempotent_and_revoke_turns_it_off(conn):
    grant(conn, "mic.listen")
    grant(conn, "mic.listen")            # granting twice must not duplicate or error
    assert enabled_senses(conn) == ["mic.listen"]
    assert check(conn, "mic_listen", {}).verdict == "allowed"

    revoke(conn, "mic.listen")
    assert enabled_senses(conn) == []
    d = check(conn, "mic_listen", {})
    assert d.verdict == "denied"
    assert "switched off" in d.reason or "never been granted" in d.reason


def test_granting_one_sense_does_not_leak_to_another(conn):
    """Each sense is approved individually. Granting screen capture must not quietly
    enable the microphone."""
    grant(conn, "screen.capture")
    assert check(conn, "screen_read", {}).verdict == "allowed"
    assert check(conn, "mic_listen", {}).verdict == "denied"
    assert check(conn, "email_read", {}).verdict == "denied"
    assert check(conn, "location_read", {}).verdict == "denied"


# ── scope: deny-list wins over allow-list ─────────────────────────────────────


def test_grant_scope_allow_list_restricts_targets(conn):
    grant(conn, "screen.capture", scope={"allow": ["vscode"]})
    assert check(conn, "screen_read", {"target": "VSCode window"}).verdict == "allowed"
    d = check(conn, "screen_read", {"target": "Banking app"})
    assert d.verdict == "denied"
    assert "outside its scope" in d.reason


def test_grant_scope_deny_list_beats_allow_list(conn):
    """A deny entry wins even when the target also matches an allow entry. Narrow
    permissions should never be widened by a broader rule sitting next to them."""
    grant(conn, "screen.capture", scope={"allow": ["code"], "deny": ["secret"]})
    assert check(conn, "screen_read", {"target": "code editor"}).verdict == "allowed"
    d = check(conn, "screen_read", {"target": "code secret vault"})
    assert d.verdict == "denied", "the deny list must win over the allow list"


def test_an_absent_scope_means_all_of_that_sense(conn):
    grant(conn, "screen.capture")            # no scope at all
    assert check(conn, "screen_read", {"target": "anything whatsoever"}).verdict == "allowed"
    assert check(conn, "screen_search", {"query": "invoice"}).verdict == "allowed"


def test_a_corrupt_scope_grant_denies_rather_than_allowing(conn):
    """Fail closed. A malformed scope must not be interpreted as 'unrestricted'."""
    grant(conn, "screen.capture", scope=None)
    conn.execute("UPDATE senses SET scope=? WHERE id=?", ("{not json", "screen.capture"))
    d = check(conn, "screen_read", {"target": "anything"})
    assert d.verdict == "denied", "a corrupt grant is a denied grant"


# ── the confirmation gate ─────────────────────────────────────────────────────


def test_always_confirm_requires_a_yes_even_when_the_sense_is_granted(conn):
    """⭐ The gate that stops FRIDAY acting on its own initiative. Irreversible or
    externally visible actions surface the exact payload and wait."""
    for tool in ALWAYS_CONFIRM:
        sense = TOOL_SENSE.get(tool)
        if sense:
            grant(conn, sense)
        d = check(conn, tool, {"target": "someone@example.com"}, scope="interactive")
        assert d.verdict == "confirmation_required", f"{tool} must confirm"
        assert not d.allowed
        assert "show the exact payload" in d.reason


def test_confirmation_only_applies_when_someone_is_present(conn):
    """In an unattended scope there is nobody to confirm, so ALWAYS_CONFIRM tools are
    denied rather than queued forever. UNATTENDED_DENY is checked before
    ALWAYS_CONFIRM, which is why heartbeat yields `denied` and not a confirmation
    nobody will ever answer."""
    grant(conn, "email.send")
    assert check(conn, "email_send", {}, scope="interactive").verdict == "confirmation_required"
    assert check(conn, "email_send", {}, scope="heartbeat").verdict == "denied"
    assert check(conn, "email_send", {}, scope="dreaming").verdict == "denied"


def test_gate_order_is_confirmation_then_unattended_then_sense(conn):
    """The order of the three gates is load-bearing and worth pinning:

      1. ALWAYS_CONFIRM + interactive  -> confirmation_required (never checked
         against the registry first, so an ungranted destructive tool still tells
         you it needs a human yes)
      2. unattended scope + UNATTENDED_DENY -> denied (checked before the registry,
         so heartbeat never even gets to ask whether the sense is granted)
      3. otherwise the sense registry decides denied / allowed

    Reordering 1 and 3 would make an ungranted `email_send` report "sense not
    granted" instead of "this needs your explicit yes", which is the less useful
    message and hides the fact that granting would not be enough.
    """
    # gate 1 fires with nothing granted at all
    d = check(conn, "email_send", {}, scope="interactive")
    assert d.verdict == "confirmation_required"
    assert "never been granted" not in d.reason

    # gate 2 fires even with the sense granted
    grant(conn, "email.send")
    assert check(conn, "email_send", {}, scope="heartbeat").verdict == "denied"

    # gate 3 is what's left
    assert check(conn, "email_read", {}, scope="interactive").verdict == "denied"
    grant(conn, "email.read")
    assert check(conn, "email_read", {}, scope="interactive").verdict == "allowed"


def test_unattended_scopes_deny_anything_touching_the_outside_world(conn):
    """Heartbeat, dreaming and eval run with nobody watching. They may think and
    they may read FRIDAY's own memory; they may not email, shell out, write files,
    post to the network or edit the calendar."""
    for sense in TOOL_SENSE.values():
        grant(conn, sense)
    for scope in ("heartbeat", "dreaming", "eval"):
        for tool in UNATTENDED_DENY:
            d = check(conn, tool, {}, scope=scope)
            assert d.verdict == "denied", f"{tool} in scope {scope} must deny"
            assert "unattended" in d.reason


def test_unattended_scope_may_still_read_and_reason(conn):
    """Denying the outside world must not paralyse the ambient modes — that is what
    makes them useful. Reading the screen and searching memory still work."""
    grant(conn, "screen.capture")
    grant(conn, "email.read")
    assert check(conn, "screen_read", {}, scope="heartbeat").verdict == "allowed"
    assert check(conn, "memory_search", {"query": "rent"}, scope="dreaming").verdict == "allowed"
    assert check(conn, "email_read", {}, scope="heartbeat").verdict == "allowed"


def test_unattended_deny_list_is_a_subset_of_always_confirm():
    """Everything denied unattended is also confirmation-gated interactively. If a
    tool were denied unattended but allowed freely interactively, that would mean the
    gate depends on who is watching rather than on what the action does."""
    assert UNATTENDED_DENY <= ALWAYS_CONFIRM
    assert ALWAYS_CONFIRM - UNATTENDED_DENY == {"delete_fact", "purge_memory"}, \
        "the two purely-destructive tools confirm interactively but are not in the " \
        "external-world deny list"


# ── the refusal message and the audit log ─────────────────────────────────────


def test_refusal_explains_itself_in_user_language(conn):
    """A denial is a conversation, not an error code. FRIDAY says what it wanted to
    do and what permission is missing."""
    d = check(conn, "mic_listen", {})
    r = refusal(d, "mic_listen")
    assert isinstance(r, dict)
    assert r.get("verdict") == "denied" or r.get("decision") == "denied"
    text = str(r)
    assert "mic.listen" in text or "microphone" in text.lower() or "sense" in text.lower()


def test_every_decision_is_written_to_the_audit_log(conn):
    """Law: FRIDAY keeps a ledger of what it did and why. A denial that is not
    logged is a denial you cannot audit afterwards."""
    check(conn, "mic_listen", {})                      # denied, ungranted
    grant(conn, "mic.listen")
    check(conn, "mic_listen", {"target": "meeting"})   # allowed

    rows = conn.execute(
        "SELECT action, decision, sense_id, target FROM audit ORDER BY rowid"
    ).fetchall()
    actions = [r["action"] for r in rows]
    # grant() is itself audited, which is correct: changing your own permissions is
    # exactly the kind of event you want to be able to look up later.
    assert "sense.grant" in actions, actions

    mic = [r for r in rows if r["action"] == "mic_listen"]
    assert len(mic) == 2, f"both the denial and the allow must be logged: {mic}"
    assert mic[0]["decision"] == "denied", "the ungranted attempt is logged as denied"
    assert mic[1]["decision"] == "allowed"
    assert mic[1]["sense_id"] == "mic.listen"
    assert mic[1]["target"] == "meeting"


def test_log_records_an_explicit_action(conn):
    log(conn, actor="cli", action="memory_write", target="lives_in",
        sense_id=None, decision="allowed", detail="direct call")
    row = conn.execute(
        "SELECT * FROM audit WHERE action='memory_write' ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["actor"] == "cli"
    assert row["target"] == "lives_in"
    assert row["detail"] == "direct call"


def test_decision_allowed_treats_redacted_as_a_pass():
    """A redacted result still ran — it ran with less data. `allowed` covers both so
    callers do not have to enumerate verdict strings."""
    assert Decision("allowed").allowed
    assert Decision("redacted", redactions=["card number"]).allowed
    assert not Decision("denied").allowed
    assert not Decision("confirmation_required").allowed
