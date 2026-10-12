"""Tests for the SMTP send connector — the FIRST write path (docs/architecture/18 §3.6).

No real network: the single egress function `friday.connectors.smtp.transport`
is stubbed at module boundary (a fake SMTP_SSL double). What is tested is the
WRITE contract — the things that make "hit send" safe to hand to a loop:

  * wire bytes == confirmed bytes (nothing added, no CC/BCC surface at all),
  * header-injection refused BEFORE any socket exists,
  * TLS-from-byte-one and the exact auth sequence,
  * server error text never echoing credentials back,
  * the gating stack: policy ALWAYS_CONFIRM for interactive scope, launcher
    refusal for 'remote' (pinning the fact that policy currently ALLOWS remote
    through — the launcher's scope gate is the load-bearing stop until the
    gateway grows a confirmation channel),
  * audit records THAT a send happened, never the body.
"""
from __future__ import annotations

import inspect
import json
import smtplib

import pytest

from friday import paths
from friday.agent import policy
from friday.agent.tools import ToolContext
from friday.connectors import get, known
from friday.store import db

PASS = "zzzz zzzz zzzz zzzz"


@pytest.fixture
def smtp_cfg(root):
    cfg = {"gmail": {"user": "me@example.com", "app_password": PASS}}
    d = paths.CONFIG
    d.mkdir(parents=True, exist_ok=True)
    (d / "connectors.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def _ctx(root, scope="interactive"):
    conn = db.connect(root / "artifacts" / "friday.db")
    return conn, ToolContext(conn=conn, scope=scope)


class _FakeSMTP:
    """Instance-recording double for smtplib.SMTP_SSL."""

    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls = []
        self.message = None
        self.fail_send = None
        _FakeSMTP.instances.append(self)

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.calls.append(("send_message",))
        self.message = msg
        if self.fail_send:
            raise self.fail_send
        return {}

    def quit(self):
        self.calls.append(("quit",))


@pytest.fixture
def fake_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr("friday.connectors.smtp.transport",
                        lambda host, port, timeout: _FakeSMTP(host, port, timeout=timeout))
    return _FakeSMTP


# ── the read/write boundary ─────────────────────────────────────────────────

def test_registered_with_send_sense_and_no_read_path():
    conn = get("gmail.smtp")
    assert conn is not None and "gmail.smtp" in known()
    assert conn.sense == "email.send"
    with pytest.raises(NotImplementedError):
        conn.fetch()
    import friday.connectors.smtp as smtp_mod
    src = inspect.getsource(smtp_mod)
    assert 'msg["Cc"' not in src and 'msg["Bcc"' not in src, \
        "the composed message must have NO cc/bcc header surface at all"
    sig = inspect.signature(smtp_mod.send)
    assert "cc" not in sig.parameters and "bcc" not in sig.parameters


def test_unconfigured_tool_returns_howto(root):
    from friday.agent.connector_tools import email_send
    conn, ctx = _ctx(root)
    out = email_send(ctx, to=["a@b.com"], subject="hi", body="there")
    assert "not configured" in out["error"]
    assert '"gmail"' in out["how_to"]
    conn.close()


# ── wire bytes == confirmed bytes ────────────────────────────────────────────

def test_wire_bytes_are_exactly_the_confirmed_bytes(smtp_cfg, fake_smtp):
    from friday.connectors.smtp import send
    body = "see you at 9:30.\n— J"
    receipt = send("me@example.com", PASS, "smtp.gmail.com",
                   to=["boss@example.com"], subject="standup time", body=body)
    srv = fake_smtp.instances[0]
    assert srv.host == "smtp.gmail.com" and srv.port == 465  # SMTP_SSL: TLS from byte one
    assert srv.calls == [("login", "me@example.com", PASS),
                         ("send_message",), ("quit",)]
    msg = srv.message
    assert msg["From"] == "me@example.com"
    assert msg["To"] == "boss@example.com"
    assert msg["Subject"] == "standup time"
    assert msg["Cc"] is None and msg["Bcc"] is None, "nothing rides along that wasn't confirmed"
    assert body in msg.get_content()
    assert receipt["sent"] and receipt["body_chars"] == len(body)


def test_one_call_one_send(smtp_cfg, fake_smtp):
    from friday.connectors.smtp import send
    send("me@example.com", PASS, "smtp.gmail.com",
         to=["a@x.com", "b@x.com"], subject="s", body="b")
    assert len(fake_smtp.instances) == 1
    assert fake_smtp.instances[0].calls.count(("send_message",)) == 1


# ── injection & cap hygiene: refused BEFORE the socket ───────────────────────

def test_header_injection_refused_before_any_socket(smtp_cfg, fake_smtp):
    from friday.connectors.smtp import send
    bad = [
        dict(to=["a@x.com"], subject="ok\r\nBcc: attacker@evil.com", body="b"),
        dict(to=["a@x.com\nBcc: attacker@evil.com"], subject="s", body="b"),
        dict(to=["not-an-address"], subject="s", body="b"),
        dict(to=["Doe, John <j@x.com>"], subject="s", body="b"),
        dict(to=[f"r{i}@x.com" for i in range(6)], subject="s", body="b"),
        dict(to=["a@x.com"], subject="s" * 201, body="b"),
        dict(to=["a@x.com"], subject="s", body="x" * 20_001),
        dict(to=[], subject="s", body="b"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            send("me@example.com", PASS, "smtp.gmail.com", **kw)
    assert fake_smtp.instances == [], "a refused send must never open a socket"


# ── the gating stack ─────────────────────────────────────────────────────────

def test_interactive_scope_requires_confirmation(smtp_cfg, root):
    conn = db.connect(root / "artifacts" / "friday.db")
    try:
        policy.grant(conn, "email.send"); conn.commit()
        d = policy.check(conn, "email_send",
                         {"to": ["a@x.com"], "subject": "s"}, scope="interactive")
        assert not d.allowed
        assert d.verdict == "confirmation_required", \
            "an irreversible, externally visible action always passes through a human yes"
    finally:
        conn.close()


def test_remote_scope_passes_policy_but_the_launcher_stops_it(smtp_cfg, fake_smtp, root):
    """PIN THE GAP: policy intentionally allows `remote` (the user is present on
    their phone), but serve has no confirm channel — so ALWAYS_CONFIRM silently
    no-ops over the gateway. The launcher's scope gate is the load-bearing stop;
    the day serve grows a confirmation round-trip this test gets relaxed, in a
    review, by deleting it — not by it drifting."""
    from friday.agent.connector_tools import email_send
    conn = db.connect(root / "artifacts" / "friday.db")
    try:
        policy.grant(conn, "email.send"); conn.commit()
        d = policy.check(conn, "email_send",
                         {"to": ["a@x.com"], "subject": "s"}, scope="remote")
        assert d.allowed, "policy allows remote — WHICH IS WHY the launcher must not"

        ctx = ToolContext(conn=conn, scope="remote")
        out = email_send(ctx, to=["a@x.com"], subject="s", body="b")
        assert not out.get("sent") and "interactive" in out["error"]
        assert fake_smtp.instances == [], "a remote-scope send must never open a socket"
    finally:
        conn.close()


def test_heartbeat_scope_denied_by_policy(smtp_cfg, root):
    conn = db.connect(root / "artifacts" / "friday.db")
    try:
        policy.grant(conn, "email.send"); conn.commit()
        d = policy.check(conn, "email_send", {"to": ["a@x.com"]}, scope="heartbeat")
        assert not d.allowed and d.verdict == "denied"
    finally:
        conn.close()


def test_send_needs_the_sense(root):
    conn = db.connect(root / "artifacts" / "friday.db")
    try:
        d = policy.check(conn, "email_send", {"to": ["a@x.com"]}, scope="interactive")
        assert not d.allowed and "email.send" in (d.sense or "") + d.reason
    finally:
        conn.close()


# ── secrets and audit ────────────────────────────────────────────────────────

def test_server_error_never_echoes_the_password(smtp_cfg, fake_smtp, root):
    """A typo'd or hostile relay can quote your credential back inside its 5xx
    text. The launcher-visible error is a type name; the audit detail likewise."""
    from friday.agent.connector_tools import email_send
    conn, ctx = _ctx(root)
    fake_smtp.instances = []  # fixture records on connect; force failure on send
    import friday.connectors.smtp as m
    orig = m.transport
    def bad(host, port, timeout):
        s = orig(host, port, timeout)
        s.fail_send = smtplib.SMTPException(f"554 rejected: {PASS} no good")
        return s
    m.transport = bad
    try:
        out = email_send(ctx, to=["a@x.com"], subject="meter reading", body="12345")
    finally:
        m.transport = orig
    assert not out["sent"] and PASS not in json.dumps(out)


def test_send_succeeded_and_failed_both_audited_body_never_logged(smtp_cfg, fake_smtp, root):
    from friday.agent.connector_tools import email_send
    conn, ctx = _ctx(root)
    body = "the whole confidential draft text that must not reach the audit log"
    out = email_send(ctx, to=["boss@example.com"], subject="meter reading", body=body)
    assert out["sent"] and out["undo"].startswith("none")
    rows = conn.execute(
        "SELECT action, target, sense_id, detail FROM audit "
        "WHERE action IN ('connector.sent', 'connector.send_failed')").fetchall()
    assert [r[0] for r in rows] == ["connector.sent"]
    assert rows[0][2] == "email.send"
    assert "boss@example.com" in rows[0][1]
    assert "meter reading" in rows[0][3]
    assert body not in rows[0][3], "the audit trail says THAT you sent, not WHAT"
    conn.close()


def test_tool_is_registered_in_phase_2():
    from friday.agent.tools import build_registry
    assert "email_send" in build_registry(phase=2).names()
    assert policy.TOOL_SENSE["email_send"] == "email.send"
