"""SMTP send — FRIDAY's FIRST write connector (docs/architecture/18 §3.6).

A read connector that misbehaves shows you wrong data. A send connector that
misbehaves publishes words in your name, to people, irreversibly. Every design
rule here follows from that asymmetry; they are LOAD-BEARING, not style:

  * EXACTLY ONE MESSAGE, EXACTLY THE CONFIRMED BYTES. `email_send` is in
    policy.ALWAYS_CONFIRM: the loop shows the user the exact tool arguments and
    demands a yes. This driver's job is then to guarantee the bytes on the wire
    are precisely those bytes — nothing added (no signature, no footer, no
    silent CC/BCC — the signature literally has no cc/bcc parameter), nothing
    rewritten. A user who confirmed a payload the wire then modified has been
    lied to, and lying about irreversible actions is the one failure a consent
    system must never have.
  * ONE CONFIRMATION = ONE SEND. `send()` issues one SMTP session per call and
    the launcher makes one driver call per tool call. No queueing, no retry
    loop in the driver (a retried send can silently double-send; a failure is
    reported and the human decides).
  * HEADER HYGIENE BEFORE THE WIRE. CR/LF in `to`/`subject` would splice
    attacker-controlled headers (Bcc: anyone) into the message. Refused with a
    ValueError raised BEFORE any socket is opened — the refusal must cost the
    network nothing. Same for the recipient-shape and size caps.
  * TLS FROM THE FIRST BYTE. smtplib.SMTP_SSL on port 465: there is no
    pre-TLS plaintext phase to downgrade. Credentials only ever cross inside
    TLS, to the host in config.
  * SECRET HYGIENE ON THE ERROR PATH. An SMTP server (a config-typo'd or
    outright hostile relay) can echo your password back inside error text.
    Every smtplib failure is re-raised as a type-name-only RuntimeError; the
    audit trail may know a send failed, never what the server said verbatim.
  * READ? NONE. A write connector has no read surface: fetch() raises
    NotImplementedError so nothing can be smuggled back through this door.
"""
from __future__ import annotations

import email.utils
import re
import smtplib
from email.message import EmailMessage

from . import Connector, config_help, load_config, register

SMTP_HOST_DEFAULT = "smtp.gmail.com"
SMTP_PORT = 465
TIMEOUT_S = 30
MAX_RECIPIENTS = 5
MAX_SUBJECT_CHARS = 200
MAX_BODY_CHARS = 20_000

# v1 accepts PLAIN mailbox addresses only (`name@host`, no display-name forms).
# A display name like `Doe, John <j@x>` would make the `To:` header line
# non-trivial to assemble and EVERY extra parsing rule on an irreversible
# action is a footgun. Users who need the fancy form can send from the laptop.
_ADDR_RE = re.compile(r"^[^@\s,<>\r\n,\"']+@[^@\s,<>\r\n,\"']+$")

_CONFIG_EXAMPLE = {
    "user": "you@gmail.com",
    "app_password": "xxxx xxxx xxxx xxxx   # Google account → Security → App passwords",
    "smtp_host": "smtp.gmail.com (optional, this is the default)",
}


def transport(host: str, port: int, timeout: int):
    """The ONE network touch in this module. Tests stub at this boundary; there
    is no other path to a socket. SMTP_SSL = TLS from the connection's first
    byte, so no upgrade dance can be refused midway."""
    return smtplib.SMTP_SSL(host, port, timeout=timeout)


def _validate(to: list[str], subject: str, body: str) -> None:
    """Every check runs BEFORE any socket exists; a refusal here is free."""
    if not isinstance(to, list) or not to:
        raise ValueError("'to' must be a non-empty list of addresses")
    if len(to) > MAX_RECIPIENTS:
        raise ValueError(f"at most {MAX_RECIPIENTS} recipients per send")
    for addr in to:
        if not isinstance(addr, str) or not _ADDR_RE.fullmatch(addr):
            raise ValueError(f"bad recipient address shape: {addr!r}")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("subject must be non-empty text")
    if len(subject) > MAX_SUBJECT_CHARS:
        raise ValueError(f"subject over {MAX_SUBJECT_CHARS} chars")
    if "\r" in subject or "\n" in subject:
        raise ValueError("subject contains CR/LF — header injection refused")
    if not isinstance(body, str):
        raise ValueError("body must be text")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"body over {MAX_BODY_CHARS} chars")


def send(user: str, password: str, host: str,
         to: list[str], subject: str, body: str) -> dict:
    """Compose and send exactly one message. `body` and `subject` are passed
    through VERBATIM: the text the user confirmed is the text on the wire."""
    _validate(to, subject, body)
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg["Date"] = email.utils.formatdate()
    msg.set_content(body)

    try:
        conn = transport(host, SMTP_PORT, TIMEOUT_S)
        try:
            conn.login(user, password)
            refused = conn.send_message(msg)
        finally:
            try:
                conn.quit()
            except Exception:
                pass
    except smtplib.SMTPAuthenticationError:
        # The fact that auth failed is safe; anything the SERVER said in reply
        # might quote the password back at us, so it never leaves this function.
        raise RuntimeError("smtp SMTPAuthenticationError: user/app-password rejected "
                           "(server text withheld in case it echoes credentials)")
    except Exception as e:
        raise RuntimeError(f"smtp {type(e).__name__}: send failed "
                           "(server text withheld in case it echoes credentials)")

    if refused:
        # refused is {address: (code, bytes-message)} — again type-free detail.
        raise RuntimeError("smtp recipients refused: "
                           + ", ".join(sorted(refused)))
    return {"sent": True, "host": host, "from": user,
            "to": list(to), "subject": subject, "body_chars": len(body)}


@register
class SmtpConnector(Connector):
    """Sends mail as the configured gmail account. The SENSE is the same
    `email.send` the policy gate was built around; the account config is the
    existing `gmail` section — the app password authenticates SMTP the same way
    it authenticates IMAP, one credential per account."""

    id = "gmail.smtp"
    sense = "email.send"

    def _cfg(self) -> dict:
        return load_config("gmail")

    def status(self) -> dict:
        cfg = self._cfg()
        if "__corrupt__" in cfg:
            return {"id": self.id, "sense": self.sense, "configured": False,
                    "corrupt": True,
                    "how_to": f"config/connectors.json is not valid JSON: {cfg['__corrupt__']}"}
        have = bool(cfg.get("user") and cfg.get("app_password"))
        return {"id": self.id, "sense": self.sense, "configured": have,
                "host": cfg.get("smtp_host", SMTP_HOST_DEFAULT),
                "how_to": config_help("gmail", _CONFIG_EXAMPLE)}

    def send(self, *, to: list[str], subject: str, body: str) -> dict:
        cfg = self._cfg()
        user, password = cfg.get("user"), cfg.get("app_password")
        if not (user and password):
            raise RuntimeError("connector 'gmail.smtp' is not configured. "
                               "Add this to config/connectors.json (placeholders):\n"
                               + config_help("gmail", _CONFIG_EXAMPLE))
        return send(user, password, cfg.get("smtp_host", SMTP_HOST_DEFAULT),
                    to=to, subject=subject, body=body)

    def fetch(self, **kw) -> list:
        # ⭐ A write connector must not acquire a read surface by accident.
        raise NotImplementedError("'gmail.smtp' sends; it has no read path. "
                                  "Reading mail is the 'gmail' (IMAP) connector.")
