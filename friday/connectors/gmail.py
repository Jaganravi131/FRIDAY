"""Gmail over IMAP, read-only by design (docs/architecture/18 §3).

Why IMAP with an app password instead of the Gmail API: it needs one config stanza
and no OAuth dance on a headless laptop, and the app password is a credential you
can revoke without touching your Google account password. IMAP is older than dirt
and every part of it is stdlib (`imaplib`, `email`), which keeps the core
dependency-free — a hard rule for this project.

Scope decisions that are LOAD-BEARING, not incidental:

  * READ-ONLY mailbox open. `imaplib` is told `readonly=True`, so FRIDAY literally
    cannot mark read, delete, or move anything — the permission is enforced by the
    protocol, not by the model remembering not to.
  * Fetch is BODY.PEEK, same reason: fetching must not change state anywhere.
  * Sending is a DIFFERENT tool on a DIFFERENT sense (email.send), fronted by SMTP
    and gated ALWAYS_CONFIRM + UNATTENDED_DENY in policy, whichever connector is
    behind it. This module contains no send path at all.
  * Everything returned is imported content. The launcher tool neutralizes; the
    truncation here is for size, not safety.
"""
from __future__ import annotations

import email
import email.header
import email.utils
import imaplib
from email.message import Message

from . import Connector, Item, config_help, load_config, register

IMAP_HOST_DEFAULT = "imap.gmail.com"
MAX_BODIES = 10
MAX_BODY_CHARS = 8_000
_CONFIG_EXAMPLE = {
    "user": "you@gmail.com",
    "app_password": "xxxx xxxx xxxx xxxx   # Google account → Security → App passwords",
    "imap_host": "imap.gmail.com (optional, this is the default)",
    "mailbox": "INBOX (optional, default)",
}


def _cfg() -> dict:
    return load_config("gmail")


def _decode_header_value(raw: str | None) -> str:
    if not raw:
        return ""
    parts = []
    for text, enc in email.header.decode_header(raw):
        if isinstance(text, bytes):
            parts.append(text.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(text)
    return " ".join(parts).strip()


def _body_text(msg: Message) -> str:
    """Pull text/plain out of the MIME tree; HTML mail degrades to a stripped
    version of its HTML part. Never raise: one malformed message must not sink the
    listing of the other nine."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disp:
                return _safe_payload(part)
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                return _strip_html(_safe_payload(part))
        return ""
    return _strip_html(_safe_payload(msg)) if msg.get_content_type() == "text/html" \
        else _safe_payload(msg)


def _safe_payload(part: Message) -> str:
    try:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    except Exception:
        return "[could not decode this part]"


def _strip_html(text: str) -> str:
    import html as _html
    import re as _re
    text = _re.sub(r"(?s)<(script|style).*?</\\1>", " ", text)
    text = _re.sub(r"(?s)<[^>]+>", " ", text)
    return _html.unescape(text)


@register
class GmailConnector(Connector):
    id = "gmail"
    sense = "email.read"

    def status(self) -> dict:
        cfg = _cfg()
        if "__corrupt__" in cfg:
            return {"connector": self.id, "configured": False,
                    "error": f"config/connectors.json is not valid JSON: {cfg['__corrupt__']}"}
        if not cfg.get("user") or not cfg.get("app_password"):
            return {"connector": self.id, "configured": False,
                    "how_to": config_help("gmail", _CONFIG_EXAMPLE)}
        return {"connector": self.id, "configured": True, "user": cfg["user"],
                "imap_host": cfg.get("imap_host") or IMAP_HOST_DEFAULT,
                "mailbox": cfg.get("mailbox") or "INBOX",
                "note": "app password present (never shown, never logged)"}

    def _open(self) -> imaplib.IMAP4_SSL:
        cfg = _cfg()
        if not cfg.get("user") or not cfg.get("app_password"):
            raise RuntimeError(config_help("gmail", _CONFIG_EXAMPLE))
        host = cfg.get("imap_host") or IMAP_HOST_DEFAULT
        imap = imaplib.IMAP4_SSL(host, timeout=20)
        try:
            imap.login(cfg["user"], cfg["app_password"])
        except Exception:
            imap.logout()
            raise
        return imap

    def fetch(self, *, limit: int = 5, since_days: int = 7,
              unseen_only: bool = False, mailbox: str | None = None) -> list[Item]:
        """Fetch recent message summaries + bodies. Raises with a user-readable
        message on misconfiguration or network failure; the launcher tool turns
        that into a tool error dict, never a crashed turn."""
        limit = max(1, min(int(limit or 5), MAX_BODIES))
        cfg = _cfg()
        box = mailbox or cfg.get("mailbox") or "INBOX"

        imap = self._open()
        try:
            # readonly=True: the protocol itself forbids STORE/EXPUNGE. This is Law 5
            # (enforce in the data layer) applied to a mail server.
            imap.select(box, readonly=True)
            criteria = ["UNSEEN"] if unseen_only else ["ALL"]
            if since_days:
                import datetime as _dt
                since = (_dt.datetime.now(_dt.timezone.utc)
                         - _dt.timedelta(days=int(since_days))).strftime("%d-%b-%Y")
                criteria += ["SINCE", since]
            typ, data = imap.search(None, *criteria)
            if typ != "OK":
                raise RuntimeError(f"IMAP search failed: {typ}")
            ids = (data[0] or b"").split()
            if not ids:
                return []
            items: list[Item] = []
            for mid in ids[-limit:][::-1]:
                # BODY.PEEK[] — fetch without setting \\Seen. Reading must not write.
                typ, parts = imap.fetch(mid, "(BODY.PEEK[])")
                if typ != "OK" or not parts or not isinstance(parts[0], tuple):
                    continue
                raw = parts[0][1] or b""
                try:
                    msg = email.message_from_bytes(raw)
                    body = (_body_text(msg) or "")[:MAX_BODY_CHARS]
                    items.append(Item(
                        connector=self.id,
                        id=mid.decode(errors="replace"),
                        title=_decode_header_value(msg.get("Subject")) or "(no subject)",
                        body=body,
                        author=_decode_header_value(msg.get("From")),
                        when=msg.get("Date") or "",
                    ))
                except Exception as e:
                    items.append(Item(connector=self.id,
                                      id=mid.decode(errors="replace"),
                                      title="[unparseable message]",
                                      body=f"[skipped: {e}]"))
            return items
        finally:
            try:
                imap.close()
            except Exception:
                pass
            try:
                imap.logout()
            except Exception:
                pass
