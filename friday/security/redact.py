"""Write-time redaction: secrets never reach disk, the database, or a cloud tier.

WHY AT WRITE TIME
-----------------
Doc 09 §4 puts redaction before persistence, and the ordering is the whole argument.
Redacting on the way OUT means the secret is already in `memory/facts/*.md`, already
in the SQLite rows, already in the JSONL traces, already embedded in the FTS index and
the vector store — and every one of those is a copy you have to remember to clean.
Markdown is truth and gets backed up, synced and (if the user ever does it) committed.
A secret that was written is a secret that leaked.

Redacting on the way IN means there is exactly one place to get it right.

WHAT IS AND IS NOT REDACTED
---------------------------
Secrets, by default: credentials, keys, tokens, connection strings, card numbers, and
the Indian identity documents this user is most likely to paste — PAN, Aadhaar, UPI.

NOT phone numbers or email addresses, by default. Those are PII, not credentials, and
they are also *the content*: "landlord: Ramesh, 98400 12345" is a fact worth
remembering, and scrubbing it would make the memory useless for the exact job it
exists to do. This system is local-first on a ₹0 budget, so the phone number stays on
the user's own machine. `redact(..., pii=True)` is the escalation path for if a tier
ever sends content off-device — that flag is the boundary, and it should be set at the
network edge, not here.

FALSE POSITIVES
---------------
Every pattern is anchored to something distinctive, because a redactor that eats
ordinary numbers gets switched off. Aadhaar requires either the word or the 4-4-4
grouping (a bare 12-digit run is an order ID half the time); card numbers must pass
Luhn; assignment-shaped secrets require an assignment. Typed markers (`[REDACTED:PAN]`)
rather than a uniform `[redacted]`, so the user can see WHAT was withheld and decide
to re-enter it deliberately — a memory system that hides its own edits cannot be
audited, and auditability is the product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MARKER = "[REDACTED:{kind}]"
_ALREADY = re.compile(r"\[REDACTED:[A-Z0-9_]+\]")


@dataclass
class RedactionReport:
    hits: dict[str, int] = field(default_factory=dict)
    #: ⭐ A pattern that raised is recorded rather than swallowed. This sits on the
    #: write path for every fact and every trace, so `redact` must never propagate an
    #: exception — but a redactor that fails SILENTLY is worse than one that fails
    #: loudly, because the secret is then stored and nobody knows. Callers can inspect
    #: this; the tests assert it stays empty.
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(self.hits.values())

    @property
    def changed(self) -> bool:
        return self.count > 0

    def summary(self) -> str:
        if not self.hits:
            return ""
        return "redacted " + ", ".join(f"{k}×{v}" for k, v in sorted(self.hits.items()))


# ── patterns ───────────────────────────────────────────────────────────────────
# Ordered: the block-shaped and longest patterns run first so their contents are not
# partially eaten by a narrower rule underneath.

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    # A private key block, including everything between the markers.
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN[ A-Z]*(?:PRIVATE KEY|OPENSSH|PGP|ENCRYPTED)[ A-Z]*-----.*?"
        r"-----END[ A-Z]*(?:PRIVATE KEY|OPENSSH|PGP|ENCRYPTED)[ A-Z]*-----",
        re.S)),

    # Provider-shaped API keys. Each prefix is distinctive enough that a bare match
    # is not a false positive.
    ("API_KEY", re.compile(
        r"\b(?:sk-(?:ant-|proj-|live-|or-)?[A-Za-z0-9_\-]{16,}"
        r"|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}"
        r"|glpat-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}"
        r"|xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,}"
        r"|hf_[A-Za-z0-9]{20,}|key-[A-Za-z0-9]{20,}"
        r"|ya29\.[A-Za-z0-9_\-]{20,}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
        r"\.[A-Za-z0-9_\-]{5,})\b")),

    # Anything on an Authorization header, or a bare Bearer/Basic prefix.
    #
    # The separator is REQUIRED unless a Bearer/Basic prefix is present. Making it
    # optional would match "Authorization required for the endpoint" and eat the word
    # "required" — the label alone is not distinctive enough without either a colon or
    # the scheme word. Minimum length is 8 rather than 16: real tokens are long, but
    # a shorter one is still a credential, and the surrounding label makes a false
    # positive unlikely enough that erring toward redaction is correct here.
    ("BEARER", re.compile(
        r"\b(?:Authorization\s*[:=]\s*(?:Bearer|Basic)?\s*"
        r"|Bearer\s+|Basic\s+"
        r"|(?:access_token|auth_token|refresh_token|token)\s*[:=]\s*)"
        r"([A-Za-z0-9._\-/+=&%]{8,})", re.I)),

    # Credentials inside a URL. The scheme is required, so a plain mention of a host
    # is not touched.
    ("CONNECTION_STRING", re.compile(
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|https?|ftp|ssh)"
        r"://[^\s:@/]{1,64}:([^\s@/]{3,128})@[^\s/]+", re.I)),

    # Assignment-shaped secrets: password = "hunter2". The key name is what makes
    # this safe — `secret`, `passwd`, `token` do not appear in ordinary prose next to
    # an equals sign very often.
    ("CREDENTIAL", re.compile(
        r"\b(?:password|passwd|pwd|passphrase|secret|secret_key|api_key|apikey"
        r"|access_key|auth_token|access_token|refresh_token|client_secret"
        r"|private_key|db_pass|database_password|encryption_key)"
        r"\s*[:=]\s*[\"']?([^\s\"',;]{6,128})", re.I)),

    # Indian tax identifier. Distinctive shape, so a bare match is reliable.
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),

    # Aadhaar: only with the word, or in 4-4-4 groups. A bare 12-digit run is far
    # more often an order or reference number than an identity document.
    ("AADHAAR", re.compile(
        r"\b(?:aadhaar|aadhar|uidai)\b[^\n]{0,30}?\b([0-9]{4})[ \-]?([0-9]{4})[ \-]?([0-9]{4})\b"
        r"|\b([0-9]{4})[ \-]([0-9]{4})[ \-]([0-9]{4})\b[^\n]{0,30}?\b(?:aadhaar|aadhar|uidai)\b"
        r"|\b(?:aadhaar|aadhar)(?:\s*(?:no|number|#|:))?\s*([0-9]{12})\b", re.I)),

    # UPI handles. Restricted to the real issuer suffixes: `\w+@\w+` alone would eat
    # every email address in the memory.
    ("UPI", re.compile(
        r"\b[A-Za-z0-9._\-]{2,64}@(?:okhdfcbank|oksbi|okicici|okaxis|paytm|ybl|ibl"
        r"|apl|axl|upi|kotak|icici|hdfcbank|sbi|baroda/mpay|fbl|tmb|dbs|unionbank"
        r"|yesbank|mahabank|centralbank|indus|airtel|jupiter|slice|amazonpay"
        r"|yapl|whatsapp|okbusiness)\b", re.I)),
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("CARD", re.compile(r"(?<![\d.])(?:\d[ \-]?){13,19}(?![\d.])")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("PHONE_IN", re.compile(
        r"(?<![\w])(?:\+91[\s\-]?|0?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?![\w])")),
)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str, *, pii: bool = False, report: RedactionReport | None = None) -> str:
    """Replace secrets with typed markers. Idempotent, and never raises.

    `pii=True` is the network-edge setting: it also scrubs card numbers, email
    addresses and Indian mobile numbers. Leave it off for local storage, where those
    are the content rather than a leak.
    """
    if not text:
        return text
    if report is None:
        report = RedactionReport()

    def mark(kind: str) -> str:
        report.hits[kind] = report.hits.get(kind, 0) + 1
        return MARKER.format(kind=kind)

    # Protect markers already present, so re-running over stored content does not
    # nest them into [REDACTED:[REDACTED:PAN]] — and, just as importantly, does not
    # COUNT them again.
    #
    # ⚠️ This protection was documented here but never implemented (`out = text`), and
    # the consequence was self-defeating: `redact()` turns `api_key: ghp_AAA…` into
    # `api_key: [REDACTED:CREDENTIAL]`, and the assignment-shaped CREDENTIAL pattern
    # then matches THAT, because its value class accepts any non-space token. So a
    # correctly redacted memory — precisely the state the system is supposed to reach —
    # scanned as containing a credential. `friday doctor` failed the secret scan on a
    # clean install, and the "fix" it told the user to apply was the thing it had
    # already done. A diagnostic that alarms on success gets ignored.
    out = text

    def _is_marker(value: str) -> bool:
        return bool(value) and _ALREADY.fullmatch(value.strip()) is not None

    for kind, pattern in _PATTERNS:
        try:
            # Where the pattern captures the secret, keep the label that identifies it
            # ("password =") and remove only the value. Where it does not — a bare API
            # key, a private key block — the whole match is the secret. Asking for
            # group(1) on those raised IndexError, which propagated out of every memory
            # write and every trace append: the redactor broke the thing it protects.
            if pattern.groups:
                def _sub(m: re.Match, k: str = kind) -> str:
                    full = m.group(0)
                    secret = m.group(1) or ""
                    if _is_marker(secret):        # already redacted: not a new hit
                        return full
                    report.hits[k] = report.hits.get(k, 0) + 1
                    return full.replace(secret, MARKER.format(kind=k)) if secret \
                        else MARKER.format(kind=k)
                out = pattern.sub(_sub, out)
            else:
                def _bare(m: re.Match, k: str = kind) -> str:
                    if _is_marker(m.group(0)):    # never nest, never re-count
                        return m.group(0)
                    return mark(k)
                out = pattern.sub(_bare, out)
        except Exception as e:                     # never break a write over a pattern
            report.errors.append(f"{kind}: {type(e).__name__}: {e}")

    if pii:
        for kind, pattern in _PII_PATTERNS:
            try:
                if kind == "CARD":
                    def _card(m: re.Match) -> str:
                        digits = re.sub(r"\D", "", m.group(0))
                        if not 13 <= len(digits) <= 19 or not _luhn(digits):
                            return m.group(0)      # not a card; leave it alone
                        return mark("CARD")
                    out = pattern.sub(_card, out)
                else:
                    out = pattern.sub(lambda m, k=kind: mark(k), out)
            except Exception as e:
                report.errors.append(f"{kind}: {type(e).__name__}: {e}")

    return out


def scan(text: str, *, pii: bool = False) -> RedactionReport:
    """Report what WOULD be redacted, without changing the text.

    Used for the audit trail and for warning a user before they paste something they
    did not mean to store.

    Text that is ALREADY redacted reports zero hits. A `[REDACTED:…]` marker is the
    evidence that redaction happened, not a credential — counting it made `doctor`'s
    secret scan fail on the cleanest possible memory.
    """
    rep = RedactionReport()
    redact(_ALREADY.sub(" ", text), pii=pii, report=rep)
    return rep


def has_secrets(text: str, *, pii: bool = False) -> bool:
    return scan(text, pii=pii).count > 0
