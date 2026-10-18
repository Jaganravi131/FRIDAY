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
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable

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
    "github_read": "github.read",
    "calendar_write": "calendar.write",
    "email_read": "email.read",
    "email_send": "email.send",
    "location_read": "location.read",
    "shell_sandboxed": "shell.exec",
    "shell": "shell.exec",
    "file_write": "fs.write",
    "http_post": "network.egress",
    # ⭐ Its own sense, not `network.egress`. Tiered permissions means the user approves
    # each new sense separately, and "may read the public web" is a different question
    # from "may POST to an endpoint" — conflating them would force one approval to cover
    # both. Off until the phase-1 registry is enabled.
    "web_read": "web.read",
    "wiki": "web.read",
    # The workspace tier (docs/architecture/18). Files are one confined folder —
    # risk is what lands IN it from connectors, which read-tools already contain.
    # Execution is a curated allowlist: granting the sense approves THE LIST, so
    # there is no per-call confirmation to fatigue-switch, and unattended use is
    # denied outright below.
    "ws_list": "workspace.files",
    "ws_read": "workspace.files",
    "ws_write": "workspace.files",
    "ws_shell": "workspace.exec",
}

#: Irreversible or externally-visible actions always require a human "yes", even
#: with the sense enabled. This is the confirmation gate from
#: docs/architecture/08, enforced here rather than hoped for.
ALWAYS_CONFIRM: set[str] = {
    "email_send", "shell", "file_write", "http_post", "calendar_write",
    "delete_fact", "purge_memory",
    # ws_shell is deliberately NOT here: the allowlist is what the user approved
    # when granting workspace.exec, and a gate that fires on every `ls` is a gate
    # that gets disabled. The curation carries the weight; see workspace.py.
}

#: Scope-sensitive: heartbeat and dreaming run unattended, so anything that
#: touches the outside world is denied outright rather than queued for a
#: confirmation nobody is there to give.
#: Denied unattended, but NOT confirmation-gated interactively — the deliberate
#: exception to `UNATTENDED_DENY <= ALWAYS_CONFIRM`, and the reason it is a named set
#: rather than a quiet omission.
#:
#: The invariant exists on the principle that a gate should depend on what an action
#: DOES, not on who is watching. For these tools, who is watching is genuinely the
#: relevant variable: fetching a public page is reversible and changes nothing anywhere,
#: so the action itself is benign. The danger is *unobserved steering* — a poisoned
#: memory file, retrieved once, sending `?d=<your data>` to a host the attacker reads.
#: A present user who can SEE the fetch is the control; a confirmation dialog per page
#: would make the tool unusable, and an unusable safety feature gets disabled.
#:
#: That argument only holds if the fetch is actually visible, so it is enforced rather
#: than assumed: `cli._announce_egress` and `serve._describe_tool` surface every egress
#: call by DEFAULT, not under --verbose, and tests assert it. Presence without
#: observability would just be a person in the room.
PRESENCE_GATED: set[str] = {"web_read", "wiki"}

#: The THIRD shape of gate: the consent was the list itself. A curated shell whose
#: programs and verbs are fixed in code means "granting workspace.exec" approves a
#: finite, auditable command surface — the user confirmed the list up front, in
#: words. A per-call confirmation on top of that fires on every `ls` and teaches
#: the user to dismiss without reading, which is the exact desensitisation the
#: ALWAYS_CONFIRM design exists to avoid. Unattended scope is still flatly denied
#: (no presence to steer by), and every invocation is audited. Same pattern as
#: PRESENCE_GATED: a named set, asserted narrowly in tests, so it cannot grow by
#: quietly omitting something from ALWAYS_CONFIRM.
ALLOWLIST_GATED: set[str] = {"ws_shell"}

UNATTENDED_DENY: set[str] = {
    "email_send", "shell", "http_post", "file_write", "calendar_write",
    # No heartbeat should spawn a process. Even allowlisted ones produce output the
    # next turn might treat as instructions, and nobody is there to notice the
    # pattern forming. Interactive scope is unimpaired.
    "ws_shell",
    # ⚠️ These are READ tools, and they are still here. That is not an oversight.
    #
    # A URL is an outbound channel: `http://evil.example/?d=<your data>` leaves the
    # machine as a request, and the attacker reads it from their own access log. Nothing
    # is POSTed, nothing is written, no form is filled — and the secret is still gone.
    # `security/injection.py` already classifies exactly this shape as `exfil_channel`.
    #
    # The SSRF guard stops the request going somewhere INTERNAL. It cannot stop the
    # request carrying something outward to a public host, because that is what reading
    # the web looks like. Interactive scope is fine: the user is present, the ledger
    # shows the fetch, and they can object. Unattended scope has nobody watching, and a
    # poisoned memory file only has to be retrieved once.
    "web_read", "wiki",
}


# ── provenance as an authority check ───────────────────────────────────────────

_STOP = frozenset("""a an the and or but if then than that this these those is are was
were be been being of to in on at by for with from as into it its i you he she they we
my your his their our me him her them us not no yes do does did done have has had
about over under again more most very just so such can could will would shall should
may might must remember note fact""".split())


def _tokens(text: str) -> set[str]:
    import re as _re

    return {t for t in _re.findall(r"[^\W_]{4,}", (text or "").lower()) if t not in _STOP}


def quote_is_user_authorised(quote: str, turns: Iterable[str]) -> bool:
    """Did the user actually say something like this?

    ⭐ This is the containment layer, and the reason it works is that a memory write
    makes a CHECKABLE claim. `memory_write(source_kind="stated", source_quote=...)`
    asserts "the user told me this". Provenance was built so `/why` could show the
    user their own words back; the same evidence answers a security question — whether
    those words are really theirs. Retrieval cannot help an attacker here, because the
    authority being tested is the turn history, which only the user writes.

    Fuzzy on purpose. Models paraphrase: the user says "my rent is 18000" and the
    write carries source_quote="rent 18000 INR". Requiring an exact substring would
    demand confirmation on every legitimate write, and a control that fires constantly
    gets switched off — which is how security features die in practice. So this asks
    whether the quote's *distinctive* tokens (4+ chars, not stopwords) are substantially
    present in what the user actually said.

    An ABSENT turn history returns True: with nothing to compare against there is no
    evidence either way, and refusing every write in that state would break the first
    turn of every session — a control that fires constantly gets switched off.

    ⚠️ A turn history that EXISTS but shares no distinctive content with the claim
    returns False, and the distinction is the whole gate. An earlier version collapsed
    the two ("no tokens found -> cannot verify -> allow"), which meant a forged write
    sailed through whenever the user's recent turns happened to be short: after "hi",
    "ok" or "thanks", the token pool was empty and ANY quote was treated as
    unverifiable rather than contradicted. The user had spoken; what they said simply
    was not this. That is evidence of forgery, not absence of evidence.

    The same principle covers the empty claim: a `stated` write that quotes NOTHING is
    not "no claim made" — "the user told me this, words unavailable" is exactly the
    shape a fabricated provenance takes, so it is not authorised. A quote with no
    distinctive words ("ok yes sure") is not authorised either: there is nothing
    checkable in it, and 'nothing distinctive' must not be read as 'nothing to deny'.

    Matching is per-sentence, not pooled over the whole history, and two aspects of
    the meaning have to agree with the quote, not just the tokens:

      * NEGATION. "I work at Acme" and "I no longer work at Acme" share every
        distinctive token, and only the negation tells a state from its
        termination. A negated quote is authorised by a negated sentence and
        nothing else — otherwise the write that should RETRACT an old fact
        quietly reinforces it instead.
      * PERSON. "I work at Acme" and "my friend works at Acme" also share almost
        every token, but one is a fact about the user and the other is not. A
        first-person claim needs a first-person sentence; "my" alone does not
        count, because "my friend" is exactly how a third party hides inside a
        possessive.
    """
    if not quote or not quote.strip():
        return False                     # a claim of authority that quotes nothing
    turns = list(turns or ())
    if not turns:
        return True                      # absent history: cannot verify either way
    q = _tokens(quote)
    if not q:
        return False                     # nothing checkable in the claim itself
    q_neg = _is_negated(quote)
    q_first = _is_first_person(quote)
    for turn in turns:
        for sentence in re.split(r"[.!?;\n]+", turn):
            s = _tokens(sentence)
            if not s:
                continue
            if len(q & s) / len(q) < 0.6:
                continue
            if _is_negated(sentence) != q_neg:
                continue                 # says the opposite of the claim
            if q_first and not _is_first_person(sentence):
                continue                 # says it about someone else
            return True                  # a sentence that actually says this
    return False


def _raw_words(text: str) -> set[str]:
    return {w.strip(".,!?;:'\"()[]").lower() for w in (text or "").split()}


def _is_negated(text: str) -> bool:
    """Does this text contain a negation marker? Checked on RAW words, not on
    `_tokens` output — the stopword filter removes 'not' and 'no' precisely
    because they carry no distinctive content, but here the whole point is the
    meaning they flip."""
    return any(w in _NEGATION or w.endswith("n't") for w in _raw_words(text))


def _is_first_person(text: str) -> bool:
    """Does this text speak AS the user? "i", "i'm", "i've" — but not "my" on
    its own: "my friend works at Acme" is a sentence about the friend."""
    return any(w == "i" or w.startswith("i'") for w in _raw_words(text))


def check(
    conn: sqlite3.Connection,
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    scope: str = "interactive",
    authority_turns: Iterable[str] | None = None,
) -> Decision:
    """The single choke point. Deterministic, table-driven, no model involved.

    `authority_turns` is what the user actually said recently. Passing it enables the
    provenance check below; omitting it leaves every other decision unchanged, so
    callers that have no turn history keep working rather than silently denying.
    """
    args = args or {}
    sense = TOOL_SENSE.get(tool_name)

    # ── provenance-gated write authority ───────────────────────────────────────
    # Only fires when the write CLAIMS the user's authority (stated / user_edit).
    # An `inferred` write is FRIDAY's own guess: it gets confidence 0, cannot outrank
    # anything real, and is already handled by the hierarchy in facts.py — blocking it
    # here would just stop FRIDAY from recording its own reasoning.
    if tool_name == "memory_write" and authority_turns is not None:
        from ..security import trust_of
        from ..security.trust import Trust

        kind = (args.get("source_kind") or "stated").lower()
        if trust_of(kind) >= Trust.USER:
            quote = args.get("source_quote") or ""
            if not quote_is_user_authorised(quote, authority_turns):
                if quote.strip():
                    detail = (
                        "the write claims the user's authority but the quoted words do not "
                        "appear in what they actually said — the justification may have come "
                        "from retrieved or imported text instead")
                else:
                    detail = (
                        "the write claims the user's authority but quotes nothing — "
                        "'the user told me this, words unavailable' is not evidence, "
                        "and the user must see what is being attributed to them")
                if scope in ("heartbeat", "dreaming", "eval"):
                    log(conn, actor=scope, action=tool_name, target=_target(args),
                        sense_id=sense, decision="denied",
                        detail="unauthorised provenance; " + detail)
                    return Decision("denied", sense,
                                    "no user present to authorise this write, and the "
                                    "quote is not theirs")
                log(conn, actor=scope, action=tool_name, target=_target(args),
                    sense_id=sense, decision="confirmation_required", detail=detail)
                return Decision("confirmation_required", sense, detail)

    # (The ALWAYS_CONFIRM gate used to sit here, before any permission check; it
    # now runs after them — see "CONFIRMATION IS THE LAST GATE" below.)

    # `remote` is NOT in this tuple, and that is deliberate: a turn arriving over the
    # gateway is the user on their phone, present and able to approve something. Only
    # scopes where nobody is watching are unattended. It still gets its own label so
    # the audit trail can distinguish network turns from keyboard ones.
    if scope in ("heartbeat", "dreaming", "eval") and tool_name in UNATTENDED_DENY:
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
            decision="denied", detail="unattended scope may not touch the outside world")
        return Decision("denied", sense, f"{scope} runs unattended; this needs you present")

    # ⭐ PERMISSIONS BEFORE CONFIRMATION. The registry checks used to sit below
    # the ALWAYS_CONFIRM gate; the gate now runs after them (see the note at the
    # single gate site further down) so a human "yes" can never stand in for a
    # permission. Sense-less tools skip straight to the gate — for them there are
    # no permission checks to run, so the gate is still the last thing before the
    # allow, and the ordering guarantee below holds unchanged.
    row = None
    if sense is not None:
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

    # ⭐ CONFIRMATION IS THE LAST GATE. Every permission check above has passed by
    # the time this fires: the sense exists, is enabled, and the target is inside
    # its granted scope (or the tool needs no sense at all). That ordering is what
    # makes the loop's contract — "a returned confirmation_required means
    # everything else already passed, so a human yes may hand the call straight to
    # the registry" — true instead of a bypass. Confirmed-but-ungranted used to
    # execute; the tests pin the new order
    # (test_gate_order_is_unattended_then_sense_then_confirmation).
    if tool_name in ALWAYS_CONFIRM and scope == "interactive":
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
            decision="confirmation_required",
            detail="irreversible or externally visible — awaiting the human yes")
        return Decision("confirmation_required", sense,
                        "irreversible or externally visible — show the exact payload first")

    if sense is None:
        # ⭐ LOGGED. This branch used to return silently, which meant the audit trail
        # recorded nothing at all during ordinary Phase 0 use: memory_search,
        # memory_write and read_artifact are the only three tools that exist, none
        # of them need a sense, and so every allow went unrecorded. `audit --today` came
        # back empty after a session that had read and written the user's memory.
        #
        # Doc 09 §8 calls the audit log "your trust anchor", and doc 10's exit test #9
        # requires that every decision and denial is visible. A trail that only
        # contains refusals is not a trust anchor — it tells you what FRIDAY was
        # stopped from doing and nothing about what it did. "FRIDAY read your memory"
        # and "FRIDAY wrote a fact about you" are precisely the events a person should
        # be able to ask about, and one indexed INSERT per tool call is free next to
        # the retrieval that follows it.
        log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=None,
            decision="allowed", detail="no sense required")
        return Decision("allowed", None, "no sense required")

    log(conn, actor=scope, action=tool_name, target=_target(args), sense_id=sense,
        decision="allowed")
    return Decision("allowed", sense)


#: A scope entry that looks like a domain (dot, no spaces) is matched against the
#: HOSTS in the target, not as a substring. "gmail.com" must not authorise
#: "gmail.com.attacker.example", and a deny on "evil.example" must still catch
#: "sub.evil.example". Everything else (window titles, app names — "vscode",
#: "code editor") keeps plain word matching, because that is what those grants
#: mean and the screen tests pin it.
_HOSTLIKE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z0-9-]+$")

#: Negation words for the authority quote check. Not a stemming exercise: these
#: are the handful of tokens that flip the meaning of an otherwise identical
#: claim ("I work at Acme" vs "I no longer work at Acme").
_NEGATION = frozenset(
    "not no never nothing nobody stop stopped quit cancelled canceled "
    "don't doesn't didn't won't wouldn't can't cannot couldn't shouldn't "
    "isn't aren't wasn't weren't hasn't haven't ain't anymore".split())


def _hosts_of(target: str) -> list[str]:
    """Candidate hosts/domains in a target string, lowercased.

    Handles URL hosts (credentials after the scheme are honoured — the host is
    what follows the @), `host:port` forms, bare domains, and email addresses
    (the domain after the @). Words that are not host-shaped are simply absent
    from the result; callers fall back to word matching for those.
    """
    from urllib.parse import urlparse

    hosts: list[str] = []
    for tok in target.replace(",", " ").split():
        t = tok.strip("<>()\"'[]")
        if not t:
            continue
        if "@" in t:                                  # an email address (or userinfo URL)
            domain = t.rsplit("@", 1)[1]
            if "." in domain:
                hosts.append(domain.split("/")[0].split(":")[0].lower())
                continue
        candidate = t if "://" in t else ("//" + t if "." in t and "/" not in t.split(".")[0] else t)
        try:
            p = urlparse(candidate)
        except ValueError:
            continue
        if p.hostname:
            hosts.append(p.hostname.lower())
    return hosts


def _entry_matches_host(entry: str, host: str) -> bool:
    """Exact host, or a genuine subdomain of it — never a substring."""
    return host == entry or host.endswith("." + entry)


def _scope_allows(scope_json: str | None, args: dict) -> bool:
    """Deny-list wins over allow-list. An absent scope means 'all of this sense'.

    The destination is read from whichever argument names it — target/app/path
    for screen and file tools, `url` for web reads, `to` for mail. An earlier
    version read only target/app/path, so `web_read(url=...)` and
    `email_send(to=...)` presented an EMPTY target, and an empty target skipped
    the allow list entirely: every scoped web/mail grant was an unscoped one for
    exactly the tools where the scope matters most.
    """
    if not scope_json:
        return True
    try:
        scope = json.loads(scope_json)
    except (json.JSONDecodeError, TypeError):
        return False           # a corrupt grant is a denied grant

    parts: list[str] = []
    for k in ("target", "app", "path", "url", "to"):
        v = args.get(k)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            parts.extend(str(x) for x in v)
        else:
            parts.append(str(v))
    target = " ".join(parts)
    t = target.lower()
    hosts = _hosts_of(target)

    deny = [str(d).strip().lower() for d in (scope.get("deny") or []) if str(d).strip()]
    allow = [str(a).strip().lower() for a in (scope.get("allow") or []) if str(a).strip()]
    if not deny and not allow:
        return True
    if not target:
        # ⭐ A scoped grant met a call that names no destination. There is nothing
        # to match against, and "cannot tell" must not mean "allowed" — that is
        # how an arg-shape the matcher doesn't know silently widens a scoped
        # grant into an unscoped one. Fail closed.
        return False

    for bad in deny:
        if _HOSTLIKE.match(bad):
            if any(_entry_matches_host(bad, h) for h in hosts):
                return False
        elif bad in t:
            return False

    if not allow:
        return True

    host_entries = [a for a in allow if _HOSTLIKE.match(a)]
    word_entries = [a for a in allow if not _HOSTLIKE.match(a)]
    if hosts and host_entries:
        # A destination-carrying call: EVERY host must be inside the allow list.
        # One out-of-scope recipient is a leak, not a partial success.
        return all(any(_entry_matches_host(a, h) for a in host_entries) for h in hosts)
    return any(a in t for a in word_entries or allow)


def _target(args: dict) -> str:
    for k in ("target", "url", "to", "path", "app", "query", "predicate"):
        if k in args:
            v = args[k]
            if isinstance(v, (list, tuple)):
                v = " ".join(str(x) for x in v)
            return str(v)[:200]
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
