"""Network egress guard — the thing that makes a web-reading tool safe to have.

WHY THIS MODULE EXISTS BEFORE THE TOOL
--------------------------------------
`web_read` is read-only, so its blast radius looks identical to retrieved memory: text
in a reply, fenced as EXTERNAL trust, neutralized, unable to instruct. That argument is
sound — and it is about the *response*.

The *request* is a different matter. A tool that fetches a URL is a tool that makes the
user's machine open a connection to an address the model chose. The model's choice can be
steered by anything it reads, and FRIDAY has already proven (Phase 0, `SECURITY.md` §3)
that a hostile quote planted in a memory file can reach the model's context. So the
attack is not "make FRIDAY say something false." It is:

    plant a fact whose value contains  http://169.254.169.254/latest/meta-data/iam/...
    wait for retrieval to surface it
    let the agent fetch it

and now a prompt injection has read a cloud credential endpoint, or probed the LAN, or
hit `127.0.0.1:8642` — FRIDAY's own gateway, from inside, where the bearer token may not
be the only thing standing between an attacker and your memory. That is SSRF, and it is
the reason a read-only tool still needs a guard with teeth.

WHAT IS BLOCKED, AND WHY
------------------------
Everything that is not a public internet address, by default:

  * loopback (127.0.0.0/8, ::1) — FRIDAY's own gateway and any local model server
  * link-local (169.254.0.0/16, fe80::/10) — **including 169.254.169.254, the cloud
    metadata endpoint**, the single most valuable SSRF target in existence
  * private (10/8, 172.16/12, 192.168/16, fc00::/7) — the LAN, routers, NAS, printers
  * CGNAT (100.64.0.0/10) — ⚠️ this is also **Tailscale's range**, so blocking it means
    FRIDAY cannot fetch from your own tailnet. That is the correct default: the tailnet
    is where your other devices live, and "the agent may reach any device you own because
    a web page told it to" is not a trade worth making implicitly. `allow_private` exists
    for when you decide otherwise, explicitly, in code you wrote.
  * multicast, reserved, unspecified (0.0.0.0), and IPv4-mapped IPv6 — the last because
    `::ffff:127.0.0.1` is loopback wearing a disguise

Also blocked: any scheme that is not http/https. `file://` reads your disk, `gopher://`
and `dict://` speak raw protocol to internal services, and `ftp://` is a data channel
nobody asked for.

Redirects are validated **per hop**. `urllib` follows them automatically, which would
otherwise let a public URL 302 to `169.254.169.254` and walk straight past the check on
the original address. This is the most commonly missed part of an SSRF guard.

RESIDUAL RISK, STATED PLAINLY
-----------------------------
This does **not** defeat DNS rebinding: a hostile resolver can pass validation and then
return a private address at connect time. Closing that properly means resolving once and
pinning the socket to the validated IP, which breaks TLS SNI and certificate hostname
verification unless you reimplement both — a worse trade than the risk for a single-user
local agent, and one that should be made deliberately rather than silently. It is
recorded here and in `SECURITY.md` §5 so it is a known gap, not an unknown one.

Nor does it rate-limit. A fetched page is capped in size and time, which bounds the
damage of one call but not of many.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

MAX_BYTES = 400_000          # ~100k tokens of HTML; far more than any useful page
MAX_SECONDS = 15
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = frozenset({"http", "https"})
USER_AGENT = "FRIDAY/0 (+https://github.com/Jaganravi131/FRIDAY) personal-agent"


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    url: str = ""
    ip: str = ""


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Return a reason string if this address must not be connected to, else ""."""
    # Unwrap IPv4-mapped IPv6 FIRST: ::ffff:127.0.0.1 is loopback wearing a disguise,
    # and checking is_loopback on the mapped form does not catch it on every platform.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_unspecified:
        # Before is_private: 0.0.0.0 and :: satisfy BOTH in Python's ipaddress, and
        # "private — your LAN" is the wrong explanation for "not a destination".
        return "unspecified (0.0.0.0 / ::) — not a destination"
    if ip.is_loopback:
        return "loopback — FRIDAY's own gateway and model server live here"
    if ip.is_link_local:
        return ("link-local — this range contains 169.254.169.254, the cloud metadata "
                "endpoint, which is the most valuable SSRF target there is")
    if ip.is_private:
        return "private — your LAN, router, NAS and printers"
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.IPv4Network("100.64.0.0/10"):
        return ("CGNAT — this is also Tailscale's range, so it means your own devices. "
                "Reach them deliberately (allow_private=True), not because a page said to")
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    return ""


def validate_url(url: str, *, allow_private: bool = False) -> Verdict:
    """Is this URL safe to fetch? Resolves the host, so it catches IP-literal tricks
    (`http://127.1`, `http://0x7f.0.0.1`, `http://[::1]`) that string matching misses.
    """
    raw = (url or "").strip()
    if not raw:
        return Verdict(False, "empty URL", raw)
    try:
        p = urllib.parse.urlsplit(raw)
    except ValueError as e:
        return Verdict(False, f"unparseable: {e}", raw)

    if p.scheme.lower() not in ALLOWED_SCHEMES:
        return Verdict(False, f"scheme '{p.scheme or '(none)'}' is not allowed — "
                              f"http/https only; file://, gopher:// and dict:// are how "
                              f"an SSRF guard gets bypassed", raw)
    if not p.hostname:
        return Verdict(False, "no hostname", raw)

    # Credentials in a URL are a phishing shape and never needed here.
    if p.username or p.password:
        return Verdict(False, "URL contains credentials", raw)

    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return Verdict(False, f"DNS resolution failed for {p.hostname}: {e}", raw)
    except (OSError, UnicodeError) as e:
        return Verdict(False, f"cannot resolve {p.hostname}: {e}", raw)

    # EVERY address the name resolves to must be public. Checking only the first is how
    # a host with one good A record and one 127.0.0.1 A record gets through.
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
        except ValueError:
            return Verdict(False, f"unparseable address from DNS for {p.hostname}", raw)
        why = _is_blocked_ip(ip)
        if why and not allow_private:
            return Verdict(False, f"{p.hostname} resolves to {ip}: {why}", raw, str(ip))

    return Verdict(True, "", raw, str(ipaddress.ip_address(infos[0][4][0].split("%")[0])))


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Validate every hop. urllib follows redirects automatically, so without this a
    public URL can 302 to 169.254.169.254 and walk past the check on the first address.
    """

    def __init__(self, allow_private: bool = False) -> None:
        self.allow_private = allow_private
        self.hops = 0
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.hops += 1
        if self.hops > MAX_REDIRECTS:
            raise urllib.error.HTTPError(req.full_url, code,
                                         f"too many redirects (>{MAX_REDIRECTS})", headers, fp)
        v = validate_url(newurl, allow_private=self.allow_private)
        if not v.ok:
            # Refuse the hop rather than following it. The message reaches the caller, so
            # the failure is loud instead of looking like an empty page.
            raise urllib.error.HTTPError(req.full_url, code,
                                         f"redirect blocked: {v.reason}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, *, allow_private: bool = False,
          max_bytes: int = MAX_BYTES, timeout: float = MAX_SECONDS) -> dict:
    """Fetch a URL safely. Returns a dict; never raises.

    The caller is responsible for treating `text` as untrusted data — which in FRIDAY
    means `neutralize()` and EXTERNAL trust, the same path retrieved memory takes. This
    function guarantees only that the *connection* went somewhere acceptable.
    """
    v = validate_url(url, allow_private=allow_private)
    if not v.ok:
        return {"ok": False, "error": f"refused: {v.reason}", "url": url}

    opener = urllib.request.build_opener(_GuardedRedirect(allow_private))
    req = urllib.request.Request(v.url, headers={"user-agent": USER_AGENT,
                                                 "accept": "text/html,application/json;q=0.9"})
    try:
        with opener.open(req, timeout=timeout) as r:
            ctype = (r.headers.get("content-type") or "").lower()
            if "html" not in ctype and "json" not in ctype and "text" not in ctype:
                return {"ok": False, "error": f"content-type '{ctype}' is not text — "
                                              f"not fetching binaries", "url": v.url}
            raw = r.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            # ⚠️ NOT `r.encoding`: http.client.HTTPResponse has no such attribute, so
            # every successful fetch raised AttributeError. The charset lives in the
            # Content-Type header, and when it is absent (common) UTF-8 with "replace"
            # is the right guess — a decode that raises is worse than a mangled
            # character, because it turns a readable page into a failed tool call.
            m = re.search(r"charset=([\w.-]+)", ctype)
            enc = m.group(1) if m else "utf-8"
            try:
                text = raw[:max_bytes].decode(enc, "replace")
            except (LookupError, UnicodeDecodeError):
                text = raw[:max_bytes].decode("utf-8", "replace")
            return {"ok": True, "url": r.geturl(), "status": r.status,
                    "content_type": ctype, "text": text, "truncated": truncated,
                    "resolved_ip": v.ip}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code} {e.reason}", "url": v.url}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"network error: {e.reason}", "url": v.url}
    except (OSError, ValueError, TimeoutError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "url": v.url}


def strip_html(html: str) -> str:
    """Crude, dependency-free HTML -> text. Not a parser, and not pretending to be one.

    Removes script/style blocks and tags, unescapes the handful of entities that matter,
    and collapses whitespace. A real extractor (readability) would be better and would
    cost a dependency; for "let the model read a page" this is enough, and the model is
    tolerant of noise in a way a template is not.
    """
    import html as _html
    import re

    t = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html)
    t = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</tr>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()
