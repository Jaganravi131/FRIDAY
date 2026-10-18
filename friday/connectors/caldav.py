"""Calendar over CalDAV, read-only by design (docs/architecture/18 §3).

Why CalDAV REPORT with an app password (iCloud/Fastmail/Nextcloud) instead of a
vendor API: one config stanza, no OAuth dance on a headless laptop, a credential
you can revoke without touching your account password, and a protocol old enough
that every piece of it is stdlib (`http.client`, `xml.etree`) — the core stays
dependency-free, which is a hard rule for this project. Google's CalDAV endpoint
requires OAuth, so Google Calendar is deliberately out of scope here; the driver
targets the providers whose plain CalDAV still works.

Scope decisions that are LOAD-BEARING, not incidental:

  * READ-ONLY IS ENFORCED BY VERB, ONLY. CalDAV writes are PUT / DELETE /
    PROPPATCH / MKCALENDAR. This driver's one network function accepts exactly
    two methods — PROPFIND and REPORT, both read-only WebDAV — and raises on any
    other before the request reaches the wire. Same shape as Gmail's
    `readonly=True`+BODY.PEEK: the permission lives in the data layer, not in a
    model remembering to behave.
  * HTTPS unless the host is loopback. Credentials travel with every request, so
    plain http to a real host is refused before the socket opens. The loopback
    carve-out exists for tests and for a Nextcloud you run at home on your own
    LAN — say so in config as `insecure_http: true` for a non-loopback LAN box.
  * The user configures the COLLECTION URL directly (e.g.
    https://caldav.icloud.com/<dsid>/calendars/<calid>/). Principal discovery is
    ~3 more PROPFIND hops of protocol surface for zero security benefit; be
    boring, be auditable.
  * Everything returned is imported content. An event DESCRIPTION is exactly the
    "your inbox told FRIDAY to…" injection surface the launcher layer withholds —
    anyone invited to your calendar can put text in front of your model.
"""
from __future__ import annotations

import base64
import datetime as _dt
import http.client
import urllib.parse
import xml.etree.ElementTree as ET

from . import Connector, Item, config_help, load_config, register

MAX_EVENTS = 15
MAX_BODY_CHARS = 4_000
READ_ONLY_METHODS = frozenset({"PROPFIND", "REPORT"})

_CONFIG_EXAMPLE = {
    "url": "https://caldav.icloud.com/<dsid>/calendars/home/   # the collection URL",
    "user": "you@icloud.com",
    "app_password": "xxxx-xxxx-xxxx-xxxx   # Apple ID → Sign-In and Security → App-Specific Passwords",
}

_C_NS = "{urn:ietf:params:xml:ns:caldav}"


def _cfg() -> dict:
    return load_config("calendar")


def _split(cfg: dict) -> urllib.parse.SplitResult:
    url = str(cfg.get("url") or "")
    parts = urllib.parse.urlsplit(url)
    loopback = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme == "http" and not (loopback or cfg.get("insecure_http")):
        raise RuntimeError(
            "calendar URL must be https:// — the app password travels with every "
            "request. (http is accepted on loopback, or set insecure_http: true "
            "for a server on your own LAN)")
    if parts.scheme not in ("https", "http"):
        raise RuntimeError(f"calendar URL scheme must be https (got {parts.scheme!r})")
    if not parts.hostname:
        raise RuntimeError("calendar URL has no host")
    return parts


def _http(parts: urllib.parse.SplitResult, method: str, body: str,
          headers: dict, cfg: dict) -> bytes:
    """The ONLY egress in this module. The verb allowlist is the read-only
    contract: anything but PROPFIND/REPORT is a write verb, and this function
    refuses it before a socket exists — Law 5 enforced in the data layer."""
    if method not in READ_ONLY_METHODS:
        raise RuntimeError(
            f"refusing CalDAV method {method!r}: this driver is read-only "
            f"({sorted(READ_ONLY_METHODS)} only — PUT/DELETE/PROPPATCH never)")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    cls = (http.client.HTTPSConnection if parts.scheme == "https"
           else http.client.HTTPConnection)
    conn = cls(parts.hostname, port, timeout=25)
    try:
        auth = base64.b64encode(
            f"{cfg['user']}:{cfg['app_password']}".encode()).decode()
        conn.request(method, parts.path or "/", body.encode("utf-8"),
                     {"Authorization": f"Basic {auth}",
                      "Content-Type": "text/xml; charset=utf-8",
                      "Depth": "1", **headers})
        resp = conn.getresponse()
        payload = resp.read()
        if resp.status == 401:
            raise RuntimeError("username or app password rejected (401)")
        if resp.status >= 400:
            raise RuntimeError(f"CalDAV server answered {resp.status} {resp.reason}")
        return payload
    finally:
        conn.close()


# ── iCalendar, the honest subset ──────────────────────────────────────────────

def _unfold(text: str) -> list[str]:
    """RFC 5545 §3.1: a line starting with space/tab continues the previous one."""
    lines, out = text.replace("\r\n", "\n").split("\n"), []
    for ln in lines:
        if ln[:1] in (" ", "\t") and out:
            out[-1] += ln[1:]
        elif ln.strip():
            out.append(ln)
    return out


def _unescape(v: str) -> str:
    return (v.replace("\\n", "\n").replace("\\N", "\n")
             .replace("\\;", ";").replace("\\,", ",").replace("\\\\", "\\"))


def _iso(v: str) -> str:
    """Normalise a DATE-TIME property into a sortable display string. Timezones
    are NOT resolved (no tz data without pulling a dependency); the raw wall time
    is shown, and 'Z' survives as the tail marker that it is UTC."""
    v = v.strip()
    try:
        if len(v) == 8 and v.isdigit():                          # DATE value
            return f"{v[:4]}-{v[4:6]}-{v[6:8]}"
        if "T" in v:                                             # DATE-TIME
            d, t = v.split("T", 1)
            z = "Z" if t.endswith("Z") else ""
            t = t.rstrip("Z")
            return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}{z}"
    except (IndexError, ValueError):
        pass
    return v


def _parse_vevent(block: list[str]) -> dict:
    props: dict[str, str] = {}
    for ln in block:
        head, _, value = ln.partition(":")
        name = head.split(";", 1)[0].upper()                     # drop ;TZID=… params
        if name in ("SUMMARY", "DTSTART", "DTEND", "LOCATION", "DESCRIPTION",
                    "UID", "ORGANIZER"):
            props[name] = _unescape(value.strip())
    if "DTSTART" in props:
        props["DTSTART"] = _iso(props["DTSTART"])
    if "DTEND" in props:
        props["DTEND"] = _iso(props["DTEND"])
    return props


def _parse_feed(payload: bytes) -> list[dict]:
    """multistatus XML -> [{'SUMMARY':…}, …]. One malformed event must not sink
    the rest of the feed — same rule as the mail parser."""
    events = []
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as e:
        raise RuntimeError(f"server response was not multistatus XML ({e})")
    for data in root.iter(f"{_C_NS}calendar-data"):
        lines = _unfold(data.text or "")
        # ⭐ EVERY VEVENT in the block, not the first. A VCALENDAR can carry many
        # events, and a containment layer that parses only event #1 is a hiding
        # spot: "friendly standup" first, "ignore previous instructions" second.
        # The test feed pairs them this way precisely so silent skipping fails.
        i = 0
        while i < len(lines):
            if lines[i].upper() == "BEGIN:VEVENT":
                try:
                    end = next(j for j in range(i + 1, len(lines))
                               if lines[j].upper() == "END:VEVENT")
                    events.append(_parse_vevent(lines[i + 1:end]))
                    i = end + 1
                except StopIteration:
                    break                    # unterminated block: drop the tail
            else:
                i += 1
    return events


def _time_range(days_back: int, days_ahead: int) -> tuple[str, str]:
    now = _dt.datetime.now(_dt.timezone.utc)
    lo = now - _dt.timedelta(days=max(0, days_back))
    hi = now + _dt.timedelta(days=max(1, days_ahead))
    fmt = "%Y%m%dT%H%M%SZ"
    return lo.strftime(fmt), hi.strftime(fmt)


_QUERY = """<?xml version="1.0" encoding="utf-8"?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop><C:calendar-data/></D:prop>
  <C:filter>
    <C:comp-filter name="VCALENDAR">
      <C:comp-filter name="VEVENT">
        <C:time-range start="{lo}" end="{hi}"/>
      </C:comp-filter>
    </C:comp-filter>
  </C:filter>
</C:calendar-query>"""


@register
class CalendarConnector(Connector):
    id = "calendar"
    sense = "calendar.read"

    def status(self) -> dict:
        cfg = _cfg()
        if "__corrupt__" in cfg:
            return {"connector": self.id, "configured": False,
                    "error": f"config/connectors.json is not valid JSON: {cfg['__corrupt__']}"}
        missing = [k for k in ("url", "user", "app_password") if not cfg.get(k)]
        if missing:
            return {"connector": self.id, "configured": False,
                    "how_to": config_help("calendar", _CONFIG_EXAMPLE)}
        host = urllib.parse.urlsplit(str(cfg["url"])).hostname or "?"
        scheme_ok = str(cfg["url"]).startswith("https://") or host in (
            "localhost", "127.0.0.1", "::1") or bool(cfg.get("insecure_http"))
        return {"connector": self.id, "configured": True, "host": host,
                "user": cfg["user"],
                "transport": "https" if str(cfg["url"]).startswith("https") else
                ("http (LAN)" if scheme_ok else "http ✗ refused — use https"),
                "note": "app password present (never shown, never logged)"}

    def fetch(self, *, limit: int = 8, days_back: int = 1,
              days_ahead: int = 14) -> list[Item]:
        """Fetch VEVENTs in the window. Raises with a user-readable message on
        misconfiguration/network failure; the launcher turns that into a dict."""
        limit = max(1, min(int(limit or 8), MAX_EVENTS))
        cfg = _cfg()
        if not all(cfg.get(k) for k in ("url", "user", "app_password")):
            raise RuntimeError(config_help("calendar", _CONFIG_EXAMPLE))
        parts = _split(cfg)
        lo, hi = _time_range(int(days_back), int(days_ahead))
        payload = _http(parts, "REPORT", _QUERY.format(lo=lo, hi=hi), {}, cfg)
        events = _parse_feed(payload)
        events.sort(key=lambda e: e.get("DTSTART", ""))
        out = []
        for e in events[:limit]:
            span = e.get("DTSTART", "")
            if e.get("DTEND"):
                span += f" → {e['DTEND']}"
            out.append(Item(
                connector=self.id,
                id=e.get("UID", "")[:80],
                title=e.get("SUMMARY") or "(untitled event)",
                body=(e.get("DESCRIPTION") or "")[:MAX_BODY_CHARS],
                author=e.get("ORGANIZER", ""),
                when=span,
                extra={"location": e.get("LOCATION", "")},
            ))
        return out
