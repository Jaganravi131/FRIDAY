"""GitHub over the REST API, read-only by design (docs/architecture/18 §3.5).

Why REST + a fine-grained personal access token instead of OAuth or the CLI:
one config stanza, a credential that is revocable in one click, and stdlib
urllib end to end — the core stays dependency-free, which is a hard rule for
this project. Use a FINE-GRAINED PAT with read-only repository access; the
driver additionally enforces read-only on its side (below), but defence in
depth means the credential itself should not carry write scopes.

Scope decisions that are LOAD-BEARING, not incidental:

  * READ-ONLY IS THE SINGLE VERB. Every mutating REST call needs POST, PATCH,
    PUT, or DELETE. This driver's one network function accepts exactly GET and
    raises on any other verb BEFORE the socket exists — same shape as IMAP
    readonly=True and CalDAV's {PROPFIND, REPORT} allowlist. A test names the
    write verbs and asserts each refusal.
  * The endpoint is FIXED to https://api.github.com. No URL from config means
    no way for a misdirected credential to be POSTed to a lookalike host —
    the token can only ever leave this process toward the one host it is for.
  * Issue and notification titles/bodies are attacker-controlled on any public
    repo — the same "inbox told FRIDAY to…" injection surface as email, just
    wearing a code review's clothes. The launcher contains title AND body.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import Connector, Item, config_help, load_config, register

API = "https://api.github.com"          # the ONLY host the token may ever reach
MAX_ITEMS = 15
MAX_BODY_CHARS = 4_000

_CONFIG_EXAMPLE = {
    "user": "octocat",
    "token": "github_pat_xxxx   # fine-grained PAT, read-only repo access — "
             "GitHub → Settings → Developer settings → Personal access tokens",
}


def _cfg() -> dict:
    return load_config("github")


def _get(path: str, cfg: dict) -> list | dict:
    """The ONLY egress in this module. GET is the entire API surface this
    connector implements; anything else is a write verb, refused before the
    request reaches the wire — Law 5 enforced in the data layer, as with the
    IMAP and CalDAV drivers."""
    if not path.startswith("/"):
        raise RuntimeError(f"refusing non-API path {path!r}")
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {cfg['token']}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "friday-connector"},
        method="GET",
    )
    assert req.get_method() == "GET"      # the allowlist, restated at the wire
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("token rejected (401) — check the fine-grained PAT")
        if e.code == 403:
            raise RuntimeError("403 — rate-limited or the token lacks read access")
        raise RuntimeError(f"github.com answered {e.code}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach api.github.com ({type(e).__name__})")


def _norm_issue(raw: dict) -> Item:
    repo = (raw.get("repository_url") or "").rsplit("/", 2)[-2:]
    return Item(
        connector="github",
        id=(raw.get("html_url") or "")[:200],
        title=raw.get("title") or "(untitled)",
        body=(raw.get("body") or "")[:MAX_BODY_CHARS],
        author=(raw.get("user") or {}).get("login", ""),
        when=str(raw.get("updated_at") or ""),
        extra={"kind": "pull_request" if "pull_request" in raw else "issue",
               "repo": "/".join(repo)},
    )


def _norm_notification(raw: dict) -> Item:
    subj = raw.get("subject") or {}
    repo = (raw.get("repository") or {}).get("full_name", "")
    body = f"[{raw.get('reason', '')}] {subj.get('type', '')} in {repo}"
    return Item(
        connector="github",
        id=str(raw.get("id") or ""),
        title=subj.get("title") or "(untitled)",
        body=body,
        author=repo,
        when=str(raw.get("updated_at") or ""),
        extra={"kind": "notification", "repo": repo},
    )


@register
class GitHubConnector(Connector):
    id = "github"
    sense = "github.read"

    def status(self) -> dict:
        cfg = _cfg()
        if "__corrupt__" in cfg:
            return {"connector": self.id, "configured": False,
                    "error": f"config/connectors.json is not valid JSON: {cfg['__corrupt__']}"}
        if not cfg.get("user") or not cfg.get("token"):
            return {"connector": self.id, "configured": False,
                    "how_to": config_help("github", _CONFIG_EXAMPLE)}
        return {"connector": self.id, "configured": True, "host": "api.github.com",
                "user": cfg["user"],
                "transport": "https",
                "note": "token present (never shown, never logged)"}

    def fetch(self, *, limit: int = 10, include_notifications: bool = True
              ) -> list[Item]:
        """Assigned open issues/PRs first, then unread notifications to fill.
        Raises with a user-readable message; the launcher turns it into a dict,
        never a crashed turn."""
        limit = max(1, min(int(limit or 10), MAX_ITEMS))
        cfg = _cfg()
        if not cfg.get("user") or not cfg.get("token"):
            raise RuntimeError(config_help("github", _CONFIG_EXAMPLE))
        out: list[Item] = []
        issues = _get(f"/issues?filter=assigned&state=open&sort=updated"
                      f"&per_page={limit}", cfg)
        for raw in (issues if isinstance(issues, list) else [])[:limit]:
            out.append(_norm_issue(raw))
        if include_notifications and len(out) < limit:
            notes = _get(f"/notifications?per_page={limit - len(out)}", cfg)
            for raw in (notes if isinstance(notes, list) else [])[:limit - len(out)]:
                out.append(_norm_notification(raw))
        out.sort(key=lambda i: i.when, reverse=True)
        return out[:limit]
