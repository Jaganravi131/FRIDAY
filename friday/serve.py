"""`friday serve` — the HTTP gateway. Stdlib only, authenticated, local-first.

WHY
---
A laptop-only agent is a toy. Exit test #12 is "reach FRIDAY from your phone", and both
reference systems agree this matters: OpenClaw is a *gateway* first (one process, every
chat channel), and OpenAI's Dots carry the same context across ChatGPT, Slack, Teams and
voice. Presence is not a nice-to-have bolted on later; it is the thing that makes an
agent part of your day instead of a window you open.

This is the seed of that — the Presence Fabric from doc 07 — and it is deliberately
small: one stdlib HTTP server, one HTML page with no build step and no CDN, and an
OpenAI-compatible endpoint so any existing client can point at FRIDAY without FRIDAY
having to implement their protocol.

SECURITY POSTURE
----------------
This process runs an agent that can read and write your memory. Exposing it is not like
exposing a web page. So:

  * **Bearer token, always.** Generated into `config/serve.token` on first run with
    0600 permissions. No token, no request — including `/v1/*`, which OpenAI-compatible
    clients handle natively via their api-key field.
  * **Binds loopback by default.** `--host tailscale` binds only the tailnet address,
    which is the doc 09 answer to device theft: reachable from your phone, invisible to
    the internet, no port forwarding, no exposed surface.
  * `--host 0.0.0.0` is allowed because a LAN phone is a legitimate setup, but it prints
    a warning naming exactly what it exposes. It is never the default.
  * **Remote turns are audited as `remote`.** `audit --today` must be able to tell you
    which turns came over the network. A gateway that cannot answer that is a gateway
    you cannot trust.
  * No CORS wildcard. A browser page on another origin cannot drive your agent.
  * The trust boundary in `friday/security/` still applies: a message arriving over HTTP
    is user input, and anything it causes to be *retrieved* is still fenced as data.

Deliberately NOT here: TLS (Tailscale encrypts the tunnel; terminating TLS locally adds
a certificate to manage for no gain on a tailnet), rate limiting, and multi-user auth.
This is one person's agent, and pretending otherwise would mean building the wrong thing
carefully.
"""

from __future__ import annotations

import json
import secrets
import socket
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 8642
MODEL_NAME = "friday-local"

#: The whole UI. No build step, no CDN, no external asset: it has to work on a laptop
#: with no internet, because that is the entire point of the project.
INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRIDAY</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0d1117;color:#e6edf3;
      font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
 header{padding:14px 18px;border-bottom:1px solid #21262d;display:flex;
        justify-content:space-between;align-items:center;gap:12px}
 header b{font-size:15px;letter-spacing:.14em}
 #state{font-size:11px;color:#7d8590}
 #bar{max-width:820px;margin:0 auto;padding:14px 18px 0;display:flex;gap:8px;
      align-items:center;flex-wrap:wrap}
 #bar span{color:#7d8590;font-size:12px}
 #bar input{flex:1;min-width:180px;background:#161b22;border:1px solid #30363d;
      border-radius:6px;color:#e6edf3;padding:7px;font:inherit;font-size:12px}
 #bar button{background:#30363d;border:0;border-radius:6px;color:#e6edf3;
      padding:7px 14px;font:inherit;font-size:12px;cursor:pointer}
 main{max-width:820px;margin:0 auto;padding:18px}
 .turn{margin:0 0 18px}
 .u{color:#79c0ff;white-space:pre-wrap}
 .a{color:#e6edf3;white-space:pre-wrap;margin-top:6px}
 .meta{color:#7d8590;font-size:11px;margin-top:6px;white-space:pre-wrap}
 .warn{color:#f0883e;white-space:pre-wrap;margin-top:6px;font-size:12px}
 .why{color:#7d8590;font-size:11px;background:none;border:1px solid #30363d;
      border-radius:4px;padding:1px 7px;margin-top:6px;cursor:pointer}
 form{position:sticky;bottom:0;background:#0d1117;border-top:1px solid #21262d;
      padding:12px 18px;display:flex;gap:8px}
 input{flex:1;background:#161b22;border:1px solid #30363d;border-radius:6px;
       color:#e6edf3;padding:10px;font:inherit}
 button[type=submit]{background:#238636;border:0;border-radius:6px;color:#fff;
       padding:0 18px;font:inherit;cursor:pointer}
 button[type=submit]:disabled{opacity:.5;cursor:progress}
</style></head><body>
<header><b>FRIDAY</b><span id="state">connecting…</span></header>
<div id="bar" hidden><span>token</span><input id="tk" type="password"
  placeholder="paste from config/serve.token"><button id="go">connect</button></div>
<main id="log"></main>
<form id="f"><input id="q" autocomplete="off" autofocus
  placeholder="ask, or tell it something to remember"><button type="submit">send</button></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),
      q=document.getElementById('q'),st=document.getElementById('state'),
      bar=document.getElementById('bar'),tk=document.getElementById('tk');

/* ⚠️ No prompt(), and no bare localStorage.
   This page is opened inside a proxied iframe (the Arena preview) as often as in a
   browser tab. A sandboxed iframe BLOCKS modal dialogs — prompt() silently returns
   null — and touching localStorage on an opaque origin THROWS SecurityError. Either
   one killed the page before the form handler was attached, so the UI rendered and
   then did nothing at all, with no error visible. Token comes from the URL fragment,
   then storage (guarded), then an inline field that works anywhere. */
const store={
  get(k){try{return localStorage.getItem(k)}catch(e){return null}},
  set(k,v){try{localStorage.setItem(k,v)}catch(e){}}
};
let token=new URLSearchParams(location.hash.slice(1)).get('token')
        || store.get('friday.token') || '';
if(token) store.set('friday.token',token);

function needToken(msg){
  st.textContent=msg; bar.hidden=false; tk.value=''; tk.focus();
}
document.getElementById('go').onclick=()=>{
  token=tk.value.trim(); if(!token)return;
  store.set('friday.token',token); bar.hidden=true; ping();
};
tk.onkeydown=e=>{if(e.key==='Enter')document.getElementById('go').click()};

async function call(path,body){
  const r=await fetch(path,{method:body?'POST':'GET',
    headers:{'content-type':'application/json','authorization':'Bearer '+token},
    body:body?JSON.stringify(body):undefined});
  if(r.status===401){needToken('token rejected — paste it again');throw new Error('401')}
  return r.json();
}
function el(cls,txt){const d=document.createElement('div');d.className=cls;
  d.textContent=txt;return d;}
async function ping(){
  try{
    const h=await call('/health');
    st.textContent=h.ready?'ready · '+h.model
                           :'degraded · '+h.model+' (no model server — see INSTALL.md §3)';
  }catch(e){/* needToken already explained it */}
}
token?ping():needToken('paste your serve token to connect');
f.onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text)return;
  q.value='';const t=document.createElement('div');t.className='turn';
  t.appendChild(el('u','you: '+text));log.appendChild(t);
  const btn=f.querySelector('button');btn.disabled=true;
  try{
    const r=await call('/api/ask',{message:text});
    t.appendChild(el('a','FRIDAY: '+r.answer));
    if(r.security&&r.security.length)r.security.forEach(s=>t.appendChild(el('warn',s)));
    if(r.ledger)t.appendChild(el('meta',r.ledger));
    const b=document.createElement('button');b.className='why';b.textContent='why?';
    b.onclick=async()=>{const w=await call('/api/why',{question:text});
      t.appendChild(el('meta',w.report||'(no provenance)'));b.remove();};
    t.appendChild(b);
  }catch(err){if(err.message!=='401')t.appendChild(el('warn','error: '+err.message));}
  btn.disabled=false;q.focus();log.scrollIntoView(false);
};
</script></body></html>
"""


# ── token ──────────────────────────────────────────────────────────────────────

def ensure_token(root: Path) -> tuple[str, Path, bool]:
    """Return (token, path, created). Created tokens get 0600 where the OS allows it."""
    path = root / "config" / "serve.token"
    if path.exists():
        return path.read_text(encoding="utf-8").strip(), path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)                 # POSIX; a no-op error on some Windows setups
    except OSError:
        pass
    return token, path, True


# ── the agent, one connection per request ──────────────────────────────────────

_local = threading.local()


def _conn():
    """sqlite3 connections cannot cross threads, so each thread keeps its own."""
    c = getattr(_local, "conn", None)
    if c is None:
        from .store import db
        from . import paths

        c = db.connect(paths.DB_PATH)
        _local.conn = c
    return c


def _agent():
    a = getattr(_local, "agent", None)
    if a is None:
        from .agent.loop import Agent
        from .llm import get_client
        from .retrieval.embedders import get_embedder
        from .retrieval.rerankers import get_reranker

        a = Agent(_conn(), client=get_client("auto"), embedder=get_embedder(),
                  reranker=get_reranker())
        _local.agent = a
    return a


#: Set once, warned once. A database built before the `remote` scope existed carries
#: the old CHECK constraint and will reject it; the fix is `friday rebuild`, which is
#: safe because artifacts/ is derived. Falling back keeps the gateway working in the
#: meantime rather than failing every request with a constraint error.
_scope_fallback = False


EGRESS_TOOLS = frozenset({"web_read", "wiki", "http_post"})


def _describe_tool(t) -> str:
    """One line per tool call; egress calls carry their destination."""
    if not isinstance(t, dict):
        return str(t)
    name = t.get("origin") or t.get("tool") or ""
    if name in EGRESS_TOOLS:
        target = t.get("target") or t.get("url") or ""
        return f"{name} → {target}" + ("  ✗ refused" if t.get("error") else "")
    return name or "unknown"


def answer(message: str, *, scope: str = "remote") -> dict:
    """Run one turn. Returns JSON-able data, never raises."""
    global _scope_fallback
    try:
        try:
            st = _agent().run(message, scope=scope)
        except sqlite3.IntegrityError as e:
            if "scope" not in str(e) or _scope_fallback:
                raise
            _scope_fallback = True
            print("[serve] this database predates the 'remote' scope; running the turn "
                  "as 'interactive' instead.\n        Run `python -m friday rebuild` so "
                  "network turns are distinguishable in the audit trail.",
                  file=sys.stderr)
            st = _agent().run(message, scope="interactive")
        return {
            "answer": st.final_text,
            "intent": getattr(getattr(st, "intent", None), "kind", None),
            "memory_used": bool(getattr(getattr(st, "intent", None), "needs_memory", True)),
            "ledger": st.compiled.printout(st.turn_idx) if st.compiled else "",
            "withheld": list(getattr(st.compiled, "withheld", []) or []),
            "security": list(getattr(st, "security_notices", []) or []),
            # Names alone are not enough for the tools that mean a request LEFT the
            # machine. The phone user gets the same observability the terminal user gets,
            # because the exfiltration risk does not care which screen you are on.
            "tools": [_describe_tool(t) for t in (st.tool_log or [])],
            "model": MODEL_NAME,
        }
    except Exception as e:
        # A gateway that 500s with a stack trace teaches a client to retry blindly.
        return {"answer": f"FRIDAY hit an error and did not guess: "
                          f"{type(e).__name__}: {e}",
                "error": True, "model": MODEL_NAME}


def why(question: str) -> dict:
    try:
        from .memory.provenance import why_query

        return {"report": why_query(_conn(), question)}
    except Exception as e:
        return {"report": f"could not build provenance: {type(e).__name__}: {e}",
                "error": True}


def health() -> dict:
    from . import paths
    from .llm import get_client

    try:
        client = get_client("auto")
        name = getattr(client, "name", type(client).__name__)
        real = "mock" not in name.lower()
    except Exception:
        name, real = "unavailable", False
    return {"status": "ok", "model": name, "ready": real,
            "db": paths.DB_PATH.exists(),
            "note": "" if real else
                    "running on the mock client — memory works, conversation does not. "
                    "Start a model server; see INSTALL.md §3."}


# ── HTTP ───────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    token: str = ""
    server_version = "FRIDAY/0"

    def log_message(self, fmt, *args):        # quieter, and never log the token
        sys.stderr.write("[serve] %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        # No wildcard CORS. A page on another origin must not be able to drive an agent
        # that can write to your memory.
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("referrer-policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _authorized(self) -> bool:
        got = self.headers.get("authorization", "")
        if got.startswith("Bearer ") and secrets.compare_digest(got[7:].strip(), self.token):
            return True
        # OpenAI-compatible clients send the key here instead.
        if secrets.compare_digest(self.headers.get("x-api-key", "").strip(), self.token):
            return True
        return False

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("content-length") or 0)
        except ValueError:
            return {}
        if n <= 0 or n > 1_000_000:            # a 1 MB cap: this is a chat gateway
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/healthz"):
            return self._json(200, health())    # unauthenticated: liveness only, no data
        if path in ("/", "/index.html"):
            return self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        if not self._authorized():
            return self._json(401, {"error": "missing or invalid bearer token"})
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "friday"}]})
        if path == "/api/status":
            from .doctor import run as doctor_run

            return self._json(200, doctor_run().as_dict())
        return self._json(404, {"error": f"no route for {path}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            return self._json(401, {"error": "missing or invalid bearer token"})
        body = self._body()

        if path == "/api/ask":
            message = str(body.get("message") or "").strip()
            if not message:
                return self._json(400, {"error": "message is required"})
            return self._json(200, answer(message))

        if path == "/api/why":
            return self._json(200, why(str(body.get("question") or "").strip()))

        if path == "/v1/chat/completions":
            # OpenAI-compatible, so an existing client — a phone app, another agent —
            # can use FRIDAY without FRIDAY implementing their protocol. Only the last
            # user message is treated as the turn; the rest is context FRIDAY already
            # keeps itself.
            msgs = body.get("messages") or []
            text = next((m.get("content") for m in reversed(msgs)
                         if m.get("role") == "user"), "")
            r = answer(str(text or "").strip())
            return self._json(200, {
                "id": f"friday-{secrets.token_hex(6)}", "object": "chat.completion",
                "model": MODEL_NAME,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": r["answer"]}}],
                "friday": {k: r[k] for k in ("intent", "memory_used", "withheld",
                                             "security", "tools") if k in r},
            })

        return self._json(404, {"error": f"no route for {path}"})


def resolve_host(spec: str) -> tuple[str, str]:
    """Return (bind_address, human_label). `tailscale` finds the tailnet address."""
    if spec in ("localhost", "loopback", "127.0.0.1"):
        return "127.0.0.1", "loopback only — this machine"
    if spec == "tailscale":
        try:
            addrs = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
            for _, _, _, _, sa in addrs:
                if sa[0].startswith(("100.", "fd00:")):      # CGNAT range Tailscale uses
                    return sa[0], f"tailnet only ({sa[0]}) — your devices, not the internet"
        except OSError:
            pass
        return "127.0.0.1", ("loopback only — could not find a 100.x tailnet address, "
                             "so `tailscale up` may not be running")
    if spec in ("0.0.0.0", "lan", "all"):
        return "0.0.0.0", ("EVERY interface, including the local network — anyone on "
                           "your Wi-Fi can attempt this port")
    return spec, f"{spec}"


def serve(*, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          token: str | None = None, quiet: bool = False) -> int:
    from . import paths

    paths.ensure_layout()
    tok, tok_path, created = (token, None, False) if token else ensure_token(paths.ROOT)
    bind, label = resolve_host(host)

    Handler.token = tok
    try:
        httpd = ThreadingHTTPServer((bind, port), Handler)
    except OSError as e:
        print(f"cannot bind {bind}:{port} — {e}", file=sys.stderr)
        print("  another FRIDAY may already be serving, or the port is taken.",
              file=sys.stderr)
        return 2

    if not quiet:
        print("FRIDAY gateway")
        print("=" * 62)
        print(f"  listening  {bind}:{port}   ({label})")
        print(f"  token      {tok_path or '(passed inline)'}"
              + ("   ← newly created, 0600" if created else ""))
        print(f"  token value {tok}")
        shown = "127.0.0.1" if bind == "0.0.0.0" else bind
        print(f"  web UI     http://{shown}:{port}/")
        # The fragment never reaches the server, so this is safe to bookmark and it
        # skips typing the token — and it is the only route that works where a modal
        # prompt would not.
        print(f"  no typing  http://{shown}:{port}/#{tok[:0]}token={tok}")
        print(f"  OpenAI API http://{bind}:{port}/v1/chat/completions")
        print(f"  health     http://{bind}:{port}/health   (no token)")
        if bind == "0.0.0.0":
            print("\n⚠️  Bound to every interface. The token is the only thing between")
            print("   your memory and your local network. Prefer --host tailscale.")
        h = health()
        if not h["ready"]:
            print(f"\n⚠️  {h['note']}")
        print("\nCtrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            print("\nstopped.")
    finally:
        httpd.server_close()
    return 0
