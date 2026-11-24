"""The unified FRIDAY HUD: FastAPI + SSE in front of the cognition core.

This module is the graft seam between the two lineages:

  * **the core** (`friday/agent`, `friday/memory`, `friday/retrieval`,
    `friday/security`, `friday/connectors`) answers the turn, enforces policy,
    compiles memory and persists the audit trail. None of that is re-implemented
    here.
  * **the HUD** (`friday/web/static`) renders it: token streaming, a live tool
    timeline, inline approvals, telemetry, voice in and out, phone layout.

The earlier design put the loop in the web process (`friday/engine.py`). That is
the one thing this integration deliberately does NOT do: there is exactly one
agent loop, and it lives in `friday/agent/loop.py`. The server's job is to run
that loop on a worker thread, translate its callbacks into SSE frames, and block
the thread at a human decision without blocking the event loop.

Why a thread and not `async def`: `Agent.run` is synchronous by contract (the CLI
and the eval harness call it directly), and its LLM client is stdlib `urllib`.
Wrapping it in `asyncio.to_thread` is the smallest correct adapter; rewriting the
core async would buy nothing and break every existing caller.

SECURITY
--------
Same posture as the stdlib gateway, because the risk is identical:

  * a bearer token is required on every route except `/`, `/static/*` and
    `/api/health` (the page has to load before it can ask for the token);
  * the token is created 0600 in `config/serve.token`;
  * there is **no CORS wildcard** — the UI is same-origin, and a page on another
    origin must not be able to drive an agent that can read your memory;
  * a turn that arrives over HTTP is recorded with `scope="remote"`, so
    `friday audit --today` can answer "which turns came over the network?".
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__, paths
from ..agent.loop import Agent
from ..llm import MockClient, get_client
from . import adapters

STATIC = Path(__file__).parent / "static"
APPROVAL_TIMEOUT_S = 190.0          # same number the CLI uses: deny after 190s
MODEL_FALLBACK = "friday-local"

RISK = {
    "memory_search": "low", "memory_write": "low", "note": "low",
    "read_artifact": "low", "clock": "low", "math": "low",
    "file_read": "medium", "file_list": "medium", "file_grep": "medium",
    "file_edit": "high",
    "web_read": "medium", "wiki": "medium",
    "ws_list": "low", "ws_read": "medium", "ws_write": "high", "ws_shell": "high",
    "email_read": "medium", "email_send": "high",
    "calendar_read": "medium", "github_read": "medium",
}


# ── request models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=200_000)
    session_id: str | None = None
    run_id: str | None = None
    approval: str | None = None


class ApprovalDecision(BaseModel):
    approved: bool
    note: str = ""


class SessionCreate(BaseModel):
    title: str = "New session"


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MemoryCreate(BaseModel):
    text: str = Field(min_length=3, max_length=4000)


# ── runtime ────────────────────────────────────────────────────────────────────

@dataclass
class Runtime:
    db_path: Path
    token: str
    started_at: float = field(default_factory=time.time)
    client: Any = None
    registry: Any = None
    runs: dict[str, threading.Event] = field(default_factory=dict)
    #: token -> {"event": threading.Event, "approved": bool, "note": str, "run_id": str}
    approvals: dict[str, dict] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def new_conn(self) -> sqlite3.Connection:
        return adapters.connect(self.db_path)

    def agent(self, bridge: RunBridge) -> Agent:
        from ..retrieval.embedders import get_embedder
        from ..retrieval.rerankers import get_reranker

        # Embedders/rerankers are stateless and expensive to construct; build once.
        if not hasattr(self, "_embedder"):
            self._embedder = get_embedder()
        if not hasattr(self, "_reranker"):
            self._reranker = get_reranker()
        return Agent(
            self.new_conn(),
            client=self.client,
            registry=self.registry,
            embedder=self._embedder,
            reranker=self._reranker,
            confirm=bridge.confirm,
            on_event=bridge.on_event,
            confirm_all=bridge.confirm_all,
        )


class RunBridge:
    """Worker-thread ⇄ event-loop adapter for one turn.

    `on_event` and `confirm` are called by the agent on the worker thread.
    Everything that touches the asyncio queue goes through
    `loop.call_soon_threadsafe`; the approval wait uses a plain
    `threading.Event`, because blocking a worker thread is fine and blocking the
    event loop is not.
    """

    def __init__(self, loop, queue: asyncio.Queue, runtime: Runtime, run_id: str,
                 approval_mode: str):
        self.loop = loop
        self.queue = queue
        self.runtime = runtime
        self.run_id = run_id
        self.approval_mode = approval_mode
        #: "always" means ask even when policy says `allowed`; "never" means
        #: auto-approve `confirmation_required`. Policy DENIALS are never
        #: overridable by either mode — that is the whole point of Law 5.
        self.confirm_all = approval_mode == "always"
        self.cancelled = False

    # -- outbound ---------------------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        event.setdefault("run_id", self.run_id)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def on_token(self, text: str) -> None:
        if text:
            self.emit({"type": "token", "text": text})

    def on_event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "policy":
            if payload.get("verdict") == "denied":
                self.emit({"type": "notice",
                           "message": f"policy denied {payload.get('tool')} "
                                      f"({payload.get('sense') or 'no sense'})"})
        elif kind == "security":
            withheld = payload.get("withheld") or []
            if withheld:
                self.emit({"type": "notice",
                           "message": f"withheld {len(withheld)} retrieved item(s) that "
                                      f"read like instructions, not memory"})
        elif kind == "tool_call":
            self.emit({"type": "tool_call", **payload})
        elif kind == "tool_result":
            self.emit({"type": "tool_result", **payload})
        elif kind == "intent":
            self.emit({"type": "notice",
                       "message": f"intent: {payload.get('kind')} · memory "
                                  f"{'on' if payload.get('needs_memory') else 'off'}"})
        # ledger / retrieved / usage / memory_sync are diagnostics the HUD does
        # not render; dropping them keeps the wire small without lying about it.

    # -- inbound ----------------------------------------------------------------
    def confirm(self, name: str, arguments: dict) -> bool:
        if self.approval_mode == "never":
            return True
        token = secrets.token_urlsafe(9)
        gate = threading.Event()
        entry = {"event": gate, "approved": False, "note": "", "run_id": self.run_id}
        with self.runtime.lock:
            self.runtime.approvals[token] = entry
        self.emit({
            "type": "approval_required",
            "token": token,
            "name": name,
            "tool": name,
            "arguments": arguments,
            "risk": RISK.get(name, "high"),
            "timeout_s": int(APPROVAL_TIMEOUT_S),
        })
        decided = gate.wait(timeout=APPROVAL_TIMEOUT_S)
        with self.runtime.lock:
            self.runtime.approvals.pop(token, None)
        approved = bool(entry["approved"]) if decided else False
        note = entry["note"] or ("" if decided else "no decision within 190s — denied by default")
        self.emit({"type": "approval_result", "token": token,
                   "approved": approved, "note": note})
        return approved

    def close(self) -> None:
        """Deny any approval still waiting so a finished run cannot keep a thread."""
        with self.runtime.lock:
            for token, entry in list(self.runtime.approvals.items()):
                if entry["run_id"] == self.run_id:
                    entry["approved"] = False
                    entry["note"] = "run ended"
                    entry["event"].set()
                    self.runtime.approvals.pop(token, None)


# ── auth ───────────────────────────────────────────────────────────────────────

def resolve_token(root: Path | None = None) -> tuple[str, Path, bool]:
    """Explicit env var wins; otherwise the same 0600 file the gateway uses."""
    from ..serve import ensure_token

    env = os.environ.get("FRIDAY_WEB_TOKEN", "").strip()
    if env:
        return env, Path("<env:FRIDAY_WEB_TOKEN>"), False
    return ensure_token(root or paths.ROOT)


def require_auth(request: Request) -> None:
    expected = getattr(request.app.state.runtime, "token", "")
    supplied = (request.headers.get("x-friday-token")
                or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                or request.query_params.get("token"))
    if not expected or not supplied or not secrets.compare_digest(str(supplied), expected):
        raise HTTPException(status_code=401, detail="Missing or invalid FRIDAY token.")


# ── app factory ────────────────────────────────────────────────────────────────

def create_app(db_path: Path | str | None = None, *,
               client: Any = None, token: str | None = None) -> FastAPI:
    db_path = Path(db_path) if db_path else paths.DB_PATH
    if token is None:
        token, _token_path, _created = resolve_token()
    else:
        token = token or ""          # empty means "no auth", used only by tests
        _token_path, _created = Path("<injected>"), False

    conn = adapters.connect(db_path)
    conn.close()

    registry = None
    try:
        from ..agent.tools import build_registry

        registry = build_registry()
    except Exception:                     # a broken connector config must not kill the UI
        registry = None

    runtime = Runtime(
        db_path=db_path,
        token=token,
        client=client if client is not None else get_client("auto"),
        registry=registry,
    )

    app = FastAPI(
        title="FRIDAY",
        version=__version__,
        description="Local-first agent: streaming loop, tools, bi-temporal memory, "
                    "human-in-the-loop approvals.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.runtime = runtime
    app.state.token_path = _token_path
    app.state.token_created = _created

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ── surfaces ───────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def index():
        page = STATIC / "index.html"
        if not page.is_file():
            return JSONResponse({"detail": "HUD assets missing"}, status_code=500)
        return FileResponse(page, media_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "name": "FRIDAY",
            "version": __version__,
            "uptime_seconds": int(time.time() - runtime.started_at),
            "simulation": isinstance(runtime.client, MockClient),
            "auth_required": bool(runtime.token),
            "model": getattr(runtime.client, "name", "unknown"),
        }

    #: Alias kept so probes written for the stdlib gateway keep working.
    @app.get("/health", include_in_schema=False)
    async def health_alias():
        return await health()

    @app.get("/api/config")
    async def config():
        name = getattr(runtime.client, "name", "unknown")
        provider = "simulation" if isinstance(runtime.client, MockClient) else name.split(":")[0]
        model = getattr(runtime.client, "model", MODEL_FALLBACK)
        return {
            "provider": provider,
            "model": model,
            "base_url": os.environ.get("FRIDAY_LLM_URL", "http://127.0.0.1:8080/v1"),
            "workspace": str(paths.ROOT),
            "approval": "dangerous",
            "simulation": isinstance(runtime.client, MockClient),
            "auth_required": bool(runtime.token),
            "version": __version__,
            "plugins": [],
            "capabilities": {"streaming": True, "approvals": True,
                             "memory": True, "voice": True},
        }

    @app.get("/api/tools", dependencies=[Depends(require_auth)])
    async def tools():
        if runtime.registry is None:
            return []
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "risk": RISK.get(spec.name, "high"),
                "parameters": spec.parameters,
                "enabled": True,
            }
            for spec in runtime.registry.specs()
        ]

    @app.get("/api/system", dependencies=[Depends(require_auth)])
    async def system():
        conn = runtime.new_conn()
        try:
            counts = adapters.stats(conn)
        finally:
            conn.close()
        return {
            **adapters.telemetry(),
            "friday": {
                "version": __version__,
                "uptime_seconds": int(time.time() - runtime.started_at),
                "provider": "simulation" if isinstance(runtime.client, MockClient)
                            else getattr(runtime.client, "name", "unknown").split(":")[0],
                "model": getattr(runtime.client, "model", MODEL_FALLBACK),
                "simulation": isinstance(runtime.client, MockClient),
                "sessions": counts["sessions"],
                "memories": counts["memories"],
                "approval": "dangerous",
            },
        }

    # ── sessions ───────────────────────────────────────────────────────────────
    @app.get("/api/sessions", dependencies=[Depends(require_auth)])
    async def list_sessions():
        conn = runtime.new_conn()
        try:
            return adapters.list_sessions(conn)
        finally:
            conn.close()

    @app.post("/api/sessions", dependencies=[Depends(require_auth)])
    async def create_session(payload: SessionCreate):
        conn = runtime.new_conn()
        try:
            return adapters.create_session(conn, payload.title)
        finally:
            conn.close()

    @app.get("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def get_session(session_id: str):
        conn = runtime.new_conn()
        try:
            session = adapters.get_session(conn, session_id)
        finally:
            conn.close()
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.patch("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def rename_session(session_id: str, payload: SessionRename):
        conn = runtime.new_conn()
        try:
            if not adapters.rename_session(conn, session_id, payload.title):
                raise HTTPException(status_code=404, detail="Session not found")
            return adapters.get_session(conn, session_id)
        finally:
            conn.close()

    @app.delete("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def delete_session(session_id: str):
        conn = runtime.new_conn()
        try:
            if not adapters.delete_session(conn, session_id):
                raise HTTPException(status_code=404, detail="Session not found")
        finally:
            conn.close()
        return {"deleted": session_id}

    # ── memory ─────────────────────────────────────────────────────────────────
    @app.get("/api/memory", dependencies=[Depends(require_auth)])
    async def list_memory(q: str = Query(default="", max_length=300)):
        conn = runtime.new_conn()
        try:
            items = adapters.list_facts(conn, limit=200)
            if q:
                needle = q.lower()
                items = [i for i in items if needle in i["text"].lower()]
            return {"items": items, "stats": adapters.stats(conn)}
        finally:
            conn.close()

    @app.post("/api/memory", dependencies=[Depends(require_auth)])
    async def add_memory(payload: MemoryCreate):
        conn = runtime.new_conn()
        try:
            report = adapters.add_fact(conn, payload.text)
        finally:
            conn.close()
        if report.get("action") == "rejected":
            raise HTTPException(status_code=409, detail=report.get("message", "rejected"))
        return {"id": report.get("id"), "text": payload.text,
                "action": report.get("action")}

    @app.delete("/api/memory/{fact_id}", dependencies=[Depends(require_auth)])
    async def delete_memory(fact_id: str):
        conn = runtime.new_conn()
        try:
            if not adapters.forget_fact(conn, fact_id):
                raise HTTPException(status_code=404, detail="Memory not found or already retracted")
        finally:
            conn.close()
        return {"deleted": fact_id}

    @app.get("/api/history/search", dependencies=[Depends(require_auth)])
    async def search_history(q: str = Query(min_length=2, max_length=300)):
        conn = runtime.new_conn()
        try:
            return {"hits": adapters.search_history(conn, q)}
        finally:
            conn.close()

    # ── the run ────────────────────────────────────────────────────────────────
    @app.post("/api/chat", dependencies=[Depends(require_auth)])
    async def chat(payload: ChatRequest, request: Request):
        approval = payload.approval or "dangerous"
        if approval not in ("never", "dangerous", "always"):
            raise HTTPException(status_code=422,
                                detail="approval must be never, dangerous or always")
        conn = runtime.new_conn()
        try:
            session_id = payload.session_id or adapters.create_session(conn)["id"]
            if adapters.get_session(conn, session_id) is None:
                session_id = adapters.create_session(conn)["id"]
        finally:
            conn.close()

        run_id = payload.run_id or uuid.uuid4().hex[:12]
        cancel = threading.Event()
        runtime.runs[run_id] = cancel
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        bridge = RunBridge(loop, queue, runtime, run_id, approval)
        started = time.monotonic()

        def work() -> None:
            try:
                agent = runtime.agent(bridge)
                state = agent.run(
                    payload.message,
                    session_id=session_id,
                    scope="remote" if session_id else "interactive",
                    on_token=bridge.on_token,
                    should_stop=cancel.is_set,
                )
                bridge.emit({
                    "type": "done",
                    "session_id": state.session_id,
                    "steps": state.turn_idx,
                    "cancelled": bool(getattr(state, "cancelled", False)),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "usage": {"total_tokens": None},
                    "message": state.final_text,
                })
            except Exception as exc:                    # noqa: BLE001 - surface, never leak
                bridge.emit({"type": "error",
                             "message": f"{type(exc).__name__}: {exc}"})
                bridge.emit({"type": "done", "session_id": session_id,
                             "steps": 0, "duration_ms":
                             int((time.monotonic() - started) * 1000),
                             "usage": {}, "message": ""})
            finally:
                bridge.close()

        task = asyncio.create_task(asyncio.to_thread(work))

        async def stream():
            try:
                yield _sse({"type": "open", "run_id": run_id,
                            "session_id": session_id})
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=12.0)
                    except TimeoutError:
                        if await request.is_disconnected():
                            cancel.set()
                            break
                        yield ": ping\n\n"          # keeps proxies and phones awake
                        continue
                    yield _sse(event)
                    if event.get("type") in ("done", "error"):
                        break
                yield _sse({"type": "close", "run_id": run_id})
            finally:
                cancel.set()
                bridge.close()
                runtime.runs.pop(run_id, None)
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass          # the stream was cancelled; nothing left to deliver
                except Exception as exc:                     # noqa: BLE001
                    from ..util import swallowed

                    swallowed("friday/web/app.py:stream.cleanup", exc)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform",
                     "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chat/{run_id}/cancel", dependencies=[Depends(require_auth)])
    async def cancel_run(run_id: str):
        event = runtime.runs.get(run_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Run not found or already finished")
        event.set()
        bridge_deny_all(runtime, run_id)
        return {"cancelling": run_id}

    # ── approvals ──────────────────────────────────────────────────────────────
    @app.get("/api/approvals", dependencies=[Depends(require_auth)])
    async def pending_approvals():
        with runtime.lock:
            return {"pending": [t for t, e in runtime.approvals.items()
                                if not e["event"].is_set()]}

    @app.post("/api/approvals/{token}", dependencies=[Depends(require_auth)])
    async def decide_approval(token: str, payload: ApprovalDecision):
        with runtime.lock:
            entry = runtime.approvals.get(token)
            if entry is None:
                raise HTTPException(status_code=404,
                                    detail="No pending approval with that token")
            if entry["event"].is_set():
                raise HTTPException(status_code=409, detail="Approval already resolved")
            entry["approved"] = bool(payload.approved)
            entry["note"] = payload.note or ("approved" if payload.approved else "denied")
            entry["event"].set()
        return {"token": token, "approved": payload.approved}

    return app


def bridge_deny_all(runtime: Runtime, run_id: str) -> None:
    with runtime.lock:
        for entry in runtime.approvals.values():
            if entry["run_id"] == run_id and not entry["event"].is_set():
                entry["approved"] = False
                entry["note"] = "run cancelled"
                entry["event"].set()


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


# ── process entry point ────────────────────────────────────────────────────────

def resolve_host(host: str) -> str:
    """Same semantics as the stdlib gateway: loopback unless explicitly widened."""
    if host == "tailscale":
        try:
            import subprocess

            out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                                 text=True, timeout=4, check=False)
            addr = out.stdout.strip().splitlines()
            if addr:
                return addr[0].strip()
        except Exception as exc:                       # noqa: BLE001
            from ..util import swallowed

            swallowed("friday/web/app.py:resolve_host.tailscale", exc)
        print("[serve] tailscale not available; binding loopback", flush=True)
        return "127.0.0.1"
    return host


def run(host: str = "127.0.0.1", port: int = 8642, *, quiet: bool = False,
        token: str | None = None) -> int:
    import uvicorn

    if token:
        token_path, created = Path("<--token>"), False
    else:
        token, token_path, created = resolve_token()
    host = resolve_host(host)
    app = create_app(token=token)
    app.state.token_path = token_path
    app.state.token_created = created

    if not quiet:
        print(f"FRIDAY HUD  ->  http://{host}:{port}/", flush=True)
        print(f"token: {token_path}  ({'created' if created else 'existing'})", flush=True)
        if host == "0.0.0.0":
            print("[serve] WARNING: bound to 0.0.0.0 — anyone who can reach this port "
                  "and has the token can drive your agent.", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
