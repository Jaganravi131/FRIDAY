"""FastAPI application: REST for state, SSE for the live agent stream."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from friday import __version__
from friday.config import Settings
from friday.engine import Agent, ApprovalRequest, PolicyApprover
from friday.llm import ProviderError, build_provider
from friday.memory import MemoryStore
from friday.tools import build_registry
from friday.tools.system import telemetry


# --------------------------------------------------------------------- models
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=200_000)
    session_id: str | None = None
    run_id: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=16, le=32_000)
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
    tags: str = ""


# -------------------------------------------------------------------- runtime
@dataclass
class Runtime:
    settings: Settings
    memory: MemoryStore
    registry: Any
    provider: Any
    runs: dict[str, asyncio.Event] = field(default_factory=dict)
    approvals: dict[str, tuple[asyncio.Future, str]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    plugin_names: list[str] = field(default_factory=list)

    def agent_for(self, approval_mode: str | None, run_id: str) -> Agent:
        """A request may tighten the approval policy, never loosen it below the server's."""
        rank = {"never": 0, "dangerous": 1, "always": 2}
        settings = self.settings
        if approval_mode:
            if approval_mode not in rank:
                raise HTTPException(status_code=422, detail="approval must be never, dangerous or always")
            effective = approval_mode if rank[approval_mode] > rank[settings.approval] else settings.approval
            if effective != settings.approval:
                settings = settings.model_copy(update={"approval": effective})
        approver = ServerApprover(self, run_id) if settings.approval != "never" else PolicyApprover(True)
        return Agent(settings, self.provider, self.registry, self.memory, approver=approver)


class ServerApprover:
    """Blocks the agent loop until a browser POSTs a decision (or the timeout fires)."""

    def __init__(self, runtime: Runtime, run_id: str, timeout: float = 190.0):
        self.runtime = runtime
        self.run_id = run_id
        self.timeout = timeout

    async def decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.runtime.approvals[request.token] = (future, self.run_id)
        try:
            approved, note = await asyncio.wait_for(future, timeout=self.timeout)
            return approved, note or ("approved" if approved else "denied by user")
        except TimeoutError:
            return False, "no decision within 190s — denied by default"
        finally:
            self.runtime.approvals.pop(request.token, None)


# ---------------------------------------------------------------- auth (opt-in)
def _auth_token() -> str:
    return os.environ.get("FRIDAY_API_TOKEN", "").strip()


def require_auth(request: Request) -> None:
    token = _auth_token()
    if not token:
        return
    supplied = request.headers.get("x-friday-token") or request.query_params.get("token")
    if supplied != token:
        raise HTTPException(status_code=401, detail="Missing or invalid FRIDAY API token.")


# ------------------------------------------------------------------ app factory
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    settings.ensure_dirs()

    runtime = Runtime(
        settings=settings,
        memory=MemoryStore(settings.db_file),
        registry=None,
        provider=None,
    )
    runtime.registry, runtime.plugin_names = build_registry(settings)
    runtime.provider = build_provider(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime
        yield
        for future, _run in runtime.approvals.values():
            if not future.done():
                future.set_result((False, "server shutting down"))
        runtime.memory.close()

    app = FastAPI(
        title="FRIDAY",
        version=__version__,
        description="Local-first AI agent with tools, memory, approvals and a HUD.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    static_dir = settings.static_dir
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # --------------------------------------------------------------- surfaces
    @app.get("/", include_in_schema=False)
    async def index():
        file = static_dir / "index.html"
        if not file.is_file():
            return JSONResponse({"detail": "UI not built"}, status_code=500)
        return FileResponse(file, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        return {
            "status": "online",
            "name": "FRIDAY",
            "version": __version__,
            "uptime_seconds": int(time.time() - runtime.started_at),
            "simulation": settings.is_simulation,
            "auth_required": bool(_auth_token()),
        }

    @app.get("/api/config")
    async def config():
        return {
            **settings.public_dict(),
            "version": __version__,
            "auth_required": bool(_auth_token()),
            "plugins": runtime.plugin_names,
            "capabilities": {"streaming": True, "approvals": True, "memory": True, "voice": True},
        }

    @app.get("/api/tools")
    async def tools():
        disabled = set(settings.disabled_tools)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "risk": tool.risk,
                "parameters": tool.parameters,
                "enabled": tool.name not in disabled,
            }
            for tool in runtime.registry.all()
        ]

    @app.get("/api/system")
    async def system():
        return {
            **telemetry(),
            "friday": {
                "version": __version__,
                "uptime_seconds": int(time.time() - runtime.started_at),
                "provider": settings.provider,
                "model": settings.model,
                "simulation": settings.is_simulation,
                "sessions": runtime.memory.stats()["sessions"],
                "memories": runtime.memory.stats()["memories"],
                "approval": settings.approval,
            },
        }

    # --------------------------------------------------------------- sessions
    @app.get("/api/sessions", dependencies=[Depends(require_auth)])
    async def list_sessions():
        return runtime.memory.list_sessions()

    @app.post("/api/sessions")
    async def create_session(payload: SessionCreate):
        session_id = runtime.memory.create_session(payload.title)
        return runtime.memory.get_session(session_id)

    @app.get("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def get_session(session_id: str):
        session = runtime.memory.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = [
            {
                **message.to_dict(),
                "created_at": message.meta.get("created_at"),
            }
            for message in runtime.memory.get_messages(session_id, limit=500)
            if message.role != "system"
        ]
        return {**session, "messages": messages}

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, payload: SessionRename):
        if not runtime.memory.rename_session(session_id, payload.title):
            raise HTTPException(status_code=404, detail="Session not found")
        return runtime.memory.get_session(session_id)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        if not runtime.memory.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    # ------------------------------------------------------------- memory/facts
    @app.get("/api/memory", dependencies=[Depends(require_auth)])
    async def list_memory(q: str = Query(default="", max_length=300)):
        return {"items": runtime.memory.recall(q, limit=50), "stats": runtime.memory.stats()}

    @app.post("/api/memory", dependencies=[Depends(require_auth)])
    async def add_memory(payload: MemoryCreate):
        memory_id = runtime.memory.remember(payload.text, payload.tags)
        return {"id": memory_id, "text": payload.text}

    @app.delete("/api/memory/{memory_id}", dependencies=[Depends(require_auth)])
    async def delete_memory(memory_id: int):
        if not runtime.memory.forget(memory_id):
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"deleted": memory_id}

    @app.get("/api/history/search", dependencies=[Depends(require_auth)])
    async def search_history(q: str = Query(min_length=2, max_length=300)):
        return {"hits": runtime.memory.search_messages(q, limit=25)}

    # -------------------------------------------------------------- the agent
    @app.post("/api/chat", dependencies=[Depends(require_auth)])
    async def chat(payload: ChatRequest, request: Request):
        session_id = runtime.memory.ensure_session(payload.session_id, title="New session")
        run_id = payload.run_id or uuid.uuid4().hex[:12]
        cancel = asyncio.Event()
        runtime.runs[run_id] = cancel

        queue: asyncio.Queue = asyncio.Queue()
        agent = runtime.agent_for(payload.approval, run_id)

        async def pump() -> None:
            try:
                async for event in agent.run(
                    session_id,
                    payload.message,
                    cancel=cancel,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                ):
                    await queue.put({**event, "run_id": run_id, "session_id": session_id})
            except ProviderError as exc:
                await queue.put(
                    {"type": "error", "message": str(exc), "run_id": run_id, "session_id": session_id}
                )
            except asyncio.CancelledError:
                await queue.put({"type": "cancelled", "run_id": run_id, "session_id": session_id})
                raise
            except Exception as exc:  # noqa: BLE001 - surface, never leak a dead stream
                await queue.put(
                    {
                        "type": "error",
                        "message": f"Agent crashed: {type(exc).__name__}: {exc}",
                        "run_id": run_id,
                        "session_id": session_id,
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(pump())

        async def stream():
            try:
                yield _sse({"type": "open", "run_id": run_id, "session_id": session_id})
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=12.0)
                    except TimeoutError:
                        if await request.is_disconnected():
                            cancel.set()
                            break
                        yield ": ping\n\n"
                        continue
                    if item is None:
                        break
                    yield _sse(item)
                    if await request.is_disconnected():
                        cancel.set()
                        break
                yield _sse({"type": "close", "run_id": run_id})
            finally:
                cancel.set()
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 - cleanup must not mask the response
                    pass
                runtime.runs.pop(run_id, None)
                for token, (future, owner) in list(runtime.approvals.items()):
                    if owner != run_id:
                        continue
                    if not future.done():
                        future.set_result((False, "run ended"))
                    runtime.approvals.pop(token, None)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/chat/{run_id}/cancel", dependencies=[Depends(require_auth)])
    async def cancel_run(run_id: str):
        event = runtime.runs.get(run_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Run not found or already finished")
        event.set()
        return {"cancelling": run_id}

    @app.post("/api/approvals/{token}", dependencies=[Depends(require_auth)])
    async def decide_approval(token: str, payload: ApprovalDecision):
        entry = runtime.approvals.get(token)
        if entry is None:
            raise HTTPException(status_code=404, detail="No pending approval with that token")
        future, _run_id = entry
        if future.done():
            raise HTTPException(status_code=409, detail="Approval already resolved")
        future.set_result((payload.approved, payload.note))
        return {"token": token, "approved": payload.approved}

    @app.get("/api/approvals", dependencies=[Depends(require_auth)])
    async def pending_approvals():
        return {"pending": [token for token, (f, _run) in runtime.approvals.items() if not f.done()]}

    return app


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


app = create_app()


def run() -> None:
    import uvicorn

    settings = Settings.load()
    uvicorn.run(
        "friday.server.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


__all__ = ["app", "create_app", "run"]
