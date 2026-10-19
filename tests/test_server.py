from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from friday.engine import ApprovalRequest
from friday.server.app import ServerApprover, create_app


@pytest.fixture
def app(settings):
    settings.approval = "never"
    application = create_app(settings)
    yield application
    application.state.runtime.memory.close()


@pytest.fixture
def client(app):
    return TestClient(app)


def sse_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_health_and_config(client):
    health = client.get("/api/health").json()
    assert health["status"] == "online"
    assert health["simulation"] is True

    config = client.get("/api/config").json()
    assert config["provider"] == "mock"
    assert "api_key" not in config
    assert config["capabilities"]["streaming"] is True


def test_ui_and_static_assets_are_served(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "F.R.I.D.A.Y." in index.text

    assert client.get("/static/style.css").status_code == 200
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "consumeSSE" in script.text


def test_tool_surface(client):
    tools = client.get("/api/tools").json()
    names = {tool["name"] for tool in tools}
    assert "shell" in names
    assert all("risk" in tool for tool in tools)


def test_chat_streams_events_and_persists(client):
    with client.stream("POST", "/api/chat", json={"message": "what is 21 * 2"}) as response:
        assert response.status_code == 200
        events = sse_events(response)

    types = [event["type"] for event in events]
    assert types[0] == "open"
    assert "tool_result" in types
    assert types[-1] == "close"
    session_id = events[0]["session_id"]
    assert session_id

    history = client.get(f"/api/sessions/{session_id}").json()
    assert history["messages"][0]["content"] == "what is 21 * 2"


def test_session_crud(client):
    created = client.post("/api/sessions", json={"title": "Test session"}).json()
    session_id = created["id"]
    assert created["title"] == "Test session"

    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"}).json()
    assert renamed["title"] == "Renamed"

    assert client.delete(f"/api/sessions/{session_id}").json()["deleted"] == session_id
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_memory_api(client):
    created = client.post("/api/memory", json={"text": "The lab code is 1973", "tags": "security"}).json()
    items = client.get("/api/memory").json()
    assert any(item["text"] == "The lab code is 1973" for item in items["items"])

    assert client.delete(f"/api/memory/{created['id']}").status_code == 200
    assert client.delete(f"/api/memory/{created['id']}").status_code == 404


def test_history_search(client):
    response = client.post("/api/chat", json={"message": "the arc reactor needs a tuning pass"})
    assert response.status_code == 200
    hits = client.get("/api/history/search", params={"q": "reactor"}).json()["hits"]
    assert hits
    assert hits[0]["session_id"]


def test_missing_session_and_bad_approval(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/approvals/nope", json={"approved": True}).status_code == 404
    response = client.post("/api/chat", json={"message": "hi", "approval": "whatever"})
    assert response.status_code in (400, 422)


def test_system_endpoint(client):
    data = client.get("/api/system").json()
    assert data["friday"]["simulation"] is True
    assert "memory" in data
    assert "cpu_count" in data


def test_api_token_is_enforced_when_configured(app, monkeypatch):
    monkeypatch.setenv("FRIDAY_API_TOKEN", "stark-industries")
    with TestClient(app) as scoped:
        assert scoped.post("/api/chat", json={"message": "hello"}).status_code == 401
        assert scoped.get("/api/health").status_code == 200
        response = scoped.post(
            "/api/chat", json={"message": "hello"}, headers={"X-FRIDAY-Token": "stark-industries"}
        )
        assert response.status_code == 200


def test_server_approver_resolves_from_the_browser(app):
    runtime = app.state.runtime
    approver = ServerApprover(runtime, "run-1", timeout=5)
    request = ApprovalRequest("tok123", "shell", {"command": "ls"}, "shell(ls)", "high", "sess")

    async def scenario():
        task = asyncio.create_task(approver.decide(request))
        await asyncio.sleep(0.02)
        future, run_id = runtime.approvals["tok123"]
        assert run_id == "run-1"
        future.set_result((True, "approved in test"))
        return await task

    approved, note = asyncio.run(scenario())
    assert approved is True
    assert note == "approved in test"
    assert "tok123" not in runtime.approvals


def test_server_approver_times_out_to_denial(app):
    approver = ServerApprover(app.state.runtime, "run-2", timeout=0.02)
    request = ApprovalRequest("tok456", "shell", {}, "shell()", "high", "sess")
    approved, note = asyncio.run(approver.decide(request))
    assert approved is False
    assert "timed out" in note or "no decision" in note


def test_chat_rejects_empty_message(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
