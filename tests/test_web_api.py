"""The HUD's REST surface, against a real SQLite store and the mock client.

These are integration tests on purpose: every one drives the same `create_app`
the CLI starts, so a route that only works under a test double would fail here.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from friday.llm import MockClient  # noqa: E402
from friday.web.app import create_app  # noqa: E402

TOKEN = "test-token"
HEAD = {"X-FRIDAY-Token": TOKEN}


@pytest.fixture
def client(root):
    app = create_app(root / "artifacts" / "friday.db", client=MockClient(), token=TOKEN)
    with TestClient(app) as c:
        yield c


def _events(response) -> list[dict]:
    out = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[5:].strip()))
    return out


# ── auth ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/api/sessions", "/api/memory", "/api/tools", "/api/system"])
def test_protected_routes_reject_a_missing_token(client, path):
    assert client.get(path).status_code == 401


def test_wrong_token_is_rejected(client):
    assert client.get("/api/sessions", headers={"X-FRIDAY-Token": "nope"}).status_code == 401


def test_bearer_scheme_is_accepted(client):
    r = client.get("/api/sessions", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_the_page_itself_needs_no_token(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "FRIDAY" in r.text


# ── configuration surfaces ────────────────────────────────────────────────────

def test_health_reports_simulation_without_a_key(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["simulation"] is True
    assert body["auth_required"] is True


def test_config_matches_the_hud_contract(client):
    body = client.get("/api/config").json()
    for key in ("provider", "model", "base_url", "workspace", "simulation",
                "approval", "auth_required", "version", "capabilities"):
        assert key in body
    assert body["capabilities"]["streaming"] is True


def test_tools_are_listed_with_a_risk(client):
    tools = client.get("/api/tools", headers=HEAD).json()
    names = {t["name"] for t in tools}
    assert {"memory_search", "memory_write", "note"} <= names
    assert all(t["risk"] in ("low", "medium", "high") for t in tools)


def test_system_reports_host_and_friday_blocks(client):
    body = client.get("/api/system", headers=HEAD).json()
    assert body["cpu_count"] >= 1
    assert "friday" in body and "sessions" in body["friday"]


# ── sessions ──────────────────────────────────────────────────────────────────

def test_session_crud_roundtrip(client):
    created = client.post("/api/sessions", headers=HEAD, json={"title": "Reactor"}).json()
    assert created["title"] == "Reactor"
    sid = created["id"]

    listed = client.get("/api/sessions", headers=HEAD).json()
    assert any(s["id"] == sid for s in listed)

    renamed = client.patch(f"/api/sessions/{sid}", headers=HEAD,
                           json={"title": "Reactor audit"}).json()
    assert renamed["title"] == "Reactor audit"

    assert client.delete(f"/api/sessions/{sid}", headers=HEAD).json() == {"deleted": sid}
    assert client.get(f"/api/sessions/{sid}", headers=HEAD).status_code == 404


def test_unknown_session_is_404(client):
    assert client.get("/api/sessions/nope", headers=HEAD).status_code == 404


# ── the run ───────────────────────────────────────────────────────────────────

def test_chat_streams_tokens_and_persists_both_halves(client):
    sid = client.post("/api/sessions", headers=HEAD, json={"title": "s"}).json()["id"]
    r = client.post("/api/chat", headers=HEAD,
                    json={"message": "hello", "session_id": sid, "run_id": "run-a"})
    assert r.status_code == 200
    events = _events(r)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "open"
    assert "token" in kinds
    assert kinds[-1] == "close"
    done = next(e for e in events if e["type"] == "done")
    assert done["cancelled"] is False

    detail = client.get(f"/api/sessions/{sid}", headers=HEAD).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"], roles
    assert detail["messages"][0]["content"] == "hello"


def test_chat_rejects_an_unknown_approval_mode(client):
    r = client.post("/api/chat", headers=HEAD,
                    json={"message": "hi", "approval": "sometimes"})
    assert r.status_code == 422


def test_cancel_of_unknown_run_is_404(client):
    assert client.post("/api/chat/nope/cancel", headers=HEAD).status_code == 404


def test_search_history_finds_the_turn(client):
    sid = client.post("/api/sessions", headers=HEAD, json={"title": "s"}).json()["id"]
    client.post("/api/chat", headers=HEAD,
                json={"message": "the reactor coolant is green", "session_id": sid})
    hits = client.get("/api/history/search?q=coolant", headers=HEAD).json()["hits"]
    assert any("coolant" in h["content"] for h in hits)


# ── memory ────────────────────────────────────────────────────────────────────

def test_memory_add_requires_auth_and_stores_a_fact(client):
    assert client.post("/api/memory", json={"text": "I like tea"}).status_code == 401
    r = client.post("/api/memory", headers=HEAD, json={"text": "I like tea"})
    assert r.status_code == 200, r.text
    body = client.get("/api/memory", headers=HEAD).json()
    assert any("tea" in item["text"] for item in body["items"])
    assert body["stats"]["memories"] >= 1


def test_memory_delete_retracts_rather_than_erasing_history(client):
    created = client.post("/api/memory", headers=HEAD,
                          json={"text": "my flight is on Tuesday"}).json()
    assert client.delete(f"/api/memory/{created['id']}", headers=HEAD).status_code == 200
    body = client.get("/api/memory", headers=HEAD).json()
    assert all(item["id"] != created["id"] for item in body["items"])
    # A retraction is still in the ledger: the stats count it.
    assert body["stats"]["retracted"] >= 1
    assert body["stats"]["facts"] >= 1

    assert client.delete(f"/api/memory/{created['id']}", headers=HEAD).status_code == 404
