"""Human-in-the-loop: a tool call must block until the browser answers.

The interesting assertions are not "the endpoint returns 200". They are:

  * the SSE stream carries `approval_required` *before* the tool runs;
  * the run does not proceed until `POST /api/approvals/{token}` resolves it;
  * a denial is visible to the model as a cancelled tool result, not a crash;
  * a policy denial is never overridable by `approval="never"`.

A fake client is used so the test controls exactly when a tool is called; the
bridge, the policy engine, the registry and the store are all real.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from friday.llm.client import Response, ToolCall  # noqa: E402
from friday.web.app import create_app  # noqa: E402

TOKEN = "t"
HEAD = {"X-FRIDAY-Token": TOKEN}


class ToolThenAnswer:
    """Calls `note` once, then answers. Records every request it saw."""

    name = "fake:tools"
    model = "fake"

    def __init__(self, tool: str = "note", args: dict | None = None):
        self.tool = tool
        self.args = args if args is not None else {"text": "a note"}
        self.requests: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=768, on_token=None):
        self.requests.append(messages)
        if len(self.requests) == 1:
            return Response(text="", model=self.model,
                            tool_calls=[ToolCall(self.tool, self.args, "call-1")])
        if on_token:
            on_token("Done.")
        return Response(text="Done.", model=self.model)


def _stream_in_thread(client, body) -> tuple[threading.Thread, dict]:
    box: dict = {}

    def go():
        r = client.post("/api/chat", headers=HEAD, json=body)
        box["events"] = [json.loads(line[5:].strip())
                         for line in r.text.splitlines() if line.startswith("data:")]

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    return thread, box


def _wait_for_approval(client, timeout=5.0) -> list[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = client.get("/api/approvals", headers=HEAD).json()["pending"]
        if pending:
            return pending
        time.sleep(0.02)
    return []


@pytest.fixture
def make_client(root):
    def build(client, token=TOKEN):
        app = create_app(root / "artifacts" / "friday.db", client=client, token=token)
        return TestClient(app)

    return build


def test_approval_blocks_the_run_until_answered(make_client):
    fake = ToolThenAnswer()
    client = make_client(fake)
    sid = client.post("/api/sessions", headers=HEAD, json={"title": "a"}).json()["id"]

    thread, box = _stream_in_thread(client, {
        "message": "note this", "session_id": sid, "run_id": "r1", "approval": "always"})
    pending = _wait_for_approval(client)
    assert pending, "the run reached the model but never asked for approval"

    # The tool must not have executed yet: the model has been called exactly once.
    assert len(fake.requests) == 1
    client.post(f"/api/approvals/{pending[0]}", headers=HEAD,
                json={"approved": True, "note": "ok"})
    thread.join(timeout=10)
    assert not thread.is_alive()

    kinds = [e["type"] for e in box["events"]]
    assert "approval_required" in kinds
    assert kinds.index("approval_required") < kinds.index("tool_call")
    assert kinds.index("tool_call") < kinds.index("tool_result")
    result = next(e for e in box["events"] if e["type"] == "tool_result")
    assert result["ok"] is True
    assert box["events"][-1]["type"] == "close"


def test_denial_is_a_cancelled_tool_result_not_a_crash(make_client):
    fake = ToolThenAnswer()
    client = make_client(fake)
    sid = client.post("/api/sessions", headers=HEAD, json={"title": "a"}).json()["id"]

    thread, box = _stream_in_thread(client, {
        "message": "note this", "session_id": sid, "run_id": "r2", "approval": "always"})
    pending = _wait_for_approval(client)
    assert pending
    client.post(f"/api/approvals/{pending[0]}", headers=HEAD,
                json={"approved": False, "note": "not now"})
    thread.join(timeout=10)

    result = next(e for e in box["events"] if e["type"] == "tool_result")
    assert result["ok"] is False
    approval = next(e for e in box["events"] if e["type"] == "approval_result")
    assert approval["approved"] is False
    assert box["events"][-1]["type"] == "close"


def test_approval_never_mode_does_not_bypass_a_policy_denial(make_client):
    """`never` suppresses the human prompt. It does NOT grant a permission.

    `email_send` needs a sense that does not exist in a fresh root, so policy
    denies it before confirmation is ever considered. The turn must still finish
    and the tool must not have run.
    """
    fake = ToolThenAnswer("email_send", {"to": "x@example.com", "subject": "s", "body": "b"})
    client = make_client(fake)
    sid = client.post("/api/sessions", headers=HEAD, json={"title": "a"}).json()["id"]

    r = client.post("/api/chat", headers=HEAD, json={
        "message": "send it", "session_id": sid, "run_id": "r3", "approval": "never"})
    events = [json.loads(line[5:].strip())
              for line in r.text.splitlines() if line.startswith("data:")]
    kinds = [e["type"] for e in events]
    assert "approval_required" not in kinds
    notices = " ".join(e.get("message", "") for e in events if e["type"] == "notice")
    assert "policy denied" in notices


def test_unknown_approval_token_is_404(make_client):
    client = make_client(ToolThenAnswer())
    assert client.post("/api/approvals/does-not-exist", headers=HEAD,
                       json={"approved": True}).status_code == 404
