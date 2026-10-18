"""Malformed remote input must not crash a handler or invoke the agent."""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from friday import serve


@pytest.fixture
def gateway(monkeypatch):
    class Handler(serve.Handler):
        token = "test-only-token"

    answer = Mock(return_value={"answer": "ok", "model": serve.MODEL_NAME})
    why = Mock(return_value={"report": "ok"})
    monkeypatch.setattr(serve, "answer", answer)
    monkeypatch.setattr(serve, "why", why)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()

    def request(body, *, path="/api/ask", headers=None):
        conn = http.client.HTTPConnection(*server.server_address, timeout=3)
        try:
            conn.request("POST", path, body=body,
                         headers=headers if headers is not None else
                         {"Authorization": "Bearer test-only-token"})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    yield request, Handler, answer, why
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


@pytest.mark.parametrize("payload", [None, [], [1], "text", 1, True])
@pytest.mark.parametrize("path", ["/api/ask", "/api/why", "/v1/chat/completions"])
def test_non_object_json_returns_400_without_running_tools(gateway, payload, path):
    request, _, answer, why = gateway
    status, body = request(json.dumps(payload), path=path)
    assert status == 400 and body.get("error")
    answer.assert_not_called()
    why.assert_not_called()


@pytest.mark.parametrize("messages", [
    None, [], "text", 1, {}, [None], [1], ["text"], [{}],
    [{"role": "assistant", "content": "not a user turn"}],
    [{"role": "user", "content": None}],
    [{"role": "user", "content": "  "}],
    [{"role": "user", "content": {"text": "not supported"}}],
    [{"role": "user", "content": [{"type": "text", "text": "not supported"}]}],
])
def test_bad_chat_messages_return_400_without_agent_call(gateway, messages):
    request, _, answer, _ = gateway
    status, body = request(json.dumps({"messages": messages}), path="/v1/chat/completions")
    assert status == 400 and body.get("error")
    answer.assert_not_called()


@pytest.mark.parametrize("path,key", [("/api/ask", "message"), ("/api/why", "question")])
@pytest.mark.parametrize("value", [None, "  ", 123, {"text": "hello"}, ["hello"]])
def test_text_fields_are_not_coerced_to_strings(gateway, path, key, value):
    request, _, answer, why = gateway
    status, body = request(json.dumps({key: value}), path=path)
    assert status == 400 and body.get("error")
    answer.assert_not_called()
    why.assert_not_called()


@pytest.mark.parametrize("headers", [{"Authorization": "Bearer café"}, {"x-api-key": "café"}])
def test_non_ascii_credentials_are_rejected_without_exception(gateway, headers):
    request, _, answer, _ = gateway
    status, body = request('{"message":"hello"}', headers=headers)
    assert status == 401 and body.get("error")
    answer.assert_not_called()


def test_an_empty_server_token_never_authorizes_an_anonymous_request(gateway):
    request, handler, answer, _ = gateway
    handler.token = ""
    status, body = request('{"message":"hello"}', headers={})
    assert status == 401 and body.get("error")
    answer.assert_not_called()


@pytest.mark.parametrize("payload", [b'{', b'{"message":"\xff"}'])
def test_invalid_json_or_utf8_is_rejected(gateway, payload):
    request, _, answer, _ = gateway
    status, body = request(payload)
    assert status == 400 and body.get("error")
    answer.assert_not_called()


def test_oversized_body_is_rejected_before_reading_it(gateway):
    request, _, answer, _ = gateway
    status, body = request(b"", headers={"Authorization": "Bearer test-only-token",
                                        "Content-Length": "1000001"})
    assert status == 413 and body.get("error")
    answer.assert_not_called()


@pytest.mark.parametrize("length", ["0", "-1", "not-a-number"])
def test_invalid_lengths_are_rejected_without_waiting_for_a_body(gateway, length):
    request, _, answer, _ = gateway
    status, body = request(b"", headers={"Authorization": "Bearer test-only-token",
                                        "Content-Length": length})
    assert status == 400 and body.get("error")
    answer.assert_not_called()


def test_unsupported_transfer_encoding_is_rejected(gateway):
    request, _, answer, _ = gateway
    status, body = request(b"", headers={"Authorization": "Bearer test-only-token",
                                        "Transfer-Encoding": "chunked"})
    assert status == 400 and body.get("error")
    answer.assert_not_called()


def test_oversized_json_integer_is_a_client_error(gateway):
    request, _, answer, _ = gateway
    # Python 3.11+ can raise ValueError (not JSONDecodeError) for digit limits.
    status, body = request('{"message":' + '9' * 10000 + '}')
    assert status == 400 and body.get("error")
    answer.assert_not_called()


def test_deeply_nested_json_is_a_client_error(gateway):
    request, _, answer, _ = gateway
    status, body = request('[' * 2000 + '0' + ']' * 2000)
    assert status == 400 and body.get("error")
    answer.assert_not_called()


@pytest.mark.parametrize("headers", [{"Authorization": "Bearer test-only-token"},
                                     {"x-api-key": "test-only-token"}])
def test_valid_chat_uses_last_user_message(gateway, headers):
    request, _, answer, _ = gateway
    payload = {"messages": [{"role": "system", "content": "ignored"},
                            {"role": "user", "content": "earlier"},
                            {"role": "user", "content": " hello "},
                            {"role": "assistant", "content": "ignored"}]}
    status, body = request(json.dumps(payload), path="/v1/chat/completions", headers=headers)
    assert status == 200 and body["choices"][0]["message"]["content"] == "ok"
    answer.assert_called_once_with("hello")


def test_valid_why_still_works(gateway):
    request, _, answer, why = gateway
    status, body = request('{"question":" my rent "}', path="/api/why")
    assert status == 200 and body["report"] == "ok"
    why.assert_called_once_with("my rent")
    answer.assert_not_called()


@pytest.mark.parametrize("contents", ["", " \n"])
def test_empty_persisted_token_is_replaced(root, contents):
    token_path = root / "config" / "serve.token"
    token_path.write_text(contents, encoding="utf-8")
    token, path, created = serve.ensure_token(root)
    assert created and len(token) >= 32
    assert path.read_text(encoding="utf-8").strip() == token
    assert serve.ensure_token(root) == (token, path, False)
