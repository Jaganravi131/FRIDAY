"""The Anthropic client: conversion, tool blocks, SSE accumulation, selection.

No network. The streaming test feeds the exact event sequence the API doc
specifies, including an `input_json_delta` split mid-token, which is the case
that breaks naive parsers.
"""

from __future__ import annotations

import json

from friday.llm.anthropic import AnthropicClient
from friday.llm.client import MockClient, get_client

# ── conversion ────────────────────────────────────────────────────────────────

def test_conversion_pulls_system_out_and_keeps_a_user_first():
    system, messages = AnthropicClient.convert([
        {"role": "system", "content": "You are FRIDAY."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert system == "You are FRIDAY."
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_conversion_pairs_tool_results_with_synthesised_tool_use():
    _, messages = AnthropicClient.convert([
        {"role": "user", "content": "what is my rent"},
        {"role": "tool", "name": "memory_search", "content": "[memory_search] rent=1200"},
        {"role": "tool", "name": "clock", "content": "[clock] 12:00"},
    ])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant, results = messages[1], messages[2]
    assert [b["type"] for b in assistant["content"]] == ["tool_use", "tool_use"]
    assert [b["type"] for b in results["content"]] == ["tool_result", "tool_result"]
    ids = [b["id"] for b in assistant["content"]]
    assert [b["tool_use_id"] for b in results["content"]] == ids


def test_conversion_inserts_a_user_turn_when_the_log_starts_with_tool_output():
    _, messages = AnthropicClient.convert([
        {"role": "tool", "name": "note", "content": "x"},
    ])
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "(session opened)"


def test_tool_schemas_are_converted_from_openai_shape():
    schemas = AnthropicClient.tool_schema([{
        "type": "function",
        "function": {"name": "note", "description": "d",
                     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}},
    }])
    assert schemas == [{
        "name": "note", "description": "d",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
    }]


# ── responses ─────────────────────────────────────────────────────────────────

def test_non_streaming_response_maps_text_and_tool_use():
    client = AnthropicClient(api_key="k")
    resp = client._parse_response({
        "model": "claude-x",
        "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "memory_search",
             "input": {"query": "rent"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    assert resp.text == "Let me check."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "memory_search"
    assert resp.tool_calls[0].arguments == {"query": "rent"}
    assert resp.tool_calls[0].id == "tu_1"
    assert resp.prompt_tokens == 10 and resp.completion_tokens == 5


def _fake_stream(events: list[dict]):
    return [f"data: {json.dumps(e)}\n".encode() for e in events]


def test_streaming_accumulates_text_and_split_json_tool_input():
    client = AnthropicClient(api_key="k")
    seen: list[str] = []
    events = [
        {"type": "message_start", "message": {"model": "claude-x",
                                              "usage": {"input_tokens": 7}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Checking"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": " now."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "tu_2", "name": "math"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"expr'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": 'ession":"2+2"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"usage": {"output_tokens": 9}}},
        {"type": "message_stop"},
    ]
    resp = client._read_stream(_fake_stream(events), seen.append)

    assert "".join(seen) == "Checking now."
    assert resp.text == "Checking now."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == {"expression": "2+2"}
    assert resp.tool_calls[0].id == "tu_2"
    assert resp.prompt_tokens == 7 and resp.completion_tokens == 9


# ── selection ────────────────────────────────────────────────────────────────

def test_anthropic_is_selected_when_the_key_is_present(monkeypatch):
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = get_client("auto")
    assert isinstance(client, AnthropicClient)
    assert client.alive() is True


def test_anthropic_without_a_key_falls_back_to_the_simulation_core(monkeypatch, capsys):
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = get_client("auto")
    assert isinstance(client, MockClient)
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_http_provider_still_wins_when_asked(monkeypatch):
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "http")
    monkeypatch.delenv("FRIDAY_LLM_URL", raising=False)
    client = get_client("auto")
    # No server is running in the test sandbox, so this is the real client in
    # "unreachable" mode — selected explicitly, never silently mocked.
    assert not isinstance(client, AnthropicClient)
    assert client.name.startswith("openai:")
