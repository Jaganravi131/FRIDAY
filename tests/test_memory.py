from __future__ import annotations

import pytest

from friday.memory import MemoryStore, _fts_query
from friday.types import Message


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    yield store
    store.close()


def test_session_lifecycle(memory):
    session_id = memory.create_session("First contact")
    assert memory.get_session(session_id)["title"] == "First contact"

    memory.rename_session(session_id, "Renamed")
    assert memory.get_session(session_id)["title"] == "Renamed"

    assert [item["id"] for item in memory.list_sessions()] == [session_id]
    assert memory.delete_session(session_id) is True
    assert memory.get_session(session_id) is None


def test_messages_keep_chronological_order(memory):
    session_id = memory.create_session()
    for index in range(5):
        memory.add_message(session_id, Message(role="user", content=f"message {index}"))

    recent = memory.get_messages(session_id, limit=2)
    assert [m.content for m in recent] == ["message 3", "message 4"]

    everything = memory.get_messages(session_id)
    assert [m.content for m in everything][:2] == ["message 0", "message 1"]


def test_tool_calls_round_trip(memory):
    session_id = memory.create_session()
    from friday.types import ToolCall

    message = Message(
        role="assistant",
        content="working",
        tool_calls=[ToolCall(id="call_1", name="clock", arguments='{"timezone": "UTC"}')],
    )
    memory.add_message(session_id, message)
    stored = memory.get_messages(session_id)[0]
    assert stored.tool_calls[0].name == "clock"
    assert stored.tool_calls[0].parsed_arguments() == {"timezone": "UTC"}


def test_full_text_search_finds_needles(memory):
    session_id = memory.create_session("Searchable")
    memory.add_message(session_id, Message(role="user", content="the reactor needs a new palladium core"))
    memory.add_message(session_id, Message(role="assistant", content="unrelated chatter about lunch"))

    hits = memory.search_messages("palladium")
    assert len(hits) == 1
    assert hits[0]["session_id"] == session_id
    assert "palladium" in hits[0]["snippet"].lower()


def test_memory_store_and_recall(memory):
    memory.remember("The deploy key lives in KMS", tags="security")
    memory.remember("The project codename is FRIDAY", tags="project")

    assert memory.stats()["memories"] == 2
    assert "KMS" in memory.memory_block()
    hits = memory.recall("deploy")
    assert any("KMS" in item["text"] for item in hits)

    assert memory.forget(hits[0]["id"]) is True
    assert memory.stats()["memories"] == 1


def test_empty_memory_is_rejected(memory):
    with pytest.raises(ValueError):
        memory.remember("   ")


def test_fts_query_is_injection_safe():
    assert _fts_query('he said "drop table" --') == '"he" OR "said" OR "drop" OR "table"'
    assert _fts_query("!!!") == '""'


def test_delete_session_cascades(memory):
    session_id = memory.create_session()
    memory.add_message(session_id, Message(role="user", content="gone soon"))
    memory.delete_session(session_id)
    assert memory.get_messages(session_id) == []
