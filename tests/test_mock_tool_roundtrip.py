"""A tool observation must not replace the human's question in the offline client."""
import pytest

from friday.agent.loop import Agent
from friday.agent.tools import build_registry
from friday.llm.client import MockClient


@pytest.mark.parametrize("question,predicate,value", [
    ("what is my monthly rent", "lease_amount_monthly", "28000 INR"),
    ("who is my manager", "manager", "Priya Raman"),
])
@pytest.mark.parametrize("role,prefix", [("user", "[tool result] "), ("tool", "")])
def test_tool_result_cannot_redirect_the_mock_question(question, predicate, value, role, prefix):
    client = MockClient()
    tools = build_registry().schemas()
    messages = [
        {"role": "system", "content": f"<recalled>\n- {predicate}: {value}\n"
         "- lease_renewal_due: 2026-10-15\n</recalled>"},
        {"role": "user", "content": question},
    ]
    first = client.chat(messages, tools=tools)
    assert first.tool_calls and first.tool_calls[0].name == "memory_search"
    # Its vocabulary must not be scored as the user's query. An observation can
    # legitimately mention a neighbouring predicate even when the question didn't.
    messages.append({"role": role, "content": prefix + '[memory_search] '
                     '{"predicate":"lease_renewal_due","object":"2026-10-15"}'})
    reply = client.chat(messages, tools=tools)
    assert not reply.tool_calls
    assert value in reply.text
    assert "2026-10-15" not in reply.text


def test_memory_write_is_not_repeated_after_its_result():
    client = MockClient()
    tools = build_registry().schemas()
    messages = [{"role": "user", "content": "my rent is 25000 INR"}]
    assert client.chat(messages, tools=tools).tool_calls[0].name == "memory_write"
    messages.append({"role": "user", "content": "[tool result] [memory_write] recorded"})
    assert not client.chat(messages, tools=tools).tool_calls
    # A genuinely new human turn must still be able to make a tool call.
    messages.append({"role": "user", "content": "what is my monthly rent"})
    assert client.chat(messages, tools=tools).tool_calls[0].name == "memory_search"


def test_seeded_monthly_rent_survives_the_real_tool_roundtrip(seeded):
    conn, _ = seeded
    st = Agent(conn, client=MockClient()).run("what is my monthly rent")
    assert "28000" in st.final_text.replace(",", "")
    assert len(st.tool_log) == 1
    assert st.done
