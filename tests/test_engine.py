from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from friday.config import Settings
from friday.engine import Agent, ApprovalRequest, PolicyApprover
from friday.llm import LLMProvider, ProviderError, build_provider
from friday.llm.base import FinishDelta, TextDelta, ToolCallDelta
from friday.memory import MemoryStore
from friday.tools import build_registry
from friday.types import Message, ToolSchema


def build_agent(settings: Settings, approver=None) -> tuple[Agent, MemoryStore]:
    settings.ensure_dirs()
    memory = MemoryStore(settings.db_file)
    registry, _plugins = build_registry(settings)
    agent = Agent(settings, build_provider(settings), registry, memory, approver=approver)
    return agent, memory


async def collect(agent: Agent, session_id: str, text: str, cancel=None) -> list[dict]:
    return [event async for event in agent.run(session_id, text, cancel=cancel)]


def kinds(events):
    return [event["type"] for event in events]


def answer_text(events) -> str:
    return "".join(event["text"] for event in events if event["type"] == "token")


def test_math_turn_runs_the_real_tool(settings):
    settings.approval = "never"
    agent, memory = build_agent(settings)
    session_id = memory.create_session()

    events = asyncio.run(collect(agent, session_id, "what is 17 * 42 + 9"))
    memory.close()

    assert kinds(events)[0] == "start"
    assert "tool_call" in kinds(events)
    assert "tool_result" in kinds(events)
    result = next(event for event in events if event["type"] == "tool_result")
    assert result["ok"] is True
    assert "723" in result["output"]

    assert "723" in answer_text(events)
    done = events[-1]
    assert done["type"] == "done"
    assert done["stop_reason"] == "stop"
    assert done["usage"]["total_tokens"] > 0


def test_transcript_and_title_are_persisted(settings):
    settings.approval = "never"
    agent, memory = build_agent(settings)
    session_id = memory.create_session()

    asyncio.run(collect(agent, session_id, "what is 2 + 2"))
    history = memory.get_messages(session_id)
    roles = [message.role for message in history]

    assert roles[0] == "user"
    assert "assistant" in roles
    assert "tool" in roles
    assert memory.get_session(session_id)["title"].startswith("what is 2 + 2")
    memory.close()


class RecordingApprover:
    def __init__(self, allow: bool):
        self.allow = allow
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        self.requests.append(request)
        return self.allow, "test decision"


def test_dangerous_tool_is_gated_and_can_be_approved(settings):
    settings.approval = "dangerous"
    approver = RecordingApprover(True)
    agent, memory = build_agent(settings, approver)
    session_id = memory.create_session()

    events = asyncio.run(collect(agent, session_id, "run this python code: print(6*7)"))
    memory.close()

    assert len(approver.requests) == 1
    assert approver.requests[0].tool == "python_exec"
    assert approver.requests[0].risk == "high"
    assert "approval_required" in kinds(events)
    assert "approval_result" in kinds(events)
    result = next(event for event in events if event["type"] == "tool_result")
    assert result["ok"] is True
    assert "42" in result["output"]


def test_denied_tool_never_executes(settings):
    settings.approval = "dangerous"
    approver = RecordingApprover(False)
    agent, memory = build_agent(settings, approver)
    session_id = memory.create_session()

    events = asyncio.run(collect(agent, session_id, "run this python code: print('must not run')"))
    memory.close()

    result = next(event for event in events if event["type"] == "tool_result")
    assert result["ok"] is False
    assert "denied" in result["output"].lower()
    assert "must not run" not in result["output"]
    assert all(event["type"] != "approval_result" or not event["approved"] for event in events)


def test_low_risk_tools_bypass_the_gate(settings):
    settings.approval = "dangerous"
    approver = RecordingApprover(False)
    agent, memory = build_agent(settings, approver)
    session_id = memory.create_session()

    asyncio.run(collect(agent, session_id, "what is 1 + 1"))
    memory.close()
    assert approver.requests == []


def test_pre_cancelled_run_does_not_call_the_model(settings):
    settings.approval = "never"
    agent, memory = build_agent(settings)
    session_id = memory.create_session()
    cancel = asyncio.Event()
    cancel.set()

    events = asyncio.run(collect(agent, session_id, "hello", cancel=cancel))
    memory.close()

    assert "cancelled" in kinds(events)
    assert events[-1]["stop_reason"] == "cancelled"
    assert not any(event["type"] == "token" for event in events)


class ChattyProvider(LLMProvider):
    name = "fake"

    def __init__(self, settings, fail_times: int = 0, capture: list | None = None):
        super().__init__(settings)
        self.fail_times = fail_times
        self.capture = capture
        self.calls = 0
        self.model = "fake"

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator:
        self.calls += 1
        if self.capture is not None:
            self.capture.append(messages)
        if self.calls <= self.fail_times:
            raise ProviderError("upstream exploded", retryable=True)
        yield TextDelta("All systems nominal.")
        yield FinishDelta(reason="stop")


def test_retryable_provider_errors_are_retried(settings, monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("friday.engine.asyncio.sleep", no_sleep)
    settings.approval = "never"
    settings.ensure_dirs()
    memory = MemoryStore(settings.db_file)
    registry, _plugins = build_registry(settings)
    provider = ChattyProvider(settings, fail_times=2)
    agent = Agent(settings, provider, registry, memory)

    events = asyncio.run(collect(agent, memory.create_session(), "status check"))
    memory.close()

    assert provider.calls == 3
    assert sum(1 for event in events if event["type"] == "notice") == 2
    assert "nominal" in answer_text(events)


def test_provider_error_is_surfaced_not_swallowed(settings):
    settings.approval = "never"
    settings.ensure_dirs()
    memory = MemoryStore(settings.db_file)
    registry, _plugins = build_registry(settings)
    agent = Agent(settings, ChattyProvider(settings, fail_times=99), registry, memory)

    events = asyncio.run(collect(agent, memory.create_session(), "anyone home?"))
    memory.close()

    assert any(event["type"] == "error" for event in events)
    assert events[-1]["stop_reason"] == "provider_error"


class AlwaysToolsProvider(LLMProvider):
    name = "loops"
    model = "loops"

    async def stream(self, messages, tools, **kwargs) -> AsyncIterator:
        if not tools:
            yield TextDelta("Wrapping up.")
            yield FinishDelta(reason="stop")
            return
        yield ToolCallDelta(index=0, id="call_x", name="clock", arguments="{}")
        yield FinishDelta(reason="tool_calls")


def test_step_budget_forces_a_final_answer(settings):
    settings.approval = "never"
    settings.max_steps = 1
    settings.ensure_dirs()
    memory = MemoryStore(settings.db_file)
    registry, _plugins = build_registry(settings)
    agent = Agent(settings, AlwaysToolsProvider(settings), registry, memory)

    events = asyncio.run(collect(agent, memory.create_session(), "loop forever"))
    memory.close()

    assert events[-1]["type"] == "done"
    assert events[-1]["stop_reason"] == "max_steps"
    assert "Wrapping up" in answer_text(events)


def test_tool_failure_is_reported_and_the_loop_continues(settings):
    settings.approval = "never"
    agent, memory = build_agent(settings)
    session_id = memory.create_session()

    events = asyncio.run(collect(agent, session_id, "read the file ghost.txt"))
    memory.close()

    result = next(event for event in events if event["type"] == "tool_result")
    assert result["ok"] is False
    assert events[-1]["stop_reason"] == "stop"


def test_memory_is_injected_into_the_system_prompt(settings):
    settings.approval = "never"
    settings.ensure_dirs()
    memory = MemoryStore(settings.db_file)
    memory.remember("The reactor runs on vibranium")
    registry, _plugins = build_registry(settings)
    captured: list = []
    agent = Agent(settings, ChattyProvider(settings, capture=captured), registry, memory)

    asyncio.run(collect(agent, memory.create_session(), "hello"))
    memory.close()

    system_prompt = captured[0][0]
    assert system_prompt.role == "system"
    assert "vibranium" in system_prompt.content
    assert "F.R.I.D.A.Y." in system_prompt.content


def test_policy_approver_denies_when_nobody_is_watching():
    approver = PolicyApprover(allow_dangerous=False)
    request = ApprovalRequest("t", "shell", {}, "shell(ls)", "high", "s")
    approved, note = asyncio.run(approver.decide(request))
    assert approved is False
    assert "denied" in note
