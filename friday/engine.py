"""The FRIDAY agent loop: stream a turn, execute tools, repeat until done.

Design notes:
- one async generator (`run`) yields structured events; the server turns them into SSE,
  the CLI prints them, tests assert on them
- every message is persisted as it happens, so a crash never loses the transcript
- tools run sequentially: simpler approvals, deterministic transcripts, and agents
  rarely need parallel I/O for correctness (a deliberate v1 trade-off)
- budgets are hard caps, not suggestions
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from friday.config import Settings
from friday.llm import LLMProvider, ProviderError
from friday.memory import MemoryStore
from friday.persona import build_system_prompt
from friday.tools.base import ToolContext, ToolRegistry
from friday.types import Message, ToolCall, ToolResult, Usage

Event = dict[str, Any]
MAX_TOOL_OUTPUT = 30_000


@dataclass
class ApprovalRequest:
    token: str
    tool: str
    arguments: dict[str, Any]
    summary: str
    risk: str
    session_id: str


class Approver(Protocol):
    async def decide(self, request: ApprovalRequest) -> tuple[bool, str]: ...


class PolicyApprover:
    """Default gate: allow low risk, block everything when no human is present to ask."""

    def __init__(self, allow_dangerous: bool = False):
        self.allow_dangerous = allow_dangerous

    async def decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        if self.allow_dangerous:
            return True, "auto-approved (approval=never)"
        return False, "denied: this action needs human approval in the current mode"


class Agent:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        tools: ToolRegistry,
        memory: MemoryStore,
        approver: Approver | None = None,
    ):
        self.settings = settings
        self.provider = provider
        self.tools = tools
        self.memory = memory
        self.approver = approver or PolicyApprover()
        self.ctx = ToolContext(settings=settings, memory=memory, session_id="")

    # ------------------------------------------------------------------ setup
    def schemas(self):
        return self.tools.schemas(self.settings.disabled_tools)

    def _needs_approval(self, tool_name: str) -> bool:
        tool = self.tools.get(tool_name)
        if tool is None:
            return False
        mode = self.settings.approval
        if mode == "never":
            return False
        if mode == "always":
            return True
        return tool.is_dangerous()

    # -------------------------------------------------------------------- run
    async def run(
        self,
        session_id: str,
        user_text: str,
        *,
        cancel: asyncio.Event | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Event]:
        self.ctx.session_id = session_id
        user_text = (user_text or "").strip()
        if not user_text:
            yield {"type": "error", "message": "Empty message."}
            return

        is_new = not self.memory.get_messages(session_id, limit=1)
        user_message = Message(role="user", content=user_text)
        self.memory.add_message(session_id, user_message)
        if is_new:
            self.memory.rename_session(session_id, _title_for(user_text))

        history = _trim_history(
            self.memory.get_messages(session_id, limit=self.settings.max_history_messages)
        )
        system_prompt = build_system_prompt(
            self.settings, session_id=session_id, memory_block=self.memory.memory_block()
        )

        total_usage = Usage()
        steps = 0
        tool_calls_made = 0
        stop_reason = "stop"
        final_text = ""

        started_at = time.perf_counter()
        yield {"type": "start", "session_id": session_id, "title": _title_for(user_text)}

        while True:
            if cancel is not None and cancel.is_set():
                stop_reason = "cancelled"
                yield {"type": "cancelled"}
                break

            if steps >= self.settings.max_steps:
                stop_reason = "max_steps"
                async for event in self._final_answer(
                    session_id, [system_prompt] + history, total_usage, temperature, max_tokens
                ):
                    if event["type"] == "token":
                        final_text += event["text"]
                    yield event
                break

            steps += 1
            yield {"type": "step", "step": steps, "max_steps": self.settings.max_steps}

            messages = [Message(role="system", content=system_prompt)] + history
            text_parts: list[str] = []
            assembled: dict[int, dict[str, Any]] = {}
            usage = Usage()
            got_delta = False
            attempt = 0
            failed = False

            while True:
                try:
                    async for delta in self.provider.stream(
                        messages, self.schemas(), temperature=temperature, max_tokens=max_tokens
                    ):
                        kind = type(delta).__name__
                        if kind == "TextDelta":
                            got_delta = True
                            text_parts.append(delta.text)
                            yield {"type": "token", "text": delta.text}
                        elif kind == "ToolCallDelta":
                            got_delta = True
                            slot = assembled.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
                            if delta.id:
                                slot["id"] = delta.id
                            if delta.name:
                                slot["name"] = delta.name
                            slot["arguments"] += delta.arguments
                        elif kind == "UsageDelta":
                            usage = usage + delta.usage
                    break
                except ProviderError as exc:
                    retryable = exc.retryable and attempt < 2 and not got_delta
                    if retryable:
                        attempt += 1
                        yield {
                            "type": "notice",
                            "message": f"Provider hiccup ({exc}). Retrying {attempt}/2…",
                        }
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    failed = True
                    stop_reason = "provider_error"
                    yield {"type": "error", "message": str(exc), "retryable": exc.retryable}
                    break
                except asyncio.CancelledError:
                    partial = "".join(text_parts)
                    if partial:
                        self.memory.add_message(
                            session_id,
                            Message(role="assistant", content=partial, meta={"cancelled": True}),
                        )
                    raise

            if failed:
                partial = "".join(text_parts).strip()
                if partial:
                    self.memory.add_message(
                        session_id,
                        Message(role="assistant", content=partial, meta={"partial": True}),
                    )
                break
            if cancel is not None and cancel.is_set():
                stop_reason = "cancelled"
                yield {"type": "cancelled"}
                break

            total_usage = total_usage + usage
            content = "".join(text_parts)
            calls = _merge_calls(assembled)

            assistant = Message(role="assistant", content=content, tool_calls=calls)
            if content or calls:
                self.memory.add_message(session_id, assistant)
                history.append(assistant)
            if content:
                final_text = content

            if not calls:
                break

            for call in calls:
                tool_calls_made += 1
                yield {
                    "type": "tool_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.parsed_arguments(),
                    "raw_arguments": call.arguments,
                }

                if tool_calls_made > self.settings.max_tool_calls:
                    stop_reason = "max_tool_calls"
                    result = ToolResult(ok=False, output="Tool budget exhausted for this request.")
                    yield {
                        "type": "tool_result",
                        "id": call.id,
                        "name": call.name,
                        "ok": False,
                        "output": result.output,
                        "duration_ms": 0,
                    }
                    tool_message = result.to_message(call)
                    self.memory.add_message(session_id, tool_message)
                    history.append(tool_message)
                    continue

                tool = self.tools.get(call.name)
                summary = tool.summarize(call.parsed_arguments()) if tool else call.name
                approved, note = True, "allowed"
                if self._needs_approval(call.name):
                    token = uuid.uuid4().hex[:16]
                    request = ApprovalRequest(
                        token=token,
                        tool=call.name,
                        arguments=call.parsed_arguments(),
                        summary=summary,
                        risk=tool.risk if tool else "unknown",
                        session_id=session_id,
                    )
                    yield {
                        "type": "approval_required",
                        "token": token,
                        "id": call.id,
                        "name": call.name,
                        "arguments": request.arguments,
                        "summary": summary,
                        "risk": request.risk,
                    }
                    approved, note = await self.approver.decide(request)
                    yield {
                        "type": "approval_result",
                        "token": token,
                        "id": call.id,
                        "approved": approved,
                        "note": note,
                    }

                if not approved:
                    result = ToolResult(
                        ok=False, output=f"Action denied ({note}). Do not retry it unchanged."
                    )
                    duration_ms = 0
                else:
                    started = time.perf_counter()
                    result = await self.tools.execute(call.name, call.parsed_arguments(), self.ctx)
                    duration_ms = int((time.perf_counter() - started) * 1000)

                output = result.output
                if len(output) > MAX_TOOL_OUTPUT:
                    output = output[:MAX_TOOL_OUTPUT] + "\n… tool output truncated"
                result.output = output

                yield {
                    "type": "tool_result",
                    "id": call.id,
                    "name": call.name,
                    "ok": result.ok,
                    "output": output,
                    "duration_ms": duration_ms,
                }

                tool_message = Message(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=output,
                    meta={"ok": result.ok},
                )
                self.memory.add_message(session_id, tool_message)
                history.append(tool_message)

            if tool_calls_made >= self.settings.max_tool_calls:
                stop_reason = "max_tool_calls"
                yield {
                    "type": "notice",
                    "message": "Tool budget reached. Wrapping up with what I have.",
                }
                async for event in self._final_answer(
                    session_id, [system_prompt] + history, total_usage, temperature, max_tokens
                ):
                    if event["type"] == "token":
                        final_text += event["text"]
                    yield event
                break

        if total_usage.total_tokens == 0:
            total_usage = Usage.estimate(history)
        yield {
            "type": "done",
            "session_id": session_id,
            "steps": steps,
            "stop_reason": stop_reason,
            "usage": total_usage.to_dict(),
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
        }

    # ------------------------------------------------------------ final answer
    async def _final_answer(
        self,
        session_id: str,
        messages: list[Message],
        usage: Usage,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[Event]:
        """One last, tool-free synthesis when the loop hits a budget."""
        collected: list[str] = []
        try:
            async for delta in self.provider.stream(
                messages, [], temperature=temperature, max_tokens=max_tokens
            ):
                if type(delta).__name__ == "TextDelta":
                    collected.append(delta.text)
                    yield {"type": "token", "text": delta.text}
                elif type(delta).__name__ == "UsageDelta":
                    usage = usage + delta.usage
        except ProviderError as exc:
            yield {"type": "error", "message": f"Could not synthesise a final answer: {exc}"}
            return
        content = "".join(collected).strip()
        if content:
            self.memory.add_message(session_id, Message(role="assistant", content=content))


def _trim_history(history: list[Message]) -> list[Message]:
    """Never start the prompt mid-exchange: a tool message without its call is invalid."""
    for index, message in enumerate(history):
        if message.role == "user":
            return history[index:]
    return history


def _merge_calls(assembled: dict[int, dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(assembled):
        slot = assembled[index]
        name = (slot.get("name") or "").strip()
        if not name:
            continue
        arguments = slot.get("arguments") or "{}"
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            arguments = "{}"
        calls.append(ToolCall(id=slot.get("id") or f"call_{index}", name=name, arguments=arguments))
    return calls


def _title_for(text: str) -> str:
    cleaned = " ".join(text.split())
    return (cleaned[:57] + "…") if len(cleaned) > 58 else (cleaned or "New session")


__all__ = ["Agent", "ApprovalRequest", "Approver", "Event", "PolicyApprover"]
