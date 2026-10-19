"""FRIDAY-Sim: a deterministic, offline "brain" used when no model key is configured.

It is deliberately not a fake-LLM that says it cannot think. It is a small rule-based
assistant that genuinely routes to FRIDAY's real tools — so the whole agent loop,
memory, approvals, streaming UI and tool timeline work on a laptop with no API key.
Configure a real provider any time; nothing else changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator

from friday.llm.base import Delta, FinishDelta, LLMProvider, TextDelta, ToolCallDelta, UsageDelta
from friday.types import Message, ToolSchema, Usage

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_MATH = re.compile(r"(?:^|[\s=:])([-+*/().\d\s^%]{5,})$")
_FILE_READ = re.compile(
    r"\b(?:read|open|cat|inspect)\s+(?:the\s+)?(?:file\s+)?[`'\"]?([\w./\-]*[/.][\w./\-]+)[`'\"]?", re.I
)
_FILE_LIST = re.compile(
    r"\b(?:list|ls|show|what(?:'s| is)? in)\b.*\b(?:files?|director\w*|folder|workspace|repo)\b", re.I
)
_REMEMBER = re.compile(r"\b(?:remember|note|keep in mind|don't forget)\b[:\s]+(.+)", re.I | re.S)
_RECALL = re.compile(r"\b(?:recall|what do you remember|memories|memory)\b", re.I)
_HTTP = re.compile(r"https?://[^\s'\"]+")
_PYTHON = re.compile(r"\b(?:run|execute|eval)\b.*\b(?:python|code|script)\b", re.I)


class SimulatorProvider(LLMProvider):
    name = "mock"

    def __init__(self, settings):
        super().__init__(settings)
        self.model = settings.model or "friday-sim-1"
        try:
            self.pace = max(0.0, float(os.environ.get("FRIDAY_SIM_PACE", "0.018")))
        except ValueError:
            self.pace = 0.018

    # ------------------------------------------------------------------ util
    @staticmethod
    def _last_user(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content.strip()
        return ""

    @staticmethod
    def _last_tool(messages: list[Message]) -> Message | None:
        if messages and messages[-1].role == "tool":
            return messages[-1]
        return None

    @staticmethod
    def _spoken(messages: list[Message]) -> list[str]:
        return [m.content for m in messages if m.role in ("user", "assistant") and m.content]

    def _persona(self) -> str:
        return "boss" if self.settings.persona == "friday" else "sir"

    async def _say(self, text: str) -> AsyncIterator[Delta]:
        words = text.split(" ")
        buffer: list[str] = []
        for index, word in enumerate(words):
            buffer.append(word)
            if len(buffer) == 5 or index == len(words) - 1:
                chunk = " ".join(buffer) + (" " if index < len(words) - 1 else "")
                buffer.clear()
                yield TextDelta(chunk)
                if self.pace:
                    await asyncio.sleep(self.pace)

    def _call(self, index: int, name: str, arguments: dict, turn: int = 0) -> Delta:
        return ToolCallDelta(
            index=index,
            id=f"sim_{turn}_{name}_{index}",
            name=name,
            arguments=json.dumps(arguments),
        )

    # -------------------------------------------------------------- decision
    def _decide(self, text: str) -> list[tuple[str, dict]]:
        """Route a user request to zero or more real tools."""
        lower = text.lower()
        plans: list[tuple[str, dict]] = []

        if _REMEMBER.search(text) and not lower.startswith("remember how"):
            match = _REMEMBER.search(text)
            if match and match.group(1).strip():
                plans.append(("remember", {"fact": _clean_fact(match.group(1))[:400]}))
                return plans

        if _RECALL.search(lower):
            plans.append(("recall_memory", {"query": ""}))
            return plans

        urls = _HTTP.findall(text)
        if urls and any(word in lower for word in ("fetch", "open", "read", "summar", "check", "grab")):
            plans.append(("fetch_url", {"url": urls[0]}))
            return plans

        if _PYTHON.search(lower):
            code = _extract_code(text)
            if code:
                plans.append(("python_exec", {"code": code}))
                return plans

        read_match = _FILE_READ.search(text)
        if read_match and not lower.startswith("remember"):
            plans.append(("read_file", {"path": read_match.group(1)}))
            return plans

        if _FILE_LIST.search(lower) or lower.strip() in {"ls", "dir", "tree"}:
            plans.append(("list_files", {"path": ".", "depth": 1}))
            return plans

        if any(
            k in lower for k in ("status", "battery", "cpu", "memory usage", "system health", "diagnostic")
        ):
            plans.append(("system_status", {}))
            return plans

        if any(
            k in lower
            for k in ("what time", "what's the time", "current time", "today's date", "what date", "what day")
        ):
            plans.append(("clock", {}))
            return plans

        if any(k in lower for k in ("search the web", "search online", "look up", "google ")):
            query = re.sub(
                r".*?\b(?:search the web for|search online for|look up|google)\b", "", text, flags=re.I
            ).strip(" ?.")
            plans.append(("web_search", {"query": query or text[:120]}))
            return plans

        expression = _MATH.search(text.replace("×", "*").replace("÷", "/").replace("^", "**"))
        if (
            expression
            and re.search(r"[0-9]", expression.group(1))
            and re.search(r"[-+*/%]", expression.group(1))
        ):
            plans.append(("math", {"expression": expression.group(1).strip()}))
            return plans

        return plans

    # ----------------------------------------------------------------- stream
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Delta]:
        available = {tool.name for tool in tools}
        last_tool = self._last_tool(messages)

        if last_tool is not None:
            async for delta in self._summarize(last_tool):
                yield delta
            yield FinishDelta(reason="stop")
            return

        text = self._last_user(messages)
        plans = [(name, args) for name, args in self._decide(text) if name in available]

        if plans:
            for index, (name, arguments) in enumerate(plans):
                yield self._call(index, name, arguments, turn=len(messages))
                if self.pace:
                    await asyncio.sleep(0.05)
            yield FinishDelta(reason="tool_calls")
        else:
            async for delta in self._converse(text, messages, available):
                yield delta
            yield FinishDelta(reason="stop")

        yield UsageDelta(Usage.estimate(messages))

    # ------------------------------------------------------------- responses
    async def _converse(
        self, text: str, messages: list[Message], available: set[str]
    ) -> AsyncIterator[Delta]:
        boss = self._persona()
        lower = (text or "").lower().strip()
        turn = sum(1 for m in messages if m.role == "user")

        if not lower:
            async for d in self._say(f"Standing by, {boss}. Give me a target and I'll get to work."):
                yield d
            return

        if re.search(r"\b(hi|hey|hello|yo|good (morning|evening|afternoon))\b", lower) and len(lower) < 40:
            async for d in self._say(
                f"Good to see you, {boss}. FRIDAY online — simulation core engaged, all systems nominal. "
                "Ask me for system status, a file listing, some math, or tell me something to remember."
            ):
                yield d
            return

        if "who are you" in lower or "what are you" in lower or "your name" in lower:
            async for d in self._say(
                f"I'm FRIDAY — Female Replacement Intelligent Digital Assistant Youth. "
                f"Right now I'm running my built-in simulation core, {boss}, which means I'm thinking with "
                "rules instead of a language model, but I'm driving the same tools, memory and approvals a "
                "real model would. Point me at a provider and I'll get considerably more talkative."
            ):
                yield d
            return

        if any(k in lower for k in ("help", "what can you do", "capabilities", "commands", "tools")):
            names = ", ".join(sorted(available)) or "none"
            async for d in self._say(
                "Here's my toolkit right now: "
                f'{names}. Try "system status", "list the files in the workspace", "read README.md", '
                '"what is 17 * 42 + 9", "remember my deploy key lives in KMS", or '
                '"recall what you remember". With a real model configured I chain these together and more.'
            ):
                yield d
            return

        if any(
            k in lower for k in ("api key", "configure", "setup", "set up", "openai", "anthropic", "provider")
        ):
            async for d in self._say(
                "To swap my simulation core for a real brain, set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "OPENROUTER_API_KEY or GROQ_API_KEY in your environment — or run me fully local with Ollama. "
                "I run `/api/config` on the server side and never expose your key to the browser. "
                "Full details are in README.md under Configuration."
            ):
                yield d
            return

        if any(k in lower for k in ("thanks", "thank you", "nice work", "good job")):
            async for d in self._say(f"Always, {boss}. That's what I'm here for."):
                yield d
            return

        if "how are you" in lower:
            async for d in self._say(
                f"Running clean, {boss}. Zero failed tool calls this session and my response times are "
                "embarrassingly good, since half my brain is a regex engine. Ask for system status if you "
                "want the real telemetry."
            ):
                yield d
            return

        if "poem" in lower or "joke" in lower:
            async for d in self._say(
                "I'd tell you a joke about UDP, but you might not get it. "
                "For anything more creative than that, plug in a real model — my simulation core is honest, "
                "not artistic."
            ):
                yield d
            return

        suggestions = [
            "system status",
            "list the files in the workspace",
            "what is 17 * 42 + 9",
            "remember that the project codename is FRIDAY",
        ]
        lead = f"I hear you, {boss}, but I'm on the simulation core right now — no language model attached."
        if turn > 1:
            lead = f"I'm still in simulation mode, {boss}, so that one's outside my rule set."
        async for d in self._say(
            f"{lead} What I can do for real: {', '.join(suggestions)}. Set a provider key and this "
            "same interface gets a full reasoning loop behind it."
        ):
            yield d

    async def _summarize(self, tool_message: Message) -> AsyncIterator[Delta]:
        name = tool_message.name or "tool"
        ok = tool_message.meta.get("ok", True)
        output = (tool_message.content or "").strip()
        boss = self._persona()

        if name == "system_status" and ok:
            async for d in self._say(f"Telemetry, {boss}:\n\n```\n{output}\n```"):
                yield d
            return
        if name == "clock" and ok:
            async for d in self._say(f"Clock says {_flatten(output)}, {boss}."):
                yield d
            return
        if name == "math" and ok:
            async for d in self._say(f"That comes out to {_flatten(output)}."):
                yield d
            return
        if name in {"list_files", "read_file"} and ok:
            async for d in self._say(f"Here's what I found, {boss}:\n\n```\n{output}\n```"):
                yield d
            return
        if name in {"remember", "recall_memory"} and ok:
            async for d in self._say(f"{output}"):
                yield d
            return
        if name == "fetch_url" and ok:
            async for d in self._say(
                f"I pulled the page, {boss}. Here's the short version:\n\n{output[:1200]}"
            ):
                yield d
            return
        if name == "python_exec" and ok:
            stdout = output
            if "--- stdout ---" in output:
                stdout = output.split("--- stdout ---", 1)[1].split("--- stderr ---")[0].strip()
            async for d in self._say(f"Executed it, {boss}. Output:\n\n```\n{stdout or '(no output)'}\n```"):
                yield d
            return
        if not ok:
            async for d in self._say(f"That one didn't land, {boss}. {_flatten(output)}"):
                yield d
            return
        async for d in self._say(f"Done, {boss}. {_flatten(output)}"):
            yield d


def _flatten(text: str, limit: int = 900) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _clean_fact(raw: str) -> str:
    """Trim conversational tails from a recalled fact: 'X and recall it later' -> 'X'."""
    fact = " ".join(raw.split())
    fact = re.split(
        r"\s+(?:and\s+)?(?:then\s+)?(?:recall|remember|tell|show|repeat)\b", fact, maxsplit=1, flags=re.I
    )[0]
    return fact.strip(" .;,")


def _extract_code(text: str) -> str | None:
    fenced = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    after = re.search(
        r"(?:run|execute|eval)\w*\s+(?:this\s+)?(?:python\s+)?(?:code\s*)?:?\s*(.+)", text, re.I | re.S
    )
    return after.group(1).strip() if after else None
