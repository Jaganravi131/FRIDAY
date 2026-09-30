"""LLM clients. One real (OpenAI-compatible, i.e. llama-server / Ollama), one mock.

The mock is not a test toy. It is what lets the whole system run — end to end, with
retrieval, ledger, tools, traces, telemetry — before you have a model downloaded.
Phase 0's exit test is "ask Monday, follow up Friday"; you should be able to prove
the *memory* half of that on day one with no 2.6 GB download and no GPU.

₹0 note: `CLOUD_TIERS_ENABLED` is False by default (docs/architecture/12 §8). When
the top of the Escalation Ladder is missing, `escalate` must degrade to *ask the
user*, never to *guess confidently*. `NoRouteError` exists to make that explicit
rather than letting it fall through to a silent hallucination.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_S


class NoRouteError(RuntimeError):
    """The ladder wanted to escalate and there is nowhere to escalate to."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class Response:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: int | None = None
    total_ms: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class Client(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ) -> Response: ...


# ── real: OpenAI-compatible HTTP ───────────────────────────────────────────────

class OpenAICompatibleClient:
    """Talks to `llama-server` or Ollama's OpenAI endpoint. Stdlib urllib only.

        llama-server -hf LiquidAI/LFM2-2.6B-GGUF:Q4_K_M -c 8192 --port 8080 --flash-attn 1

    Tool calling goes through the server's own parser (`--jinja` on llama.cpp). If
    the model or server doesn't support it, `tools` is ignored and we fall back to
    the prompt-embedded protocol in `_PROMPT_TOOL_HINT` — which is what a 2.6B model
    needs anyway, since native function calling at that size is unreliable.
    """

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,
        timeout: float = LLM_TIMEOUT_S,
        native_tools: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.native_tools = native_tools
        self.name = f"openai:{self.base_url}:{model}"

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url.replace("/v1", "") + "/health", timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=768) -> Response:
        import time

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        use_native = bool(tools) and self.native_tools
        if use_native:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        elif tools:
            messages = list(messages)
            messages.append({"role": "system", "content": _prompt_tool_hint(tools)})

        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            # A 400 mentioning tools usually means the server has no tool parser.
            if tools and e.code == 400:
                self.native_tools = False
                return self.chat(messages, tools=tools, temperature=temperature,
                                 max_tokens=max_tokens)
            raise RuntimeError(f"LLM HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach the local model at {self.base_url} ({e.reason}). "
                f"Is llama-server running? See docs/architecture/12 §11."
            ) from e
        total_ms = int((time.perf_counter() - t0) * 1000)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = [_parse_call(c) for c in (msg.get("tool_calls") or [])]
        text = msg.get("content") or ""
        if not calls and not use_native and tools:
            text, calls = _parse_prompt_tools(text)
        usage = data.get("usage") or {}
        return Response(
            text=text,
            tool_calls=[c for c in calls if c],
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_ms=total_ms,
            raw=data,
        )


def _parse_call(c: dict) -> ToolCall | None:
    fn = c.get("function") or {}
    name = fn.get("name")
    if not name:
        return None
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    return ToolCall(name=name, arguments=args or {}, id=c.get("id", ""))


# ── prompt-embedded tool protocol (small models) ───────────────────────────────

_TOOL_RE = re.compile(
    r"<tool>(?P<body>.*?)</tool>", re.DOTALL | re.IGNORECASE
)


def _prompt_tool_hint(tools: list[dict]) -> str:
    """Describe the tools in plain text and ask for a strict, parseable format.

    Small models are much better at emitting `<tool>{"name":…}</tool>` than at
    satisfying a JSON-schema tool parser, and this path needs no server support.
    """
    lines = ["You may call these tools. To call one, output exactly:"]
    lines.append('<tool>{"name": "TOOL_NAME", "arguments": {...}}</tool>')
    lines.append("Output at most one tool call, then stop. If no tool is needed, answer normally.\n")
    for t in tools:
        fn = t.get("function", t)
        params = ", ".join((fn.get("parameters") or {}).get("properties", {}).keys())
        doc = (fn.get("description") or "").strip().split("\n")[0]
        lines.append(f"- {fn.get('name')}({params}): {doc}")
    return "\n".join(lines)


def _parse_prompt_tools(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool calls from free text. Returns (text_with_calls_removed, calls)."""
    calls: list[ToolCall] = []
    def _sub(m: re.Match) -> str:
        body = m.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            name = re.search(r'"?name"?\s*[:=]\s*"?([\w_]+)', body)
            if not name:
                return ""
            obj = {"name": name.group(1), "arguments": {}}
        if isinstance(obj, dict) and obj.get("name"):
            calls.append(ToolCall(
                name=str(obj["name"]),
                arguments=obj.get("arguments") or obj.get("args") or {},
            ))
        return ""
    cleaned = _TOOL_RE.sub(_sub, text).strip()
    return cleaned, calls


# ── mock ───────────────────────────────────────────────────────────────────────

class MockClient:
    """Deterministic, offline, zero-dependency.

    It is intentionally *retrieval-driven*: it reads the `<recalled>` block the
    Ledger produced and answers from it. That makes it a genuine test of the memory
    path — if retrieval finds the fact, the mock answers correctly; if retrieval
    returns `[]`, the mock says it doesn't know. Which is exactly the behaviour
    Law 4 demands, and exactly what the needle test measures.
    """

    name = "mock"

    def __init__(self, latency_ms: int = 0):
        self.latency_ms = latency_ms
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=768) -> Response:
        import time

        t0 = time.perf_counter()
        self.calls.append(messages)
        blob = "\n".join(m.get("content") or "" for m in messages)
        recalled = _between(blob, "<recalled>", "</recalled>")
        scratch = _between(blob, "<scratch>", "</scratch>")
        query = _last_user(messages)
        # The clock reading travels in the compiled context, exactly where a real model
        # would find it — see _now_from for why the mock must not read a clock itself.
        now = _now_from(scratch)

        wants_write = _wants_write(query)
        if wants_write and tools and _has_tool(tools, "memory_write"):
            pred, obj = wants_write
            return Response(
                text="",
                tool_calls=[ToolCall("memory_write", {"predicate": pred, "object": obj,
                                                      "source_quote": query})],
                model=self.name, total_ms=_ms(t0, self.latency_ms),
            )
        if _wants_search(query) and tools and _has_tool(tools, "memory_search"):
            return Response(
                text="",
                tool_calls=[ToolCall("memory_search", {"query": query})],
                model=self.name, total_ms=_ms(t0, self.latency_ms),
            )
        # ⭐ A clock question is answered from the injected reading, BEFORE the memory
        # branch. The tool contract withholds memory_search for it, so falling through
        # to "I don't have anything in memory" is truthful and useless — the user asked
        # the time, not whether FRIDAY remembers it.
        if now and _CLOCKY.search(query.lower()):
            return Response(text=f"It's {now}.", model=self.name,
                            total_ms=_ms(t0, self.latency_ms))

        # "who are you" is answered from the IDENTITY slot, which is in the compiled
        # context — SOUL.md is the stable prefix, so it is always there. Saying "I have
        # no memory about that" when the answer is literally in the prompt is the mock
        # failing to read its own context, which is the one thing it must model a real
        # client doing.
        if _META.search(query.strip().lower()):
            ident = _between(blob, "<identity>", "</identity>").strip()
            line = next((l.strip() for l in ident.splitlines()
                         if l.strip() and not l.strip().startswith("#")), "")
            return Response(
                text=(f"{line[:200]}" if line else
                      "I'm FRIDAY. Your memory is in Markdown you can read and edit."),
                model=self.name, total_ms=_ms(t0, self.latency_ms))

        # Arithmetic needs neither memory nor a model.
        arith = _safe_arith(query)
        if arith is not None:
            return Response(text=f"{arith}", model=self.name,
                            total_ms=_ms(t0, self.latency_ms))

        # A greeting or an acknowledgement is not a question and needs no memory.
        # Answering "hi" with "I don't have anything in memory about that" is the
        # single fastest way to make an assistant feel like a database.
        g = _SOCIAL.match(query.strip())
        if g:
            return Response(text=_social_reply(g.group(0)),
                            model=self.name, total_ms=_ms(t0, self.latency_ms))

        if recalled.strip():
            facts = _facts_from(recalled)
            if facts:
                return Response(text=_answer(query, facts, now=now), model=self.name,
                                total_ms=_ms(t0, self.latency_ms))
        return Response(
            text=(
                "I don't have anything in memory about that. "
                "(Retrieval returned no fact above the score floor — I'd rather say so "
                "than invent one.)"
            ),
            model=self.name, total_ms=_ms(t0, self.latency_ms),
        )


def _ms(t0: float, extra: int) -> int:
    import time

    return int((time.perf_counter() - t0) * 1000) + extra


def _between(s: str, a: str, b: str) -> str:
    i = s.find(a)
    if i < 0:
        return ""
    j = s.find(b, i + len(a))
    return s[i + len(a): j if j >= 0 else len(s)]


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


_FACT_LINE = re.compile(r"^-\s*([\w_\-\.]+):\s*(.+?)(?:\s*\[.*\])?$")


def _facts_from(recalled: str) -> list[tuple[str, str]]:
    out = []
    for line in recalled.splitlines():
        line = line.strip()
        if line.startswith("> "):  # a source quote, not a fact
            continue
        m = _FACT_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


_CLOCKY = re.compile(
    r"\b(?:what(?:'s|\s+is)?\s+(?:the\s+)?(?:time|date|day)|what\s+time|today'?s\s+date"
    r"|what\s+day\s+is|the\s+time\b|\bdate\b)\b", re.I
)
_ASK = re.compile(r"\b(what|which|who|where|when|how much|how many|is|are|do i|tell me)\b", re.I)
_WRITE = re.compile(
    r"\b(?:my|i am|i'm|i live|i moved|i work|call me|remember that|i prefer|my name is)\b", re.I
)
_SEARCHY = re.compile(r"\b(my|remember|earlier|last time|before|used to|what did)\b", re.I)


def _wants_search(q: str) -> bool:
    return bool(_ASK.search(q) and _SEARCHY.search(q))


def _wants_write(q: str) -> tuple[str, str] | None:
    """Very small extraction, only for the mock's benefit. The real system uses the
    model's own tool call."""
    if not _WRITE.search(q):
        return None
    ql = q.lower().strip().rstrip(".")
    if "my name is" in ql:
        return "name", q.split("my name is", 1)[1].strip().title()
    if "i live in" in ql:
        return "lives_in", q.split("i live in", 1)[1].strip()
    if "i moved to" in ql:
        return "lives_in", q.split("i moved to", 1)[1].strip()
    if "i work at" in ql:
        return "employer", q.split("i work at", 1)[1].strip()
    if "i prefer" in ql:
        return "prefers", q.split("i prefer", 1)[1].strip()
    if "my rent is" in ql:
        return "lease_amount_monthly", q.split("my rent is", 1)[1].strip()
    return None


def _has_tool(tools: list[dict], name: str) -> bool:
    return any((t.get("function", t).get("name") == name) for t in tools)


_NOW_LINE = re.compile(r"^\[now\]\s*(.+)$", re.MULTILINE)


def _now_from(scratch: str) -> str | None:
    """Pull the injected clock reading back out of the context.

    The mock reads it from the SAME place a real model would — the compiled context —
    rather than calling datetime itself. That matters: if the mock had its own clock,
    it would keep working when the injection broke, and the no-model path would stop
    testing the thing it exists to test.
    """
    m = _NOW_LINE.search(scratch or "")
    return m.group(1).strip() if m else None


#: Greetings and acknowledgements, and what to say back. The mock is not a toy: it is
#: the path that has to feel like an assistant before any model is downloaded, and
#: "hi" -> "I don't have anything in memory about that" would make Phase 0 unusable to
#: actually talk to, which is the one habit doc 10 says decides whether this ships.
_SOCIAL = re.compile(
    r"^(?:hi|hey|hello|yo|namaste|vanakkam|good\s+(?:morning|afternoon|evening|night)"
    r"|thanks|thank\s+you|thx|ok|okay|k|sure|alright|got\s+it|makes\s+sense"
    r"|bye|goodbye|see\s+you|how\s+are\s+you|what'?s\s+up)(?:[!.,\s]*(?:friday|there|da|mach|bro))?[\s!.,]*$",
    re.IGNORECASE,
)
_SOCIAL_REPLY = {
    "hi": "Hi. What do you need?",
    "hey": "Hey. What's up?",
    "hello": "Hello. I'm here.",
    "namaste": "Namaste. What can I do for you?",
    "vanakkam": "Vanakkam. What do you need?",
    "thanks": "Any time.",
    "thank you": "Any time.",
    "ok": "Alright.",
    "okay": "Alright.",
    "bye": "Bye. I'll keep the memory warm.",
    "how are you": "Running. Ask me something and I'll be more use.",
}


_META = re.compile(
    r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do"
    r"|help|commands?|are\s+you\s+(?:real|human|there)|what\s+model\s+are\s+you"
    r"|how\s+do\s+you\s+work)(?:\??)[\s!.,]*$", re.IGNORECASE)

# No exponentiation on purpose. `**` is right-associative and cheap to write but
# unbounded to evaluate: 9**9**9 is a one-line denial of service against a laptop
# that is also the user's only machine. Nobody asks a personal assistant for a power
# tower, and the four arithmetic operators cover every realistic question.
_ARITH_OK = re.compile(r"^[\s(]*[-+]?[\d.,]+(?:\s*[-+*/x×%]\s*[-+]?[\d.,]+\s*[)]*)+[\s?=]*$")


def _safe_arith(query: str):
    """Evaluate a plain arithmetic question, or return None.

    ⚠️ Parsed with `ast` and walked to allow ONLY numbers and the five arithmetic
    operators. `eval()` on a user string is how a personal agent with filesystem
    access becomes a remote code execution hole, and the regex pre-check is not
    sufficient on its own — it is defence in depth, not the defence.
    """
    import ast as _ast

    q = (query or "").strip().rstrip("?=").strip()
    low = q.lower()
    for lead in ("what is", "what's", "calculate", "compute", "how much is"):
        if low.startswith(lead):
            q = q[len(lead):].strip()
            break
    if not q or not _ARITH_OK.match(q):
        return None
    expr = q.replace("x", "*").replace("×", "*").replace(",", "")
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    allowed = (_ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Constant,
               _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv, _ast.Mod)
    for node in _ast.walk(tree):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, _ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        val = eval(compile(tree, "<arith>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return None
    if isinstance(val, float):
        return f"{val:,.2f}".rstrip("0").rstrip(".") if val != int(val) else f"{int(val):,}"
    return f"{val:,}"


def _social_reply(matched: str) -> str:
    key = matched.lower().strip().rstrip("!.,")
    return _SOCIAL_REPLY.get(key, "Noted.")


def _answer(query: str, facts: list[tuple[str, str]], now: str | None = None) -> str:
    ql = query.lower()
    if now and _CLOCKY.search(ql):
        return f"It's {now}."
    # prefer a fact whose predicate shares a word with the question
    qwords = set(re.findall(r"[a-z]{3,}", ql))
    best = None
    for pred, obj in facts:
        pw = set(re.findall(r"[a-z]{3,}", pred.lower().replace("_", " ")))
        score = len(pw & qwords)
        if "RETRACTED" in obj:
            score -= 2   # historical: only surface it if asked about the past
        if any(k in ql for k in ("used to", "before", "previously", "last year")):
            score += 2 if "RETRACTED" in obj else 0
        if best is None or score > best[0]:
            best = (score, pred, obj)
    if best is None or best[0] <= 0:
        pred, obj = facts[0]
    else:
        _, pred, obj = best
    hist = " (this was retracted — it's what I used to believe)" if "RETRACTED" in obj else ""
    return f"{pred.replace('_',' ')}: {obj.split('  [')[0].strip()}{hist}"


# ── selection ──────────────────────────────────────────────────────────────────

def get_client(prefer: str = "auto", **kw: Any) -> Client:
    """Real server if reachable, else the mock. Never raises."""
    if prefer in ("auto", "http"):
        try:
            c = OpenAICompatibleClient(**kw)
            if c.alive():
                return c
        except Exception:
            pass
        if prefer == "http":
            return OpenAICompatibleClient(**kw)
    return MockClient()
