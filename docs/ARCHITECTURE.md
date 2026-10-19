# Architecture

This document explains how FRIDAY actually works: the turn lifecycle, the event protocol, the data
model, the security boundaries and the deliberate trade-offs. If you are extending the agent, start
here.

---

## 1. Layers

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Interface        friday/server/static/*        CLI: friday/cli.py           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Transport        FastAPI REST + SSE            (friday/server/app.py)       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Runtime          Agent loop, approval gate     (friday/engine.py)           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Cognition        Provider abstraction          (friday/llm/)                │
│ Capability       Tool registry + tools         (friday/tools/)              │
│ State            SQLite + FTS5                 (friday/memory.py)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each layer depends only downward. The agent never imports FastAPI; the server never reaches into a
tool; tools receive an explicit `ToolContext` rather than reading globals.

---

## 2. A turn, step by step

`Agent.run(session_id, user_text)` is an **async generator of events**.

1. **Persist the user message** and derive a session title on first contact.
2. **Assemble context**:
   - system prompt (persona + live UTC time + workspace + session id),
   - the long-term memory block (most recent facts),
   - the transcript, capped at `max_history_messages`, trimmed so it always starts on a user turn —
     a tool result without its preceding tool call is an invalid request.
3. **Model turn**. Stream deltas:
   - `TextDelta` → forwarded immediately as `token` events,
   - `ToolCallDelta` → accumulated by index (the wire format sends names once and arguments in
     fragments),
   - `UsageDelta` → summed,
   - `FinishDelta` → records the stop reason.
   On a retryable provider error with no output yet, retry up to twice with backoff. Never retry
   after tokens have been emitted — that would duplicate text.
4. **Persist the assistant message** (content and/or tool calls).
5. **For each tool call**, sequentially:
   - emit `tool_call`,
   - if the approval policy covers this tool's risk, emit `approval_required` and await the
     approver (browser future, terminal prompt, or policy denial),
   - execute through the registry (argument validation, per-tool timeout, exception capture),
   - emit `tool_result`,
   - persist the tool message and append it to the working history.
6. **Loop** until the model answers without tool calls, a budget is exhausted, the client cancels,
   or the provider fails.
7. **Emit `done`** with step count, usage and wall time.

If the step budget is hit, FRIDAY makes one last **tool-free** call to synthesise an answer from what
it has. If the tool budget is hit, the same. If a provider error occurs after partial text, that text
is persisted with `meta.partial = true` so the transcript never lies about what was said.

### Why sequential tools?

Parallel tool execution is faster but makes approvals, transcript ordering and failure semantics
much harder to reason about. FRIDAY v1 chose determinism. The registry is already async, so a
scheduler can be added without changing tool contracts.

---

## 3. Event protocol (SSE)

One JSON object per `data:` line. The server injects `run_id` and `session_id` into every event.

| Event | Payload (beyond type/ids) | Meaning |
|---|---|---|
| `open` | — | Stream established |
| `start` | `title` | Turn started, session resolved |
| `step` | `step`, `max_steps` | A model turn begins |
| `token` | `text` | Incremental assistant text |
| `message` | `role`, `content` | Complete assistant message, only for non-streamed text |
| `tool_call` | `id`, `name`, `arguments`, `raw_arguments` | Model requested a tool |
| `tool_result` | `id`, `name`, `ok`, `output`, `duration_ms` | Tool finished or failed |
| `approval_required` | `token`, `id`, `name`, `arguments`, `summary`, `risk` | Blocked, waiting on a human |
| `approval_result` | `token`, `approved`, `note` | Decision delivered to the loop |
| `notice` | `message` | Non-fatal advisory (e.g. retry) |
| `error` | `message`, `retryable` | Fatal for this turn |
| `cancelled` | — | Operator stopped the run |
| `done` | `steps`, `stop_reason`, `usage`, `duration_ms` | Terminal |
| `close` | — | Stream closed |

Client-side, the HUD finalises the current assistant block whenever a `tool_call` arrives, so text
and tool cards interleave in chronological order across multiple steps.

### Approvals, mechanically

1. Engine emits `approval_required` with a random token.
2. The server's `ServerApprover` registers a `Future` under that token and awaits it with a 190 s
   timeout.
3. The browser `POST /api/approvals/{token}` resolves it (409 if already resolved, 404 if unknown).
4. On timeout, client disconnect or server shutdown, the future resolves to **deny**.
5. The engine emits `approval_result` and either executes or reports a denial to the model.

A request's `approval` field can only move the policy toward stricter
(`never < dangerous < always`); the server's configured policy is the floor. A browser cannot
silently disable a guardrail the operator set.

---

## 4. Providers

`LLMProvider.stream()` yields normalized deltas. Two concrete backends plus a simulator:

- **OpenAI-compatible** (`openai_compat.py`): Chat Completions with `stream: true`, tool schemas,
  and incremental `tool_calls`. Works with OpenAI, OpenRouter, Groq, Ollama, LM Studio, vLLM and
  anything else that implements the wire format. `stream_options.include_usage` is requested where
  supported.
- **Anthropic** (`anthropic.py`): Messages API. Converts the neutral transcript to content blocks,
  maps `tool_use`/`tool_result`, converts `input_json_delta` fragments back into the same tool-call
  index space, and emits usage at `message_delta`.
- **Simulator** (`simulator.py`): deterministic intent routing to real tools, plus a small
  conversational rule set. It exists so the runtime is testable and demonstrable offline, and it is
  explicit about what it is.

Adding a provider means implementing one async generator and registering it in `build_provider`.

Provider selection (`Settings._resolve_provider`) is deliberately conservative: if a keyed provider
is requested but no key exists and no custom base URL was supplied, FRIDAY falls back to the
simulator and says so through `/api/config` rather than failing at the first message.

---

## 5. State

SQLite (WAL mode) with four tables:

```text
sessions(id PK, title, created_at, updated_at, meta)
messages(id PK, session_id FK→sessions ON DELETE CASCADE, seq, role, content,
         tool_calls JSON, tool_call_id, name, meta JSON, created_at)
memories(id PK, text, tags, session_id, created_at)
messages_fts / memories_fts  — FTS5 (porter, unicode61) with snippets
```

- Messages are stored individually so a crash or cancellation never loses the transcript.
- `seq` is assigned inside the write lock; reads return the most recent *N* in chronological order.
- If SQLite lacks FTS5, the store records that and falls back to `LIKE` search — degraded, not broken.
- Long-term facts are injected into the system prompt every turn. `search_history` exposes the FTS
  index as a tool, so the agent can recall past conversations.

The store uses one connection behind a re-entrant lock. Local SQLite is microseconds; the agent
awaits models, not disks. This is intentional, documented and tested.

---

## 6. Security boundaries

| Boundary | Enforcement | Limits |
|---|---|---|
| Filesystem | `resolve_in_workspace()` resolves symlinks and checks containment | Workspace only; no per-file ACLs |
| Network | `assert_public_url()` resolves DNS and rejects private/loopback/link-local/reserved IPs; validated on every redirect hop; ports 80/443 | DNS rebinding is mitigated but not eliminated |
| Shell | Denylist regexes for catastrophic commands, sanitised environment, timeout | **Not a sandbox.** Runs as the server user |
| Python | `python -I` subprocess, temp file, timeout, sanitised env | Same privileges as the server |
| Human gate | Risk tiers + approval modes + server policy floor | Timeout / disconnect ⇒ deny |
| API | Optional `FRIDAY_API_TOKEN` on mutating and sensitive read endpoints | Single shared token, no per-user identity |
| Secrets | Keys live in env/server only; `/api/config` is sanitised | Process environment |

The honest summary: **FRIDAY is safe for a trusted operator, not for arbitrary internet users.** The
approval gate is the real security boundary; everything else is defence in depth. Containers are the
correct deployment unit for anything exposed.

---

## 7. Interface

The HUD is intentionally dependency-free:

- `index.html` — structure, inline SVG icon sprite, semantic landmarks, ARIA on live regions.
- `style.css` — design tokens, ambient reactor light, responsive grid, reduced-motion support.
- `app.js` — state machine, SSE reader, markdown renderer (HTML-escaped before formatting),
  approval cards, gauges, session browser, memory bank, Web Speech API integration.

Token streaming is rendered through `requestAnimationFrame` rather than per-delta DOM writes. The
markdown renderer is deliberately small: fences, headings, lists, tables, quotes, links, emphasis.
It escapes first and only then inserts a known set of tags, so model output cannot inject script.

There is no framework and no build step; editing the three files and refreshing is the entire
development loop.

---

## 8. Testing

70 tests across five suites:

- **config** — precedence, provider resolution, secret hygiene.
- **memory** — lifecycle, ordering, FTS, cascading deletes, injection-safe queries.
- **tools** — argument validation, jail escapes, denylist, math AST safety, SSRF, HTML extraction,
  timeouts, plugin-disabled tools.
- **engine** — real tool execution, approval grant/denial, pre-cancellation, retries, partial
  failures, step budgets, memory injection, provider errors surfaced.
- **server** — static assets, chat SSE, session CRUD, memory API, auth enforcement, approver
  resolution and timeout.

Run with `pytest -q`; lint with `ruff check friday tests`.

---

## 9. Known trade-offs

1. **One shared token, no users.** Fine for a personal assistant; insufficient for a team.
2. **Sequential tools.** Simpler approvals, slower agents.
3. **Truncation, not compaction.** Long sessions lose old context at the message cap.
4. **Approval timeout is hard-coded** (190 s) and not configurable yet.
5. **Simulation core is rule-based.** It routes real tools but is not a language model and should
   never be mistaken for one.
6. **No streaming token accounting mid-run.** Usage arrives at the end of a provider turn.

Each is a conscious v1 choice, listed so the next engineer can disagree with the reasoning rather
than rediscover the gap.
