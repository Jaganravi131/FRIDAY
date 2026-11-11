# The unified build

*What was combined, why, and what was deliberately left on the cutting-room floor.*

`docs/INTEGRATION.md` contains the measurement: file counts, containment, the
eleven-conflict `merge-tree`, and the capability split. This document is the
decision that followed, written as a map of the code that now exists.

---

## 1. The shape

Two lineages were cut from the same one-line commit and never shared a commit
again:

| Lineage | What it actually was | Fate |
|---|---|---|
| **Cognition** (`arena/ad9dff8f-friday`) | memory compiler, bi-temporal facts, retrieval, ledger, policy, connectors, heartbeat, 34 test modules | **became the base** |
| **Interface** (`arena/95e7eb54-friday`) | HUD, SSE, approvals UI, providers, extra tools, Docker/CI | **became the surface** |

The resulting tree is not a hybrid. It is the cognition tree with a web package
added, one provider added, four tools added, and a handful of callbacks threaded
through the existing loop. Everything else in the interface branch was retired.

The rejected alternative was a textual merge. `git merge-tree` reported 11
conflicts — ten `add/add` at package roots (`.github/workflows/ci.yml`,
`.gitignore`, `friday/{__init__,__main__,cli,config}.py`, `friday/llm/__init__.py`,
`pyproject.toml`, `tests/conftest.py`, `tests/test_cli.py`) plus `README.md` — because
both trees independently claimed the top-level `friday` package (111 modules vs 30).
Resolving those conflicts produces a tree where neither lineage's tests can be
trusted, which is worse than either branch alone.

---

## 2. The three seams

Everything that crossed from the interface lineage into the core crosses at one of
three places, and each is small enough to review in one sitting.

### Seam 1 — transport (`friday/web/`)

`friday/web/app.py` is a FastAPI application with SSE. It:

* runs `friday.agent.loop.Agent` on a worker thread through
  `asyncio.to_thread` — the loop is synchronous by contract, and the CLI/eval
  callers depend on that;
* translates agent callbacks into the HUD's event vocabulary
  (`open`, `token`, `tool_call`, `tool_result`, `approval_required`,
  `approval_result`, `notice`, `error`, `done`, `close`, plus a 12 s `: ping`);
* blocks the **worker thread** — never the event loop — on a `threading.Event`
  until `POST /api/approvals/{token}` resolves, denies after 190 s, and denies
  every pending approval when the run ends;
* creates sessions with `scope="remote"` so `friday audit --today` can tell network
  turns from keyboard turns. A pre-gateway database whose CHECK constraint predates
  `remote` falls back to `interactive` instead of failing the request.

`friday/web/adapters.py` translates between the HUD's simple model and the real
store:

| HUD | Core | Translation |
|---|---|---|
| session title | `sessions.topic` or first user turn | derived on read; never invented in the DB |
| `message_count` | `COUNT(turns)` | — |
| `updated_at` | `MAX(turns.ts)` | falls back to `sessions.started_at` |
| memory text | `facts.predicate` + `facts.object` | `predicate: object` |
| add memory | `assert_fact(Fact(predicate="note"))` | redaction + reconciliation + Markdown + FTS + supervision all happen |
| delete memory | `retract_fact()` | strikes the Markdown line, closes the row, keeps FTS for history |

The last two rows matter. The HUD's textarea cannot produce the
`(subject, predicate, object)` triple the fact compiler wants, so it says so
honestly: the predicate is `note`. It does **not** bypass `assert_fact`, which is
the single chokepoint every other write passes through.

### Seam 2 — providers (`friday/llm/anthropic.py`)

The interface lineage's Anthropic provider was built on `httpx`, its own `Message`
type and an async-generator protocol. Porting it as-is would have created a second
agent protocol inside one loop.

The port instead implements the mature `Client` protocol with stdlib `urllib`:

```python
def chat(messages, *, tools=None, temperature=0.2, max_tokens=768,
         on_token=None) -> Response
```

Two real protocol mismatches are handled explicitly:

* **Tool results.** Anthropic requires `tool_result` to answer a preceding
  `tool_use` with the same id. FRIDAY's transcript keeps only
  `{"role": "tool", "name", "content"}`. The converter reconstructs a paired
  `assistant(tool_use)` → `user(tool_result)` exchange, grouping consecutive tool
  messages into one turn.
* **Streaming tool calls.** `input_json_delta` fragments are not valid JSON until
  the block closes. They are accumulated per block index and parsed once — the
  same discipline the OpenAI reader uses, for the same reason.

Provider selection lives in `get_client`: explicit env (`FRIDAY_LLM_PROVIDER`) →
local server that answers `/health` → simulation core. A configured-but-keyless
Anthropic provider prints one line and falls back rather than handing back a client
that fails on first use.

### Seam 3 — tools (`friday/agent/extra_tools.py`)

The core already had the workspace tier (`ws_list/read/write/shell`) behind
`FRIDAY_TOOLS_PHASE=2`. Only four tools were genuinely missing:

| Tool | Why it exists | Phase | Sense |
|---|---|---|---|
| `file_edit` | exact, unique substring replacement — re-emitting a whole file to change a line is how you lose the rest of it | 2 | `workspace.files` |
| `file_grep` | regex search without `ws_shell`'s single-command allowlist | 2 | `workspace.files` |
| `math` | deterministic arithmetic; a small model cannot be trusted with `17*42+9` | 1 | none |
| `clock` | explicit time, not only when the intent classifier notices | 1 | none |

All paths go through `workspace._confined`, the existing resolve-then-contain
check. `file_edit` refuses ambiguous matches rather than guessing which one was
meant.

The phase ceiling is preserved: phase 0 is still four tools, not eleven. Tool
definitions are a fixed per-turn token tax, and the architecture doc defends the
small number.

---

## 3. What was deliberately left behind

| Artifact | Why it was not ported |
|---|---|
| `friday/engine.py` (the interface lineage's loop) | Two loops, two behaviours. The core loop is stricter: policy before confirmation, rung-0 capping before history, explicit outcome labels. |
| `friday/memory.py` (SQLite + FTS5) | Strictly weaker than the bi-temporal compiler. Keeping it as a second store means two truths. |
| `friday/server/app.py` (FastAPI routes) | Its JSON contract was kept; the implementation was rewritten against the real store. The old routes read a schema that no longer exists. |
| `CORS allow_origins=["*"]` | A page on another origin must not be able to drive an agent that can read your memory. Same-origin only. |
| `friday/tools/files.py` | Duplicated `ws_*` with a different confinement implementation. Only the two operations the core lacked were ported. |
| `tests/test_core.py` etc. (interface tests) | They tested the retired engine/memory. Equivalent coverage now lives in `test_web_api.py`, `test_web_approvals.py`, `test_llm_anthropic.py`, `test_extra_tools.py`. |

---

## 4. Callbacks added to the core

The loop changes are small and all optional:

```python
Agent(..., confirm_all=False)          # "always" approval mode
Agent.run(..., on_token=None,          # stream deltas to the HUD
                should_stop=None)      # cooperative cancellation
```

* `on_token` is passed to `client.chat` **only if** the client's signature declares
  it (`inspect.signature`), so every existing test double and third-party client
  keeps working unchanged.
* `should_stop` is checked at the top of each iteration; a stopped run keeps its
  partial text, persists the turn, and records `outcome="cancelled"` in the trace.
* `confirm_all` asks on `allowed` calls too. It does **not** ask on `denied` ones:
  a policy denial is final.
* `tool_call` / `tool_result` events were added for the timeline, emitted at the
  point the call actually runs (so an `approval_required` always precedes the
  `tool_call` it gates), and emitted for denials/declines too.

The five SQL/behavioural invariants the core tests pin — policy ordering, tool
message roles, outcome vocabulary, rung-0 capping, both halves of the exchange in
`turns` — are untouched.

---

## 5. Packaging

* **Core `dependencies = []`** is preserved. `pip install friday` still installs
  nothing, and `friday serve --classic` still runs the stdlib gateway.
* `[serve]` gains what the HUD already declared (`fastapi`, `uvicorn`).
* `[dev]` gains `httpx` for Starlette's `TestClient`.
* `friday.web.static` is package data, so a wheel install has the HUD.
* `friday serve` prefers the HUD when importable and falls back with a one-line
  explanation when it is not. `--classic` is explicit.

---

## 6. Verification

| Gate | Result |
|---|---|
| `pytest` | **892 passed, 2 xfailed** (was 839 + 2) |
| `ruff check friday tests` | clean (pyflakes + import order) |
| wheel build + install into a scratch venv | checked by `make wheel` |
| gateway tests (`test_gateway_validation.py`) | untouched and passing — the stdlib gateway still owns its 400 paths |
| HUD smoke | `POST /api/chat` streams `open → notice → token… → done → close`; approval flow blocks, resolves, and denies correctly |

The two xfails are pre-existing and documented; they are not new debt.

---

## 7. The one open question

`friday-dev`'s synonym bridge and the mature concept lexicon disagree on held-out
paraphrases (0.25 vs 0.58, 0.42 vs 0.50) while the committed set says the opposite
(1.00 vs 0.963). The module ships unreferenced. Settling it requires a blind
holdout of 40 paraphrases and 10 unanswerable questions, run by a human who has not
seen the lexicon. Until then, the mature lexicon is what retrieval uses — it wins on
the only measurements that were not used to build it.
