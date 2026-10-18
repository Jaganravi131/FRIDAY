# 01 — Research Landscape (September 2026)

What actually exists, what to steal, and what to ignore. Everything here was verified against
current sources — see [11-research-sources.md](./11-research-sources.md).

---

## 1. The four projects you must study

### 1.1 OpenClaw (~247K stars) — the pattern library
Peter Steinberger's local-first agent runtime. Exploded from 9K → 60K stars in days.

**Steal these patterns:**
- **The file-based memory layout.** `SOUL.md` (personality), `AGENTS.md` (rules),
  `MEMORY.md` (durable facts), `HEARTBEAT.md` (proactive checklist), `memory/YYYY-MM-DD.md`
  (daily append-only logs). Simple, human-readable, git-versioned. This is now the
  de-facto convention across the whole personal-agent ecosystem.
- **The Heartbeat.** A cron-triggered *agentic turn* (default every 30 min). It reads
  `HEARTBEAT.md`, decides if anything needs attention, acts or replies `HEARTBEAT_OK` —
  which the gateway **suppresses and never delivers**. This is *the* proactivity pattern.
- **Daily logs are NOT auto-injected.** They're retrieved on demand via memory tools.
  Keeps routine conversation lean.

**Do NOT copy:**
- The capability surface. OpenClaw gives the agent a terminal, arbitrary code execution, and a
  plugin marketplace. That is a large blast radius for something running unattended with your
  real accounts. Security researchers documented serious issues; the ecosystem found
  **confirmed malicious payloads in 76 of 3,984 agent skills**.

---

### 1.2 Hermes Agent (Nous Research, MIT, Feb 2026) — the learning loop
The most important reference for *your* stated goal ("an intelligent system that builds itself").

**Three-layer memory:**
| Tier | Mechanism | Stores | Constraint |
|---|---|---|---|
| 1 — High-signal state | `USER.md` + `MEMORY.md` | profile, preferences, conventions | **1,375 + 2,200 chars** — guaranteed always in context |
| 2 — Cross-session search | SQLite + FTS5 | full conversation history | no limit |
| 3 — External | mem0 / vector stores / Honcho | optional | provider-dependent |

Note the **character budgets** on Tier 1. That's a deliberate, hard constraint — it forces
high-signal density. Steal that discipline.

**The closed learning loop** (their actual differentiator):
1. Completes a task
2. **Autonomously writes a reusable `SKILL.md`** from the experience
3. **Refines that skill during subsequent use**
4. Periodic self-nudges to persist knowledge to `MEMORY.md`
5. FTS5 cross-session recall
6. Honcho dialectic user modelling (12 identity layers, evolves over time)

Claimed result: agents with 20+ self-created skills complete similar tasks **~40% faster**
(40% less token consumption and wall-clock — *not* 40% better output). Be precise about that
distinction when you report your own numbers.

Also: **no external vector DB required**. FTS5 + tiny markdown files. That is exactly right
for a 16 GB single-user machine.

---

### 1.3 `roiguri/jarvis-agent` — the safety counterweight
A hand-rolled LangGraph agent over Gemini that **deliberately rejects OpenClaw's capability
surface**. Read this repo for its discipline:

> *"Where OpenClaw gives an agent a terminal, arbitrary code execution, and a plugin
> marketplace, Jarvis has none of that. Every action Jarvis can take is an explicit,
> registered tool — and nothing else. There is no shell, no exec, no filesystem access beyond
> a path-validated memory directory. Its blast radius is exactly its tool set."*

Two properties worth copying exactly:
- **Sandboxed** — blast radius = tool set, by construction.
- **Deterministic** — hand-rolled `StateGraph` (not a prebuilt ReAct loop), low temperature,
  per-scope tool whitelists, explicit skill activation. Auditable.

**Also steal:** the **two-scope runtime sharing one memory**. An *interactive* scope
(conversational, sees today's proactive notifications) and a *heartbeat* scope (terse, sees
today's chat, stays silent unless its tick-ack says notify). Awareness flows both ways so the
two never duplicate work. That solves the "FRIDAY tells me something it already told me" bug.

And **progressive tool disclosure**: tools grouped into *skills* that stay **hidden** until the
model calls `activate_skill`. Keeps the prompt small and the tool surface relevant. Critical on
a small local model where every tool definition is a real tax.

---

### 1.4 screenpipe (YC S26, ~20K stars) — don't build ambient capture
Continuous local screen + audio capture. **Just use it.**

**Architecture worth understanding:**
1. **Event-driven capture** — listens for OS events (app switch, click, typing pause, scroll,
   clipboard) and captures **only when something meaningful happens**. Screenshot paired with
   the **accessibility tree** (structured text the OS already knows: buttons, labels, fields),
   falling back to **OCR** when a11y data isn't available (remote desktops, games).
2. Local **Whisper** for audio; speaker diarisation.
3. **SQLite + FTS5** locally; JPEGs on disk. ~300 MB / 8 hr vs ~2 GB naive.
4. REST API on `localhost:3030`.
5. **MCP server** — `npx screenpipe-mcp`.
6. **Pipes** — scheduled AI agents defined as markdown files.

Cost: **5–10% CPU, 0.5–3 GB RAM, ~5–20 GB/month.** Works on Windows/macOS/Linux, offline.

**The genuinely important idea — deterministic AI data permissions:**
Per-pipe YAML frontmatter (`allow-apps`, `deny-apps`, `deny-windows`, `allow-content-types`,
`time-range`, `allow-raw-sql`, `allow-frames`) enforced at **three OS-level layers**:
skill gating (the AI never *learns* denied endpoints exist) → agent interception (blocked
before execution) → server middleware (per-pipe cryptographic tokens).

> *"Enforced deterministically at the OS level via three layers — not by prompting the AI to
> behave. Even a compromised agent cannot access denied data."*

**This is exactly the model for FRIDAY's Sense Registry.** ([09](./09-security-privacy-trust.md))

Caveats: telemetry (PostHog + Sentry) is **on by default** — disable in Settings → Privacy.
The signed desktop app is now subscription (~$25/mo); source is available for personal
non-commercial use.

---

### 1.5 Stanford OpenJarvis (Apache-2.0, Mar 2026) — the local-first proof
From Stanford's Hazy Research + Scaling Intelligence labs ("Intelligence Per Watt").

**The headline finding that should shape your whole design:**
> Local models already handle **88.7% of single-turn queries.**

OpenJarvis runs inference, agents, memory *and learning* fully on-device and lands within
**3.2 pp** of the best cloud model at **~800× lower marginal API cost** and **~4× lower
latency**. The residual gap concentrates in reasoning- and research-heavy tasks — exactly the
~11% you should escalate.

**Five swappable primitives, one declarative TOML "spec":**
`Intelligence` (model/weights/quant) · `Engine` (Ollama/vLLM/llama.cpp/Apple FM) ·
`Agents` (ReAct or CodeAct loop, tool policy, turn limits) · `Tools & Memory` (25+ connectors,
32+ channels, native MCP, interchangeable memory backends) · `Learning` (LoRA / DSPy / GEPA /
LLM-guided spec search).

Two specs can differ *only* in model+engine and run the same behaviour on a Mac Mini and a
workstation without rewriting prompts. **Adopt this discipline** — it's how you swap your
Kaggle-fine-tuned GGUF in without touching a prompt.

**LLM-guided spec search** (their real contribution): a frontier cloud model acts as a *teacher
at search time only*, reads your traces, diagnoses failure clusters, proposes edits across all
five primitives. An edit is accepted only if it improves the target failure cluster without
regressing elsewhere — they call this **the gate** (default tolerance **1%**). Recovers
**13–32 pp** of the local↔cloud gap vs ~5 pp for prompt-only optimizers, at 7–11× lower cost.
At 100 queries/day the amortised teacher cost drops below **$0.001/query** within six months.

> **This "gate" is the direct ancestor of FRIDAY's Constitutional Eval Gate.** ([06](./06-INNOVATION-self-improvement-loop.md))

**The swap test** (brutal and important): dropping Qwen3.5-9B into existing frameworks
(OpenClaw, Hermes) loses **25–39 pp** of accuracy. Under an OpenJarvis spec, the residual drop
shrinks to **5.6–16.5 pp** — recovering **56–77%** of the portability loss.
**Translation: scaffolding matters as much as the model.** A separate Render benchmark found the
same model (Opus 4.5) varying by **17 problems on SWE-bench** purely from agent scaffolding.

That is your justification for spending Phase 1–2 on the Memory Compiler and Attention Ledger
instead of chasing a bigger model.

---

## 2. Memory frameworks: the honest comparison

| | **Mem0** | **Zep / Graphiti** | **Letta (MemGPT)** | **Cognee** |
|---|---|---|---|---|
| Architecture | Hybrid vector + graph + KV | **Temporal knowledge graph** | OS-inspired core/recall/archival | Typed KG (ECL pipeline) |
| LongMemEval (GPT-4o) | **49.0%** | **63.8%** | N/A (runtime) | — |
| Retrieval latency p95 | 7–10 s (write-heavy path) | ~200 ms–1 s | variable | higher |
| Temporal reasoning | **Weak** — facts *updated*, not versioned | **Strong** — validity intervals on edges | Indirect (agent self-manages) | Structural, not time-windowed |
| Standout | OpenMemory MCP local server | Fact invalidation | **Sleep-time compute** | Fully local, 6-line setup |
| Self-host | Yes (OSS core) | Yes (Graphiti CE) | Yes (server is OSS) | Yes |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

**Three findings that change FRIDAY's design:**

1. **The 15-point LongMemEval gap is entirely about time.** Zep stores *"Kendra loves Adidas
   (as of March 2026)"* as a fact with a **validity window**. When it changes, Graphiti
   **invalidates the old edge with a timestamp rather than overwriting**. Ask *"where did they
   live last year?"* and the graph answers; a flat vector store cannot. → **FRIDAY adopts
   bi-temporal facts.** ([03](./03-memory-architecture.md))

2. **Letta's "sleep-time compute" is the idea to steal.** A background *sleeper* agent
   reorganises memory while the primary agent is idle, so **consolidation cost never lands in
   the user-facing latency budget.** Most memory systems do their thinking on the critical
   path. → **FRIDAY's Dreaming phase.** ([08](./08-proactive-heartbeat.md))

3. **Retrieval latency by strategy** — budget against these:
   | Strategy | Typical latency |
   |---|---|
   | Vector-only | 10–50 ms |
   | Graph traversal | 50–150 ms |
   | Multi-strategy parallel | 100–600 ms |
   | LLM synthesis ("reflect") | 800–3,000 ms |
   | Memory **ingestion** (write path) | 500–2,000 ms |

   → **Never put ingestion or LLM-synthesis on the turn path.** Both go to the Dreaming phase.

**Verdict for FRIDAY: build it, don't adopt it.** Mem0/Zep/Letta all assume a server and a
vector DB. You have 16 GB, one user, and a hard requirement that memory be human-editable.
The right build is **Markdown-as-source-of-truth + SQLite FTS5 + sqlite-vec + a bi-temporal
fact table** — Zep's *semantics* with Hermes's *substrate*. Steal the model, not the dependency.

> ⚠️ **The practitioner warning worth heeding.** From a developer who'd seen a dozen of these:
> *"They all start with chat logs as the corpus and end up with an agent that knows your venting
> habits but couldn't fill out a basic shipping form. The corpus that actually moves the needle
> is the browser profile, autofill tables, history, bookmarks. That's where your real
> preferences live, not where you complain about them. The three-layer memory architecture
> matters way less than what you choose to feed it on day one."*
>
> **Corollary: your Day-1 corpus choice matters more than your memory architecture.**
> Feed FRIDAY your *behaviour* (screenpipe, browser history, files, calendar), not just your
> *conversations*.

---

## 3. Realtime voice: what's real in 2026

### The measured latency budget
| Component | Latency (p50/p95) |
|---|---|
| **Human turn-gap baseline** (Stivers et al., PNAS 2009) | **200–250 ms** ← the number to beat |
| LiveKit SFU edge audio (Opus 48k, RED/FEC) | 15–25 ms one-way |
| Deepgram Nova-3 streaming STT | 110–160 ms |
| Groq LPU (Llama-3.3-70B) TTFT | 65–95 ms |
| Cartesia Sonic-2 / ElevenLabs Flash v2.5 TTFA | **85–135 ms** |
| Fixie Ultravox v0.7 (audio-in → text-out) | 150–220 ms TTFT |
| Full duplex speech-to-speech (Moshi / OpenAI Realtime) | **300–550 ms end-to-end** |

Humans take turns at ~200 ms. A cascade pipeline that lands under ~800 ms feels natural.
Above ~1.5 s it feels like a phone tree.

### Framework choice
| Framework | Abstraction | Transport coupling | Realtime models covered | Best for |
|---|---|---|---|---|
| **Pipecat** | Frame-processor pipeline | **Transport-agnostic** | OpenAI, Gemini, Nova Sonic, Azure Voice Live + cascaded STT/LLM/TTS | Mixing realtime models with classic cascades in one codebase |
| **LiveKit Agents** (1.5.x) | `AgentSession` / `RealtimeModel` plugin per provider | **Coupled to LiveKit rooms/SFU** | OpenAI, Gemini (incl. video), Nova Sonic, xAI | Concurrency, telephony/SIP, multi-party, scale |
| **TEN Framework** | Extension graph, pluggable turn-detection | Loosely coupled | Growing; strong multimodal/vision | Voice+vision where turn-detection must be swappable |
| **OpenAI Realtime** | Single-vendor speech-to-speech | Managed | `gpt-realtime-2.1` | Fastest path to highest conversational quality |

LiveKit Agents hit **1.0 in April 2025**; the 1.5.x line ships an **adaptive turn-detection
model** and **native MCP tool support**. OpenAI's Realtime API went **GA 28 Aug 2025** with
`gpt-realtime`, SIP calling, remote MCP, and image inputs. LiveKit raised a **$100M Series C at
a $1B valuation in Jan 2026**.

> **FRIDAY's choice: start with Pipecat** (transport-agnostic, lets you mix a local cascade
> with a cloud realtime model), **graduate to LiveKit Agents** when you want phone/SIP,
> multi-room, or the agent-as-participant model. Both are Apache-2.0-ish and Python-native.

### The three voice bugs that will eat your week

**Bug 1 — Barge-in needs exact milliseconds, not "cancel".**
OpenAI's community documented this repeatedly: cancelling the response stops *future* audio,
but the model's conversation history still contains the **full, uninterrupted response it
generated**. As far as the model knows, you *heard* all of it — including the part that was
cut off and never played. Correct sequence:
1. On `input_audio_buffer.speech_started` → stop local playback **immediately**
2. Measure **exactly how many ms of assistant audio were actually played**
3. Send `conversation.item.truncate` with that item's ID, content index, and
   `audio_end_ms` = **actual played duration** (not full length, not zero)

Gemini Live handles this differently: the **server** sets an `interrupted` flag; the client just
clears its playback queue. There is no upstream truncate equivalent — an adapter must locally
discard queued audio.

**Bug 2 — Failing to clear the client playout buffer.** Cited as *"the most common interruption
bug we fix on prototypes."* The model has usually streamed several hundred ms ahead of what the
caller has heard.

**Bug 3 — Echo cancellation.** Non-negotiable on the client. Without AEC the agent hears its
own voice and **barges in on itself**. Also: capture at **16-bit PCM 24 kHz** (OpenAI's native
rate) — 48 kHz wastes bandwidth and adds resampling latency. Gemini Live is fixed at
**16 kHz input / 24 kHz output**, so a cross-provider adapter needs anti-alias-filtered
resampling.

### Turn detection: the actual lever
Don't use fixed silence timeouts. Use **semantic turn detection** — a model that evaluates
*linguistic completeness*:
```python
turn_detection=MultilingualModel(),   # LiveKit's adaptive model
min_endpointing_delay=0.4,            # floor: don't clip mid-sentence pauses
max_endpointing_delay=3.0,            # ceiling: for trailing-off speakers
preemptive_generation=True,           # start LLM inference on PARTIAL transcripts
```
`preemptive_generation` is the trick that gets you under 400 ms: **start generating before the
turn is confirmed complete**, and cancel if the user keeps talking.

**Barge-in debounce physics:** Silero VAD runs on 30–32 ms frames. Set
`interrupt_speech_duration = 0.25–0.35 s` so coughs and background clicks don't false-trigger.
LiveKit's stack then mutes the media track, cancels LLM token streaming, and clears the TTS
buffer in **under 40 ms**.

**Pre-warm the VAD.** Instantiating Silero inside the WebRTC entrypoint adds **~1 s cold start**
to the first turn. Load it once per worker process at container init.

### Session limits (matters for an always-on assistant)
| | OpenAI Realtime | Gemini Live |
|---|---|---|
| Max session | 60 min/connection, no documented resumption | 15 min audio-only / **2 min audio+video** default |
| Extension | — | `contextWindowCompression`; `SessionResumptionUpdate` with a token valid ~2 h; `GoAway` warns of termination |
| Turn detection | `semantic_vad` (default, model-judged), `server_vad` (energy), or `null` (push-to-talk) | `automaticActivityDetection` with start/end sensitivity LOW/MED/HIGH, `silenceDurationMs`, `prefixPaddingMs` |
| Native transcription | input via whisper / gpt-4o-transcribe | **bidirectional native** — `inputAudioTranscription` + `outputAudioTranscription`, no sidecar ASR |
| Reasoning control | none exposed | `thinkingConfig` — ⚠️ a mis-set `thinkingLevel` **silently degrades barge-in** |
| Audio format | PCM16, configurable, 24 kHz typical | PCM16 fixed 16 kHz in / 24 kHz out |

→ **Design implication:** an always-on FRIDAY *must* have a **session resumption layer**.
Gemini's 2-minute audio+video limit means you cannot hold a permanent video session; you need
duty-cycled vision with resumption tokens.

### Security note on Realtime APIs
The OpenAI Realtime API **lacks client-side authentication** — it is insecure to connect
directly from a browser. Two mitigations: (a) WebRTC with a **short-lived ephemeral token
minted server-side**, or (b) a **relay** where the browser talks to your server and your server
holds the real key. → FRIDAY's Presence Fabric is a relay by design. Never ship an API key to a
client.

---

## 4. MCP: the state of the standard

**Current spec: `2026-07-28`** — the biggest change since launch.

| Version | Status | Headline |
|---|---|---|
| 2024-11-05 | Final | Initial: client-server, JSON-RPC 2.0, tools/resources/prompts, stdio + HTTP+SSE |
| 2025-03-26 | Final | OAuth 2.1, **Streamable HTTP**, tool annotations, audio content, batching |
| 2025-06-18 | Final | Structured tool output, **elicitation**, resource links, batching removed, security best practices |
| 2025-11-25 | Final | OIDC Discovery, icon metadata, Client ID Metadata Documents, experimental **tasks**, JSON Schema 2020-12 |
| **2026-07-28** | **Current** | **Stateless core**, Multi Round-Trip Requests (MRTR), **MCP Apps**, Tasks extension, extensions framework, auth hardening, formal deprecation policy |

**What "stateless core" means concretely:** the `initialize`/`initialized` handshake and the
`Mcp-Session-Id` header are **gone**. Every request is self-describing — protocol version,
client info and capabilities travel inline in a `_meta` field on every request. Standard HTTP
headers (`Mcp-Method`, `Mcp-Name`) allow routing without deep packet inspection. Scales on
ordinary round-robin load balancers.

**Deprecations to avoid:** `Roots`, `Sampling`, and `Logging` are formally deprecated. New
implementations should use **tool parameters**, **direct provider APIs**, and **stderr /
OpenTelemetry** respectively.

**Ecosystem scale:** ~97M monthly SDK downloads; 10K+ active public servers (Anthropic, Dec
2025); official Registry ~9,652 latest records; 15,926 GitHub repos with the `mcp-server` topic;
`modelcontextprotocol/servers` at 86K stars. Governed by the **Agentic AI Foundation** (Linux
Foundation), 460+ members.

**⚠️ The security reckoning — read this before installing any server:**
- **30+ CVEs filed Jan–Feb 2026**, ranging from path traversal to a **CVSS 9.6 RCE**.
- AgentSeal found **security issues in 66% of servers** they scanned.
- Enkrypt AI found **critical vulnerabilities in 1/3 of the top 1,000** MCP servers (Oct 2025).
- Snyk found **confirmed malicious payloads in 76 of 3,984 agent skills**.
- Registry counts include many abandoned/duplicate listings.

**FRIDAY policy:** allowlist, not marketplace. Every MCP server is (a) pinned by version,
(b) scanned (`mcp-scan`/`agent-scan` from Invariant Labs → Snyk, or Cisco's open-source MCP
Scanner), (c) run in the sandbox, (d) granted a scoped token. Default-deny.

**Related protocols:** **A2A** (agent↔agent, Google → Linux Foundation, v1.0 with gRPC, 50+
launch partners) and **ACP** (IBM, converging with A2A). Analogy that holds: **MCP is USB-C
(device connection), A2A is TCP/IP (peer communication).** FRIDAY needs MCP now; A2A only if
you go multi-agent later.

---

## 5. Context engineering: the 2026 consensus

### The four operations
**Write** (persist outside the window) · **Select** (retrieve only what's relevant now) ·
**Compress** (reduce tokens, keep signal) · **Isolate** (give sub-tasks their own clean window).

Every framework maps onto these — Anthropic's framing (structured note-taking / just-in-time /
compaction / sub-agents), LangChain's (Store & State / dynamic selection / lifecycle
summarisation middleware / handoffs), GraphRAG's (long-term memory / hybrid retrieval / context
pyramid / MCP).

### The token budget
```
Context window (~200K)
├── Fixed costs        system prompt + AGENTS.md (LEAN) + tool definitions (minimal set)
├── Working set        JIT-retrieved code/docs + recent transcript + scratchpad pointers
└── Reserved headroom  space for the next tool result + reasoning
```
> *"An agent that runs its window to 99% full has no room to think: the next big tool result
> forces a panicked compaction at exactly the wrong moment."*

**Budget to a fraction of the rated window, not its ceiling.** Chroma's data shows accuracy
loss beginning around **50,000 tokens of genuinely relevant information** — even inside windows
rated 200K–1M. Retrieved knowledge should be **2,000–4,000 tokens** if you rerank and filter
properly; easily 10× that if raw chunks get dumped in.

### The compaction decision tree
```
Task horizon 1–2 turns?  → just prompt it well + hand it the right 1–2 files. STOP.
Long-running?
  ├─ Grabs wrong/too many files   → structural retrieval, load JIT
  ├─ Degrades after many turns    → offload >20K to disk + cap tool output + compact at 85%
  ├─ Dithers between similar tools→ CUT the tool set; load tool groups dynamically
  └─ Ignores conventions          → trim AGENTS.md to durable rules; layer the rest JIT
```

**Set the compaction threshold at ~70–75% of the window, not 95–98%.** Compacting at 98%
leaves the model "context-anxious" and produces compressed, incomplete summaries — the failure
Devin documented. Early compaction gives it output tokens to write a good summary.

**Offload long-lived facts to external memory ON WRITE, not at compaction time.** Architectural
decisions, preferences, environment constants and conventions go to structured memory files the
moment they're established. Relying on compaction to capture them produces fragile,
*retroactive* extraction. External memory must be the source of truth for durable facts —
never the compaction summary.

### For code specifically
**Structural retrieval beats embeddings.** Returning a symbol's actual definition plus its call
sites lifted **precision@5 from 0.14 → 0.48** in Sourcegraph benchmarks. Use embeddings for
prose; use code intelligence (LSP/tree-sitter/ctags) for code.

### The JIT retrieval pipeline that actually works
Four parts, all small, and skipping any one is where rot creeps in:
1. **Intent classifier** — is retrieval even needed this turn?
2. **Query rewriter** — conversational turn → search-ready query
3. **Retrieve wide → rerank → ship narrow**
4. **Freshness gate** — drop chunks whose source changed since embedding

And the most important detail is **the tool docstring**:
```python
@tool
def search_policies(query: str) -> list[dict]:
    """Search the policy corpus. Use ONLY when the user asks about
    refunds, shipping, returns, or warranty. Returns at most 3 chunks."""
```
> *"Without that sentence, the model will call the retriever on every turn 'just to be sure',
> and your JIT system quietly turns back into pre-packing."*

Plus a **hard score floor** — better to return `[]` than noise.

---

## 6. Self-improvement: the taxonomy (this is your research surface)

Four categories, ordered by risk. **Work down the list, not up.**

| Category | What changes | Frameworks | Best result | Maturity |
|---|---|---|---|---|
| **1. Prompt self-optimisation** | instructions + examples; **weights frozen** | DSPy, TextGrad, **GEPA**, Trace | GEPA: **+10% avg over RL, 35× fewer rollouts** | ✅ pip-installable, MIT, production |
| **2. Agent self-improvement** | reasoning strategy; model frozen | Reflexion, LATS, AFlow, EvoAgentX | LATS **94.4%** pass@1 HumanEval; DGM **20%→50%** SWE-bench | ⚠️ reference code, adapt don't depend |
| **3. Training-time** | **model weights** | SEAL, SPIN, Self-Rewarding | SEAL: model writes its own finetuning data, **32.7%→47.0%** QA, beats GPT-4 data | 🔬 research-grade |
| **4. Architecture/agent search** | the agent's own code/design | ADAS, **Darwin Gödel Machine**, EvoAgentX | DGM self-modified **20%→50%** SWE-bench | 🔬 risky |

**Key numbers:**
- **DSPy MIPROv2** raised ReAct accuracy on HotPotQA **24% → 51%** with gpt-4o-mini — a 2×
  improvement from *prompt optimisation alone*, no model change.
- **DSPy** improved prompt-evaluation accuracy **46.2% → 64.0%** and refinement **85.0% → 90.0%**.
- **GEPA** (Databricks/UC Berkeley, **ICLR 2026 Oral**) outperforms GRPO by **10% average, up to
  20%** on specific tasks, using **up to 35× fewer rollouts**, and beats MIPROv2 by >10%.
- **TextGrad** — published in **Nature**; +20% on LeetCode-Hard. Static since mid-2025 (stable,
  not actively developed).
- **Reflexion** — verbal self-reflection in episodic memory → **91% pass@1 HumanEval** vs GPT-4's
  80% baseline.
- A production reflection loop delivered **+34.2%** accuracy: runtime reflection (3 iterations
  max) **+12.1%** in 2 days; failure-trace clustering + targeted fixes **+2.6%** on a weekly cycle.

**GEPA's actual mechanism** (worth understanding deeply — it's the core of FRIDAY's Dreaming):
treats prompts as *organisms*. Samples execution trajectories → **reflects on them in natural
language** to diagnose failures → proposes mutations → selects along a **Pareto front** (not
greedy, to avoid local optima). Candidates derive from ancestors via reflective mutation or
crossover, accumulating lessons along a genetic tree. Train/validation split prevents
overfitting. Multi-objective (Pareto-optimal across grader dimensions, not a single average).

> *The key innovation is **reflection**: instead of random mutations, GEPA uses the LLM to reason
> about **why** a prompt failed and propose targeted fixes.*

```python
import gepa
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a summarization assistant..."},
    trainset=train_data,
    valset=val_data,
    adapter=your_eval_adapter,     # bridges YOUR graders to GEPA
    reflection_lm="gpt-5",         # strong model as teacher — search time only
    max_metric_calls=20,
    track_best_outputs=True,
)
best_prompt = result.best_candidate["system_prompt"]
```

### ⚠️ The critical caveat nobody mentions
> **LLMs cannot reliably self-correct reasoning without external feedback** — and performance
> *sometimes degrades* after self-correction (Huang et al., ICLR 2024). Pure intrinsic
> self-reflection is unreliable.

**Effective self-improvement requires external signals**: test results, user ratings, automated
metrics. This is *precisely* why FRIDAY's Constitutional Eval Gate exists — you must supply the
external signal, and the agent must not be able to edit it. ([06](./06-INNOVATION-self-improvement-loop.md))

### Relevant 2026 papers for your research surface
| Paper | Venue | Idea |
|---|---|---|
| **GEPA: Reflective Prompt Evolution Can Outperform RL** | ICLR 2026 **Oral** | arXiv 2507.19457 |
| **Self-Improvements in Modern Agentic Systems: A Survey** | arXiv **2607.13104** (Ren, …, **Schmidhuber**) | the definitive curated library |
| **Experiential Reflective Learning (ERL)** | ICLR 2026 MemAgents Workshop | arXiv 2603.24639 — reflects on trajectories to generate **transferable heuristics**; explicitly notes fine-tuning has overhead and "provides no guidance when the current state fails to match any stored context" |
| **Self-Improving LLM Agents at Test-Time (TT-SI)** | arXiv 2510.07841 | uncertainty estimator → self-data-augmentation → **test-time LoRA** (rank=8, α=16, all linear layers, 5 epochs, lr 1e-4, bs=1). Ran on **Qwen2.5-1.5B** — i.e. *your* size class |
| **Towards Self-Evolving Agents: Dual-Process (DPA)** | Electronics 15(6):1232, 2026 | separates **fast response** from **deep reflection**; evolvable long-term memory; frozen backbone. Future work explicitly proposes **combining memory evolution with LoRA to internalise high-confidence patterns into weights** ← *this is literally FRIDAY's S2→S4 stage* |
| **Self-Adapting Language Models (SEAL)** | NeurIPS 2025 / MIT | model writes its own finetuning data ("self-edits") |
| **Skill Self-Play** | arXiv 2026 | co-evolving skills push the capability frontier |
| **Socratic-SWE** | arXiv 2026 | self-evolving coding agents via **trace-derived agent skills** |
| **FORGE** | CAIS 2026 | self-evolving agent memory **with no weight updates** via population broadcast |
| **Voyager** | NVIDIA/Caltech 2023 | persistent skill library + auto curriculum — **3.3× items, 15.3× faster milestones**. The ancestor of every skills system |
| **Eureka** | ICLR 2024 | LLM-generated reward functions; beat human experts on 83% of tasks |

**Governance ladder** (use this ordering):
1. Start with **prompt optimisation** (DSPy, GEPA) — safest: weights never change, prompts are
   human-readable and auditable, changes fully reversible.
2. Move to **agent-level** (Reflexion, AFlow) **with approval gates**.
3. Use **training-time** (SPIN, Self-Rewarding, SEAL) **only when you control the weights AND
   have an evaluation pipeline the system cannot touch.**

---

## 7. What to skip entirely

| Skip | Why |
|---|---|
| **Pinecone / Weaviate / managed vector DBs** | You have one user and 16 GB. `sqlite-vec` is a single file, zero-ops, fast enough. Cloud vector DBs add latency, cost and a data-exfiltration path. |
| **Neo4j** | Heavy JVM, real ops burden. Your bi-temporal graph fits in SQLite tables + a Markdown adjacency list. Revisit only past ~100K entities. |
| **LangChain as a dependency** | Fine as a *reference*; as a dependency it abstracts away exactly the layer you said you want to own. Hand-roll the graph (~400 lines). |
| **CrewAI / AutoGen multi-agent** | Multi-agent is for parallelisable team workloads. A personal assistant with a shared memory is faster and cheaper as **one agent with isolated sub-contexts**. Add sub-agents only for genuinely parallel research tasks. |
| **A dedicated "Emotion Engine" module** | 2024 thinking. Modern multimodal models do tone/prosody natively; Gemini Live has **affective dialog** built in (adapts response style/tone to match user expression). Don't build a bolt-on sentiment classifier — it will be worse than the LLM's own judgement and adds latency. Instead: capture *affect as a fact* in memory ("user is stressed about X until <date>"). |
| **YOLO for ambient vision** | Wrong tool. You need **document/screen understanding**, which is a VLM job (Qwen3-VL-4B: DocVQA **95.3**). YOLO is for object detection/tracking — only relevant if you later do robotics or AR. |
| **Building your own screen capture** | screenpipe already does event-driven capture + a11y tree + OCR fallback + diarisation + MCP, at 5–10% CPU. Months of work you don't need to do. |
| **gRPC for the client gateway** | Over-engineering for one user. WSS + WebRTC covers everything. gRPC only makes sense between your own backend services, which you shouldn't have yet. |
| **MQTT (for now)** | Only when you add real IoT hardware. Revisit in Phase 6+. |
| **Microservices / Kubernetes** | Law 8. One Python process. |

---

## 8. Naming & identity (small thing, big effect on "feels real")

The research is consistent that identity lives in a file, not in code. `SOUL.md` is now the
convention across OpenClaw, Hermes, jarvis.pm and others — and Hermes even provides a migration
path that imports it.

Two practical rules:
- **Keep `AGENTS.md` lean.** It's prepended to *every* turn — every line is a recurring tax on
  your context budget. Durable, project-wide rules only; layer task-specific detail JIT.
- **Voice instructions must forbid markdown.** Everything the model says in voice mode is
  spoken aloud: *"Speak in short sentences. Never use markdown, bullet points, or tables."*
  Otherwise FRIDAY says "dash dash dash" out loud.
