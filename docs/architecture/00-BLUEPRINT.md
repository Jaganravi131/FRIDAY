# PROJECT FRIDAY — Master Blueprint

> **Objective (locked):** Build a *clear, innovative, context-aware AI assistant* first.
> Task management and integrations are **Phase 3+**. Do not build them early.
>
> **Secondary objective:** an intelligent system that **builds itself** — and a genuine
> architecture-research surface (fine-tuning, transformers, self-improvement loops).

---

## 0. Read this first: the one-paragraph thesis

Most "JARVIS" projects fail because they are a **chatbot with tools bolted on**. FRIDAY's
thesis is different:

> **FRIDAY is a compiler whose input is your life and whose output is a progressively
> smaller, cheaper, more personal model.**

Every interaction you have produces a *trace*. That trace flows down a pipeline and is
*compiled* — first into episodes, then into facts, then into skills, then into prompt
optimizations, and finally into **model weights**. Each stage is cheaper, faster and more
durable than the one before, and each has a **gate** that can reject and roll back.

That single idea is what makes FRIDAY (a) context-aware, (b) self-improving, and
(c) a real research project — instead of a wrapper.

---

## 1. What actually differentiates FRIDAY (the 8 innovations)

| # | Innovation | One-liner | Doc |
|---|---|---|---|
| 1 | **The Memory Compiler (5-stage cascade)** | Trace → Episode → Fact → Skill → Weight, with gates + rollback | [02](./02-INNOVATION-memory-compiler.md) |
| 2 | **Markdown-as-Source-of-Truth + derived indices** | You can *open a file and correct FRIDAY*. Indices are build artifacts. | [02](./02-INNOVATION-memory-compiler.md) |
| 3 | **Bi-temporal fact model** | Every fact has `valid_from` / `valid_to`. "Where did I live last year?" is answerable. | [03](./03-memory-architecture.md) |
| 4 | **The Attention Ledger** | Context assembly as a *budgeted, cached, deterministic* schedule — not a stuffed prompt | [04](./04-INNOVATION-attention-ledger.md) |
| 5 | **The Escalation Ladder** | 6-tier model routing; cheapest tier that can do the job wins. Confidence-gated. | [05](./05-model-tiers-hardware.md) |
| 6 | **Constitutional Eval Gate ("Shadow Mode")** | FRIDAY may edit itself, but **never** its evaluation suite. Changes A/B in shadow before promotion. | [06](./06-INNOVATION-self-improvement-loop.md) |
| 7 | **Presence Fabric** | One *room*, many *surfaces*. Start on phone, finish on laptop, mid-sentence. | [07](./07-realtime-voice-presence.md) |
| 8 | **The Sense Registry** | Tiered consent, enforced **deterministically in the data layer** — not by prompting the model to behave. | [09](./09-security-privacy-trust.md) |

Plus the **circadian rhythm** that ties them together: [08](./08-proactive-heartbeat.md).

---

## 2. System diagram

```
                        ┌──────────────────────────────────────────────┐
                        │              SURFACES (thin clients)         │
                        │  PWA · Tauri desktop · Expo mobile · CLI/TUI │
                        │  Telegram/WhatsApp (later) · wake-word node  │
                        └───────────────────────┬──────────────────────┘
                                                │  WSS / WebRTC  (Tailscale mesh)
                        ┌───────────────────────▼──────────────────────┐
                        │            PRESENCE FABRIC  (gateway)        │
                        │  one session bus · device registry · audio    │
                        │  relay · handoff · fan-out · authn/authz      │
                        └───────────────────────┬──────────────────────┘
                                                │
      ┌─────────────────────────────────────────┼──────────────────────────────────────┐
      │                                         │                                      │
┌─────▼──────────────┐            ┌─────────────▼──────────────┐         ┌─────────────▼────────────┐
│  PERCEPTION LAYER  │            │      COGNITION CORE        │         │      ACTION LAYER        │
│                    │            │                            │         │                          │
│ STT (whisper)      │  evidence  │  ┌──────────────────────┐  │  tool   │  MCP client              │
│ VAD (silero)       ├───────────►│  │ ATTENTION LEDGER     │  │ call    │  (2026-07-28 stateless)  │
│ Wake word (µWW)    │            │  │ budget · assemble ·  │  ├────────►│                          │
│ Vision (Qwen3-VL)  │            │  │ cache-stable prefix  │  │         │  Skills (agentskills.io) │
│ Screen (screenpipe)│            │  └──────────┬───────────┘  │         │  Sandbox (Docker/gVisor) │
│ Clipboard/files    │            │             │              │         │  Browser (Playwright)    │
│ Calendar/email     │            │  ┌──────────▼───────────┐  │         │  OS control              │
│                    │            │  │ ESCALATION LADDER    │  │         │  Notifications → all     │
│  ┌──────────────┐  │            │  │ L0 router → L1 local │  │         │  surfaces                │
│  │SENSE REGISTRY│◄─┼── consent  │  │ → L2 cloud → L3 deep │  │         │                          │
│  │ tiered perms │  │            │  └──────────┬───────────┘  │         │  ┌────────────────────┐  │
│  │ redaction    │  │            │             │              │         │  │ CONFIRMATION GATE  │  │
│  │ audit log    │  │            │  ┌──────────▼───────────┐  │         │  │ irreversible ops   │  │
│  └──────────────┘  │            │  │ AGENT LOOP (ReAct +  │  │         │  │ require human ack  │  │
└────────────────────┘            │  │ plan/critique/reflect)│ │         │  └────────────────────┘  │
                                  │  └──────────┬───────────┘  │         └──────────────────────────┘
                                  └─────────────┼──────────────┘
                                                │ read/write
                        ┌───────────────────────▼──────────────────────┐
                        │          THE MEMORY COMPILER                 │
                        │                                              │
                        │  S0 TRACE ─► S1 EPISODE ─► S2 FACT           │
                        │  (append-only   (summarised)  (bi-temporal,   │
                        │   JSONL)                         MD + graph) │
                        │                                  │           │
                        │            S4 WEIGHT ◄── S3 SKILL┘           │
                        │            (QLoRA LoRA)   (SKILL.md)         │
                        │                                              │
                        │  Store of truth: Markdown (git-versioned)    │
                        │  Derived indices: SQLite FTS5 + sqlite-vec   │
                        │  (rebuildable at any time — never sacred)    │
                        └───────────────────────┬──────────────────────┘
                                                │
                        ┌───────────────────────▼──────────────────────┐
                        │      THE CRONOS LOOP  (off-hours compute)    │
                        │                                              │
                        │  HEARTBEAT   every 15–30 min while awake     │
                        │              read checklist → act or shut up │
                        │                                              │
                        │  DREAMING    nightly                         │
                        │              consolidate · dedupe · decay ·  │
                        │              GEPA prompt evolution · build   │
                        │              tomorrow's cached prefix        │
                        │                                              │
                        │  EVAL GATE   locked suite. Agent CANNOT edit │
                        │              Accept iff: failure cluster ↑   │
                        │              AND held-out regression ≤ 1%    │
                        │                                              │
                        │  DISTILL     weekly, Kaggle free T4 (30h/wk) │
                        │              traces → QLoRA → GGUF → swap in │
                        └──────────────────────────────────────────────┘
```

---

## 3. The design laws (non-negotiable)

These came out of the research and are what most hobby JARVIS projects get wrong.

### Law 1 — Proactivity is a *heartbeat*, not a cron job
A scheduler that runs a script is automation. A scheduler that runs an **agentic turn** which
decides *whether to speak* is a companion. The tick must be able to answer "nothing to do"
and be **suppressed**. Otherwise FRIDAY becomes noise and you will turn it off in a week.
→ [08](./08-proactive-heartbeat.md)

### Law 2 — Do NOT summarise your conversation history by default
This is the most counter-intuitive finding in the research and it matters enormously:

- Rewriting history **breaks prompt caching** (the prefix changes, so every turn re-prefills).
- In one production eval, summarisation dropped in-session recall from **92% → ~33%** while
  *costing twice as much* as keeping everything.
- What **does** pay off: **capping** each tool output at a fixed size before it enters history.
  Same eval: **−38% cost/turn, won 14 of 15 trajectories, recall unchanged.**

> **Rule: shrink the prefix, don't rewrite it.** Cap → truncate → offload-to-file-with-pointer
> → *only then* summarise, and only when you can name the constraint (window full, cost, or
> measured quality rot).
> → [04](./04-INNOVATION-attention-ledger.md)

### Law 3 — Just-in-time retrieval beats pre-packing
Don't guess what's relevant at turn start. Keep **pointers** (file paths, IDs, queries) and
let the agent pull what it needs, when it needs it. Pre-packing's question — *"what might be
relevant in some future turn?"* — is unanswerable. JIT's question — *"what is the minimum the
model needs for this sub-step?"* — is answerable.
Hybrid is best: a small stable identity block up front, JIT for everything else.

### Law 4 — Retrieval without a reranker is noise
`embed → top-k → dump into prompt` is the failure mode. The working pipeline is
**embed → retrieve wide (k=20) → cross-encoder rerank → ship narrow (top-3) → hard score floor →
return `[]` rather than noise.** And put a **contract in the tool docstring** ("Use ONLY when…")
or the model calls the retriever every turn and you're back to pre-packing.

### Law 5 — Enforce permissions in the data layer, never in the prompt
Telling an LLM "don't read the banking app" is not a control. Three deterministic layers are:
**capability gating** (the model never learns the denied endpoint exists) →
**execution interception** (blocked before the call) →
**middleware with per-scope cryptographic tokens** (the data server refuses).
A fully compromised/prompt-injected FRIDAY must still be *unable* to read denied data.
→ [09](./09-security-privacy-trust.md)

### Law 6 — The agent must not be able to edit its own evaluator
Self-improvement without a locked gate is self-delusion. The eval suite lives in a
**read-only, git-signed path** that the agent's sandbox cannot mount writable. Every self-edit
runs in **shadow mode** first (new policy executed in parallel, outputs compared, not served).
→ [06](./06-INNOVATION-self-improvement-loop.md)

### Law 7 — Memory files are human-readable and human-editable
If FRIDAY believes something wrong about you, you should be able to open `MEMORY.md` in
Notepad, fix the line, and have FRIDAY detect the change and re-index **only that part**.
This is the single highest-trust feature you can build, and it costs nothing.
Black-box vector memory is why people stop trusting their assistants.

### Law 8 — One language for the brain, one process for the truth
Single-user system. Resist microservices. **One Python process** owns cognition;
**one SQLite file + one git repo of Markdown** owns truth. Add processes only when a
capability genuinely needs isolation (voice worker, sandbox, capture).

---

## 4. Recommended stack (for *your* machine)

**Python core + TypeScript surfaces.** Python is non-negotiable for the self-improvement
layer — DSPy/GEPA, Unsloth, PEFT, torch, sentence-transformers, llama.cpp bindings all live
there. TypeScript for everything a human touches.

| Layer | Choice | Why |
|---|---|---|
| OS / dev | **WSL2 Ubuntu 24.04** on Win 11, `networkingMode=mirrored` | Whole AI ecosystem is Linux-first; mirrored networking makes `localhost` work both ways |
| Inference server | **llama.cpp `llama-server`** (Vulkan build) running **Windows-native** | iGPU for fast *prefill*, CPU for *decode*; native avoids WSL GPU translation loss |
| Model manager | **Ollama** (convenience) / raw GGUF (control) | Both; Ollama for pulling, llama.cpp for the tuned server |
| Agent core | **Python 3.12**, FastAPI + `uvloop` | Async, WebSocket-native, one process |
| Agent loop | **Hand-rolled `StateGraph`** (LangGraph optional) | You said you want to *own* the intelligence. A hand-rolled graph is ~400 lines and you'll understand every branch |
| Protocol | **MCP, spec `2026-07-28`** (stateless core) via **FastMCP 3.0** | De-facto standard, ~10K public servers, 97M+ monthly SDK downloads |
| Skills | **agentskills.io** `SKILL.md` format | Portable, git-diffable, human-readable |
| Store of truth | **Markdown in git** | Versioned, diffable, hand-editable, rollback = `git revert` |
| Indices | **SQLite + FTS5 + `sqlite-vec`** | Zero-ops, single file, sub-ms keyword search, rebuildable |
| Embeddings | **Qwen3-Embedding-0.6B** (GGUF, local) | Tiny, strong, runs on your CPU |
| Reranker | **bge-reranker-base** (ONNX, CPU) | The step that turns retrieval from noise into signal |
| STT | **faster-whisper `small`/`base`** (CTranslate2, CPU) | Real-time-ish on Ryzen 5 |
| VAD | **Silero VAD v5** | 30ms frames, tiny |
| Wake word | **microWakeWord** or **openWakeWord** | Custom "Friday" wake word, ESP32-capable later |
| TTS (local) | **Piper** | Instant, offline, decent |
| TTS (natural) | **Cartesia Sonic** / **ElevenLabs Flash v2.5** (~85–135ms TTFA) | For the "realistic" requirement — local TTS still sounds robotic |
| Voice transport | **Pipecat** (start) → **LiveKit Agents** (scale) | Pipecat is transport-agnostic and mixes realtime models with cascades; LiveKit when you want SIP/phone/multi-room |
| Ambient capture | **screenpipe** (Windows-supported, local SQLite, MCP server) | Don't build this. Event-driven, ~5–10% CPU, ~5–10 GB/mo |
| Scheduler | **APScheduler** | Heartbeat + circadian jobs |
| Self-improvement | **DSPy + GEPA**, **Unsloth QLoRA** | Prompt evolution + weight distillation |
| Training GPU | **Kaggle free tier — 2× T4, 30 h/week, resets Sunday UTC** | $0 and *predictable* quota (Colab free is demand-throttled, 90-min idle, 12-h cap) |
| Cross-device network | **Tailscale** | Private mesh, phone→laptop with zero exposed ports. Non-negotiable for a personal agent |
| Observability | **OpenTelemetry → JSONL** (self-host Langfuse later) | Every turn: tokens, latency, tier used, tool calls, cache hit |
| Desktop client | **Tauri v2** (Rust shell, web UI) | ~10 MB vs Electron's ~150 MB; matters on 16 GB RAM |
| Mobile client | **Expo / React Native** → or **PWA** first | PWA on day 1 (zero build), native later for wake word + background audio |
| Web client | **React + Vite PWA** | Same codebase as Tauri |

---

## 5. Honest hardware verdict (Ryzen 5 + Radeon iGPU, 16 GB RAM, 512 GB, Win 11)

Full numbers and the model menu in → **[05-model-tiers-hardware.md](./05-model-tiers-hardware.md)**

### The good news
| What | Measured/expected | Verdict |
|---|---|---|
| **Qwen3-4B Q4_K_XL, CPU-only** | ~**20 t/s** generation | ✅ Your daily driver. Comfortably faster than reading speed |
| **Qwen3-1.7B Q8** | ~**25–40 t/s** | ✅ Router / classifier / fast path |
| **Qwen3-0.6B Q8** | ~**60–86 t/s** | ✅ Intent gate, redaction, tagging — runs constantly for free |
| **iGPU (Vulkan/ROCm) prefill** | ~**54 t/s** prompt processing on a Radeon 780M-class iGPU | ✅ Offload prefill to iGPU → **huge** win for RAG/long context |
| **Qwen3-VL-4B** (vision) | MMMU 67.4, DocVQA **95.3**, ~3.5 GB @ Q4 | ✅ Best small VLM; screen/doc reading is viable |
| **Whisper small (CTranslate2)** | sub-second chunks on CPU | ✅ Real-time-ish STT |
| **Embeddings + reranker + VAD + wake word** | all run together, <1 GB | ✅ The entire perception stack is free |
| **512 GB SSD** | plenty | ✅ This is your real superpower — store *everything* |

### The hard limits (know these before you waste a week)
| Constraint | Reality |
|---|---|
| **iGPU decode is NOT faster than CPU.** | On a Radeon 780M-class iGPU, dense 7B/12B decode measured **4.5–5.3 t/s** on GPU vs **5.3 t/s** on CPU. The iGPU wins **prefill**, not generation. Don't expect magic. |
| **Qwen3-30B-A3B will NOT fit.** | It's a 19 GB download; people with exactly 16 GB report *"Q4 can't be loaded"*. Windows alone eats 4–5 GB. **Do not plan around it.** |
| **16 GB is the binding constraint, not the CPU.** | Windows + WSL2 + browser + a 4–5 GB model + capture = you will be swapping. |
| **Fine-tuning locally is not viable.** | No CUDA, iGPU ROCm is experimental on Windows, CPU training is days-scale. **Use Kaggle's free 30 GPU-hours/week instead.** |
| **Always-on full capture + inference will thermally throttle a VivoBook.** | Duty-cycle the capture; don't run 24/7 on a thin-and-light. |

### The two highest-ROI moves you can make
1. **Check if your VivoBook has a free SODIMM slot → go to 32 GB (~₹4–6k).**
   This single upgrade unlocks Qwen3-30B-A3B MoE at ~**27 t/s CPU-only** (measured on
   comparable hardware) and removes every memory constraint in this document.
   It is the difference between "4B assistant" and "30B assistant". **Do this first.**
2. **Use Kaggle's free 2×T4 / 30 h per week for all training.**
   QLoRA + Unsloth on a T4 (15 GB) comfortably handles **Qwen3-4B** (~5 GB with Unsloth)
   and even 7–8B at short sequence lengths. A reference pipeline fine-tuned a 1.5B model in
   **70 seconds** on a free T4 and exported straight to GGUF → Ollama.
   **Your self-improvement loop can therefore run for ₹0.**

---

## 6. Phased roadmap

> **Do not skip phases.** Each phase produces something you use daily *and* the substrate
> the next phase needs. The most common failure is building the integrations first and
> having nothing worth integrating.

### **Phase 0 — Foundation (Week 1)**
*Deliverable: you can talk to FRIDAY in a terminal and it remembers.*
- WSL2 + mirrored networking + `.wslconfig` memory cap
- `llama-server` (Vulkan) on Windows; benchmark **your** machine (`llama-bench`) — real numbers, not mine
- Repo skeleton, `SOUL.md`, `AGENTS.md`, `USER.md`, `MEMORY.md`
- SQLite + FTS5 + sqlite-vec schema
- Single-turn ReAct loop with 3 tools: `memory_search`, `memory_write`, `shell_sandboxed`
- **Exit test:** ask it something on Monday, ask a follow-up on Friday, it recalls correctly.

### **Phase 1 — The Memory Compiler (Weeks 2–4)** ← *the core objective*
*Deliverable: FRIDAY has a model of you that improves without being told.*
- S0→S1→S2 pipeline: trace → episode → bi-temporal fact
- Markdown-as-source-of-truth + `watchdog` file-watcher → incremental re-index
- Nightly **Dreaming** job: consolidate, dedupe, decay, merge
- Retrieval: embed → wide → rerank → narrow → score floor
- **Exit test:** tell it a fact, contradict it two weeks later, ask "what did I used to think?" —
  it answers *both* correctly with dates.

### **Phase 2 — The Attention Ledger (Weeks 4–6)** ← *"clear" + "context-aware"*
*Deliverable: responses that are actually about you, at low cost and low latency.*
- Deterministic context compiler with a token budget per slot
- Stable cached prefix (identity + senses) vs JIT zone (memory + retrieval + scratch)
- Tool-output capping (the −38% cost lever)
- Offload-over-20K-to-file-with-pointer
- Telemetry: tokens, cache-hit %, TTFT, per-slot utilisation
- **Exit test:** 200-turn session; turn 200 is as coherent as turn 5, and cost/turn is flat.

### **Phase 3 — Perception & Presence (Weeks 6–9)** ← *"all three modes equal"*
*Deliverable: voice, vision and cross-device continuity.*
- screenpipe integration via MCP + Sense Registry with tiered consent
- Voice: VAD → wake word → faster-whisper → LLM → Piper/Cartesia, **with correct barge-in**
- Presence Fabric: one session bus, phone + laptop in the same room
- **Exit test:** start a question walking (phone), sit down (laptop) — the answer is on the
  laptop screen. Interrupt FRIDAY mid-sentence; it stops in <200 ms and doesn't think it
  finished the sentence.

### **Phase 4 — Proactivity (Weeks 9–11)**
*Deliverable: FRIDAY speaks first, and you're glad it did.*
- Heartbeat with `HEARTBEAT_OK` suppression
- Circadian schedule; morning briefing compiled overnight
- Confirmation gate for irreversible actions
- **Exit test:** one week where FRIDAY initiates ≥5 useful things and **zero** annoying things.
  Measure the ratio. Tune until it's >5:1.

### **Phase 5 — Self-Improvement (Weeks 11–16)** ← *the research*
*Deliverable: FRIDAY measurably gets better at your tasks without you editing prompts.*
- Locked eval suite (agent has **no write access**)
- GEPA prompt evolution against it, nightly
- Skill synthesis after complex tasks + **shadow-mode** A/B before promotion
- Weekly QLoRA distillation on Kaggle T4 → GGUF → swap into the L1 slot
- **Exit test:** publish a before/after number. "FRIDAY completed my recurring task 40%
  faster / with 30% fewer tokens after 30 days" — measured by the gate, not by vibes.

### **Phase 6 — Skills & Task Management (Week 16+)**
*Now* you add calendar, email, notes, finance, home. By this point FRIDAY already knows you,
so each integration lands with context instead of being a dumb API wrapper.

---

## 7. Repository layout (target)

```
FRIDAY/
├── docs/architecture/          ← you are here
├── soul/                       ← FRIDAY's identity (Markdown, git-versioned, hand-editable)
│   ├── SOUL.md                 ← personality, voice, boundaries
│   ├── AGENTS.md               ← operating rules (keep LEAN — every line is a recurring tax)
│   ├── USER.md                 ← who you are (≤1,400 chars, high-signal)
│   ├── MEMORY.md               ← durable facts (≤2,200 chars core + linked files)
│   └── HEARTBEAT.md            ← the proactive checklist
├── memory/
│   ├── traces/  YYYY-MM-DD.jsonl       S0  append-only raw
│   ├── episodes/ YYYY-MM-DD.md         S1  summarised days
│   ├── facts/   *.md                   S2  bi-temporal, human-editable
│   └── daily/   YYYY-MM-DD.md          ephemeral log (NOT auto-injected)
├── skills/                     ← S3  procedural memory, agentskills.io format
│   └── <skill>/SKILL.md
├── core/                       ← Python cognition core
│   ├── ledger/                 ← Attention Ledger (innovation #4)
│   ├── compiler/               ← Memory Compiler (innovation #1)
│   ├── ladder/                 ← Escalation Ladder (innovation #5)
│   ├── loop/                   ← agent loop, StateGraph
│   ├── retrieval/              ← embed → rerank → floor
│   ├── senses/                 ← Sense Registry (innovation #8)
│   └── cronos/                 ← heartbeat + dreaming + eval gate
├── gateway/                    ← Presence Fabric (innovation #7)
├── voice/                      ← VAD/STT/TTS/wake-word worker
├── eval/                       ← THE LOCKED SUITE. Read-only to the agent.
├── train/                      ← QLoRA/Unsloth notebooks for Kaggle
├── surfaces/
│   ├── web/                    ← React+Vite PWA
│   ├── desktop/                ← Tauri v2 wrapper
│   └── mobile/                 ← Expo (later)
└── mcp/                        ← your own MCP servers
```

---

## 8. Success metrics (instrument from day 1)

If you can't measure it, you're building a toy.

| Dimension | Metric | Target |
|---|---|---|
| **Latency** | Time-to-first-audio (voice) | < 800 ms local, < 1.2 s cloud |
| | Time-to-first-token (text) | < 400 ms |
| | Barge-in stop | < 200 ms |
| **Context** | Prompt cache-hit rate | > 85% |
| | Tokens/turn (steady state) | flat, not growing |
| | Retrieved-precision@3 | > 0.7 |
| **Memory** | LongMemEval-style probe score | track the trend, don't chase a number |
| | Fact contradiction rate | ↓ over time |
| **Proactivity** | Useful-initiation : annoying-initiation | > 5 : 1 |
| | Heartbeat suppression rate | > 80% (it should usually stay quiet) |
| **Self-improvement** | Eval-gate delta after 30 days | ≥ +10% on your task cluster |
| | Tokens to complete recurring task | ↓ ≥ 30% |
| **Cost** | $/day | < $1 with hybrid routing |
| **Trust** | % of denied-data accesses blocked | 100% (deterministic — test it adversarially) |

---

## 9. Top 10 ways this project dies (and the antidote)

1. **You build integrations before memory.** → Phase order is law.
2. **FRIDAY talks too much.** → `HEARTBEAT_OK` suppression + a hard "annoyance budget".
3. **Context bloat kills latency and cost.** → Attention Ledger + Law 2 (cap, don't summarise).
4. **Memory becomes a swamp of duplicates.** → Nightly consolidation with dedupe + decay.
5. **It hallucinates facts about you.** → Bi-temporal facts with provenance links back to traces.
6. **Prompt injection via a webpage/email it reads.** → Quarantine untrusted content, Law 5
   deterministic permissions, confirmation gate on irreversible ops.
7. **You run out of RAM.** → 32 GB upgrade, or duty-cycle models (unload L2 after idle).
8. **Self-improvement makes it worse and you don't notice.** → Locked eval gate + shadow mode.
9. **Voice feels robotic so you stop using it.** → Budget for Cartesia/ElevenLabs; local TTS
   is the fallback, not the goal. "Natural and realistic" is a paid feature.
10. **You lose trust once and never recover.** → Law 7: every memory is a file you can read
    and edit. Audit log for every action. `git log` is your undo button.

---

## 10. Where to go next

Read in this order:

1. **[01-research-landscape.md](./01-research-landscape.md)** — what exists in 2026, what to steal, what to skip
2. **[02-INNOVATION-memory-compiler.md](./02-INNOVATION-memory-compiler.md)** — the core novelty
3. **[05-model-tiers-hardware.md](./05-model-tiers-hardware.md)** — your exact model menu + benchmark script
4. **[04-INNOVATION-attention-ledger.md](./04-INNOVATION-attention-ledger.md)** — why responses will be *clear*
5. **[10-phase-0-implementation.md](./10-phase-0-implementation.md)** — start typing code
