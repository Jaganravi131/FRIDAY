# PROJECT FRIDAY — Master Blueprint

> **Objective (locked):** Build a *clear, innovative, context-aware AI assistant* first.
> Task management and integrations are **Phase 3+**. Do not build them early.
>
> **Secondary objective:** an intelligent system that **builds itself** — and a genuine
> architecture-research surface (fine-tuning, transformers, self-improvement loops).

---

> ## ⚡ PIVOT — [12-LINEAR-ATTENTION-PIVOT.md](./12-LINEAR-ATTENTION-PIVOT.md)
>
> Two things changed after this blueprint was written, and both reshape it:
>
> 1. **The goal sharpened** from *"use a good small model"* to **"build the best lightweight model
>    that is also very accurate."**
> 2. **The budget is ₹0.** Fully local + free tiers. No cloud API, no paid TTS.
>
> **Doc 12 is the amendment to this document.** Read it after this one. In brief:
>
> - FRIDAY moves to a **hybrid linear-attention backbone** (Gated DeltaNet / Mamba-2 class, ~3:1
>   linear-to-attention) so the Attention Ledger becomes a **fixed-size physical state** instead of
>   a growing text prefix. `num_ctx` goes from 8–12K to **32K–128K on the same 16 GB of RAM.**
> - **RetNet specifically is rejected** — its fixed input-independent decay yields *"near-zero
>   recall even when full-attention layers are added,"* and recall is FRIDAY's entire product.
>   Use **LFM2 / LFM2.5** (official GGUFs, official llama.cpp on-device docs, 30 t/s at 2.6B,
>   38 t/s at LFM2-8B-A1B, vision variants) and **RWKV-7**.
> - **Innovation #9** below is the actual research contribution.
> - **Law 2 is amended** (architecture-contingent) and **Laws 2b/2c are added.**
> - **Phases 3.5 and 5.5 are inserted** into the roadmap; Phase 0 now begins with a bake-off.
>
> **Then [13-GATED-DELTANET2-ERASE-WRITE.md](./13-GATED-DELTANET2-ERASE-WRITE.md) specifies the
> RSC's operator:** NVIDIA's **Gated DeltaNet-2** (May 2026) decouples the delta rule's single
> scalar gate into **channel-wise erase `b_t`** and **channel-wise write `w_t`**, and its own
> ablation reports **the erase gate accounts for most of the gain** — RULER needle retrieval
> **63 → 90**. Under a scalar tie you *cannot erase hard without writing hard*, which is why both
> the 92%→33% summarisation collapse and RetNet's fixed-γ failure presented as recall failures:
> **both were coupling failures.** Its successor **EDA** (Jun 2026) decouples erase/write
> *addresses* rather than strengths — *"GDN-2 decouples **how strongly**; EDA decouples **where**"* —
> which is exactly FRIDAY's bi-temporal correction case, and the two compose. This yields
> **Innovation #9b**: the compiler's decay function, retraction pairs and salience labels are
> **free supervision for the three gate branches.**
>
> ⚠️ **GDN-2 does not change Phase 0 and cannot be the backbone.** The NVlabs repo is a *training*
> harness (`pretrain.py`, `lit_gpt/`, Dockerfile) with **no inference server, no GGUF, no llama.cpp
> support and no checkpoint** — there is no path to serving a custom layer on a 16 GB CPU laptop,
> and you cannot insert GDN-2 layers into a frozen LFM2. It belongs in **Phase 3.5**, inside the
> RSC, running offline on a Kaggle T4 against a frozen backbone — where every blocker disappears.
>
> Everything else in this blueprint stands unchanged.

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

## 1. What actually differentiates FRIDAY (the 10 innovations)

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
| **9** | ⭐ **The Two-Memory Split + Retention State Compiler** | A **lossy O(1) recurrent state** carries conversational *context*; a **lossless external store** carries *facts*. A trained ~40M-param gated-delta module compiles one into the other — and its **salience head feeds S2 fact extraction in the same forward pass.** Compression and memory, unified. | [12](./12-LINEAR-ATTENTION-PIVOT.md) |
| **9b** | ⭐ **Gate supervision from a bi-temporal knowledge base** | **GDN-2/EDA** expose three gate branches — **decay α_t, erase b_t/e_t, write w_t**. The Memory Compiler already computes three matching signals: the **decay function** ([02 §4](./02-INNOVATION-memory-compiler.md)), **retraction events** — which yield `(key_old, key_new)` pairs, free supervision for EDA's *independently-addressed* erase — and the **salience head**. A recurrent memory operator whose forgetting is supervised by what the compiler decided to keep. | [13](./13-GATED-DELTANET2-ERASE-WRITE.md) |

Plus the **circadian rhythm** that ties them together: [08](./08-proactive-heartbeat.md).

> **Why #9 is the research contribution and not just an optimisation.** The literature states the
> problem exactly: *"a hybrid paired with retrieval **sidesteps** the recall gap rather than solving
> it: you do not ask the recurrent state to memorize a fact you can fetch from an index."* FRIDAY
> already **has** the index — that's innovations #1–#3. So the pivot doesn't add a workaround; it
> finds that the memory architecture designed for a different reason is precisely the missing half
> of a linear-attention system. And nobody has published a Retention State Compiler with a
> fact-extraction salience side-channel for a personal assistant.

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

> ### ⚡ AMENDED by the linear-attention pivot — [12 §1.2](./12-LINEAR-ATTENTION-PIVOT.md)
>
> **This law is a property of KV-prefix caching in *attention* models. It is architecture-contingent,
> not universal.** With a recurrent-state backbone there is **no prefix to cache**, so rewriting the
> state is not a cache miss — it is simply the next recurrent step. Compaction stops being a
> semantic break and becomes a **learned, continuous operation.**
>
> Consequences: the **6-rung compaction ladder collapses** to Rung 0 (cap tool output — still free,
> still worth it, still −38% cost) plus the **Retention State Compiler**. The `transcript` slot
> becomes a **fixed-size state over unbounded history**. Compaction-at-75%-of-window **never fires**,
> because the window never fills.
>
> **The empirical warning still stands, and it's the reason for the amendment's shape:** the 92% →
> ~33% recall collapse happened because a *lossy* compressor was asked to preserve *verbatim*
> detail. A recurrent state is lossy in exactly the same way — *"what it loses first is associative
> recall: retrieving a specific earlier token verbatim."* So the RSC is trained with an explicit
> `L_recall` term and the external store remains the guaranteed-lossless path. See Law 9.
>
> **If the backbone stays a Transformer** (Qwen3-4B), this law applies in full, unamended.

### Law 2b — ⭐ Never ask a lossy state to be a database *(added by [12](./12-LINEAR-ATTENTION-PIVOT.md))*

The single most important structural rule of the pivot. Two memories, two substrates:

| | **RETENTION STATE** (lossy, O(1)) | **EXTERNAL STORE** (lossless, exact) |
|---|---|---|
| Substrate | the linear-attention layers / RSC output | SQLite + FTS5 + sqlite-vec, Markdown-as-truth |
| Carries | conversational gist, current thread, your register, what FRIDAY just said | **FACTS**, names, numbers, dates, verbatim quotes, provenance, bi-temporal validity |
| Cost | fixed, no growth | exact lookup, 10–150 ms |
| Fails at | **verbatim recall** ⚠️ | nothing — it's a database |
| So we | **never** ask it to recall a fact | **always** ask it to recall a fact |

This isn't a workaround for a weakness. It's a clean separation of concerns that happens to map
exactly onto the architecture's strengths — and it has a theoretical warrant. **ICLR 2025
(*RNNs Are Not Transformers (Yet)*) proves** that RNNs with *o(n)*-bit memory cannot solve
in-context retrieval even with chain-of-thought, **and** gives two sufficient fixes:
**(a)** a function-call primitive for in-context retrieval lifts them to *all polynomial-time
solvable problems* — that is FRIDAY's `memory_search`; **(b)** one Transformer layer at the end
closes the representation gap — that is the hybrid's attention layer. Empirically, *"In-Context RAG
allows all the models to reach near-perfect accuracy."*

**FRIDAY implements both (a) and (b).** → [12 §3](./12-LINEAR-ATTENTION-PIVOT.md)

### Law 2c — ⭐ Recall is a property of a *checkpoint*, not an architecture *(added by [12](./12-LINEAR-ATTENTION-PIVOT.md))*

> *"Chain-of-thought fine-tuning has been shown to **degrade** long-range recall in hybrids, so
> base-model recall numbers do not transfer to the reasoning checkpoint you deploy.
> **Measure after post-training.**"*

**Every time you QLoRA, continue-pretrain, or GEPA-optimise a model, re-run the needle-recall
test.** Otherwise the self-improvement loop can silently destroy the one capability FRIDAY exists
to provide — and the Constitutional Eval Gate will pass it, because the gate measures task success,
not retrieval.

This is why `eval/suites/personal-recall.yaml` is weighted **0.25** in [06](./06-INNOVATION-self-improvement-loop.md)
and why [12 §5.3](./12-LINEAR-ATTENTION-PIVOT.md) makes the recall test the deciding benchmark of
the whole bake-off. The literature's repair is targeted (**QK-Restore**), not "add more data."

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
| ⭐ **LFM2-2.6B (hybrid linear-attn) Q4** | ~**30 t/s** CPU-only | ✅ **Preferred daily driver.** Better quality/speed than Qwen3-4B on your machine |
| ⭐ **LFM2-8B-A1B (hybrid MoE)** | ~**38 t/s** — **8B stored, 1B active** | ✅ If it fits beside Windows, this is the best model you can run |
| ⭐ **LFM2.5-1.2B-Instruct Q4** | ~0.9 GB resident | ✅ Tiny footprint → **huge `num_ctx` headroom**. The O(1)-state payoff |
| ⭐ **LFM2-VL-1.6B / LFM2.5-VL-450M** | official GGUF + `mmproj` | ✅ Vision **in the same family** — one tokenizer, one serving path |
| ⭐ **RWKV-7 World 2.9B Q8** | in llama.cpp via @MollySophia | ✅ Benchmark it against LFM2 |
| ⭐ **KV cache, hybrid vs Transformer** | hybrid state is **fixed-size** | ✅ **`num_ctx` 8–12K → 32K–128K on the same 16 GB.** The Attention Ledger stops rationing and starts routing. *(Nemotron 3.5 Lightning 30B-A3B: ~6 KB/token because its 6 attention layers use only 2 KV heads)* |

### The hard limits (know these before you waste a week)
| Constraint | Reality |
|---|---|
| **iGPU decode is NOT faster than CPU.** | On a Radeon 780M-class iGPU, dense 7B/12B decode measured **4.5–5.3 t/s** on GPU vs **5.3 t/s** on CPU. The iGPU wins **prefill**, not generation. Don't expect magic. |
| **Qwen3-30B-A3B will NOT fit.** | It's a 19 GB download; people with exactly 16 GB report *"Q4 can't be loaded"*. Windows alone eats 4–5 GB. **Do not plan around it.** |
| **16 GB is the binding constraint, not the CPU.** | Windows + WSL2 + browser + a 4–5 GB model + capture = you will be swapping. |
| **Fine-tuning locally is not viable.** | No CUDA, iGPU ROCm is experimental on Windows, CPU training is days-scale. **Use Kaggle's free 30 GPU-hours/week instead.** |
| **Always-on full capture + inference will thermally throttle a VivoBook.** | Duty-cycle the capture; don't run 24/7 on a thin-and-light. |
| ⭐ **From-scratch pretraining is off the table.** | Kaggle's 30 T4-h/week ≈ **1.1B tokens/week** ≈ 55B/year *if you use every hour forever*. The 2026 rule is **100:1 minimum, ideally 500–1000:1** tokens-per-parameter (LFM2.5-350M used **80,000:1**). Chinchilla-optimal 1B alone is ~600 H100-hours / 50B tokens. **A from-scratch 150M model on 15B tokens is worse than GPT-2.** Continue-pretrain an existing over-trained small model instead → [12 §7](./12-LINEAR-ATTENTION-PIVOT.md) |
| ⭐ **₹0 means no natural voice.** | You asked for *"very natural and realistic."* **Piper is robotic.** Cartesia Sonic / ElevenLabs Flash (85–135 ms TTFA, genuinely human) cost money. This is the one place the budget and the stated goal genuinely conflict — decide consciously, don't discover it in Week 8 → [12 §8.1](./12-LINEAR-ATTENTION-PIVOT.md) |
| ⭐ **₹0 means no cloud GEPA teacher.** | OpenJarvis's 13–32 pp recovery came from a *frontier* teacher reflecting on failures. Mitigate with **behavioural reflection** — your S0 traces carry external signals (barge-ins, rephrases, "no I meant…", tool failures, gate failures) that are a *better* feedback function than a model's opinion. Huang et al.: LLMs can't self-correct without external feedback anyway → [12 §8.2](./12-LINEAR-ATTENTION-PIVOT.md) |

### The two highest-ROI moves you can make
1. **⚡ NEW — Run the backbone bake-off before writing any agent code.** *(₹0, one afternoon,
   decides everything downstream.)* Benchmark **LFM2.5-1.2B / LFM2-2.6B / LFM2-8B-A1B /
   RWKV-7-World-2.9B** against a **Qwen3-4B Transformer baseline** on: speed, RAM at `num_ctx=8192`
   vs `32768`, and above all **the needle-recall test** (8K-token context, a fact planted at 5
   depths, exact-match score). That last test is the one where linear-attention models collapse and
   hybrids hold — it directly measures whether FRIDAY will remember what you told it 200 turns ago.
   Write the numbers into `docs/architecture/BENCHMARKS.md`. **Pick the backbone with data, not
   vibes.** → [12 §5.3, §11](./12-LINEAR-ATTENTION-PIVOT.md)
2. **Use Kaggle's free 2×T4 / 30 h per week for all training.**
   QLoRA + Unsloth on a T4 (15 GB) comfortably handles **Qwen3-4B** (~5 GB with Unsloth)
   and even 7–8B at short sequence lengths. A reference pipeline fine-tuned a 1.5B model in
   **70 seconds** on a free T4 and exported straight to GGUF → Ollama.
   **Your self-improvement loop can therefore run for ₹0.**
   ⭐ **And the Retention State Compiler fits too:** ~40M trainable params against a **frozen 4-bit
   backbone** ≈ **2–6 GB VRAM, minutes per run** — dozens of experiments a week. That's a ~10,000×
   smaller problem than Microsoft's 512-MI200-GPU RetNet training run, because you're not teaching
   it *language* — you're teaching it *what to remember about you.* → [12 §6.2](./12-LINEAR-ATTENTION-PIVOT.md)
3. *(Was #1, now conditional)* **Check if your VivoBook has a free SODIMM slot → 32 GB (~₹4–6k).**
   Still the biggest single hardware win — it unlocks Qwen3-30B-A3B MoE at ~27 t/s CPU-only.
   **But the pivot lowers its urgency:** with a fixed-size recurrent state, a hybrid model gets
   32K–128K of usable context *inside 16 GB*, which recovers much of what the upgrade would have
   bought. Re-run the bake-off at both RAM sizes before spending.
   ⚠️ Worth revisiting later: **Nemotron 3.5 Lightning 30B-A3B** — hybrid Mamba-2 + attention + MoE,
   52 layers (23 Mamba-2, 23 MoE, 6 attention), 262K context in local GGUF, **~6 KB/token KV cache**,
   Q5_K_M at 26.6 GB or AD-IQ4_NL at 19.7 GB shared-memory. At 32 GB this becomes realistic —
   a 30B-class model on your laptop, which the pure-attention menu could not offer.

---

## 6. Phased roadmap

> **Do not skip phases.** Each phase produces something you use daily *and* the substrate
> the next phase needs. The most common failure is building the integrations first and
> having nothing worth integrating.

### **Phase 0 — Foundation (Week 1)**
*Deliverable: you can talk to FRIDAY in a terminal and it remembers.*
- ⚡ **Days 1–2: the backbone bake-off** ([12 §11](./12-LINEAR-ATTENTION-PIVOT.md)). Download the
  official GGUFs (`winget install llama.cpp`), measure speed / RAM-at-8K / RAM-at-32K, and **write
  the needle-recall test** (`scripts/eval_recall.py`, ~40 lines). Record everything in
  `docs/architecture/BENCHMARKS.md`. **Keep a Qwen3-4B Transformer baseline** — you need it to prove
  the hybrid helps. Decision rule: hybrid ≥0.9 needle recall at 8K *and* ≥25 t/s → use it; hybrids
  collapse (<0.6) → stay on Qwen3-4B and let the Ledger + external store do the work.
- WSL2 + mirrored networking + `.wslconfig` memory cap
- `llama-server` (Vulkan) on Windows; benchmark **your** machine (`llama-bench`) — real numbers, not mine
- Repo skeleton, `SOUL.md`, `AGENTS.md`, `USER.md`, `MEMORY.md`
- SQLite + FTS5 + sqlite-vec schema
- Single-turn ReAct loop with 3 tools: `memory_search`, `memory_write`, `shell_sandboxed`
- **Exit test:** ask it something on Monday, ask a follow-up on Friday, it recalls correctly —
  *and* `BENCHMARKS.md` exists with your own recall numbers.

### **Phase 1 — The Memory Compiler (Weeks 2–4)** ← *the core objective*
*Deliverable: FRIDAY has a model of you that improves without being told.*
- S0→S1→S2 pipeline: trace → episode → bi-temporal fact
- ⚡ **Add the salience-label tap NOW.** Every time S2 extracts a fact, record the **source span**
  (`trace_id`, `char_start`, `char_end`). These are **free supervision labels for Track B's
  salience head** — and if you start collecting them in Week 12 instead of Week 2 you will have no
  training data when you need it. Cost: one extra column, one extra insert.
- ⚡ **And record retraction pairs.** Every correction yields `(key_old, key_new, span_old, span_new)`
  from the bi-temporal `valid_to` / `superseded_by` write you're already doing. These are **free
  supervision for `L_erase`** — the term that makes EDA's independently-addressed erase trainable
  ([13 §5.1](./13-GATED-DELTANET2-ERASE-WRITE.md)). **Neither of these can be backfilled.**
- Markdown-as-source-of-truth + `watchdog` file-watcher → incremental re-index
- Nightly **Dreaming** job: consolidate, dedupe, decay, merge
- Retrieval: embed → wide → rerank → narrow → score floor
- **Exit test:** tell it a fact, contradict it two weeks later, ask "what did I used to think?" —
  it answers *both* correctly with dates. **Plus:** ≥100 salience-labelled spans **and ≥20 retraction pairs** in the DB.

### **Phase 2 — The Attention Ledger (Weeks 4–6)** ← *"clear" + "context-aware"*
*Deliverable: responses that are actually about you, at low cost and low latency.*
- Deterministic context compiler with a token budget per slot
- Stable cached prefix (identity + senses) vs JIT zone (memory + retrieval + scratch)
- Tool-output capping (the −38% cost lever) — **Rung 0 survives the pivot unchanged; it's free**
- ⚡ **The build splits by backbone:**
  - **Hybrid (LFM2/RWKV)** → Law 2 largely dissolves. `num_ctx` 32K+, so the transcript slot is
    generous and rungs 1–3, 5 rarely fire. Build **Rung 0 + Rung 4 (offload-with-pointer)** only,
    and leave a `state` slot stub where the RSC will plug in at Phase 3.5.
  - **Transformer (Qwen3-4B)** → build the **full 6-rung ladder** as specified in [04](./04-INNOVATION-attention-ledger.md).
- Telemetry: tokens, cache-hit %, TTFT, per-slot utilisation, **and needle-recall at session end**
- **Exit test:** 200-turn session; turn 200 is as coherent as turn 5, and cost/turn is flat.

### **Phase 3 — Perception & Presence (Weeks 6–9)** ← *"all three modes equal"*
*Deliverable: voice, vision and cross-device continuity.*
- screenpipe integration via MCP + Sense Registry with tiered consent
- Voice: VAD → wake word → faster-whisper → LLM → **Piper** (₹0), **with correct barge-in**
  ⚠️ *Set the expectation now: Piper is not "natural and realistic." If voice quality turns out to
  matter more than you expected, ~₹400/month of Cartesia/ElevenLabs Flash transforms the experience
  more than anything else in this document. Decide at Week 9 with the system in hand.*
- ⚡ Vision: benchmark **LFM2-VL-1.6B / LFM2.5-VL-450M** (same family as the backbone → one
  tokenizer, one serving path) against **Qwen3-VL-4B** (DocVQA 95.3) **on your own documents.**
  Benchmark sheets lie; your invoices and screenshots don't.
- Presence Fabric: one session bus, phone + laptop in the same room
- **Exit test:** start a question walking (phone), sit down (laptop) — the answer is on the
  laptop screen. Interrupt FRIDAY mid-sentence; it stops in <200 ms and doesn't think it
  finished the sentence.

### **⭐ Phase 3.5 — The Retention State Compiler (Weeks 8–12)** ← *the research contribution*
*Deliverable: history stops being rationed. The Attention Ledger becomes a physical state matrix.*
- ⚡ **Operator: Gated DeltaNet-2, not generic Gated DeltaNet** ([13](./13-GATED-DELTANET2-ERASE-WRITE.md)).
  Channel-wise **erase** gate `b_t` on the key axis + channel-wise **write** gate `w_t` on the value
  axis + KDA's channel-wise decay: `S_t = (I − k_t(b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t(w_t ⊙ v_t)ᵀ`.
  Block: `GDN-2 → MLP → SWA(w=2048) → MLP`. 16 heads, d_k=d_v=128, **~1 MB state/layer**.
  **NVlabs' ablation says the erase gate accounts for most of the gain** — and erasing accumulated
  transient noise is FRIDAY's dominant memory problem, while durable content doesn't need the state
  at all (it goes to the store). ~40M trainable params, backbone **frozen in 4-bit**
- ⚡ **Use the published code, don't write kernels.** [`NVlabs/GatedDeltaNet-2`](https://github.com/NVlabs/GatedDeltaNet-2)
  ships PyTorch + Triton kernels + the chunkwise WY form + a gate-aware backward. **Extract only the
  mixer** — their harness is `lit_gpt`-based pretraining and you want your own trainer. Fallback:
  [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) has battle-tested
  **GDN + KDA** kernels; start at rung 3 by adding a channel-wise erase gate to `fla`'s GDN.
  ⚠️ **T4 is sm_75; the NVlabs kernels are tuned for Hopper** — expect poor occupancy, and plan a
  pure-PyTorch recurrent path for correctness checks (the RSC is offline, so throughput is moot).
- ⚡ **Run the ablation ladder, in order** ([13 §5.2](./13-GATED-DELTANET2-ERASE-WRITE.md)):
  **KDA (rung 2) → +channel-wise erase (3) → +channel-wise write (4) → +independent erase address
  EDA (5).** GDN-2 **recovers KDA exactly** when both gates tie to a scalar, so rung 2 is a *free
  control condition* and every rung is a one-variable change. If rung 3 is where your recall jumps —
  as the published ablation predicts — **stop there and ship.** Do not start at rung 5.
- ⚡ **Five-term loss**, of which **two are new and free**: `L_reconstruction` (KL vs the
  full-history teacher), **`L_recall`** (needle exact-match — *without this the RSC learns to drop
  verbatim detail*), `L_salience` (BCE vs Phase 1's labels → supervises `w_t`), ⭐ **`L_erase`**
  (bi-temporal correction: the state must not return a superseded value — supervises EDA's `e_t`),
  ⭐ **`L_decay`** (align α_t with the hand-specified decay function, then anneal the weight to 0 so
  it learns freely — a working prior instead of a cold start)
- ⭐ **The synthesis that makes this FRIDAY's and not NVIDIA's:** the compiler's **decay function**,
  **retraction pairs** and **salience labels** supervise the operator's **three gate branches.**
  Retractions yield `(key_old, key_new)` — and `key_old ≠ key_new` is *precisely* why EDA's
  independently-addressed erase is needed, since a write-anchored erase (GDN-2) cannot reach it.
- Salience side-channel: while compressing, the RSC flags spans worth writing to the lossless store.
  **Compression and fact extraction in one forward pass** — this unifies S1 with the Ledger's
  transcript slot.
- Ship behind a feature flag; **A/B against the compaction ladder on the locked eval suite**
- ⚠️ **Design for swappability.** Four substantive papers in five months (GDN-2 May, EDA Jun,
  query-erase Aug, TERN Sep). One `RecurrentOperator` protocol, the state tensor shape
  (16×128×128) as the contract, the operator chosen by **YAML flag**, and every rung permanently a
  row in the eval table. Pin the arXiv ID and date you implemented against.
- **Exit test:** RSC beats the compaction ladder on the eval suite **and** holds needle recall
  ≥ baseline, **and `L_erase` improves under EDA vs GDN-2** (if it doesn't, the address-level
  coupling wasn't binding on your data — publish that too). Design so RSC failure degrades to
  *"FRIDAY searches memory more often,"* never *"FRIDAY forgets."*

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
- ⚡ **GEPA with *behavioural* reflection, not a cloud teacher** (₹0 constraint). Your S0 traces
  carry external signals — barge-ins, immediate rephrases, "no I meant…", unnecessary escalations,
  tool failures, gate failures. **These are a better feedback function than a model's opinion**, and
  they're free. Fall back to **DSPy MIPROv2** (Bayesian search over instructions + bootstrapped
  demos; raised HotPotQA ReAct accuracy **24% → 51%**) if reflection quality is too low.
  Optional: one ₹200 burst buys a lot of frontier-teacher reflection tokens for the *weekly* run only.
- Skill synthesis after complex tasks + **shadow-mode** A/B before promotion
- Weekly QLoRA distillation on Kaggle T4 → GGUF → swap into the L1 slot
- ⚡ **Continue-pretrain / QLoRA the *existing* over-trained backbone on your personal corpus —
  never from scratch.** LFM2.5-350M was trained at **80,000:1** tokens-per-parameter; 30 Kaggle
  hours buys ~1–3B tokens of continued pretraining, which is *meaningful* on top of that base and
  *hopeless* underneath it.
- ⚡ **Law 2c: re-run the needle-recall test after EVERY training run.** CoT post-training has been
  shown to *degrade* long-range recall in hybrids, and the Constitutional Eval Gate will **not**
  catch it — the gate measures task success, not retrieval. A checkpoint that scores higher on the
  eval suite and lower on recall is a regression, not a promotion.
- **Exit test:** publish a before/after number. "FRIDAY completed my recurring task 40%
  faster / with 30% fewer tokens after 30 days" — measured by the gate, not by vibes —
  **with needle recall flat or better.**

### **⭐ Phase 5.5 — The Tiny Specialists (Weeks 14–20)** ← *"best lightweight model, very accurate"*
*Deliverable: six narrow models that beat one general model at FRIDAY's job.*

This is the honest answer to "very accurate." **Not** *"the best 200M model in the world"* —
**"the best model in the world at being Jagan's assistant,"** where *the job* is defined by your
locked eval suite. Each of these is a tractable Kaggle job, and each measurably improves the system:

| Specialist | Params | Why a small model beats a general one |
|---|---|---|
| **The RSC** (Phase 3.5) | 10–100M | A compressor tuned for *your* conversation distribution. No general model is |
| **The L0 router** | 100–600M | 6-way classification + a confidence scalar, trained on *your* escalation traces. Sub-5 ms, zero output tokens |
| **The redactor** | 100–400M | India-specific PII NER. You have the labels; cloud models are worse at PAN / Aadhaar / IFSC than a tuned small model |
| **The salience head** | <10M | Binary: *did this span produce a fact?* Labels are free from Phase 1 |
| **The fact extractor** | 400M–1B | `(subject, predicate, object, valid_from)` from a span. Highly structured, highly repetitive, entirely your domain |
| **The wake-word + intent model** | <5M | Must run on an ESP32. Tiny is the requirement |

**Exit test:** each specialist beats the general backbone at its narrow task, measured, with the
gate's held-out set.

### **Phase 6 — Custom architecture (6+ months, *optional, research-only*)**
⚡ *Rescoped.* Originally "train the memory-operations router; distil S1–S3 into small models."
Now: **a 100–300M hybrid from scratch, purely to understand the architecture.**

- **3 Gated DeltaNet (or Mamba-2) : 1 Sliding Window Attention (w=2048)** — the production
  consensus ratio (Kimi K3, Qwen3.5, Nemotron 3, Granite 4, Jamba, Falcon-H1, Zamba2, Samba).
  **Not pure RetNet.** Optional: 1 full-attention layer at the very end (ICLR 2025: provably
  sufficient to close the representation gap).
- Short convolution on Q/K/V of the linear layers (Samba Table 10: helps SWA a lot, helps GLA less —
  GLA already has channel-level fine-grained decay). SwiGLU FFN; RoPE inside the SWA layers only.
- **Reuse LFM2's or Qwen3's tokenizer.** Do not train your own — it wastes parameters and data and
  you lose the ability to distil from existing models.
- Corpus: FineWeb-Edu / SlimPajama subset + **Tamil** (indic-nlp, OSCAR-Tamil) for your
  code-switching reality + synthetic FRIDAY-shaped `(history, fact, turn)` dialogues + your own S0
  traces once you have ~6 months.
- **Do a 200M-param validation run FIRST**, then scale only if the loss curve justifies it.
  Checkpoint on a wall-clock interval and copy off the VM immediately — free Kaggle can reclaim it
  without warning.
- **Exit test:** a reproducible benchmark vs the borrowed backbone, written up honestly.

> ⚠️ **Never let Phase 6 block Phases 0–5.** A from-scratch model on your compute budget will be
> *worse than GPT-2* at general language. Its value is understanding, not accuracy. You said you
> want both a working assistant and architectural research — so keep them on separate clocks.
> → [12 §7.1](./12-LINEAR-ATTENTION-PIVOT.md) has the full token/GPU-hour arithmetic.

### **Phase 7 — Skills & Task Management (Week 16+)**
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
