# FRIDAY

**A personal AI agent that compiles your life into a progressively smaller, cheaper, more personal
model.**

Not a chatbot with tools bolted on. FRIDAY is built around one thesis:

> Every interaction produces a **trace**. That trace flows down a pipeline and is *compiled* —
> into episodes, then facts, then skills, then prompt optimizations, and finally into **model
> weights**. Each stage is cheaper, faster and more durable than the one before, and each has a
> **gate** that can reject and roll back.

That single idea makes FRIDAY context-aware, self-improving, and a genuine research project.

---

## 📐 The Blueprint

**Status: architecture complete, implementation not started.**

Everything below is researched, sourced, and sized against real hardware.

| Doc | What's in it |
|---|---|
| **[00-BLUEPRINT.md](docs/architecture/00-BLUEPRINT.md)** | ⭐ **Start here.** Thesis, the 8 innovations, system diagram, 8 design laws, recommended stack, hardware verdict, 7-phase roadmap, repo layout, success metrics, top 10 ways this dies |
| [01-research-landscape.md](docs/architecture/01-research-landscape.md) | What exists in Sept 2026. The 5 projects to study, memory frameworks compared, realtime voice measured latencies, MCP spec state, context-engineering consensus, the self-improvement taxonomy, **what to skip entirely** |
| **[02-INNOVATION-memory-compiler.md](docs/architecture/02-INNOVATION-memory-compiler.md)** | ⭐ **The core novelty.** The 5-stage cascade (Trace → Episode → Fact → Skill → Weight), bi-temporal facts, Markdown-as-source-of-truth, the 7 gates, the decay function, the retrieval path, your transformer-research menu |
| [03-memory-architecture.md](docs/architecture/03-memory-architecture.md) | Full SQLite schema, storage topology, the 4 scopes, context slot budgets, Tamil+English handling, cross-device sync, backup & recovery, sizing on 512 GB |
| **[04-INNOVATION-attention-ledger.md](docs/architecture/04-INNOVATION-attention-ledger.md)** | ⭐ Why answers will be *clear*. The cached-prefix / JIT-zone split, the 6-rung compaction ladder, the 4-part retrieval pipeline, progressive tool disclosure, **reference implementation**, telemetry, 9 anti-patterns |
| **[05-model-tiers-hardware.md](docs/architecture/05-model-tiers-hardware.md)** | ⭐ **Measured numbers for this exact machine class.** What works, what doesn't, the exact model menu with tokens/sec, the Escalation Ladder, WSL2 setup, the benchmark script, the two upgrades worth making |
| **[06-INNOVATION-self-improvement-loop.md](docs/architecture/06-INNOVATION-self-improvement-loop.md)** | ⭐ The Constitutional Eval Gate & Shadow Mode. The full eval suite (5 suites, real YAML), grader calibration, the 4 self-improvement loops in build order, the governance ladder, the gate log artifact |
| [07-realtime-voice-presence.md](docs/architecture/07-realtime-voice-presence.md) | Voice that feels human (measured latency budgets, turn detection, **the 3 bugs that eat your week**) + the Presence Fabric (rooms/surfaces, capability union, handoff, output routing) |
| [08-proactive-heartbeat.md](docs/architecture/08-proactive-heartbeat.md) | The Heartbeat, `HEARTBEAT_OK` suppression, the circadian rhythm, two scopes / one memory, the confirmation gate, the annoyance-budget controller, the always-on problem |
| [09-security-privacy-trust.md](docs/architecture/09-security-privacy-trust.md) | The Sense Registry (tiered consent), 3-layer deterministic enforcement, India-specific PII redaction, prompt-injection defence, MCP supply-chain policy, sandboxing, the audit log, **the trust features**, threat model |
| **[10-phase-0-implementation.md](docs/architecture/10-phase-0-implementation.md)** | ⭐ **Start coding here.** A day-by-day Week 1: benchmark your machine, WSL2 config, repo skeleton, real `SOUL.md`/`AGENTS.md`/`USER.md`, the Markdown→SQLite compiler, retrieval, the agent loop, a 12-point exit test |
| [11-research-sources.md](docs/architecture/11-research-sources.md) | Every source, grouped, with the specific data taken from each. Plus two claims to treat carefully |

---

## The 8 innovations

| # | Innovation | One-liner |
|---|---|---|
| 1 | **The Memory Compiler** | Trace → Episode → Fact → Skill → Weight, with gates + rollback |
| 2 | **Markdown-as-source-of-truth** | You can open a file and correct FRIDAY. Indices are build artifacts. |
| 3 | **Bi-temporal facts** | Every fact has `valid_from`/`valid_to` *and* `asserted_at`/`retracted_at` |
| 4 | **The Attention Ledger** | Context assembly as a budgeted, cached, deterministic schedule |
| 5 | **The Escalation Ladder** | 6-tier model routing. Cheapest tier that can do the job wins. |
| 6 | **Constitutional Eval Gate** | FRIDAY may edit itself. It may never edit its examiner. |
| 7 | **Presence Fabric** | One *room*, many *surfaces*. Phone → laptop, mid-sentence. |
| 8 | **The Sense Registry** | Tiered consent, enforced deterministically in the data layer |

---

## The design laws

1. **Proactivity is a heartbeat, not a cron job.** It must be able to decide *not* to speak.
2. **Don't summarise conversation history by default.** Shrinking the prefix by *rewriting* it
   breaks prompt caching and dropped recall from 92% → ~33% in one production eval. **Cap** tool
   output instead: −38% cost/turn, recall unchanged. *Shrink the prefix, don't rewrite it.*
3. **Just-in-time retrieval beats pre-packing.** Keep pointers; pull on demand.
4. **Retrieval without a reranker is noise.** Wide → rerank → narrow → hard score floor → `[]`
   beats noise.
5. **Enforce permissions in the data layer, never in the prompt.** Three layers: capability
   gating, execution interception, data-layer middleware.
6. **The agent must not be able to edit its own evaluator.** Locked suite, shadow mode, gate.
7. **Memory files are human-readable and human-editable.** Git is the undo button.
8. **One language for the brain, one process for the truth.** Resist microservices.

---

## Target hardware

**ASUS VivoBook · AMD Ryzen 5 · Radeon iGPU · 16 GB RAM · 512 GB SSD · Windows 11**

| | |
|---|---|
| ✅ **Daily driver** | Qwen3-4B Q4 — **~20 t/s** CPU-only |
| ✅ **Free tier** | Qwen3-0.6B Q8 — **~60–86 t/s**, runs on every turn |
| ✅ **Vision** | Qwen3-VL-4B — DocVQA **95.3**, ~3.5 GB |
| ✅ **iGPU prefill** | ~54 t/s (vs 14–16 CPU) — offload prefill, keep decode on CPU |
| ✅ **Training** | Kaggle free tier — **2× T4, 30 GPU-h/week, ₹0** |
| ⚠️ **iGPU decode** | **Not faster than CPU** (4.5–5.3 t/s either way) |
| ❌ **Qwen3-30B-A3B** | **Won't load** at 16 GB |
| ❌ **Local fine-tuning** | No CUDA; ROCm-on-Windows experimental |

**Highest-ROI upgrade: 32 GB RAM (~₹4–6k) if a SODIMM slot is free.** Unlocks Qwen3-30B-A3B MoE
at ~**27 t/s** CPU-only and removes every memory constraint in the docs.

---

## Stack

**Python core + TypeScript surfaces.**

`WSL2 Ubuntu 24.04` (mirrored networking) · `llama-server` Vulkan, Windows-native · `FastAPI` ·
hand-rolled `StateGraph` · **MCP `2026-07-28`** (stateless) via `FastMCP 3.0` ·
**agentskills.io** `SKILL.md` · **Markdown in git** as truth · `SQLite + FTS5 + sqlite-vec` as
indices · `Qwen3-Embedding-0.6B` / `bge-m3` · `bge-reranker-base` · `faster-whisper` ·
`Silero VAD` · `microWakeWord` · `Piper` (local) / `Cartesia Sonic` (natural) · `Pipecat` →
`LiveKit Agents` · `screenpipe` for ambient capture · `APScheduler` · **`DSPy` + `GEPA`** ·
**`Unsloth` QLoRA on Kaggle T4** · `Tailscale` · `Tauri v2` · `React + Vite` PWA · `Expo`

---

## Roadmap

| Phase | Weeks | Deliverable | Exit test |
|---|---|---|---|
| **0 — Foundation** | 1 | Talk to FRIDAY in a terminal; it remembers | Monday fact → Friday follow-up |
| **1 — Memory Compiler** | 2–4 | A model of you that improves unprompted | Contradict a fact → both answers correct, with dates |
| **2 — Attention Ledger** | 4–6 | Answers that are actually about you, cheap and fast | 200-turn session; turn 200 as coherent as turn 5; flat cost |
| **3 — Perception & Presence** | 6–9 | Voice, vision, cross-device continuity | Ask walking (phone) → answer on laptop; barge-in <200 ms |
| **4 — Proactivity** | 9–11 | FRIDAY speaks first, and you're glad | 1 week: ≥5 useful, **0** annoying initiations |
| **5 — Self-Improvement** | 11–16 | Measurably better without you editing prompts | A published before/after number from the gate |
| **6 — Skills & Tasks** | 16+ | Calendar, email, notes, finance, home | Each lands *with* context, not as a dumb API wrapper |

> **Do not skip phases.** The most common failure is building integrations first and having
> nothing worth integrating.

---

## The one habit that decides whether this ships

**Talk to FRIDAY every day starting Day 6 of Phase 0, even when it's bad.**

Every personal-AI project dies the same way: six weeks of infrastructure, never using the thing,
losing the feedback loop, abandoning it. The Memory Compiler only compiles *what you feed it*.
A week of real conversation beats a month of architecture.

---

## Metrics instrumented from day 1

| | Target |
|---|---|
| Time-to-first-audio (voice) | < 800 ms local / < 1.2 s cloud |
| Time-to-first-token (text) | < 400 ms |
| Barge-in stop | < 200 ms |
| Prompt cache-hit rate | > 85% |
| Tokens/turn over a 200-turn session | **flat** |
| Retrieval precision@3 | > 0.70 |
| Useful : annoying proactive initiations | > 5 : 1 |
| Heartbeat suppression rate | > 80% |
| Eval-gate delta after 30 days | ≥ +10% |
| Denied-data accesses that succeeded | **0** |
| Cost per day | < ₹100 |
