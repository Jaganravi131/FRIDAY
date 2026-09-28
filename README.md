# FRIDAY

> ### ⚡ NEW — read [12-LINEAR-ATTENTION-PIVOT.md](docs/architecture/12-LINEAR-ATTENTION-PIVOT.md) first
>
> The goal moved to **"build the best lightweight model that is very accurate"** at a **₹0**
> budget. Doc 12 re-architects FRIDAY around a **hybrid linear-attention backbone** so the
> Attention Ledger becomes a *physical state matrix* instead of a text prefix.
>
> **The pivot is right; RetNet specifically is wrong** — its fixed input-independent decay yields
> *"near-zero recall even when full-attention layers are added"* (arXiv 2507.06457), and recall is
> FRIDAY's entire product. The replacement is a **Gated DeltaNet / Mamba-2 hybrid at ~3:1**
> (the shape Kimi K3, Qwen3.5, Nemotron 3, Granite 4, Jamba, Falcon-H1 and Zamba2 all shipped),
> running today in llama.cpp as **LFM2 / LFM2.5** — 30 t/s at 2.6B, 38 t/s at LFM2-8B-A1B,
> official GGUFs, official on-device docs, vision variants included.
>
> **The innovation is Track B: the Retention State Compiler** — a trained ~40M-param gated-delta
> module that compresses unbounded history into a fixed-size state *and* emits a salience
> side-channel feeding fact extraction. Compression and memory in one forward pass. Fits a free
> Kaggle T4. Nobody has published this for a personal assistant.
>
> ### ⚡ UPDATE — [13-GATED-DELTANET2-ERASE-WRITE.md](docs/architecture/13-GATED-DELTANET2-ERASE-WRITE.md)
>
> The RSC's operator is now specified: **NVIDIA's Gated DeltaNet-2** ([arXiv 2605.22791](https://arxiv.org/abs/2605.22791),
> May 2026, code on NVlabs) which **decouples erase from write** into separate channel-wise gates.
> Its own ablation says **the erase gate accounts for most of the gain** — and erasing accumulated
> transient noise is exactly FRIDAY's dominant memory problem. It moves RULER needle retrieval
> **63 → 90**, i.e. it targets the precise benchmark doc 12 named as decisive.
>
> **Three corrections:** (1) *"permanent associative records"* is a category error — 90 ≠ 100 and
> nothing in a fixed-size state is permanent, so **Law 2b stands**; (2) it **cannot** go in Phase 0 —
> the NVlabs repo is a *training* harness with **no inference server, no GGUF, no llama.cpp support**,
> and you cannot put GDN-2 layers inside a frozen LFM2; (3) **SWA is a layer, not your retrieval
> store.** It goes in **Phase 3.5 as the RSC's operator**, with a frozen backbone — where every
> blocker disappears.
>
> **Plus a successor that fits FRIDAY better: EDA** ([2606.26560](https://arxiv.org/abs/2606.26560),
> Jun 2026) — *"GDN-2 decouples **how strongly** erase and write are applied; EDA decouples
> **where**."* GDN-2 still cannot clear a stale association stored at a *different address* than the
> current write, which is precisely FRIDAY's bi-temporal correction case. They compose.
>
> ⭐ **And the genuinely new idea:** GDN-2 has three gate branches (decay α_t, erase b_t, write w_t)
> and FRIDAY's Memory Compiler already computes three matching signals — the **decay function**,
> **retraction events**, and the **salience head**. Retractions yield `(key_old, key_new)` pairs:
> free supervision for EDA's independently-addressed erase. **A recurrent memory operator supervised
> by a bi-temporal knowledge base.** → [13 §4.1](docs/architecture/13-GATED-DELTANET2-ERASE-WRITE.md)
>
> **Phase 0 is unchanged.** The only immediate action: start recording retraction pairs in Phase 1 —
> they're `L_erase`'s training data and you can't backfill them.

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
| **[12-LINEAR-ATTENTION-PIVOT.md](docs/architecture/12-LINEAR-ATTENTION-PIVOT.md)** | ⭐ **NEW — read after 00.** The hybrid linear-attention pivot: why RetNet specifically is the wrong choice (proven recall failure), the two-memory split, **Track A** production backbone (LFM2/LFM2.5, RWKV-7), **Track B** the Retention State Compiler, **Track C** custom architecture + the honest ₹0/from-scratch token math, the ₹0 budget consequences, and the revised roadmap |
| **[13-GATED-DELTANET2-ERASE-WRITE.md](docs/architecture/13-GATED-DELTANET2-ERASE-WRITE.md)** | ⭐ **NEW — read after 12.** Specifies the RSC's recurrent operator: **Gated DeltaNet-2** (decoupled channel-wise erase/write gates, NVIDIA May 2026) and its successor **EDA** (decoupled erase/write *addresses*, Jun 2026). Verified math and benchmarks, the full DeltaNet→KDA→GDN-2→EDA lineage, three corrections to the proposal, the **gate-supervision synthesis**, a 5-rung ablation ladder, the T4/Triton engineering reality, and the swappable-operator design |

---

## The 9 innovations

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
| **9** | ⭐ **The Two-Memory Split + Retention State Compiler** | A **lossy O(1) recurrent state** carries conversational *context*; a **lossless external store** carries *facts*. A trained gated-delta module compiles one into the other — and its **salience head feeds fact extraction in the same forward pass**. Compression and memory, unified. *([12](docs/architecture/12-LINEAR-ATTENTION-PIVOT.md))* |
| **9b** | ⭐ **Gate supervision from a bi-temporal knowledge base** | GDN-2/EDA expose three gate branches — **decay α_t, erase b_t, write w_t**. FRIDAY's Memory Compiler already computes three matching signals: the **decay function**, **retraction events** (which yield `(key_old, key_new)` pairs — free supervision for EDA's *independently-addressed* erase), and the **salience head**. A recurrent memory operator whose forgetting is supervised by what the compiler decided to keep. *([13](docs/architecture/13-GATED-DELTANET2-ERASE-WRITE.md))* |

> Innovation 9 rests on a distinction the literature hands us directly: *"a hybrid paired with
> retrieval **sidesteps** the recall gap rather than solving it: you do not ask the recurrent state
> to memorize a fact you can fetch from an index."* FRIDAY already has the index.
>
> Innovation 9b is the part that isn't in any paper I found: 9's store was built to *answer
> questions*, and it turns out to also be a **supervision signal for the state's forgetting**.
> Decoupled erase/write is what makes that supervision expressible at all — under a scalar tie
> (Gated DeltaNet, KDA) there is no separate erase knob to supervise.

---

## The design laws

1. **Proactivity is a heartbeat, not a cron job.** It must be able to decide *not* to speak.
2. **Don't summarise conversation history by default.** *(For attention backbones.)* Shrinking the
   prefix by *rewriting* it breaks prompt caching and dropped recall from 92% → ~33% in one
   production eval. **Cap** tool output instead: −38% cost/turn, recall unchanged. *Shrink the
   prefix, don't rewrite it.*
   **Δ With a recurrent-state backbone this law largely dissolves** — there is no prefix to
   cache-break, and compaction becomes a *learned, continuous* operation instead of a semantic
   break. See [12 §1.2](docs/architecture/12-LINEAR-ATTENTION-PIVOT.md). Rung 0 (cap tool output)
   stays regardless — it's free and it works.
3. **Just-in-time retrieval beats pre-packing.** Keep pointers; pull on demand.
4. **Retrieval without a reranker is noise.** Wide → rerank → narrow → hard score floor → `[]`
   beats noise.
5. **Enforce permissions in the data layer, never in the prompt.** Three layers: capability
   gating, execution interception, data-layer middleware.
6. **The agent must not be able to edit its own evaluator.** Locked suite, shadow mode, gate.
7. **Memory files are human-readable and human-editable.** Git is the undo button.
8. **One language for the brain, one process for the truth.** Resist microservices.
9. ⭐ **Never ask a lossy state to be a database.** The recurrent state holds *context*; the
   external store holds *facts*. Verbatim recall is the one capability linear attention provably
   lacks — so route it to the index, always. ICLR 2025 proves this is *sufficient*: in-context RAG
   plus one attention layer closes the representation gap entirely.
10. ⭐ **Recall is a property of a checkpoint, not an architecture.** CoT post-training has been
    shown to *degrade* long-range recall in hybrids. **Re-run the needle test after every
    fine-tune**, or you will silently destroy the capability that made the model worth using.

---

## Target hardware

**ASUS VivoBook · AMD Ryzen 5 · Radeon iGPU · 16 GB RAM · 512 GB SSD · Windows 11**

| | |
|---|---|
| ✅ **Hybrid linear-attention (preferred)** | **LFM2-2.6B Q4 → 30 t/s** · **LFM2-8B-A1B → 38 t/s** (8B with only 1B active) · LFM2.5-1.2B → tiny footprint, huge `num_ctx` headroom. Official GGUFs + official llama.cpp on-device docs, vision variants included |
| ✅ **The real prize: no KV-cache growth** | A hybrid's state is **fixed-size**, so `num_ctx` goes from 8–12K (Transformer, KV-bound) to **32K–128K on the same 16 GB**. The Attention Ledger stops rationing and starts routing |
| ✅ **Transformer baseline (keep for comparison)** | Qwen3-4B Q4 — ~20 t/s. You need it to *prove* the hybrid helps |
| ✅ **Free tier** | Qwen3-0.6B / LFM2-350M — **30–86 t/s**, runs on every turn |
| ✅ **Vision** | LFM2-VL-1.6B / LFM2.5-VL-450M (same family) or Qwen3-VL-4B (DocVQA **95.3**) |
| ✅ **iGPU prefill** | ~54 t/s (vs 14–16 CPU) — offload prefill, keep decode on CPU |
| ✅ **Training** | Kaggle free tier — **2× T4, 30 GPU-h/week, ₹0**. ≈ **1.1B tokens/week** of training throughput |
| ⚠️ **iGPU decode** | **Not faster than CPU** (4.5–5.3 t/s either way) |
| ⚠️ **₹0 budget costs you natural voice** | Piper is robotic. This is the one place the budget and "very natural and realistic" genuinely conflict |
| ❌ **From-scratch pretraining** | 30 T4-h/week ≈ 55B tokens/**year** at best. A from-scratch 150M model on 15B tokens is **worse than GPT-2**. **Continue-pretrain LFM2.5-350M (trained at 80,000:1) instead** |
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
| **0 — Foundation** | 1 | Talk to FRIDAY in a terminal; it remembers. **Δ Now starts with the backbone bake-off** ([12 §5.3](docs/architecture/12-LINEAR-ATTENTION-PIVOT.md)) | Monday fact → Friday follow-up; `BENCHMARKS.md` with your own needle-recall numbers |
| **1 — Memory Compiler** | 2–4 | A model of you that improves unprompted. **Δ Plus the salience-label tap — start collecting in Week 2, not Week 12** | Contradict a fact → both answers correct, with dates |
| **2 — Attention Ledger** | 4–6 | Answers that are actually about you, cheap and fast. **Δ Ladder shortens to Rung 0 + Rung 4 if the backbone is a hybrid** | 200-turn session; turn 200 as coherent as turn 5; flat cost |
| **3 — Perception & Presence** | 6–9 | Voice, vision, cross-device continuity | Ask walking (phone) → answer on laptop; barge-in <200 ms |
| **⭐ 3.5 — Retention State Compiler** | 8–12 | **Track B.** History compressed into a **fixed-size state** + a salience side-channel feeding fact extraction | RSC beats the compaction ladder on the locked eval suite *and* holds needle recall ≥ the baseline |
| **4 — Proactivity** | 9–11 | FRIDAY speaks first, and you're glad | 1 week: ≥5 useful, **0** annoying initiations |
| **5 — Self-Improvement** | 11–16 | Measurably better without you editing prompts. **Δ Behavioural reflection, not a cloud teacher** | A published before/after number from the gate; **needle recall re-measured after every training run** |
| **⭐ 5.5 — Tiny specialists** | 14–20 | **Track C-lite.** Router, redactor, salience head, fact extractor, wake word — 10M–1B params each | Each specialist beats the general backbone at its narrow task |
| **6 — Custom architecture** *(optional, research)* | 6+ mo | 100–300M hybrid (3 Gated DeltaNet : 1 SWA) on `flash-linear-attention`, continued-pretrained on your corpus | A reproducible benchmark vs the borrowed backbone. **Never blocks production** |
| **7 — Skills & Tasks** | 16+ | Calendar, email, notes, finance, home | Each lands *with* context, not as a dumb API wrapper |

> **Do not skip phases.** The most common failure is building integrations first and having
> nothing worth integrating.
>
> **Do not let Track C block Track A.** The custom architecture is for understanding; the borrowed
> hybrid is for shipping. Both goals are yours — keep them on separate clocks.

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
