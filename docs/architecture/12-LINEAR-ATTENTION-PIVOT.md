# 12 — The Linear-Attention Pivot: Retention State Architecture

> **Trigger:** the goal moved from *"use a good small model"* to **"build the best lightweight
> model that is also very accurate"** — and a proposal to swap standard attention for
> **RetNet's recurrent, O(1) state engine**, turning the Attention Ledger into a *physical state
> matrix* instead of a text prefix.
>
> **Budget:** ₹0 cloud. Free tiers only.
>
> **Verdict: the instinct is right. The specific architecture is wrong. The replacement is
> better than either.**

---

## 1. What the proposal gets RIGHT (and it's a lot)

### 1.1 The KV cache is genuinely your binding constraint

This is correct and I under-weighted it in [05](./05-model-tiers-hardware.md). Your bottleneck is
**RAM bandwidth and RAM capacity**, and the KV cache is the part of the bill that *grows*:

```
KV cache bytes = 2 (K and V) × n_layers × n_kv_heads × head_dim × 2 bytes × context_length
```

On a 16 GB machine with Windows taking 5 GB, **the KV cache is what forces `num_ctx` down to 8K**
and therefore what forces the Attention Ledger to be stingy. A fixed-size recurrent state deletes
that entire term.

The RetNet paper measured it: at 6.7B, RetNet's **additional** inference memory beyond the weights
is *"almost negligible (i.e., about 3%) while the model weights occupy 97%"* — and throughput is
**length-invariant**. Transformer memory grows with every token; retention's doesn't.

**Real production proof this matters:** NVIDIA's Nemotron 3.5 Lightning 30B-A3B — a hybrid
Mamba-2 + attention + MoE model with 52 layers (23 Mamba-2, 23 MoE, 6 attention) — has a KV cache
of **~6 KB per token** in llama.cpp, because its six attention layers use only two KV heads.
That's what makes a 30B model servable on consumer hardware. IBM reports **>70% memory reduction**
on Granite 4.0-H.

### 1.2 Law 2 genuinely does dissolve

My Law 2 said *"shrink the prefix, don't rewrite it — summarising breaks prompt caching."* That
law is a property of **KV-prefix caching in attention models**. It is real, but it is
*architecture-contingent*.

With a recurrent state:
- There is no prefix to cache. The state **is** the cache.
- Rewriting the state is not a cache miss — it's just the next recurrent step.
- Compaction stops being a semantic break and becomes a **learned, continuous operation**.

So the peer is right: **the entire 6-rung compaction ladder in [04 §Law B](./04-INNOVATION-attention-ledger.md)
collapses.** Rungs 1–5 (cap, truncate, dedup, offload, evict) become *one* learned operation:
compile history into state. That is a real architectural simplification, and it's the strongest
part of the proposal.

### 1.3 The "Memory Compiler as physical state matrix" framing is genuinely novel

My S1 stage said "summarise the day into narrative Markdown." That's a *text* compaction, done by
an LLM, at night, lossily.

The proposal says: **the compaction is a differentiable module that emits a fixed-size state
vector, trained end-to-end, running at O(1) per token.**

That is strictly more interesting, and it's the actual innovation here. It also unifies two things
I had as separate stages — S1 (episode) and the Ledger's transcript slot — into one mechanism.

### 1.4 The decay-as-forgetting intuition is elegant

*"Transient chatter decays naturally via the decay factor γ; important high-weight inputs are
intercepted by the sqlite-vec engine."*

That maps beautifully onto my [02 §4 decay function](./02-INNOVATION-memory-compiler.md) — where I
hand-tuned half-lives per predicate class. A learned γ is the same idea, optimised instead of
guessed.

**But — see §2.3. This exact mechanism is what breaks FRIDAY.**

---

## 2. What the proposal gets WRONG

### 2.1 RetNet is the *worst* linear architecture for this use case, and it's measurable

RetNet's decay is **fixed and input-independent**: `α_t = γ`, with γ ∈ (0,1) learned once and
shared across all positions. That gives the simplest decay and the fastest inference — and
**no content adaptivity**. It cannot decide *what* to keep.

From **A Systematic Analysis of Hybrid Linear Attention** (arXiv 2507.06457):

> *"Architectures that expose their hidden state to a learned, token-wise gate — **GatedDeltaNet
> and HGRN-2** — consistently perform best in recall once hybridised, **surpassing the Transformer
> baseline by 2–5 percentage points** at the optimal ratio. By contrast, **RetNet's fixed
> exponential decay fails to protect long-range cues, yielding near-zero recall even when
> full-attention layers are added.**"*

Read that twice. RetNet doesn't just lose to alternatives — **adding attention layers to RetNet
doesn't fix it.** The fixed decay destroys the long-range cue before the attention layer can see
it.

Benchmark table from the same paper (ARC-c / ARC-e / HellaSwag / LAMBADA / OBQA / PIQA):

| Model | Avg |
|---|---|
| **RetNet 6-1** | **0.547** |

And from Songlin Yang's Mamba-2 / retention survey, on synthetic recall tasks:

| Model | Compress | Fuzzy Recall | In-Context Recall | Memorize | Noisy Recall | Selective Copy | **Average** |
|---|---|---|---|---|---|---|---|
| **Transformer** | 51.6 | 29.8 | **94.1** | 85.2 | 86.8 | **99.6** | **74.5** |
| **Mamba** | 52.7 | **6.7** | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| **DeltaNet** | — | — | **perfect recall** | — | — | — | — |

The progression is clear and it has a name:
```
vanilla linear attention   S_t = S_{t-1} + v_t k_t^⊤        → unstable, underperforms badly
RetNet (Gen-2, fixed γ)    S_t = γ S_{t-1} + v_t k_t^⊤      → stable, but NOT selective
GLA (Gen-2, learned gate)  S_t = G_t ⊙ S_{t-1} + v_t k_t^⊤  → per-token diagonal gate
Mamba-2 / RWKV-6           data-dependent scalar gate       → same design axis
DeltaNet                   delta rule (overwrite by key)    → perfect synthetic recall
Gated DeltaNet             gate + delta rule                → beats Mamba2 AND DeltaNet
```
**RetNet is generation 2 of 5.** It is the 2023 stepping stone that the field has since passed.
Building FRIDAY on it in 2026 is like choosing RNNs over LSTMs in 2016.

### 2.2 The recall gap is *proven*, not empirical

**RNNs Are Not Transformers (Yet)** — ICLR 2025:

> *"We prove that RNNs are **not expressive enough** to solve [associative recall, IsTree] while
> Transformers can solve them with ease… CoT improves the representation power of RNNs, but **to
> close the gap with Transformers, CoT alone is not enough**."*
>
> The bottleneck is **in-context retrieval**, and it follows directly from memory efficiency: for a
> model with at most *o(n)* bits in memory, streaming complexity gives **impossibility results**.

And **NeurIPS 2025 Spotlight, "Achilles' Heel of Mamba"**: SSMs *"systematically fail on copy and
recall tasks that pose no problem for Transformers. This weakness is **not an implementation
detail that can be corrected with more data or better tuning**."*

IBM's Granite 4 results confirm it in production: *"pure SSM models match or exceed Transformers on
many tasks, but Mamba and Mamba-2 models remain significantly behind on tasks requiring strong
copy or in-context learning"* — notably **five-shot MMLU**, where the model must generalise from
examples in the prompt.

### 2.3 Why this is catastrophic *specifically for FRIDAY*

Here is the part that matters. Look at what FRIDAY actually does:

| FRIDAY's job | Required capability |
|---|---|
| "What's my rent?" | **exact recall of a specific earlier token** |
| "Remember that thing about the lease?" | **associative recall** |
| "Why do you think I live in Velachery?" | retrieval + provenance |
| Following a 10-step instruction | in-context learning |
| Not repeating itself mid-session | verbatim retention of what it just said |
| Code editing | **copy tasks** |

**Every single one of these is precisely the capability that pure linear attention provably lacks.**

The peer's proposal — *"history isn't cached as a massive KV-text prefix, it's compressed into a
fixed-size recurrent state"* — is architecturally elegant and would **delete FRIDAY's core value
proposition.** A fixed-size state is *"by construction, a lossy memory"*, and *"what it loses first
is associative recall: retrieving a specific earlier token verbatim."*

> **FRIDAY's entire product is associative recall about you.** You cannot build a memory assistant
> on an architecture proven incapable of memory.

### 2.4 "RetNet-2.7B" as a downloadable model barely exists

The config is real (paper Table 2: **2.7B = 32 layers, hidden 2560, FFN 5120, 10 heads, 100B
tokens, lr 3e-4**). The checkpoints are not, in any usable sense:

| Artifact | Reality |
|---|---|
| Microsoft's original | Trained on **512 AMD MI200 GPUs**, TorchScale. Research comparison against Transformer baselines at 1.3B/2.7B/6.7B — **not a released product**. Code MIT-licensed. |
| [`fla-hub/retnet-2.7B-100B`](https://huggingface.co/collections/fla-hub/retnet) | Exists. **58 downloads. 1 like.** A research reproduction from the Flash-Linear-Attention hub. |
| `fla-hub/retnet-1.3B-100B` | 685 downloads. |
| The first community HF attempt | Author's own words: *"undertrained (loss=8!)"* |
| Frenos' own RetNet | Pre-trained on **only 4B tokens** due to compute constraints. *"Performs on par with industry-standard models of its size (~3B)."* |

**100B tokens is badly undertrained by 2026 standards.** TinyLlama did 3T. And the original paper's
own reviewers noted the 6.7B at 100B tokens was undertrained — *"if you look at Figure 1 in the
LLaMA paper, the curve (even for 7B) is still very steep at 100B tokens."* RetNet is also
**worse than Transformer at 1.3B**; it only crosses over above ~2B.

### 2.5 The field's verdict

> *"As of 2026 there is **no publicly available frontier-scale large language model that uses pure
> retention as its sole sequence operator**, and the dominant hybrid pattern in efficient
> architectures today is to **interleave attention layers with linear-state operators** rather than
> committing fully to one or the other."*

Everyone shipped the same shape:

| Model (lab) | Layout | Scale |
|---|---|---|
| **Kimi K3** (Moonshot, Jul 2026) | **3 Kimi Delta Attention : 1 gated MLA** (69 KDA / 24 MLA over 93 text layers) | 2.8T total, 104B active, 1M ctx — largest open-weight model |
| **Qwen3.5-397B-A17B** (Alibaba) | **3 Gated DeltaNet : 1 full attention** | 397B / 17B active |
| **Nemotron 3 Ultra** (NVIDIA, Jun 2026) | Mostly Mamba-2, minority attention, hybrid latent MoE | 550B / ~55B active, 1M ctx |
| **Granite 4.0/4.1** (IBM) | ~**9 Mamba-2 : 1 attention** | H-Small 32B / ~9B active; **>70% memory reduction** |
| **Jamba 1.5** (AI21) | Mamba + attention blocks + MoE | 398B/94B; Mini 52B/12B, 256K ctx |
| **Falcon-H1** (TII) | attention + Mamba-2 heads **in parallel inside one mixer block** | **0.5B → 34B** |
| **Zamba2 / Zamba2-VL** (Zyphra) | Mamba-2 backbone + **shared** attention blocks | **1.2B → 7B**, Apache 2.0, incl. **vision-language** |
| **Samba** (ICLR 2025) | Mamba + **Sliding Window Attention** (w=2048) + SwiGLU, layer-wise interleaved | linear complexity, *potentially infinite* length extrapolation |
| **Gated DeltaNet-H1/H2** (ICLR 2025) | GDN + SWA; or Mamba2 + GDN + SWA | beats Mamba2 and DeltaNet on in-context retrieval |

**The consensus ratio is 3:1 linear-to-attention.** Not 100% linear. Not 100% attention.

### 2.6 One more warning the proposal misses

> *"**Recall is a property of a checkpoint, not an architecture.** Chain-of-thought fine-tuning has
> been shown to degrade long-range recall in hybrids, so base-model recall numbers do not transfer
> to the reasoning checkpoint you deploy. **Measure after post-training.**"*

The repair proposed in that literature is targeted (**QK-Restore**), not "add more data."

**Consequence for FRIDAY:** when you QLoRA/continue-pretrain a hybrid model on your personal
corpus, you may *destroy* the recall that made it worth using. Your eval suite must measure recall
**before and after every training run.** This is exactly what the Constitutional Eval Gate's
held-out regression check is for — and it's why `eval/suites/personal-recall.yaml` is weighted 0.25.

---

## 3. The synthesis: **two memories, two substrates**

Here is the resolution, and it is not a compromise — it is the *correct* design, and the literature
hands it to us directly:

> *"**A hybrid paired with retrieval sidesteps the recall gap rather than solving it: you do not ask
> the recurrent state to memorize a fact you can fetch from an index.** The architecture's weak
> axis, exact lookup, is precisely what an external store is good at."*

**FRIDAY already has the external store.** That's the entire Memory Compiler. So:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE TWO-MEMORY SPLIT                                 │
├─────────────────────────────────┬───────────────────────────────────────┤
│   RETENTION STATE (lossy, O(1)) │   EXTERNAL STORE (lossless, exact)    │
│   ─ the linear-attention layers │   ─ SQLite + FTS5 + sqlite-vec        │
│                                 │   ─ Markdown-as-truth, git-versioned  │
├─────────────────────────────────┼───────────────────────────────────────┤
│ carries:                        │ carries:                              │
│  · conversational gist          │  · FACTS ("rent is ₹28,000")          │
│  · current topic & thread       │  · NAMES, NUMBERS, DATES              │
│  · your register & mood         │  · verbatim quotes + provenance       │
│  · what FRIDAY just said        │  · bi-temporal validity windows       │
│  · priming for the next turn    │  · skills, episodes, traces           │
├─────────────────────────────────┼───────────────────────────────────────┤
│ cost: fixed, ~O(1), no growth   │ cost: exact lookup, 10–150 ms         │
│ fails at: verbatim recall  ⚠️   │ fails at: nothing (it's a database)   │
│ so we NEVER ask it to recall    │ so we ALWAYS ask it to recall         │
└─────────────────────────────────┴───────────────────────────────────────┘
```

**The recurrent state holds *context*. The external store holds *facts*.**
Never ask the state to be a database. Never ask the database to be a vibe.

This is not a workaround for a weakness — it's a **clean separation of concerns that happens to
map exactly onto the architecture's strengths.** And it has a theoretical warrant:

> ICLR 2025 proves two sufficient fixes for the RNN retrieval bottleneck:
> **(a)** *"allowing RNNs to invoke function calls to perform a primitive of in-context retrieval
> based on regular expression is sufficient to boost their representation power to solve **all
> polynomial-time solvable problems** with CoT"* — **that is FRIDAY's `memory_search` tool.**
> **(b)** *"adding **one Transformer layer at the end** of the RNN architecture is sufficient to
> close the representation gap."* — **that is the hybrid's attention layer.**
>
> And empirically: *"In-Context RAG allows all the models to reach near-perfect accuracy."*

**FRIDAY implements both (a) and (b).** That is not luck; it's the design falling out of the theory.

---

## 4. The revised architecture: three tracks

```
╔══════════════════════════════════════════════════════════════════════════╗
║  TRACK A — PRODUCTION BACKBONE  (Phase 0–3, ships, ₹0)                    ║
║  Use an EXISTING hybrid linear-attention model with official GGUF support ║
║  → LFM2.5 family (Liquid AI) and/or RWKV-7                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TRACK B — THE INNOVATION  (Phase 2+, the real research)                  ║
║  RETENTION STATE COMPILER: a trained module that compiles FRIDAY's        ║
║  history into a fixed-size state vector. The Memory Compiler made         ║
║  physical — exactly what the peer proposed, built on a gated architecture ║
║  that doesn't destroy recall.                                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TRACK C — CUSTOM ARCHITECTURE  (Phase 5+, ambitious)                     ║
║  A small hybrid (Gated DeltaNet / Mamba-2 + SWA) CONTINUE-PRETRAINED      ║
║  on your personal corpus. Not from scratch — that math doesn't close.     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 5. Track A — the production backbone

### 5.1 Why LFM2 / LFM2.5 is the right starting point

Liquid AI's **LFM2** (*"a new generation of **hybrid** models, designed for **on-device**
deployment"*) hits every one of your constraints:

| Requirement | LFM2 answer |
|---|---|
| **Runs in llama.cpp** | ✅ **Official Liquid AI docs for llama.cpp on-device**, incl. Windows CPU (`llama-*-bin-win-avx2-x64.zip`) and Vulkan |
| **Official GGUFs** | ✅ `LiquidAI/LFM2-350M-GGUF`, `LFM2-700M-GGUF`, `LFM2-2.6B-GGUF`, `LFM2.5-350M-GGUF`, `LFM2.5-1.2B-Instruct-GGUF` |
| **Vision in the same family** | ✅ `LFM2-VL-450M-GGUF`, `LFM2-VL-1.6B-GGUF`, `LFM2.5-VL-450M-GGUF` — **multimodal without a second stack** |
| **Measured on your hardware class** | ✅ From the CPU-only llama.cpp table: **LFM2-2.6B Q8_0 → 24 t/s**; **LFM2-2.6B i1-Q4_K_M → 30 t/s**; **LFM2-8B-A1B UD-Q4_K_XL → 38 t/s** |
| **Runs on a ₹0 budget** | ✅ Local, no API |
| **QAD quantisation** | ✅ `LFM2.5-350M-QAD-Q4_0.gguf` — quantisation-aware, so less quality loss at 4-bit |
| **Over-trained, not under-trained** | ✅ **LFM2.5-350M was trained at 80,000:1 tokens-per-parameter.** Compare Chinchilla's 20:1. This is the *opposite* of RetNet-2.7B's undertrained 100B tokens |

```bash
# The entire Track A setup, ₹0:
winget install llama.cpp     # Windows native
llama-server -hf LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M -c 8192 --port 8080 -ngl 99 --flash-attn 1
# or the bigger one:
llama-server -hf LiquidAI/LFM2-2.6B-GGUF:Q4_K_M -c 8192 --port 8080
# vision:
llama-server -m LFM2-VL-1.6B-Q8_0.gguf --mmproj mmproj-LFM2-VL-1.6B-Q8_0.gguf -c 4096 --port 8081
```

### 5.2 The revised model menu (replaces [05 §3](./05-model-tiers-hardware.md))

| Role | Model | Quant | RAM | Speed (CPU) | Why |
|---|---|---|---|---|---|
| **L0 gate / router / redaction** | `LFM2-350M` **or** `qwen3:0.6b` | Q4/Q8 | 0.2–0.7 GB | **30–86 t/s** | Benchmark both; LFM2-350M's QAD quant may win |
| **L1 main brain — option A** | **`LFM2.5-1.2B-Instruct`** | Q4_K_M | ~0.9 GB | est. **30–45 t/s** | Tiny footprint → huge `num_ctx` headroom. **The O(1)-state payoff** |
| **L1 main brain — option B** | **`LFM2-2.6B`** | i1-Q4_K_M | ~1.8 GB | **30 t/s** | Best quality/speed on your machine. **Recommended default** |
| **L1 main brain — option C** | **`LFM2-8B-A1B`** | UD-Q4_K_XL | ~5 GB | **38 t/s** | 8B with **1B active**. If it fits alongside Windows, this wins |
| **L1 alt (dense, for comparison)** | `qwen3:4b` | UD-Q4_K_XL | 2.6 GB | 20 t/s | **Keep it. You need a Transformer baseline to prove the hybrid helps** |
| **L1 alt** | `RWKV-7 World 2.9B` | Q8_0 | ~3 GB | ? t/s | In llama.cpp via @MollySophia's work. **Benchmark it** |
| **Vision** | `LFM2-VL-1.6B` or `LFM2.5-VL-450M` | Q8/Q4 | 0.5–1.6 GB | — | Same family = same tokenizer, same serving path |
| **Vision alt (stronger)** | `qwen3-vl:4b` | Q4 | 3.5 GB | 10–15 t/s | DocVQA **95.3**, MMMU 67.4. Benchmark against LFM2-VL on *your* documents |
| **Embeddings** | `Qwen3-Embedding-0.6B` / `bge-m3` | — | 0.6 GB | fast | bge-m3 for Tamil |
| **Reranker** | `bge-reranker-base` (ONNX) | — | 0.4 GB | 50–200 ms | Non-negotiable. Still the highest-value 400 MB |

### 5.3 The one benchmark that decides Track A

**Run this before writing any agent code.** It is the entire question, in one table:

```bash
# For each candidate: measure (a) speed, (b) RAM at num_ctx=8192, (c) RAM at num_ctx=32768,
# (d) RECALL. The recall test is the one that matters.

for M in LFM2.5-1.2B-Instruct LFM2-2.6B LFM2-8B-A1B RWKV-7-World-2.9B Qwen3-4B; do
  llama-bench -m $M.gguf -ngl 99 -fa 1 -p 512 -n 128          # speed
  llama-bench -m $M.gguf -ngl 99 -fa 1 -p 8192 -n 128         # long-prefill speed
done

# ── THE RECALL TEST (write this yourself, ~40 lines) ──────────────────────────
# Generate 20 conversations of exactly 8,000 tokens where a specific fact
# ("the API key was 7f3a9c2b") appears at position 500 / 2000 / 4000 / 6000 / 7500,
# then ask for it at the end. Score exact-match.
#
# This is the "needle in a haystack" test, and it is the ONE benchmark where
# linear-attention models collapse and hybrids hold. It directly measures
# whether FRIDAY will remember what you told it 200 turns ago.
python scripts/eval_recall.py --models "$CANDIDATES" --needles 5 --trials 20
```

**Decision rule:**
- If a hybrid scores ≥0.9 on needle recall at 8K **and** ≥25 t/s → **use it.** Track A wins.
- If hybrids collapse on recall (<0.6) → the recall gap is real on your scale, **stay with
  Qwen3-4B** and let the Attention Ledger + external store do the work. Track B becomes the
  research, not the product.
- If LFM2-8B-A1B fits in RAM and recalls well → **that's your answer**, 38 t/s at 8B quality.

> ⚠️ And per §2.6: **re-run this test after every fine-tune.** Recall is a property of a
> checkpoint, not an architecture.

### 5.4 What the O(1) state actually buys you on 16 GB

This is the payoff, and it's large:

| | Transformer (Qwen3-4B) | Hybrid linear (LFM2-2.6B class) |
|---|---|---|
| KV cache at 8K ctx | ~1.5–2.5 GB | **~fixed, small** |
| KV cache at 32K ctx | ~6–10 GB → **won't fit** | **~same as 8K** |
| Max practical `num_ctx` on 16 GB | 8–12K | **32K–128K** |
| Attention Ledger budget | tight, 8K working | **generous** |
| Long-session degradation | compaction ladder required | **minimal** |
| Law 2 (cap, don't summarise) | load-bearing | **largely moot** |

**The real prize isn't speed — it's that the Attention Ledger stops being a rationing system and
becomes a routing system.** With 32K+ of usable context on 16 GB of RAM, you can keep the whole
day's conversation resident instead of evicting it. That is a qualitative change in how
context-aware FRIDAY feels.

---

## 6. Track B — The Retention State Compiler ⭐

**This is the innovation. This is what you should actually build.**

### 6.1 What it is

A **trained module** — small, ~10–100M parameters — that compiles FRIDAY's interaction history
into a **fixed-size state tensor** which is then injected into the backbone as a soft prompt /
prefix-state.

```
      conversation history (unbounded, lossless, on disk)
                    │
                    ▼
   ┌────────────────────────────────────────────────┐
   │     RETENTION STATE COMPILER  (RSC)            │
   │     ~40M params, gated linear recurrence       │
   │                                                │
   │  S_t = G_t ⊙ S_{t-1} + β_t (v_t k_t^⊤ - S_{t-1} k_t) k_t^⊤   │
   │        └ gated ┘     └ delta rule (overwrite by key) ┘        │
   │                                                │
   │  + a salience head that emits a SIDE CHANNEL:  │
   │    "this span contains a durable fact" ────────┼──► S2 fact extraction
   │                                                │    (external store)
   │  output: S ∈ R^{L × d}   (fixed size!)         │
   └────────────────────┬───────────────────────────┘
                        │ injected as prefix-state / soft prompt
                        ▼
   ┌────────────────────────────────────────────────┐
   │   BACKBONE (LFM2 hybrid, FROZEN)               │
   │   sees: [RSC state] + [stable identity prefix] │
   │         + [retrieved facts] + [current turn]   │
   └────────────────────────────────────────────────┘
```

**Note the gate: `G_t` is *learned and token-wise*, and the update uses the *delta rule*.**
That is Gated DeltaNet, not RetNet — the difference between "surpasses Transformer baseline by
2–5 pp" and "near-zero recall even with attention added."

**Note the salience side-channel.** This is the piece that makes it FRIDAY's Memory Compiler and
not just a context compressor: while the RSC compresses, it *also* flags spans worth writing to
the lossless external store. **Compression and extraction happen in one forward pass.** That is
the unification of my S1 stage and the Ledger's transcript slot that the peer intuited — done
correctly.

### 6.2 Why this is tractable on ₹0 / Kaggle free T4

| Property | Value |
|---|---|
| Parameters to train | ~10–100M (the RSC), **backbone frozen** |
| VRAM | **~2–6 GB on a T4** with the backbone in 4-bit |
| Time per run | minutes to a few hours |
| Kaggle budget | 30 GPU-hours/week → **many runs per week** |
| Loss | **distillation / alignment**, not next-token prediction from scratch |
| Data | your own S0 traces + synthetic conversations |

Compare to training a 2.7B RetNet from scratch on 100B tokens, which took Microsoft **512 AMD
MI200 GPUs**. The RSC is a ~10,000× smaller problem because **you're not teaching it language —
someone already did that. You're teaching it *what to remember about you*.**

### 6.3 Training objective

Three losses, jointly:

```python
L_total = λ₁·L_reconstruction + λ₂·L_recall + λ₃·L_salience

# 1. RECONSTRUCTION — the compressed state must let the frozen backbone
#    behave as if it had seen the full history.
#    Teacher: backbone with FULL history in context.
#    Student: backbone with RSC state + short recent window.
#    Loss: KL(teacher_logits || student_logits) on the continuation.
L_reconstruction = KL( p_backbone(y | full_history) || p_backbone(y | RSC(history), recent) )

# 2. RECALL — the guard rail. Explicitly tests the thing linear attention is bad at.
#    Synthetic + real needles embedded in the history; exact-match required.
#    WITHOUT THIS TERM THE RSC WILL LEARN TO DROP VERBATIM DETAIL.
L_recall = -log p(exact_needle_tokens | RSC(history), needle_query)

# 3. SALIENCE — the side channel, supervised by the Memory Compiler's own output.
#    Labels come free: any span that produced an S2 fact is salient.
L_salience = BCE(salience_head(h_t), produced_fact_t)
```

**`L_recall` is the whole design.** It's the architectural answer to §2.3 — instead of hoping the
recurrent state retains verbatim detail, we **train it to** and **measure it**, while the external
store remains the guaranteed-lossless path.

**`L_salience` labels are free.** Every time the Memory Compiler extracts a fact from a span, that
span gets a positive label. Your existing pipeline generates the supervision.

### 6.4 What it replaces

| Before ([04](./04-INNOVATION-attention-ledger.md)) | After |
|---|---|
| 6-rung compaction ladder (cap → truncate → dedup → offload → evict → summarise) | **Rung 0 + the RSC.** Cap tool outputs (still free, still worth it); everything else is compiled into state |
| `transcript` slot: last N turns, budgeted | `state` slot: **fixed size, unbounded history** |
| Nightly S1 episode summarisation by an LLM | RSC **salience head**, continuous, ~free |
| Prompt-cache fragility (Law 2) | **Moot.** No prefix to cache-break |
| Compaction at 75% of window | **Never.** The window doesn't fill |

**The Attention Ledger survives — it just gets easier.** Identity, senses, skill index, retrieved
docs and reserve are all still budgeted slots. The transcript slot becomes a fixed-size state.
The Ledger's job shifts from *rationing* to *routing*.

### 6.5 Honest limitations

- **The RSC is trained on your distribution.** It will be excellent at compressing *FRIDAY's*
  conversations and mediocre at arbitrary text. That's fine — it's a personal model.
- **It couples you to a backbone.** The state is aligned to a specific frozen model's embedding
  space. Swap the backbone → retrain the RSC. Mitigate by keeping the RSC's output in a
  backbone-agnostic projected space.
- **`L_recall` will never reach attention's exactness.** Accept it. The external store is the
  guarantee; the RSC is the *convenience*. Design so that RSC failure degrades to "FRIDAY has to
  search memory more often," not "FRIDAY forgets."
- **Nobody has published this exact thing for a personal assistant.** That's the point — but it
  also means no reference implementation. Budget for debugging.

---

## 7. Track C — the custom architecture (and the honest math)

### 7.1 What ₹0 / Kaggle free tier actually buys

Hard numbers, so you can decide with your eyes open:

| Reference | Compute | Tokens |
|---|---|---|
| **Chinchilla-optimal 1B** | ~600 H100-hours (~$1.5K spot) | **50B** |
| Chinchilla-optimal 3B | ~5,500 H100-hours (~$14K) | 60B |
| **RetNet paper (1.3B/2.7B/6.7B)** | **512 AMD MI200 GPUs** | **100B** |
| TinyLlama 1.1B | ~90 days on 16× A100 | **3T** |
| A real practitioner's 200M-param validation run | 50 h H100, $75 | 10B |
| 8× H100 throughput (small model) | 485K tokens/sec | — |
| **Your budget: 2× T4, 30 h/week** | T4 ≈ 65 TFLOPS FP16, ~1/15 of an H100 | see below |

**Your throughput, realistically:** a T4 running a 150–500M model with mixed precision and a
well-tuned dataloader gets maybe **5–20K tokens/sec**. Take 10K:
```
30 h/week × 3600 s × 10,000 tok/s  ≈  1.1 B tokens / week
                                     ≈  55 B tokens / year (if you use EVERY hour, forever)
```

**Against the 2026 rule:** *"train at least 100:1, ideally 500–1,000:1 for small models"*
(Chinchilla's 20:1 is considered undertrained for anything you'll actually serve; Llama 3 used
~200:1; **LFM2.5-350M used 80,000:1**).

| If you train from scratch | Params | Tokens at 100:1 | Time at your budget |
|---|---|---|---|
| 55B tokens/yr | **~550M** | 55B | **1 year of every free GPU hour** |
| 55B tokens/yr | ~150M (at 350:1) | 55B | 1 year |
| Realistic with a life | **~100–200M** | 10–20B | **3–6 months** |

**A from-scratch 150M model trained on 15B tokens will not be "very accurate."** It will be
worse than GPT-2. That is the honest answer, and I'd rather give it to you now than in month four.

### 7.2 The path that DOES work: continue-pretrain, don't start from zero

```
❌  train a 400M hybrid from scratch on 20B tokens
     → worse than GPT-2. Months of free-tier quota. Dead end.

✅  take LFM2.5-350M (already trained at 80,000:1 tokens-per-parameter by people
    with real compute) and CONTINUE-PRETRAIN + QLoRA it on your personal corpus
     → world-class base capability + hyper-specialisation on YOU
     → 30 Kaggle T4-hours = ~1–3B tokens of continued pretraining = meaningful
     → this is "very accurate at being Jagan's assistant," which is the actual goal
```

**Cost comparison from the literature:**
| Approach | Data | Cost |
|---|---|---|
| Train from scratch (sub-1B) | millions of samples | **$500–$5,000** + weeks–months |
| **Fine-tune** | 500–10,000 samples | **$10–$100** + hours–days |
| **Distill** | teacher-generated | **$200–$2,000** + days–weeks |

At ₹0, only the middle two are available. **Use them.**

### 7.3 If you still want to build an architecture from scratch — do it at this scale

This is the version I'd genuinely recommend as a *research* project, and it's feasible:

```
ARCHITECTURE  (target: 100–300M params)
  · 12–20 layers
  · 3 Gated DeltaNet (or Mamba-2) : 1 Sliding Window Attention (w=2048)
      ← the production consensus ratio. NOT pure RetNet. See §2.5
  · SwiGLU FFN, RoPE inside the SWA layers only (Samba's choice)
  · Short convolution on Q/K/V of the linear layers (Samba Table 10: helps SWA a lot,
    helps GLA less because GLA already has channel-level fine-grained decay)
  · Optional: 1 full-attention layer AT THE END (ICLR 2025: provably sufficient to
    close the representation gap)
  · Vocab: reuse LFM2's or Qwen3's tokenizer. DO NOT train your own — it wastes
    parameters and data, and you lose the ability to distil from existing models.

CORPUS  (target: 5–15B tokens, from open sets — ₹0)
  · FineWeb-Edu / SlimPajama / The Pile subset — general language
  · + Tamil corpus (indic-nlp, OSCAR-Tamil) — your code-switching reality
  · + synthetic FRIDAY-shaped dialogues — assistant-with-memory conversations
    generated by prompting a small model to produce (history, fact, turn) triples
  · + your own S0 traces once you have ~6 months (small but the most valuable part)

TRAINING  (Kaggle 2×T4, 30 h/week)
  · torchscale (MIT — has the RetNet/retention implementation) OR
    flash-linear-attention (`fla`) — hardware-efficient kernels for
    RetNet, GLA, Based, HGRN2, RWKV6, GSA, Mamba2, DeltaNet, Gated DeltaNet, RWKV7
  · Use `fla`. It has Gated DeltaNet kernels already written and benchmarked.
    Writing your own Triton kernels on a T4 is a month of your life.
  · checkpoint on a WALL-CLOCK interval and copy off the VM immediately —
    "free Colab can reclaim the VM without warning"
  · 200M-param validation run FIRST (a real practitioner: 50 h H100 / $75 / 10B tokens),
    then scale only if the curve justifies it
```

**Expected outcome, stated honestly:** a ~200M hybrid that is **worse than LFM2.5-350M at
everything general**, and that you understand completely. Whether that trade is worth it depends
on whether your goal is *a good assistant* or *understanding architectures*. **You said both — so
do Track A for the assistant and Track C for the understanding, and never let Track C block
Track A.**

### 7.4 The version of Track C that could actually beat a frontier model

Not "best 200M model in the world." **Best model in the world at one narrow job.** Your locked
eval suite defines the job. Candidates where a tiny specialised model genuinely wins:

| Narrow model | Params | Why a small model beats a big one |
|---|---|---|
| **The RSC** (Track B) | 10–100M | It's a compressor for *your* conversation distribution. No general model is tuned for that |
| **The L0 router** | 100–600M | 6-way classification + a scalar. Trained on *your* escalation traces. Sub-5ms, zero tokens |
| **The redactor** | 100–400M | India-specific PII NER. You have the labels; cloud models are worse at PAN/Aadhaar/IFSC than a tuned small model |
| **The salience head** | <10M | Binary: did this span produce a fact? Free labels from your own pipeline |
| **The fact extractor** | 400M–1B | (subject, predicate, object, valid_from) from a span. Highly structured, highly repetitive, *your* domain |
| **The wake-word + intent model** | <5M | Runs on an ESP32. Must be tiny |

**Six tiny specialised models + one borrowed general backbone will beat one medium general model
at FRIDAY's job, on your hardware, for ₹0.** That's the real "best lightweight model that is very
accurate" — it's a *system*, not a single checkpoint.

---

## 8. The ₹0 budget: what it changes

You picked **₹0 — fully local and free tiers only.** Consequences, stated plainly:

### 8.1 What you lose
| Feature | Impact | Mitigation |
|---|---|---|
| **Natural voice** | ⚠️ **This is the real loss.** You asked for "very natural and realistic." **Piper sounds robotic.** Cartesia Sonic / ElevenLabs Flash (85–135 ms TTFA, genuinely human) cost money | Accept robotic local TTS; **or** reconsider — voice is the one place where ~₹400/month transforms the experience more than anything else in this document |
| **L2/L3 cloud escalation** | The top of the Escalation Ladder is empty. Hard reasoning caps at your local model | **Make the local tier better instead**: Track B's RSC gives it far more usable context, which recovers a lot |
| **Deepgram Nova-3 STT** | 110–160 ms, best-in-class | faster-whisper `small`/`base` on CPU. Slower but free. Test `whisper-turbo`-class local variants |
| **Cloud teacher for GEPA** | ⚠️ **This one matters.** GEPA's reflection step wants a *strong* model. OpenJarvis's 13–32 pp recovery came from a frontier teacher | See 8.2 |
| **Frontier model for SEAL-style training-data synthesis** | Weaker synthetic data | Use your best local model + heavy filtering; accept lower yield |

### 8.2 The GEPA teacher problem, and three ₹0 answers

GEPA *"uses the LLM to reason about why a prompt failed and propose targeted fixes."* With a 4B
local model as the reflector, reflection quality drops — but it doesn't go to zero. Options:

1. **Behavioural reflection instead of verbal reflection.** Your S0 traces already carry
   *external* signals: barge-ins, immediate rephrases, "no I meant…", unnecessary escalations,
   tool failures, eval-gate failures. **These are a better feedback function than a model's
   opinion** — and they're free. Remember Huang et al.: LLMs can't self-correct without external
   feedback anyway. A weak reflector with strong external signal beats a strong reflector with
   none.
2. **DSPy MIPROv2 instead of GEPA.** Bayesian search over instructions + bootstrapped
   demonstrations. Less reflection-dependent. Raised HotPotQA ReAct accuracy **24% → 51%**.
3. **Occasional paid burst.** One ₹200 top-up buys a lot of GEPA reflection tokens. Consider it
   for the *weekly* Dreaming run only — the rest stays local. (Your call; the architecture
   supports it either way.)

### 8.3 What ₹0 does NOT cost you

**The training loop is free.** Kaggle's 30 GPU-hours/week is the single most important fact in
this pivot:
- Track B (RSC, ~40M params, frozen 4-bit backbone): **~2–6 GB VRAM, minutes per run.** Dozens of
  experiments a week.
- Track C QLoRA / continued pretraining of a 350M–1.2B model: **comfortable on a T4.**
- The reference: **Qwen2.5-1.5B + LoRA r=16 on 106 examples → 70 seconds on a free T4 → 940 MB
  GGUF → Ollama. Total cost ₹0.**

**₹0 buys you a genuinely self-improving system. It does not buy you a natural voice.**
Those are different problems with different price tags.

---

## 9. Revised roadmap

Changes to [00-BLUEPRINT §6](./00-BLUEPRINT.md), marked **Δ**:

| Phase | Weeks | Δ from before |
|---|---|---|
| **0 — Foundation** | 1 | **Δ Day 0 now includes the Track-A bake-off**: benchmark LFM2.5-1.2B / LFM2-2.6B / LFM2-8B-A1B / RWKV-7-2.9B / **Qwen3-4B (Transformer baseline)** on speed, RAM-at-32K, and **the needle-recall test**. Write `BENCHMARKS.md`. Pick the backbone with data, not vibes |
| **1 — Memory Compiler** | 2–4 | **Δ Add the salience-label tap**: every extracted S2 fact records its source span. These are Track B's free training labels. **Start collecting them in Week 2, not Week 12** |
| **2 — Attention Ledger** | 4–6 | **Δ Split by backbone.** If Track A is a hybrid: the transcript slot becomes large/absent and the ladder shortens to Rung 0 + Rung 4. If it's Qwen3-4B: build the full ladder as specified |
| **3 — Perception & Presence** | 6–9 | **Δ Use `LFM2-VL` if it beats `qwen3-vl:4b` on *your* documents** — same family, one serving path. **Δ Voice is local-only: set expectations now, Piper is not "natural"** |
| **3.5 — NEW: Track B RSC** | 8–12 | **Δ The innovation.** Build the Retention State Compiler: gated-delta recurrence + salience head, distilled against the frozen backbone with an explicit `L_recall` term. Kaggle T4. Ship it behind a flag; A/B it against the compaction ladder on the locked eval suite |
| **4 — Proactivity** | 9–11 | unchanged |
| **5 — Self-Improvement** | 11–16 | **Δ GEPA uses behavioural reflection (8.2), not a cloud teacher.** **Δ Continued-pretraining/QLoRA of the *existing* backbone on your corpus — not from scratch.** **Δ Re-run the needle-recall test after every training run** (recall is a checkpoint property) |
| **5.5 — NEW: Track C tiny specialists** | 14–20 | **Δ Train the six narrow models from §7.4** — router, redactor, salience head, fact extractor. Each is a tractable Kaggle job and each measurably improves the system |
| **6 — Custom architecture** | 6+ months | **Δ Optional, research-only, never blocks production.** 100–300M hybrid (3 GDN : 1 SWA) using `flash-linear-attention` kernels |
| **7 — Skills & Tasks** | was 6 | renumbered; still last |

---

## 10. Decision summary

| Question | Answer |
|---|---|
| **Is the linear-attention pivot right?** | **Yes.** The KV cache is your binding constraint; O(1) state deletes it; Law 2 dissolves; the Ledger stops rationing and starts routing |
| **Is RetNet the right architecture?** | **No.** Fixed input-independent decay → *"near-zero recall even when full-attention layers are added."* Worst-in-class benchmark avg (0.547). It's generation 2 of 5. No usable checkpoints (58 downloads). Undertrained at 100B tokens |
| **What replaces it?** | **Gated DeltaNet / Mamba-2 / Kimi-Delta-class hybrids with ~3:1 linear-to-attention.** Kimi K3, Qwen3.5, Nemotron 3, Granite 4, Jamba, Falcon-H1, Zamba2, Samba — *everyone* shipped this shape |
| **What runs on your machine today, ₹0, in llama.cpp?** | **LFM2 / LFM2.5 (Liquid AI hybrid, official GGUFs, official llama.cpp on-device docs, incl. vision variants)** and **RWKV-7** (2.9B World in llama.cpp). LFM2-2.6B at **30 t/s**, LFM2-8B-A1B at **38 t/s** |
| **Does the recall gap kill FRIDAY?** | **Pure linear attention would.** The hybrid + external-store pairing **sidesteps it** — and ICLR 2025 *proves* that in-context RAG plus one attention layer is sufficient to close the representation gap. FRIDAY's `memory_search` **is** that RAG |
| **Where's the innovation?** | **Track B: the Retention State Compiler.** A trained gated-delta module emitting a fixed-size state *plus* a salience side-channel that feeds S2 fact extraction. Compression and memory extraction in one forward pass. ~40M params, fits a free T4, nobody has published this for a personal assistant |
| **Can I build a model from scratch?** | **Yes, at 100–300M params, and it will be worse than GPT-2.** Worth it for understanding, not for production. **Continue-pretraining LFM2.5-350M (trained at 80,000:1) on your corpus is the path that actually delivers "lightweight + very accurate"** |
| **Can it be "very accurate"?** | **Yes — at its job.** Not "best 200M model in the world"; **best model in the world at being Jagan's assistant**, defined by your locked eval suite. Plus **six tiny specialists** (router, redactor, salience, fact extractor, RSC, wake word) that genuinely beat general models at their narrow tasks |
| **What does ₹0 cost me?** | **Natural voice.** Piper is robotic. That's the one place where the budget and your stated goal ("very natural and realistic") genuinely conflict. Everything else — including the whole training loop — is free on Kaggle's 30 h/week |

---

## 11. The immediate next step

**Run the bake-off.** Everything downstream depends on it, it takes one afternoon, and it costs
nothing:

```bash
# 1. Install
winget install llama.cpp

# 2. Download candidates (all official GGUF)
llama-cli -hf LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M --version   # just to fetch
llama-cli -hf LiquidAI/LFM2-2.6B-GGUF:Q4_K_M
llama-cli -hf LiquidAI/LFM2-700M-GGUF:Q4_K_M
llama-cli -hf LiquidAI/LFM2-VL-1.6B-GGUF:Q8_0
# RWKV-7: hf.co/RWKV/RWKV-7-World-2.9B-v3-... GGUF
# Transformer baseline (you MUST have this to prove the hybrid helps):
ollama pull qwen3:4b

# 3. Speed + RAM
for M in <each>.gguf; do
  llama-bench -m $M -ngl 99 -fa 1 -p 512  -n 128
  llama-bench -m $M -ngl 99 -fa 1 -p 8192 -n 128
done    # watch Task Manager RSS at each -c setting

# 4. THE RECALL TEST  ← the one that decides everything
python scripts/eval_recall.py    # 8K-token contexts, needle at 5 depths, exact-match score

# 5. Write docs/architecture/BENCHMARKS.md with YOUR numbers
```

Then send me `BENCHMARKS.md` and I'll design Phase 0 around the winner.

**If the needle test surprises you** — hybrids collapsing where Qwen3-4B holds — that is the most
interesting possible outcome, because it means the recall gap is real at your scale and Track B's
`L_recall` term is not optional decoration but the entire load-bearing wall. Either result is a
good result. That's what makes this a research project rather than an integration exercise.
