# 13 — The Erase/Write Split: Gated DeltaNet-2 as the Memory Compiler's Operator

> **Trigger:** a proposal to upgrade FRIDAY from a "standard gated hybrid" to a **Decoupled
> Gated DeltaNet-2 layer + Sliding Window Attention**, so the model gets *"dedicated channel-wise
> axes to actively wipe transient noise without scrambling your permanent associative records."*
>
> **Verdict: the architecture is real, the code is real, and the instinct is better than I gave it
> credit for — the paper's own ablation says *erase* is where the gain is, which is precisely
> FRIDAY's need. But it does not go in Phase 0, and "permanent associative records" is a category
> error that would reintroduce the exact failure mode [12 §2.3](./12-LINEAR-ATTENTION-PIVOT.md) warned about.**
>
> **Placement: GDN-2 becomes the recurrent operator inside the Retention State Compiler
> (Phase 3.5), not the backbone. And there is a June-2026 successor — EDA — that fits FRIDAY's
> bi-temporal correction requirement better than GDN-2 does.**

---

## 1. Verified: GDN-2 is real, and the numbers are good

**Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention**
Ali Hatamizadeh, Yejin Choi, Jan Kautz — **NVIDIA**. [arXiv 2605.22791](https://arxiv.org/abs/2605.22791), submitted **21 May 2026**, CC BY 4.0.
Code: [`github.com/NVlabs/GatedDeltaNet-2`](https://github.com/NVlabs/GatedDeltaNet-2) — **323 stars, 33 forks**, last commit **29 Aug 2026** (still active).

**The claim being attacked** (from the abstract, verbatim):

> *"The hard part is not just what to forget, but **how to edit this compressed memory without
> scrambling existing associations**… the active edit still uses a **single scalar gate** to control
> two different things, how much old content to erase on the key side and how much new content to
> commit on the value side."*

That sentence is FRIDAY's problem statement. Verbatim.

### 1.1 The lineage — and where the coupling actually lives

| Method | Decay | Erase | Write | Date |
|---|---|---|---|---|
| DeltaNet | none | **scalar β_t** | **scalar β_t** | — |
| Gated DeltaNet | **scalar** α_t | scalar β_t | scalar β_t | ICLR 2025 |
| **KDA** (Kimi Delta Attention) | **channel-wise** α_t ∈ (0,1]^d_k | scalar β_t | scalar β_t | Team et al. 2025 |
| ⭐ **Gated DeltaNet-2** | channel-wise α_t | **channel-wise b_t ∈ [0,1]^d_k** | **channel-wise w_t ∈ [0,1]^d_v** | **May 2026** |
| ⭐ **EDA** (Erase-then-Delta) | channel-wise D_t | **independently *addressed*** e_t | at k_t | **Jun 2026** |
| "The Query Knows What to Forget" | — | **second erase direction from q_t**, orthogonal to k_t | rank-one, gated by b_t | Aug 2026 |
| **TERN** | phase-conditioned | erase-then-delta + **seasonal reference** + online adaptation | | **Sep 2026** |

**Read the last two columns of the GDN-2 row and then the EDA row.** That gap is the whole rest of
this document.

### 1.2 Gated Delta Rule-2 — the exact math

Given erase gate **b**_t ∈ [0,1]^{d_k}, write gate **w**_t ∈ [0,1]^{d_v}, channel-wise decay
D_t = Diag(**α**_t):

```
S_t = ( I − k_t (b_t ⊙ k_t)ᵀ ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
      └──────── active edit ────────┘└decay┘        └── write ──┘
```

Three things worth noticing, all load-bearing for FRIDAY:

1. **Decay is applied *before* the active edit.** Broad clearing first, targeted edit second.
2. **The write direction stays `k_t`.** The *left* factor of the erase term is still the raw key —
   so the delta rule's "edit the association *at* key k_t" property survives. The *right* factor
   becomes `b_t ⊙ k_t`, which is what makes the **read direction channel-selective**.
3. **The write term uses `z_t = w_t ⊙ v_t`**, making the value update channel-selective.

**Both gates are sigmoid projections of the token representation, from *separate linear branches*.**
Decay α_t, erase b_t, write w_t — three branches, three decisions.

**Tied subspaces (this is what makes it a safe, falsifiable upgrade):**
```
b_t = β_t·1_(d_k)  and  w_t = β_t·1_(d_v)   →  recovers KDA exactly
+ α_t = α_t·1_(d_k)                          →  recovers Gated DeltaNet
```
**So KDA is your free control condition.** You can ablate from scalar-tie upward and know exactly
what each gate bought you. That's rare and it's valuable.

**Block design (the hybrid):** `GDN-2 → MLP → SWA → MLP`, repeating.
- Q/K paths: linear projection → short causal conv → SiLU → **L2 normalisation** (stability)
- V path: linear projection → short conv → SiLU
- Output: **RMS-normalised**, multiplied by a **separate SiLU output gate**, then output projection
- **16 heads, d_k = d_v = 128** → state = **262,144 floats per layer ≈ 1 MB (fp32)**, *"similar to
  Mamba-2/3"*

**Training stays parallel:** chunkwise **WY form** with channel-wise decay absorbed into
**asymmetric erase factors** (`ē = γ ⊙ (B ⊙ K)`, `Z = W ⊙ V`) plus a **gate-aware backward pass**
fused in Triton. **Decoupling the gates does *not* break the KDA kernel.** That's the key
engineering result — no throughput cliff.

### 1.3 Results — real numbers, with the honest framing

**Aggregate (1.3B params, 100B FineWeb-Edu tokens, matched state size):**

| Model | Avg |
|---|---|
| Gated DeltaNet | 52.25 |
| **Gated DeltaNet-2** | **53.97** |

Beats **Mamba-2, Gated DeltaNet, KDA, and Mamba-3** across language modelling, commonsense
reasoning, and retrieval — in **both recurrent-only and hybrid** settings.

**Where the gain actually is:**

| Benchmark | Movement |
|---|---|
| ⭐ **RULER Single-NIAH-3 @ 8K** | **63 → 90** |
| **RULER Multi-Key NIAH-1** | substantial gain over KDA (prior best recurrent) |
| Gains concentrated in | S-NIAH-1, S-NIAH-2 @8K, **all** S-NIAH-3 lengths, longer MK-NIAH-1 |

> ### ⭐ This is the single most important number in this document.
>
> **NIAH is Needle-In-A-Haystack.** That is *exactly* the test
> [12 §5.3](./12-LINEAR-ATTENTION-PIVOT.md) named as the decisive benchmark — the one I said you
> must write before choosing a backbone. GDN-2 moves **63 → 90 on multi-depth needle retrieval**,
> and the paper's own explanation is: *"With **SWA handling local evidence**, this decoupled
> recurrent update **preserves longer-range associations** more effectively than a scalar delta
> gate."*
>
> **The proposal targets precisely FRIDAY's binding constraint.** That's not a lucky guess — that's
> the right instinct, and it deserves more credit than "upgrade the layer."

**The ablation that decides FRIDAY's design:**

> *"Ablations confirm both gates contribute, with the **erase gate b_t accounting for most of the
> gain** — consistent with its role in selectively protecting or revising key-side associations."*
> — NVlabs README

**Cost:** throughput **38.0 → 36.1 Kt/s** vs KDA — **~5%**, described as *"a modest training-cost
increase."* No inference cliff (decay is absorbed into the same chunkwise form).

**⚠️ Two caveats, stated plainly:**
1. **Outside replication is still missing.** All results are NVIDIA's own, on NVIDIA's benchmarks.
   Four months old as of this writing. The repo has 323 stars, not a production ecosystem.
2. **It was validated at 1.3B / 100B tokens.** That is *the same undertrained scale I criticised
   RetNet-2.7B for in [12 §2.4](./12-LINEAR-ATTENTION-PIVOT.md).* I have to apply my own standard.
   **The distinction:** this is a **controlled architecture comparison** — Mamba-2, GDN, KDA,
   Mamba-3 and a Transformer baseline were all trained *identically* at 1.3B/100B, same optimizer,
   same state size. That makes it a **valid statement about the update rule**, and the update rule
   is what FRIDAY wants to port. It is **not** a claim that a GDN-2 LLM is production-ready.
   **Take the math. Do not expect a downloadable frontier model.**

---

## 2. The successor you didn't cite — and it fits FRIDAY better

**Erase-then-Delta Attention (EDA): Decoupling Erase and Write *Addresses* in Delta-Rule Linear Attention**
Xiao Li, Chengruidong Zhang, Hao Luo, Xi Lin, Zekun Wang, Zihan Qiu, Yunfei … (Alibaba-group
author list; references Qwen's Zheng/Liu/Zhou). [arXiv 2606.26560](https://arxiv.org/abs/2606.26560), **18 June 2026**.

### 2.1 EDA's critique of GDN-2 is exact and it is *about FRIDAY*

> *"GDN-2 separates key-side erase and value-side write gates, allowing the model to assign
> different **strengths** to erasing and writing inside the delta residual. The active edit,
> however, **remains organized around the current write key**. … Thus GDN-2 relaxes the
> **gate-level** coupling, while the **address-level** coupling between erasure and writing
> remains."*
>
> *"If **stale information is stored at an address different from the current write address**, the
> diagonal gate can decay feature channels but **cannot selectively remove that stale association**
> before writing elsewhere."*

**Now translate that into FRIDAY's Phase 1 exit test:**

> *Tell it a fact, contradict it two weeks later, ask "what did I used to think?" — it answers
> **both** correctly with dates.*

```
Week 1:  "I live at 12 Marina Road"     → written at address k("my address")
Week 3:  "I moved to 48 Velachery Main" → written at address k("my new address")
                                         ↑ DIFFERENT ADDRESS
```

**GDN-2 cannot clear the Week-1 association while writing the Week-3 one.** Its erase direction is
`b_t ⊙ k_t` — still built from the *current* write key. It can decay channels globally, or correct
whatever sits at the address it's writing to. It cannot reach sideways and delete the stale
association at a *different* address.

**EDA can.** It inserts an **independently addressed erase operator** before the standard delta
write, separating memory management into **three levels of specificity**:

| Level | Mechanism | What it does |
|---|---|---|
| 1 | **diagonal decay** `D_t` | broad, channel-wise clearing of context |
| 2 | **independent directional erasure** `γ_t·e_t` | ⭐ **removes a stale association at an address of its choosing** |
| 3 | **write-coupled correction** `β_t·k_t` | the standard delta-rule corrective write |

The erase address is a **factorised map with a 16-dimensional intermediate per head** (per TERN's
description: `e_t = W₂W₁z_t / ‖W₂W₁z_t‖₂`), *"so stale associations can be cleared along a
direction that is decoupled from the key being written."*

**And the empirical finding that makes this more than a nice idea:**

> *"Empirical analysis reveals that the model learns a **near-orthogonal separation between erase
> and write addressing**, indicating that the two operations serve **genuinely different roles**."*
> *"The extra address acts as a **conditional cleanup path** rather than merely stronger
> forgetting."*

The model, left to itself, learns to erase *somewhere else* than where it writes. Near-orthogonal.
That's the degree of freedom FRIDAY needs.

### 2.2 EDA's numbers

| Model | MMLU | MMLU-Pro | GSM8K | MATH | BBH | EvalPlus | **Avg** |
|---|---|---|---|---|---|---|---|
| GDN | 67.43 | 40.55 | 75.93 | 45.94 | 64.25 | 50.09 | 57.37 |
| KDA | 67.32 | 40.60 | 76.21 | 46.87 | 65.04 | 53.82 | 58.31 |
| **EDA** | **68.12** | **41.71** | 75.99 | **49.28** | **65.81** | 51.78 | **58.45** |

At **dense 2.5B**, EDA has the strongest average among all dense models; **+0.63 over KDA** (same
channel-wise gated delta backbone, minus the independent erase address). The larger
**MoE 25B-A2.8B** setting *"performs best on most benchmarks… across knowledge-heavy,
reasoning-heavy, and code-oriented tasks."* Midtrained further on 80B tokens at 32K sequence length.

**Framing, from EDA's own related-work section — this is the sentence to remember:**

> *"The two designs are therefore **orthogonal in spirit**: **GDN-2 decouples *how strongly* erase
> and write are applied, while EDA decouples *where* erasing and writing are applied.**"*

**They compose.** That is not a footnote — that is FRIDAY's design.

### 2.3 The field has already moved twice more

- **"The Query Knows What to Forget: A Second Erase Direction for Linear Attention"**
  ([arXiv 2608.13668](https://arxiv.org/pdf/2608.13668), Aug 2026). Notes that in GDN-2 *"the erase
  vector is still built from the current key, although its channels are gated independently"* — and
  adds a **second erase direction derived from `q_t`** (what you're *asking about*, not what you're
  writing), keeping a single rank-one update with the correction **constrained orthogonal to the
  key**.
  ⭐ **For FRIDAY this is a big deal: query-driven forgetting.** "What's my rent?" should erase
  whatever is stale *about rent*, addressed by the query rather than by the incoming token.
- **TERN** ([arXiv 2609.18407](https://arxiv.org/pdf/2609.18407), **Sep 2026** — three weeks old).
  Adopts the erase-then-delta form with **phase-conditioned erase and write, an explicit *seasonal
  reference*, and online adaptation.** Reports **erase-then-delta is the best rule on all three
  datasets**, ahead of KDA and GDN-2.
  ⭐ **"Seasonal reference" is temporal periodicity baked into the memory rule.** FRIDAY's Memory
  Compiler already has a **decay function with per-predicate half-lives** and a **circadian
  rhythm** ([02 §4](./02-INNOVATION-memory-compiler.md), [08](./08-proactive-heartbeat.md)).
  TERN is doing in-architecture what FRIDAY does in policy.

**The lesson: this is a *fast-moving* target — four substantive papers in five months.**
Design for **swappability**, not for GDN-2 specifically. See §6.

---

## 3. Three corrections to the proposal

### 3.1 ⚠️ "Permanent associative records" — this is the category error

The proposal says the decoupled gates let the model *"wipe transient noise without scrambling your
**permanent** associative records."*

**Nothing in a recurrent state is permanent.** Not with GDN-2, not with EDA, not with anything.
A fixed-size state is *"by construction a lossy memory"* and *"what it loses first is associative
recall: retrieving a specific earlier token verbatim"* ([12 §2.3](./12-LINEAR-ATTENTION-PIVOT.md)).
GDN-2 moves needle retrieval from **63 → 90**. **Ninety is not one hundred.** The other ten
percent of the time it still scrambles the record — and the failure is *silent*.

If FRIDAY treats the state as its permanent record, you have rebuilt exactly the failure mode that
killed the RetNet proposal, just at a higher recall ceiling. **Law 2b stands unamended:**

> **Never ask a lossy state to be a database.** The recurrent state holds *context*; the external
> store holds *facts*.

**What GDN-2/EDA actually buy you, stated correctly:**

| ❌ Wrong framing | ✅ Right framing |
|---|---|
| "the state becomes a permanent record" | the state **forgets more aggressively with less collateral damage** |
| "erase gate protects permanent memory" | erase gate lets FRIDAY **wipe transient chatter hard** while the *associative structure* it still needs survives |
| "we no longer need the external store" | the external store becomes **more** load-bearing, because we can now afford to decay the state harder |

**The decoupling raises the recall ceiling; the external store is what removes it.** You need both,
and they are not substitutes. 90 vs 100 is the entire argument for Law 2b.

### 3.2 ⚠️ It does not go in Phase 0 — and here is the hard blocker

The proposal calls this a **"Phase 0 Integration: The Decoupled Delta State Gate… this foundational
module."** I have to push back firmly, because this is the failure mode
[10](./10-phase-0-implementation.md) exists to prevent.

**Phase 0's exit test is: *"ask it something on Monday, ask a follow-up on Friday, it recalls
correctly."*** A custom recurrent layer does not help you pass that. It helps you pass a Phase 3.5
exit test. Building it in Week 1 means you have no assistant in Week 1.

**But there's a harder, non-negotiable blocker:**

> ### The NVlabs repo is a *training* harness. It cannot serve a model on your laptop.
>
> Repo contents: `pretrain.py`, `data.py`, `cache.py`, `Dockerfile`, `scripts/`, `paper/`,
> **`lit_gpt/`** (model definitions for Lightning's lit-gpt pretraining harness).
>
> **What is absent: any inference server. Any GGUF export. Any llama.cpp support. Any released
> checkpoint.** `Model: N/A` in the paper summary.

Your backbone runs via **`llama-server` on a Windows/WSL2 CPU with 16 GB of RAM.** There is no path
from `NVlabs/GatedDeltaNet-2` to a GGUF that `llama-server` will load. Getting a *custom* layer into
llama.cpp means writing a ggml implementation of the recurrence — that is months, and it is
somebody else's research project, not your assistant.

**Also, a factual correction:** the proposal says to run *"a Decoupled Gated DeltaNet-2 Core Layer
Mesh… using an open-weight base like **LFM2-2.6B** or **LFM2.5-1.2B**."* **You cannot put GDN-2
layers inside LFM2.** LFM2 is a *different* hybrid with its own gated-delta-family recurrence and
its own frozen weights. These are not composable that way. LFM2 is the **backbone**; GDN-2 is a
layer **you train separately and bolt onto it as a module.** That distinction is the whole design.

### 3.3 ⚠️ SWA ≠ your retrieval store — the diagram conflates them

The proposed flow ends: `SWA Retrieval → Lossless JIT Index Store (sqlite-vec + FTS5)`.

**Those are two different kinds of thing and they cannot be stacked:**

| | **SWA** (Sliding Window Attention) | **sqlite-vec + FTS5** |
|---|---|---|
| What it is | a **layer inside the network** | a **database on disk** |
| Range | the last **w = 2048** tokens | **everything, forever** |
| How the model reaches it | automatically, every forward pass | via a **tool call** (`memory_search`) |
| Gradient | **differentiable** | **not differentiable** |
| Cost | free (it's a layer) | 10–150 ms + a round trip |
| Fails at | anything outside the window | nothing — it's a database |

In the GDN-2 hybrid, SWA's job is stated plainly: *"**SWA handling local evidence**"* while the
recurrent operator *"preserves longer-range associations."* It is **local precision over recent
tokens.** It is not a retrieval interface to an index, and the index is not a layer.

**Why this matters concretely:** if you draw them as one pipeline you will end up trying to make
the external store differentiable, or trying to make SWA reach beyond its window. Both are dead
ends. The correct picture has them as **two parallel paths that meet in the prompt**:

```
                       ┌──────────────────────────────────────────┐
   recent ≤2048 tok ──►│ SWA layer (inside the network, local)    │──┐
                       └──────────────────────────────────────────┘  │
                                                                     ├─► assembled context
   unbounded history ──► RSC (GDN-2/EDA state, fixed size) ──────────┤
                                                                     │
   explicit query ─────► memory_search → FTS5/sqlite-vec ────────────┘
                       (tool call, NOT a layer, NOT differentiable)
```

---

## 4. ⭐ Where it *does* go — and the part neither of us had yet

**GDN-2 (then EDA) becomes the recurrent operator inside the Retention State Compiler.**

That placement dissolves every blocker in §3.2:

| Blocker for the backbone | Non-issue for the RSC |
|---|---|
| No llama.cpp/GGUF support | **The RSC never gets served by llama.cpp.** It runs in PyTorch on Kaggle T4 (and optionally locally, offline, in a Python process) |
| No released checkpoint | **You are training it.** That's the point |
| Custom layer = months of kernel work | **NVlabs already shipped PyTorch + Triton kernels + chunkwise WY + gate-aware backward.** You import, you don't write |
| LFM2 is a frozen different architecture | **Correct — and that's the design.** RSC output is injected into frozen LFM2 as a prefix-state/soft prompt |
| ~5% throughput cost | Irrelevant. The RSC compiles **offline / between turns**, not in the decode path |

### 4.1 The synthesis: the Memory Compiler *is* the gate supervision

Here is the part that neither the original blueprint nor the proposal had. GDN-2 gives you **three
separate branches** — decay α_t, erase b_t, write w_t — each *"from separate linear branches."*
FRIDAY's Memory Compiler already computes **three exactly corresponding signals**, for unrelated
reasons, and has been computing them since Phase 1:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║   GDN-2 / EDA BRANCH          FRIDAY'S EXISTING SIGNAL      LABEL SOURCE   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║   channel-wise decay α_t  ◄── the DECAY FUNCTION            computed, not  ║
║   "clear broad context"       per-predicate half-lives      learned:       ║
║                               (02 §4)                       age → α        ║
║                                                                           ║
║ ⭐ independently-addressed ◄── RETRACTION events            free: every    ║
║    erase e_t  (EDA)           valid_to set on a fact;       S2 fact that   ║
║    "remove the stale          superseded_by links           gets corrected ║
║     association *elsewhere*"                                gives you      ║
║                                                             (old_addr,     ║
║                                                              new_addr)     ║
║                                                             PAIRS          ║
║                                                                           ║
║   channel-wise write w_t  ◄── the SALIENCE HEAD             free: every    ║
║   "commit only what          "did this span produce an      span that      ║
║    should persist"            S2 fact?"                     yielded a fact ║
║                                                             is positive    ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Read the middle row again.** EDA's contribution is an erase operator at an address *decoupled
from the write address* — and FRIDAY's bi-temporal fact store generates
**(superseded_address, new_address)** pairs as a natural byproduct of every correction a user makes.
That is **free supervision for exactly the mechanism EDA adds**, and it comes from a schema I
designed in [03](./03-memory-architecture.md) for a completely different reason.

**This is Innovation #9 becoming concrete.** Not "compress history into a state" — that's a
context-compressor, and there are several published. Rather:

> **A recurrent memory operator whose decay, erase and write gates are supervised by a
> bi-temporal knowledge base.** The compiler's *policy* (decay half-lives, retractions, salience)
> becomes the *gradient signal* for the state's *mechanism*.

I have not seen this published for a personal assistant, and the three-way alignment is too clean
to be a coincidence. It works because FRIDAY's memory model was already **operational** — it
computes decay, records retractions, and knows which spans matter. Most memory frameworks store;
FRIDAY's *decides*, and deciding is exactly what a gate needs supervision for.

### 4.2 Why the erase-heavy ablation is FRIDAY-shaped

Recall the NVlabs finding: **"the erase gate b_t accounts for most of the gain."**

That is not a generic result. It is generic-*for-FRIDAY*:

| Gate | FRIDAY's need | Why |
|---|---|---|
| **Erase (b_t / e_t)** | ⭐⭐⭐ **HIGH** | A personal assistant's dominant memory problem is **accumulated transient noise** — 200 turns of chatter, tool spam, abandoned threads, mid-sentence corrections. FRIDAY needs to *wipe hard* |
| **Write (w_t)** | ⭐ **LOW** | Durable content **does not need to go into the state at all** — it goes to the external store and comes back via retrieval. The state's write path is a convenience, not the record |
| **Decay (α_t)** | ⭐⭐ MEDIUM | Broad context clearing. Already specified by the decay function |

**Under a scalar tie (GDN/KDA), you cannot erase hard without writing hard.** β_t is one number.
Every attempt to wipe transient chatter also scrambles what you were trying to keep — which is
*precisely* the 92% → 33% recall collapse from [04](./04-INNOVATION-attention-ledger.md), and
*precisely* RetNet's fixed-γ failure in [12 §2.1](./12-LINEAR-ATTENTION-PIVOT.md). Both were
**coupling** failures, and both showed up as recall failures.

**The decoupling is the fix for the failure mode this project has now hit twice from two different
directions.** That is the strongest argument for the proposal, and it's stronger than the one made
for it.

---

## 5. The revised RSC specification

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RETENTION STATE COMPILER v2 — Gated Delta Rule-2 + independent erase address │
│  ~40M trainable params. Backbone FROZEN. Runs offline / between turns.        │
├──────────────────────────────────────────────────────────────────────────────┤
│  Per layer:  GDN-2/EDA recurrent mixer → MLP → SWA(w=2048) → MLP              │
│  Heads 16 · d_k = d_v = 128 · state 262,144 floats ≈ 1 MB fp32 per layer      │
│                                                                              │
│  q,k: Linear → short causal conv → SiLU → L2-normalise                        │
│  v  : Linear → short conv → SiLU                                              │
│  α_t: Linear (log-decay projection)        ◄── SUPERVISED by decay function   │
│  b_t: Linear → sigmoid  (d_k)              ◄── erase strength, channel-wise   │
│  e_t: W₂·W₁·z_t, ‖·‖₂-normalised, 16-d      ◄── SUPERVISED by retraction pairs │
│       intermediate per head (EDA)                                             │
│  w_t: Linear → sigmoid  (d_v)              ◄── SUPERVISED by salience head    │
│  out: RMSNorm → ⊙ SiLU output gate → Linear                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  S_t = (I − k_t(b_t ⊙ k_t)ᵀ)(I − γ_t e_t ê_tᵀ) D_t S_{t−1} + k_t(w_t ⊙ v_t)ᵀ   │
│        └─ GDN-2 channel erase ─┘└─ EDA addressed erase ─┘└decay┘└─ write ─┘    │
│                                                                              │
│  Read: o_t = S_tᵀ q_t   (⊕ SWA over the last 2048 tokens for local evidence)  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
     injected into frozen      salience head →        erase-address probe →
     LFM2 as prefix-state      S2 fact extraction     "what did we forget?"
                               (lossless store)        (telemetry + eval)
```

> ⚠️ The composed update rule above is **my construction**, not a published equation. GDN-2's rule
> is verified verbatim (§1.2); EDA's independent erase is verified structurally (three levels of
> specificity, factorised 16-d address, near-orthogonal to the write key) but **I have not
> reproduced EDA's exact equation — read §3.3 of
> [2606.26560](https://arxiv.org/html/2606.26560) before implementing.** EDA states the two designs
> are *"orthogonal in spirit"* and its own baseline **is** a channel-wise gated delta rule with the
> `e_t = b_t ⊙ k_t`, `z_t = w_t ⊙ v_t` substitution — i.e. GDN-2-shaped — so composition is
> intended, but verify the ordering and the normalisation.

### 5.1 Loss function (extends [12 §6.3](./12-LINEAR-ATTENTION-PIVOT.md))

```python
L_total = λ₁·L_reconstruction + λ₂·L_recall + λ₃·L_salience
        + λ₄·L_erase   + λ₅·L_decay                       # ← NEW, and free

# 1. RECONSTRUCTION — KL(teacher with full history || student with RSC state + recent window)
# 2. L_recall        — explicit needle exact-match. WITHOUT THIS THE RSC LEARNS TO DROP DETAIL.
# 3. L_salience      — BCE(salience_head(h_t), produced_fact_t)      → supervises w_t
# 4. ⭐ L_erase      — the bi-temporal correction loss. FREE LABELS.
#      For every retraction event (fact_old.valid_to := now, fact_new asserted):
#        · the state must NOT return the superseded value when queried at "now"
#        · the state MUST still support "what did I used to think?" → routed to the
#          external store, NOT the state. Assert that in the eval, not the loss.
L_erase = -log( 1 - p_state(value_old | query(key_old), t=now) )
#      This is the loss that makes EDA's independent erase address *necessary*:
#      key_old ≠ key_new, so a write-anchored erase (GDN-2) cannot reach it.

# 5. ⭐ L_decay      — align α_t with the hand-specified decay function, then let it drift.
#      Curriculum: strong λ₅ early (α_t ≈ policy half-lives), anneal to 0
#      (α_t learned freely). Gives the model a working prior instead of a cold start.
L_decay = ‖α_t − decay_policy(age_t, predicate_class_t)‖²
```

**`L_erase` is the new load-bearing term**, and it is testable with data FRIDAY already produces.
It is also the term that decides GDN-2 vs EDA empirically: **if `L_erase` plateaus under GDN-2 and
drops under EDA, the address-level coupling was real for your data.** That's a publishable result
either way.

### 5.2 The ablation ladder — and why it's unusually clean

Because GDN-2 **recovers KDA exactly** when both gates tie to a scalar, you get a monotone ladder
where every rung is a real, measurable, one-variable change:

| Rung | Config | What it isolates | Expected |
|---|---|---|---|
| 0 | RetNet-style **fixed γ** | the [12](./12-LINEAR-ATTENTION-PIVOT.md) baseline | ❌ near-zero long-range recall |
| 1 | **scalar** decay + **scalar** β (= **Gated DeltaNet**) | adaptive forgetting | recall improves, still coupled |
| 2 | **channel-wise** decay + scalar β (= **KDA**) | fine-grained forgetting | further gain |
| 3 | 2 + **channel-wise erase b_t**, write tied | ⭐ **the erase gate alone** — NVlabs says *most of the gain is here* | **biggest single jump** |
| 4 | 3 + **channel-wise write w_t** (= **full GDN-2**) | the write gate's marginal value | small gain |
| 5 | 4 + **independent erase address e_t** (= **EDA**) | ⭐ **address-level** decoupling → `L_erase` | the bi-temporal win |
| 6 | 5 + **query-driven erase** (2608.13668) | forgetting addressed by what you *ask* | speculative |
| — | **+ SWA** at every rung | local evidence | GDN-2 paper runs hybrid; SWA is what lets the recurrent part specialise in long-range |

**Run 2 → 3 → 4 → 5 in that order.** Each is a few Kaggle hours. If rung 3 is where FRIDAY's recall
jumps (as NVlabs' ablation predicts), you have learned the most important thing about your own
system for about six GPU-hours — and you can stop there and ship.

**Do not start at rung 5.** EDA is the most interesting and the least validated, and if you begin
there you will not know which mechanism helped.

### 5.3 Engineering reality on a free T4

| Concern | Assessment |
|---|---|
| **Triton on T4** | T4 is **sm_75 (Turing)**. Triton supports sm_70+, so the NVlabs kernels should compile. **They are tuned for Hopper** — expect poor occupancy. Plan a **pure-PyTorch recurrent fallback** for correctness checks and for the RSC's own inference (which is offline, so throughput doesn't matter) |
| **Chunkwise WY on 16 GB VRAM** | T4 has 15 GB. At ~40M trainable params + frozen 4-bit backbone + chunked sequences this fits comfortably ([12 §6.2](./12-LINEAR-ATTENTION-PIVOT.md): 2–6 GB) |
| **`lit_gpt` dependency** | NVlabs' harness is lit-gpt-based. You want **only the layer**, not their training loop. Extract the mixer + kernels; keep your own PyTorch trainer. Check `LICENSE` (NVIDIA source release — confirm terms before redistributing derivatives) |
| **Alternative if Triton fights you** | [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) (`fla`) already has **Gated DeltaNet** and **KDA** kernels and is battle-tested on consumer GPUs. **Start at rung 3 using `fla`'s GDN + a channel-wise erase gate you add yourself** — smaller diff, better-supported kernels, same science |
| **Sequence length for training** | Needle tests need 8K contexts. T4 memory + 8K + chunkwise is feasible at ~40M params but **watch it**. Curriculum: train at 2K, evaluate at 8K, extend only if the curve demands it |
| **~5% throughput cost** | Irrelevant — the RSC is not in the decode path |

---

## 6. Design for swappability — this target moves fast

Four substantive papers in five months (GDN-2 May, EDA Jun, query-erase Aug, TERN Sep). **Do not
hard-code the update rule.**

```python
# friday/rsc/operators.py — one interface, five implementations, one config flag
class RecurrentOperator(Protocol):
    def forward(self, S_prev, q, k, v, *, alpha, b, w, e) -> tuple[State, Output]: ...

OPERATORS = {
    "retnet":       FixedDecayOperator,        # rung 0 — the control that must lose
    "gdn":          GatedDeltaNetOperator,     # rung 1 — from `fla`
    "kda":          KimiDeltaOperator,         # rung 2 — from `fla`
    "gdn2":         GatedDeltaNet2Operator,    # rung 3/4 — NVlabs kernels
    "eda":          EraseThenDeltaOperator,    # rung 5 — independent erase address
}
# Config, not code, decides which runs. Every rung is a YAML flag and an eval row.
```

**Three rules that follow:**
1. **The state tensor's shape is the contract** (`n_heads × d_k × d_v`, 16×128×128). Any operator
   that produces it can be swapped without touching the backbone interface.
2. **Every rung gets a row in the eval table**, permanently. When rung 6 appears in November you
   add a flag, not a rewrite.
3. **Pin the paper you implemented against, with the arXiv ID and date, in the config.** In six
   months "GDN-2" will mean something different to different people.

---

## 7. Revised roadmap (amends [12 §9](./12-LINEAR-ATTENTION-PIVOT.md))

| Phase | Δ |
|---|---|
| **0 — Foundation** | **UNCHANGED. No custom layers.** Still: bake-off on *existing* GGUFs (LFM2.5-1.2B / LFM2-2.6B / LFM2-8B-A1B / RWKV-7 / Qwen3-4B baseline), needle-recall test, `BENCHMARKS.md`. **Exit test is a Friday follow-up recalling a Monday fact — not a working recurrent operator** |
| **1 — Memory Compiler** | **Δ One extra field.** The salience-label tap now also records **retraction pairs** `(fact_old.trace_span, fact_new.trace_span, key_old, key_new)`. These are `L_erase`'s free supervision. Same cost as before: one column, one insert |
| **2 — Attention Ledger** | unchanged |
| **3 — Perception & Presence** | unchanged |
| **⭐ 3.5 — RSC** | **Δ Substantially upgraded.** Operator = GDN-2 (rung 3/4) via NVlabs kernels or `fla`; ablation ladder 2→3→4; five-term loss incl. `L_erase` and `L_decay`. **EDA (rung 5) is a stretch goal inside this phase, gated on `L_erase` plateauing** |
| **4 — Proactivity** | unchanged |
| **5 — Self-Improvement** | unchanged; **Law 2c still applies** — re-run needle recall after every RSC training run too |
| **5.5 — Tiny specialists** | unchanged |
| **6 — Custom architecture** | **Δ Now has a concrete, current recipe.** If you build the 100–300M hybrid, the mixer is **GDN-2 or EDA**, not "Gated DeltaNet" generically — and the block is `GDN-2 → MLP → SWA(w=2048) → MLP`, exactly as published. **Still research-only, still never blocks production** |
| **7 — Skills & Tasks** | unchanged |

**The one-line version: nothing about Phase 0 changes. GDN-2 makes Phase 3.5 much better and
Phase 6 much more concrete. It does not become Phase 0.**

---

## 8. Decision summary

| Question | Answer |
|---|---|
| **Is GDN-2 real?** | **Yes.** NVIDIA, Hatamizadeh/Choi/Kautz, [arXiv 2605.22791](https://arxiv.org/abs/2605.22791), 21 May 2026, CC BY 4.0, [`NVlabs/GatedDeltaNet-2`](https://github.com/NVlabs/GatedDeltaNet-2) 323★, last commit 29 Aug 2026 |
| **Is it better than what I specified?** | **Yes.** I said "Gated DeltaNet / Mamba-2 class." GDN-2 beats **GDN, KDA, Mamba-2 and Mamba-3** at matched 1.3B/100B, and its hybrid block *is* `GDN-2 → MLP → SWA → MLP` — the SWA pairing you proposed is the paper's own design, not an addition |
| **Is the erase/write instinct right?** | **More right than argued.** NVlabs' ablation: **the erase gate accounts for most of the gain.** FRIDAY's dominant problem is accumulated transient noise, and durable content doesn't need the state at all (it goes to the store). **Erase is FRIDAY's lever; write barely matters** |
| **Does it fix the recall gap?** | **It raises the ceiling; it does not remove it.** S-NIAH-3 **63 → 90**. Ninety is not one hundred, and the failure is silent. **Law 2b stands** |
| **Is "permanent associative records" right?** | **No — this is the one error that would sink it.** Nothing in a fixed-size state is permanent. Treating it as the record rebuilds the RetNet failure mode at a higher ceiling. The decoupling lets FRIDAY **forget harder with less collateral damage**; the external store is what makes forgetting *safe* |
| **Does it go in Phase 0?** | **No, and it can't.** The NVlabs repo is a **training harness** (`pretrain.py`, `lit_gpt/`, Dockerfile) with **no inference server, no GGUF, no llama.cpp support, no checkpoint.** There is no path to serving it on a 16 GB CPU laptop |
| **Where does it go?** | **Phase 3.5 — as the RSC's recurrent operator.** The RSC runs in PyTorch on Kaggle T4 with a **frozen** LFM2 backbone, offline, out of the decode path. Every blocker disappears |
| **Can GDN-2 layers go *inside* LFM2-2.6B?** | **No.** Different architecture, frozen weights. LFM2 is the backbone; the RSC is a module bolted onto it. That separation is the design, not a limitation |
| **Is SWA the retrieval store?** | **No.** SWA is a **layer** over the last ~2048 tokens (*"local evidence"*). sqlite-vec/FTS5 is a **database** reached by a **tool call**. Two parallel paths that meet in the prompt — not a pipeline |
| **What's better than GDN-2 for FRIDAY?** | ⭐ **EDA** ([2606.26560](https://arxiv.org/abs/2606.26560), Jun 2026). GDN-2 decouples **how strongly**; EDA decouples **where**. GDN-2's erase direction is still built from the current write key — so it **cannot clear a stale association at a different address.** That is *exactly* FRIDAY's bi-temporal correction case. And they **compose**: EDA's own baseline is GDN-2-shaped |
| **What's the genuinely new idea here?** | ⭐ **The Memory Compiler is the gate supervision.** GDN-2 has three branches (decay α_t, erase b_t, write w_t); FRIDAY already computes three matching signals — the **decay function**, **retraction events**, and the **salience head**. Retractions yield `(key_old, key_new)` pairs, which is free supervision for EDA's independently-addressed erase. **A recurrent memory operator supervised by a bi-temporal knowledge base.** |
| **Is it validated?** | **Not externally.** All results are NVIDIA's, four months old, at 1.3B/100B tokens — *the same undertrained scale I criticised RetNet for.* It is a valid **controlled architecture comparison** (all baselines trained identically), so the **update rule** ports. It is **not** a production model claim |
| **What do I do Monday?** | **Nothing about this.** Run the Phase 0 bake-off from [12 §11](./12-LINEAR-ATTENTION-PIVOT.md). The RSC is Phase 3.5, weeks 8–12. **Phase 1's only change: start recording retraction pairs now**, because that's `L_erase`'s training data and you can't backfill it |
