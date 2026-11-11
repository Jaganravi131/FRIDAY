# 15 · The free-intelligence ladder: closing on Astra at ₹0

> Written 2026-09-30, after the competitive analysis in doc 14. Sources for the
> time-sensitive claims (free-tier limits, Vulkan performance, model sizes) are inline;
> all of them shift, so re-verify before acting. Everything here is ₹0.

Doc 14 concluded that FRIDAY cannot match GPT-6 Astra and should stop competing on that
axis. That is still true of the *model*. This document is about the question underneath
it: **how much intelligence can be assembled for zero rupees, and where does it actually
come from?**

The answer is more than doc 14 assumed, because doc 14 (and the first draft of
`INSTALL.md`) made an arithmetic error worth correcting in public.

## 0 · The correction that changes the ceiling

FRIDAY's target machine is a Ryzen 5 with a **Radeon iGPU and no CUDA**. The earlier
reading was "no CUDA ⇒ CPU decode ⇒ stay at 1.2B–3B." That is wrong.

llama.cpp has a mature **Vulkan** backend that runs on AMD and Intel iGPUs **on Windows,
with no ROCm and no extra drivers** — the Vulkan loader ships with every AMD graphics
driver. Measured field reports:

| Setup | Prefill | Generation | Source |
|---|---|---|---|
| Gemma4 **26B** Q4_K_M, Radeon 780M iGPU, Vulkan | ~209–239 t/s | **~21–25 t/s** | [llama.cpp discussion #24222](https://github.com/ggml-org/llama.cpp/discussions/24222), [dev.to field report](https://dev.to/hrodrig/21-toks-gemma-4-on-a-ryzen-mini-pc-llamacpp-vulkan-and-the-messy-truth-about-local-chat-m82) |
| Llama **8B** Q8_0, Radeon 760M, Vulkan | ~236 t/s | ~9.8 t/s | [dev.to](https://dev.to/hrodrig/21-toks-gemma-4-on-a-ryzen-mini-pc-llamacpp-vulkan-and-the-messy-truth-about-local-chat-m82) |
| 8B Q4, **CPU only** | ~45–50 t/s | ~10–18 t/s | [promptquorum](https://www.promptquorum.com/local-llms/best-beginner-local-llm-models) |

The same report measures a **5–6× uplift over the Ollama CPU path** on identical
hardware. On 16 GB the realistic local ceiling is therefore not 1.2B — it is
**Qwen3 14B Q4 (~8.5 GB)** or **Gemma 3 12B Q4 (~6.7 GB)** as a daily driver
([model sizes by RAM tier](https://www.frankx.ai/blog/best-local-llm-2026)). That is
roughly a **10× parameter jump** over what doc 14 recommended, for zero rupees, and it
was available the whole time.

`friday doctor` previously reported "no CUDA. Expected." and stopped. It now identifies
the GPU and the Vulkan loader and says which of the four cases you are in. **The cost of
the original error was not a wrong number in a document — it was a 10× smaller model on
the user's actual laptop.** That is the argument for `doctor` existing at all.

### Why Vulkan helps FRIDAY more than it helps a chat app

Generation tok/s is the number everyone quotes, but FRIDAY is **prefill-heavy**: every
turn re-prefills the ledger block — identity, senses, user, core memory, recalled slots.
Vulkan's prefill advantage (~200–285 t/s vs ~45–50 on CPU) lands exactly there, and it
compounds with the byte-stable prefix the Attention Ledger enforces (doc 05), because a
stable prefix is what lets llama.cpp's prompt cache skip the prefill entirely on
follow-up turns. **A 5× faster prefill plus a cacheable prefix is worth more to this
project than any model swap.**

## 1 · The six levers, ranked by intelligence per rupee

Ranked by what they buy, not by how interesting they are to build.

---

### Lever 1 — Right-size the local model on Vulkan. *(do this first; it is a config change)*

**Cost:** ₹0, one afternoon. **Gain:** ~10× parameters, 5× prefill.

Already specified in `INSTALL.md` §3. The only engineering work in FRIDAY is that
`FRIDAY_CTX` and the ledger's slot budgets should be re-tuned for a 12–14B model with a
larger usable context — the current 8192 default was chosen for a 1.2B model on CPU.

Nothing else in the architecture changes. This is the highest ratio in the document and
it is nearly free.

---

### Lever 2 — Rent frontier reasoning for the hard 5%, behind an egress lattice.

**Cost:** ₹0. **Gain:** the only route to genuinely Astra-class reasoning on this budget.

The free tiers available in 2026, with the catch that decides whether FRIDAY may use
them:

| Provider | Free allowance | The catch | Admissible for FRIDAY? |
|---|---|---|---|
| **Groq** | `gpt-oss-120b` at 30 RPM, 1,000 RPD, 8,000 TPM, **200,000 TPD** | The **token** cap binds first: at ~2,500 tokens/request that is **~80 real requests/day**, not 1,000. Sources disagree on whether 120b is still flagged free as of Sep 2026 — verify in the console | ✅ **Yes.** Groq's services agreement does **not** permit training on your inputs or outputs unless you explicitly allow it; data retention listed as none ([requesty](https://www.requesty.ai/models/groq/openai-gpt-oss-120b), [merginit](https://merginit.com/blog/16062026-free-ai-api-inference-comparison), [limits](https://www.grizzlypeaksoftware.com/articles/p/groq-api-free-tier-limits-in-2026-what-you-actually-get-uwysd6mb)) |
| **Google Gemini** | Flash / Flash-Lite free of charge, ~1,500 RPD, 1M TPM | **Google's own pricing table marks free-tier content as used to improve its products.** The paid tier says no | ❌ **No.** This is a personal memory system. "Free" paid for with your diary is not free |
| **OpenRouter** | 20 RPM, **50 RPD** unfunded | 1,000 RPD only after purchasing $10 of credit — no longer pure zero | ⚠️ Marginal. 50/day is a rounding error; the funded tier breaks the ₹0 rule |
| **Cerebras** | $5 credit, 1M tokens/day | Credit expires; not a permanent tier | ⚠️ Good for a burst, not a foundation |

([tier comparison](https://continuumcode.ai/guides/free-llm-api/),
[stacking strategy](https://klymentiev.com/blog/best-free-llm-2026))

So: **one admissible destination, ~80 requests a day, a 120B MoE reasoning model.** That
is enough for the hard 5% of turns — multi-step planning, a difficult synthesis, a
question the local 12B fumbles — and nowhere near enough to run FRIDAY on. Which dictates
the design: **a router, not a replacement.**

#### The architectural piece this requires: an egress lattice

FRIDAY already has a **trust lattice** governing what may be *believed* from a source
(doc 06, `security/policy.py`). Sending context to Groq is the mirror image and needs the
same treatment: a lattice governing what may be *sent* to a destination.

This is not a new subsystem, it is the existing one turned around — and FRIDAY is
unusually well placed to build it, because **`redact()` already runs at the write
boundary.** Most systems cannot use a cloud API for personal memory safely at all.
FRIDAY can, because the thing that makes it safe already exists and is already tested.

Concretely, `friday/security/egress.py`:

```
DESTINATION  trust class    required redaction    approval
local        trusted        none                  none
groq-free    vetted-free    CREDENTIAL+PII        first use, then remembered
<anything>   unvetted       —                     DENIED (no ad-hoc destinations)
```

Rules, all enforced in the data layer per Law 5 — never in prose:

1. **Local by default.** A turn egresses only if the router decides it must and the
   destination is on the vetted list.
2. **Redact at egress, not only at write.** The context that leaves is passed through
   `redact(pii=True)`. What is verified as leaving is the *actual serialized payload*,
   not the intent — same discipline as the write path.
3. **Unattended turns may never egress.** `UNATTENDED_DENY` already blocks writes from
   heartbeat/dreaming/eval scopes; egress joins that list. A background turn sending
   your memory to a third party with nobody watching is precisely the failure mode doc 09
   exists to prevent.
4. **Every egress is an audit row.** `audit --today` must answer "what left this machine,
   to whom, when." If it cannot, the feature does not ship.
5. **The ledger gains an `egress` slot**, so the fact that a turn was routed out is
   visible in the same printout the user already reads every turn.

Worth stating plainly: this is a **policy decision the user owns, not a technical one.**
The architecture's job is to make the choice explicit, reversible, auditable, and
structurally incapable of leaking more than the user approved. It is not to make the
choice for them.

---

### Lever 3 — Test-time compute, spent where time is free.

**Cost:** ₹0, wall-clock only. **Gain:** the most under-used lever on this list.

Reasoning quality can be bought at *inference* time instead of training time: sample k
candidate answers and take the majority (self-consistency), generate N and rank them with
a verifier (best-of-N), or critique-and-refine in a loop. On a CPU this is slow. On an
iGPU via Vulkan it is merely slow. Either way it is free.

**FRIDAY has a place to put it that most systems do not: the `dreaming` scope.** That
scope is unattended and latency-insensitive by definition — it runs when nobody is
waiting. So it can burn minutes of compute per conclusion, and `UNATTENDED_DENY` already
guarantees it can *reason* without being able to *act*. The output is not a chat reply;
it is a candidate fact written through the normal provenance path, which means a
slow expensive inference overnight becomes an auditable claim you can inspect with
`/why` in the morning.

This is doc 10's self-improvement loop with compute substituted for money. **Spending
time instead of rupees is the only currency FRIDAY has in abundance.**

---

### Lever 4 — Retrieval quality. The highest ROI for a *memory* assistant.

**Cost:** ₹0, pure engineering. **Gain:** larger than a model swap, for this workload.

For a RAG system, answer quality tracks retrieval quality far more closely than it
tracks model size. FRIDAY currently has FTS5 plus a **hashing embedder** and a lexical
reranker. Four upgrades, in order of value:

1. **Hybrid retrieval with RRF fusion.** BM25 (FTS5, already there) and dense vectors
   fail on different queries; reciprocal-rank fusion combines them for a measured gain
   that costs nothing but code. FRIDAY has both halves and does not fuse them.
2. **Query expansion / HyDE.** Rewrite the question, or have the model write a
   hypothetical answer and embed *that*.

   > **Correction, and a lesson about diagnosing from symptoms.** This lever was
   > originally justified by the gateway bug where `who is my manager` scored below the
   > floor while `my manager` retrieved fine, and it blamed "the hashing embedder
   > diluting on interrogative tokens". That diagnosis was **wrong**. The embedder was
   > not involved: the *lexical* reranker divides coverage by the number of query words,
   > and `when`/`who`/`where`/`why`/`how`/`which` were absent from the stopword list it
   > used, so a question word halved the score of an otherwise perfect match. Six words
   > in a frozenset, not a modelling limitation. Unifying the three stopword lists into
   > `db.CONTENT_STOPWORDS` took interrogative recall@5 from 0.545 to **1.000** with no
   > new dependency and no model at all.
   >
   > Query expansion remains worth doing — but for **paraphrase**, which is still 0.000
   > and is genuinely a semantics problem ("who do I report to" shares no token with a
   > fact about a manager). The cheap version is lever 4, a real embedder; HyDE is the
   > version that also needs a model good enough to write a plausible hypothetical
   > answer, which is a higher bar than it sounds like.
3. **Contextual retrieval.** Prepend chunk-specific context to each passage *before*
   embedding it. Cheap, and one of the better-measured retrieval improvements available.
4. **A real cross-encoder reranker.** Small `ms-marco-MiniLM`-class rerankers run on a
   CPU in tens of milliseconds. This needs `onnxruntime` or `sentence-transformers`, so
   it belongs in the **optional** dependency tier FRIDAY already has — the core stays
   stdlib-only, and `doctor` reports which tier you are on.

None of this makes the model smarter. All of it makes the model *look* smarter, which for
an assistant whose job is remembering you is the same thing.

---

### Lever 5 — Distillation: turn the free frontier tier into local weights.

**Cost:** ₹0 (Kaggle gives 2× T4, ~30 GPU-hours/week). **Gain:** compounding, slow.

The loop: use Lever 2's ~80 free frontier requests a day to generate high-quality
reasoning traces **on your own domain** — your facts, your phrasing, your Tamil/English
code-switching — then QLoRA the local model on them. Over months, capability that was
rented becomes owned, and runs offline.

FRIDAY's unusual advantage, from doc 14: **the system generates its own supervision
signal.** Every turn already records the ledger block, the retrieved context, the tools
called, the policy verdict and the audit row. That is a labelled dataset of "given this
context, correct behaviour was X," accumulating for free on your own machine.

Targets, in priority order — all *behaviour*, not capability:

1. **"I don't know" calibration.** The most valuable behaviour in an assistant that
   remembers your life. Base models are trained to be helpful, which means they answer.
2. **Tool-call discipline.** Well-formed calls, only when the context warrants one,
   never an invented tool.
3. **Fence adherence.** Treating `<recalled>` as data — the PREFIX_RULE behaviour that
   had to be debugged by hand.
4. **Provenance-shaped answers.** Answering from the retrieved block and citing it.
5. **Code-switching.** A data problem, not a capability problem — the ideal LoRA target.

⚠️ **Two things to verify before running this.** First, whether the provider's terms
permit *you training on their outputs* — Groq's no-training clause covers them training
on your data, which is not the same question. Second, the licence of the base model
(Qwen3 is Apache 2.0 and clean; Gemma and Llama are restricted open-weights). Build the
evaluation suite **before** the first run, and make **refusal rate the headline metric** —
a model that confabulates less is worth more here than one that scores higher anywhere.

---

### Lever 6 — Agentic scaffolding. Where Dots and OpenClaw actually get their power.

**Cost:** ₹0, pure engineering. **Gain:** large, and it is not the model.

Neither reference system is impressive because of its model. Both are impressive because
of what the model can *reach*. FRIDAY has eight tools. Free, keyless additions:

- **Wikipedia / Wikidata APIs** — no key, no limit worth hitting
- **arXiv API** — no key
- **DuckDuckGo HTML** — no key
- **A calculator and a Python sandbox** — local, free, and the single most reliable way
  to stop a model doing arithmetic badly
- **The filesystem and the calendar**, under the existing tiered-sense approval

Then the loops that make tools into agency: plan → execute → verify, and a reflection
pass that checks its own answer before returning it. Both are free at runtime and both
are where the perceived intelligence jump actually comes from.

## 2 · What this adds up to

Stacked, at ₹0:

| | Doc 14's assumption | After this ladder |
|---|---|---|
| Local model | 1.2B on CPU | **12–14B Q4 on Vulkan iGPU** |
| Prefill | ~45 t/s | **~200–285 t/s**, and cacheable |
| Hard reasoning | none | **120B MoE, ~80 requests/day**, redacted at egress |
| Inference strategy | single pass | **self-consistency / best-of-N in the dreaming scope** |
| Retrieval | FTS + hashing embedder | **hybrid RRF + HyDE + contextual + cross-encoder** |
| Weights | frozen | **QLoRA on free Kaggle T4, distilled from your own audit log** |
| Reach | 8 tools | **8 + free keyless web/math/calendar, with plan-verify loops** |

That is not Astra. It will not hold a 1.05M-token context or out-reason a frontier model
on a hard proof. But it is a **12–14B local model with frontier-class backup on demand,
better memory than either competitor, and provenance neither has** — assembled for zero
rupees, on a laptop, with the user's data under their own control.

## 3 · The honest ceiling

What none of this buys, and what should not be promised:

- **Sustained autonomous agency.** Dots runs an always-on agent with its own cloud
  computer. FRIDAY's equivalent would be unattended action on your machine, and doc 09's
  threat model says no — correctly. GPT-6.1 Astra was *cancelled* over safety concerns
  after autonomous hacking incidents. That is the news to design against, not the
  capability to envy.
- **Frontier reasoning on demand.** ~80 requests a day is a scalpel, not an engine. Free
  rosters also shift: one source reports `gpt-oss-120b` unflagged from Groq's free tier
  in Sep 2026. **Anything built on a free tier must degrade gracefully when it vanishes** —
  which is exactly the fallback discipline `get_client("auto")` already implements for
  the model server.
- **Long context.** 8192–32K local against 1.05M. FRIDAY's answer is retrieval plus the
  RSC state block, which is a different mechanism aimed at the same felt experience. It
  is not the same thing and should not be described as such.

## 4 · Ordered plan

By value ÷ effort, all ₹0:

1. **Vulkan + Gemma 3 12B** (or Qwen3 14B). A config change and a re-tuned
   `FRIDAY_CTX`. Biggest single jump in the document, smallest effort.
2. **Retrieval: hybrid RRF fusion + query rewriting.** Fixes a real observed bug
   (`who is my manager`), and raises every answer that follows.
3. **Wire the heartbeat scheduler.** `HEARTBEAT.md` exists and `UNATTENDED_DENY` already
   makes it safe. Smallest change with the largest shift in how the system *feels*.
4. **Build the fine-tuning evaluation suite.** Refusal rate, tool-call validity, fence
   adherence, recall. Must exist before any training run or the runs are unmeasurable.
5. **Test-time compute in the dreaming scope.** Self-consistency first; write conclusions
   back through the provenance path.
6. **Egress lattice + the Groq router.** The largest build on this list, and the only one
   that needs a policy decision from the user before a line of code. Do not start it
   before that decision is made explicitly.
7. **First QLoRA run** on free Kaggle T4, targeting behaviours 1–3 from Lever 5.
8. **Then** the doc 13 architecture work — GDN-2 + SWA hybrid, custom erase gate. Months,
   and it needs everything above to have proven the evaluation harness first.
