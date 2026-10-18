# 02 — INNOVATION #1: The Memory Compiler

> **The central architectural idea of FRIDAY.**
>
> FRIDAY is not a chatbot with a database. It is a **compiler** whose input is your life and
> whose output is a progressively smaller, cheaper, faster, more personal model.

---

## 1. Why "compiler"

Every optimising compiler you've ever used does the same thing: it takes a verbose, redundant,
human-written input and produces a dense, fast, machine-optimised output through a **pipeline of
stages**, each with **passes**, **gates**, and the ability to **reject and roll back**.

Memory in an AI agent has exactly the same shape problem:

| | Compiler | FRIDAY |
|---|---|---|
| Input | source code | your raw interaction traces |
| Output | machine code | an internalised personal model |
| Verbosity | redundant source | 40,000 tokens of chat about one decision |
| Density | tight instructions | one line: *"prefers TypeScript over JavaScript"* |
| Cost gradient | compile once, run forever | consolidate once, recall instantly |
| Correctness | type checker, tests | **eval gate** |
| Rollback | `git revert`, rebuild | `git revert`, re-index |

Nobody in the personal-agent space frames it this way. They build "memory layers" — static tiers
you query. FRIDAY builds a **pipeline that transforms**. That reframing is the innovation, and it
has concrete engineering consequences.

---

## 2. The five stages

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          THE MEMORY COMPILER                                   │
│                                                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ S0 TRACE │──►│S1 EPISODE│──►│ S2 FACT  │──►│ S3 SKILL │──►│S4 WEIGHT │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│   raw signal     narrative      knowledge      procedure      intuition       │
│                                                                                │
│   JSONL          Markdown       Markdown +     SKILL.md       LoRA adapter    │
│   append-only    per-day        bi-temporal    + code         → GGUF          │
│   never edited   summarised     rows                                           │
│                                                                                │
│  ─────────────────────────────────────────────────────────────────────────►   │
│   CHEAP · LOSSLESS · SLOW TO USE              EXPENSIVE · LOSSY · INSTANT     │
│   high volume, low value density              low volume, high value density  │
│                                                                                │
│         ▲                                     ▲                                │
│         │  every turn (write path, async)     │  nightly / weekly (Dreaming)   │
└─────────┼─────────────────────────────────────┼────────────────────────────────┘
          │                                     │
     NEVER on the                          NEVER on the
     turn critical path                    turn critical path
```

### S0 — TRACE (raw signal)

**What:** every interaction, verbatim. Append-only JSONL, partitioned by day.

```jsonl
{"id":"t_01J8ZK...","ts":"2026-09-29T14:32:07+05:30","scope":"interactive","surface":"mobile","mode":"voice","channel":"wss","turn":42,
 "input":{"audio_ms":2340,"transcript":"remind me the thing about the Chennai lease","confidence":0.94,"vad_frames":78},
 "context_snapshot":{"ledger_hash":"a91f...","slot_tokens":{"identity":612,"senses":180,"memory":410,"retrieval":0,"scratch":1240,"reserve":2000},"cache_hit":true},
 "reasoning":{"tier":"L1","model":"qwen3:4b","ttft_ms":312,"tokens_out":187,"tool_calls":[{"name":"memory_search","args":{"q":"Chennai lease"},"ms":41,"hits":3}]},
 "output":{"text":"...","audio_ms":4100},
 "affect":{"valence":-0.2,"arousal":0.4,"note":"mild frustration, self-reported tired"},
 "feedback":{"explicit":null,"implicit":"accepted","barge_in":false,"followup_within_60s":true},
 "senses_used":["calendar.read","files.read:~/Documents/lease"],
 "redactions":[{"field":"input.transcript","pattern":"PAN","action":"tokenised"}]}
```

**Design rules:**
- **Append-only. Never edited. Never summarised in place.** This is your ground truth and your
  debugging record. If FRIDAY ever behaves strangely, the answer is in here.
- **Strip media blobs before persisting.** Store a path + hash, not the bytes.
  (jarvis-agent does exactly this — "media blobs are stripped before persistence".)
- **Redact at write time**, not read time. If a PAN/Aadhaar/card number enters the trace,
  tokenise it *before* it lands on disk. See [09](./09-security-privacy-trust.md).
- **Record the ledger hash + slot token counts.** This is what makes the Attention Ledger
  measurable and lets you correlate "bad answer" with "context was starved in slot X".
- **Record implicit feedback signals** — `barge_in`, `followup_within_60s`, `accepted`,
  `regenerated`, `corrected`. These are your **free reward signal** for S3/S4. This is the
  detail most projects miss: *you don't need a thumbs-up button, you need to observe behaviour.*

**Retention:** 90 days hot on SSD, then archive to a compressed cold file. At your volume this
is maybe 2–5 GB/year. You have 512 GB. Keep it all.

---

### S1 — EPISODE (narrative)

**What:** each day compressed from ~40K tokens of trace into ~800 tokens of narrative.

```markdown
<!-- memory/episodes/2026-09-29.md -->
---
date: 2026-09-29
turns: 47
duration_active: 3h12m
dominant_topics: [flat-lease-renewal, friday-architecture, gym-schedule]
affect_arc: "flat/frustrated AM → focused PM"
compiled_at: 2026-09-30T03:00:00+05:30
compiler_version: 0.4.1
source_traces: 1423
confidence: 0.88
---

## What happened
Jagan spent the morning on the Chennai flat lease renewal — the broker hasn't sent the
draft, third follow-up today. Frustrated. Afternoon switched to FRIDAY architecture;
long focused block, ~2h, no interruptions.

## Decisions made
- FRIDAY's core language: **Python** (decided after weighing TS; reason: the
  self-improvement stack — DSPy/Unsloth/torch — is Python-native). [→ fact f_092]
- Rejected Pinecone; going SQLite + sqlite-vec. [→ fact f_093]

## Open loops  ← these feed HEARTBEAT.md
- [ ] Broker's lease draft — not received as of 18:00. **Chase tomorrow 10:00.**
- [ ] Decide on 32 GB RAM upgrade for the VivoBook — was researching prices.
- [ ] Phase 0 exit test not yet run.

## What I (FRIDAY) got wrong
- Twice answered about the *old* lease amount. The fact was stale. [→ invalidation f_041]
- Over-explained WebRTC when a one-liner was wanted. Barge-in at 1.2s. [→ skill s_017 revision]
```

**The critical pass: "What FRIDAY got wrong".**
This section is the engine of self-improvement. It is generated by a **reflection pass** over
the day's traces using the implicit feedback signals from S0:
- turns where the user **barged in** → FRIDAY was too slow or too verbose
- turns where the user **immediately rephrased** → FRIDAY misunderstood
- turns where the user said **"no, I meant…"** → a fact was wrong or stale
- turns where FRIDAY **escalated to L2 unnecessarily** → router mis-calibrated
- turns where FRIDAY **failed to escalate and got it wrong** → router under-confident

> ⚠️ **This reflection is NOT self-correction of reasoning.** Huang et al. (ICLR 2024) showed
> LLMs cannot reliably self-correct reasoning without external feedback and sometimes *degrade*.
> What makes this valid is that the signal is **external and behavioural**: barge-in events,
> rephrase rates, escalation outcomes, eval scores. FRIDAY is not asked "was that good?" —
> it is shown what *you did*.

**Design rules:**
- Compiled by a **local model** (Qwen3-4B is fine — this is summarisation, not reasoning) during
  the Dreaming phase. Never on the turn path. (Letta's *sleep-time compute* principle.)
- **Bidirectional links**: `episode → facts it produced`, `fact → episodes that support it`.
  This is your **provenance chain** — the answer to "why do you believe that about me?"
- Episodes are **not auto-injected into context**. Retrieved on demand. (OpenClaw's daily-log
  discipline.)

---

### S2 — FACT (knowledge)

**What:** atomic, bi-temporal, human-editable statements about you and your world.

This is where FRIDAY stops being a summariser and starts being *context-aware*.

#### The bi-temporal model

Every fact carries **two independent time axes**:

| Axis | Meaning | Example |
|---|---|---|
| **Valid time** (`valid_from`, `valid_to`) | when the fact was **true in the world** | "lives in Chennai" — true from 2023-06 |
| **Transaction time** (`asserted_at`, `retracted_at`) | when **FRIDAY believed** it | FRIDAY learned this on 2024-01-12 from a calendar entry |

Why both? Because they diverge constantly:
- You *moved* in June but only *mentioned* it in September → valid_from = June, asserted_at = Sept.
- FRIDAY *believed* you liked Python, then you corrected it → the old row gets `retracted_at`,
  and a new row is created. **Nothing is deleted.**

```sql
CREATE TABLE facts (
  id            TEXT PRIMARY KEY,          -- f_092
  subject       TEXT NOT NULL,             -- 'user' | 'person:Amma' | 'project:FRIDAY' | 'device:vivobook'
  predicate     TEXT NOT NULL,             -- 'prefers_language'
  object        TEXT NOT NULL,             -- 'Python'
  object_type   TEXT,                      -- 'str'|'num'|'date'|'entity'|'bool'
  confidence    REAL DEFAULT 0.5,          -- 0..1, decays
  salience      REAL DEFAULT 0.5,          -- how much this matters (not how recent)

  -- valid time (the world)
  valid_from    TEXT,
  valid_to      TEXT,                      -- NULL = currently true

  -- transaction time (FRIDAY's belief)
  asserted_at   TEXT NOT NULL,
  retracted_at  TEXT,                      -- NULL = FRIDAY still believes this
  superseded_by TEXT REFERENCES facts(id), -- the correction chain

  -- provenance (Law: no belief without a source)
  source_kind   TEXT NOT NULL,             -- 'stated'|'observed'|'inferred'|'imported'|'user_edit'
  source_refs   TEXT NOT NULL,             -- JSON array of trace/episode/file ids
  source_quote  TEXT,                      -- the literal words, if stated

  -- lifecycle
  review_after  TEXT,                      -- when to re-verify (decays confidence)
  access_count  INTEGER DEFAULT 0,         -- recall frequency → salience signal
  last_accessed TEXT,
  embedding_id  TEXT,                      -- pointer into sqlite-vec
  fts_rowid     INTEGER                    -- pointer into FTS5
);

-- the two queries that matter
CREATE INDEX idx_facts_current  ON facts(subject, predicate) WHERE retracted_at IS NULL;
CREATE INDEX idx_facts_valid    ON facts(subject, valid_from, valid_to);
```

**This buys you the four questions a flat vector store cannot answer:**
1. *"What's my current lease amount?"* → `valid_to IS NULL AND retracted_at IS NULL`
2. *"What was my lease amount last year?"* → point-in-time query on valid time
3. *"When did I start preferring TypeScript?"* → `valid_from`
4. *"Why did you think I lived in Coimbatore?"* → the retracted row + `source_refs` + `source_quote`

> This is precisely the 15-point LongMemEval gap between Zep (63.8%) and Mem0 (49.0%):
> **temporal validity windows.** We get Zep's semantics without Zep's Neo4j dependency.

#### Invalidation, not overwrite

When a contradicting observation arrives, run the **reconciliation pass**:

```
new observation O(subject=S, predicate=P, object=V_new)
  │
  ├─ query current facts for (S, P) where retracted_at IS NULL
  │
  ├─ no existing fact ────────────► CREATE (confidence from source_kind)
  │
  ├─ existing V == V_new ─────────► REINFORCE
  │                                  confidence ↑, access_count ↑, review_after pushed out
  │
  ├─ existing V != V_new
  │    │
  │    ├─ P is single-valued        ┌──► source_kind(new) > source_kind(old)?
  │    │   (e.g. 'lives_in',        │      yes ─► INVALIDATE old (retracted_at=now,
  │    │    'current_lease')        │                     superseded_by=new)
  │    │                            │              CREATE new (valid_from=now)
  │    │                            │      no  ─► QUARANTINE → ask the user
  │    │                            └──► "You mentioned X, but I had Y. Which is right?"
  │    │
  │    └─ P is multi-valued         ──► CREATE alongside (e.g. 'likes_food' can hold many)
  │        (e.g. 'likes', 'knows')
  │
  └─ existing fact is stale (review_after passed) ──► DECAY confidence; if < 0.15, ARCHIVE
```

**Source-kind trust hierarchy** (this ordering is a policy decision — tune it):
```
user_edit (you hand-edited the file)   1.00   ← absolute, never auto-overridden
stated    (you said it explicitly)     0.90
imported  (calendar/email/bank API)    0.80
observed  (screenpipe/behaviour)       0.60
inferred  (FRIDAY deduced it)          0.40
```

> **The `user_edit` tier is the trust feature.** If you open `memory/facts/housing.md` and
> change a line by hand, FRIDAY must (a) detect the change via file-watch, (b) re-index **only
> that part**, (c) set `source_kind='user_edit'` and `confidence=1.0`, and (d) **never
> auto-overwrite it again** without asking. That single behaviour is worth more for long-term
> trust than any amount of retrieval accuracy.

#### Markdown as source of truth, SQLite as build artifact

This is **Innovation #2** and it inverts how everyone does it.

```markdown
<!-- memory/facts/housing.md -->
---
domain: housing
version: 14
last_compiled: 2026-09-30T03:00:00+05:30
---

## Current residence
- lives_in: **Chennai, Tamil Nadu** (since 2023-06) [f_012 · stated · conf 0.95]
  <!-- src: t_00421 "i moved to chennai in june" -->
- address_locality: Velachery [f_013 · observed · conf 0.70]
  <!-- src: screenpipe:maps, 2026-08 -->

## Lease
- lease_amount_monthly: **₹28,000** (valid 2025-06-01 → ) [f_041 · imported · conf 0.90]
  <!-- src: bank-api:txn-recurring -->
- ~~lease_amount_monthly: ₹24,000 (valid 2023-06 → 2025-05-31)~~ [f_040 · **retracted 2026-09-29** · superseded_by f_041]
- landlord: Ramesh (contact via WhatsApp only, does not answer calls) [f_044 · stated · conf 0.95]

## Open
- lease_renewal_due: 2026-10-15 [f_045 · imported]
```

**The pipeline direction:**
```
   Markdown  ──(compile)──►  SQLite facts table + FTS5 + sqlite-vec
      ▲                              │
      │                              │  queries run HERE (fast)
      └────(you edit here)───────────┘  truth lives HERE (readable)
```

- **Truth lives in Markdown**, git-versioned. `git log memory/facts/` is FRIDAY's belief history.
  `git revert` is your undo. `git diff` shows exactly what the nightly consolidation changed —
  you can *review* what FRIDAY learned about you while you slept.
- **Indices are build artifacts.** Delete `friday.db` and run `friday compile` — it rebuilds
  perfectly from Markdown. **Indices are never sacred.** This kills the "my vector DB is
  corrupted / I can't inspect what it knows" class of problem permanently.
- **File-watcher for incremental recompile.** `watchdog` on `memory/`; hash each file; on change,
  re-parse and re-index **only that file**. Hand-edits take effect in <1 s without a restart.

Why this is better than a database-first design:
| | DB-first (everyone else) | Markdown-first (FRIDAY) |
|---|---|---|
| Can you read what it knows? | Need a query tool | Open in Notepad |
| Can you correct it? | Need an admin UI | Type in the file |
| Can you diff its beliefs? | No | `git diff` |
| Can you roll back? | Backup/restore | `git revert` |
| Can you sync across machines? | DB replication | `git push` |
| Does it survive a framework change? | Migration | It's text |
| Query speed | fast | fast (compiled index) |

---

### S3 — SKILL (procedure)

**What:** when FRIDAY solves something hard, or does something twice, it writes down *how*.

Format: **agentskills.io `SKILL.md`** — portable across Hermes, Claude Code, OpenClaw and
others, so anything FRIDAY learns is not locked into FRIDAY.

```markdown
<!-- skills/lease-chase/SKILL.md -->
---
name: lease-chase
version: 3
description: Chase the broker for the pending lease draft without being annoying
triggers:
  - "lease draft"
  - "broker hasn't replied"
when_to_use: >
  Use ONLY when there is an outstanding document request to the landlord/broker
  that has gone unanswered for >24h. Do NOT use for rent payment issues.
success_criteria:
  - a follow-up message was sent, OR
  - the user was told why sending one now would be counterproductive
cost:
  max_tool_calls: 6
  max_wall_seconds: 90
permissions: [messaging.draft, messaging.send:whatsapp:broker, calendar.read]
provenance:
  distilled_from: [t_01122, t_01180, t_01203, ep_2026-09-27, ep_2026-09-29]
  first_created: 2026-09-27
  revisions: [{v: 2, why: "user said the 2nd reminder was too formal"},
              {v: 3, why: "added the 'don't chase on Sunday' rule after barge-in"}]
eval:
  shadow_pass_rate: 0.83
  promoted_at: 2026-09-29T03:00:00+05:30
---

## Procedure
1. Check `facts/housing.md` → is `lease_renewal_due` within 21 days? If not, lower urgency.
2. Check the last 5 WhatsApp messages to the broker (messaging.read).
3. **Gate: do not send on a Sunday, and not before 10:00 or after 19:00 IST.**
   ← learned v3: user barged in and said "not now" on a Sunday morning.
4. Draft in the user's register: short, plain, no "Dear Sir", no "kindly".
   ← learned v2: user called the formal draft "too much".
5. Show the draft → **confirmation gate** → send.
6. Write an open loop to HEARTBEAT.md with a 48h re-check.

## Anti-patterns
- Do not escalate to calling the landlord. Fact f_044: he does not answer calls.
- Do not draft more than 2 sentences.
```

**Design rules:**
- **Skills are created by a distillation pass, not mid-conversation.** After a complex task
  completes (measured by: tool calls > N, wall time > T, or user said "perfect"), the Dreaming
  phase proposes a skill.
- **Progressive disclosure.** Skills stay **hidden** from the prompt until activated. Only their
  `name` + `when_to_use` are visible in a compact index (~30 tokens each). The model calls
  `activate_skill("lease-chase")` to load the body. On a 4B local model with a tight budget this
  is essential — 40 fully-expanded skills would consume your entire window.
- **`when_to_use` is a contract, not a description.** Copy the retrieval-tool lesson:
  *"Use ONLY when…"* and *"Do NOT use for…"*. Without the negative clause the model activates
  skills reflexively.
- **Revisions carry a `why`.** Every skill edit records the trace that caused it. That's your
  audit trail and your training data for S4.
- **Shadow before promotion** (Innovation #6). A newly-written or revised skill runs in
  **shadow mode**: executed in parallel with the incumbent policy, outputs compared, *not
  served to you*. Promote only when `shadow_pass_rate` clears the gate.

---

### S4 — WEIGHT (intuition)

**What:** skills and facts that have proven stable get **internalised into the model** so they
stop costing context tokens.

This is the stage nobody in the personal-agent space has, and it's where your
"can I do fine-tuning / transformer innovation" question gets its answer.

#### The economics of internalisation

A fact in S2 costs **~40 context tokens every single turn** it's relevant.
A skill in S3 costs **~300 tokens** when activated.
The same knowledge in S4 costs **0 tokens** — it's in the weights.

```
Knowledge that is:            Lives in:      Cost/turn:
  volatile, contextual          S2 fact       ~40 tokens
  procedural, situational       S3 skill      ~300 tokens (on activation)
  stable, pervasive, stylistic  S4 weight     0 tokens  ← the win
```

**What belongs in weights:** your *register* (how you write — short, plain, no "kindly"),
your *domain vocabulary* (Tamil/English code-switching patterns, your project names, your
colleagues), your *formatting habits*, your *refusal style*. Things that are true on **every
turn** and never need to be looked up.

**What must NOT go in weights:** anything with a `valid_to`. Facts change; weights don't.
Never bake "lease is ₹28,000" into an adapter — you'll be retraining weekly. Weights carry
**style and procedure**, memory carries **facts**.

> This split is exactly what the Dual-Process Agent paper (Electronics 15(6):1232, 2026)
> proposes as future work: *"Combining memory evolution with lightweight parameter updates such
> as LoRA could allow persistent high-confidence patterns to be internalized into model weights
> while preserving the flexibility of external memory."* FRIDAY implements it.

#### The pipeline

```
S2 facts + S3 skills + S0 traces
        │
        ▼
 ┌──────────────────────────────┐
 │ 1. HARVEST (local, cheap)    │   filter: confidence ≥ 0.85, age ≥ 14 days,
 │                              │           access_count ≥ 5, valid_to IS NULL
 │                              │   → candidate knowledge set
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 2. SYNTHESISE TRAINING DATA  │   the SEAL idea: the model writes its own
 │    (cloud teacher, search    │   finetuning data. Use a strong cloud model
 │     time only — OpenJarvis)  │   to turn (trace, outcome) pairs into
 │                              │   (instruction, ideal_response) pairs in
 │                              │   YOUR register. ~200–2,000 examples.
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 3. TRAIN (Kaggle free T4)    │   Unsloth QLoRA, Qwen3-4B base,
 │    30 GPU-hours/week, ₹0     │   rank=16, α=32, all linear layers,
 │                              │   3 epochs, lr 2e-4, bs 1, grad-accum 8
 │                              │   ← a 1.5B reference run took 70 SECONDS
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 4. EXPORT → GGUF → quantise  │   merge adapter, convert, Q4_K_XL
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 5. GATE  ← the hard part     │   run the LOCKED eval suite against
 │                              │   candidate vs incumbent, in shadow.
 │                              │   ACCEPT iff:
 │                              │     target cluster  improves ≥ 5%
 │                              │     held-out set    regresses ≤ 1%
 │                              │     latency         not worse > 10%
 │                              │   else REJECT + log why
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 6. PROMOTE (atomic swap)     │   Modelfile points at new GGUF;
 │                              │   keep previous 3 versions; instant
 │                              │   rollback via symlink flip
 └──────────────────────────────┘
```

**Cadence:** weekly at most. Daily is waste — you don't have enough new stable knowledge.

**The teacher is used at search time only.** This is OpenJarvis's key economic insight: at
100 queries/day the amortised teacher cost falls **below $0.001/query within six months**. At
inference time FRIDAY runs 100% local. You pay a few cents of cloud API a week to make your
local model permanently better.

#### Where you can do genuine transformer research

This is your answer to *"can I work on architectural innovation such as fine-tuning transformers?"*
— **yes, and here's what's actually tractable:**

| Project | Feasible on your hardware? | Why it's interesting |
|---|---|---|
| **QLoRA on Qwen3-4B / Qwen2.5-1.5B** | ✅ Kaggle free T4 | The core loop. Real, publishable-if-you-measure-it result |
| **Train a tiny transformer from scratch** (character/token-level, ~10–50M params, on your own trace corpus) | ✅ **your 512 GB SSD + CPU** | You will understand attention, positional encoding, KV-cache and tokenisation *viscerally*. Nobody who hasn't done this actually understands LLMs |
| **Implement a LoRA adapter by hand** (no PEFT library — just the two low-rank matrices and the merge) | ✅ | Demystifies fine-tuning completely. ~150 lines of PyTorch |
| **Test-time LoRA (TT-SI, arXiv 2510.07841)** | ⚠️ Kaggle for experiments, local for the 1.5B case | The paper used **Qwen2.5-1.5B** — exactly your size class. Uncertainty estimator → self-augment → temporary gradient updates at inference. Genuinely novel to apply to a personal assistant |
| **Speculative decoding with your own draft model** | ✅ local, big latency win | Qwen3-0.6B drafting for Qwen3-4B. Measured `-hfrd Qwen3-0.6B-GGUF:Q8_0 --draft-max 16` gives real speedups on iGPU/CPU |
| **Adapter composition / merging** — one adapter per domain (work, personal, code, Tamil) hot-swapped by the router | ✅ | "Superficial Self-Improved Reasoners Benefit from Model Merging" (EMNLP 2025). Novel for a personal agent: *mood/domain-conditioned adapters* |
| **Distil your own router** — train a 0.6B classifier on your escalation traces to predict "does this need L2?" | ✅ | Cheap, measurable, immediately useful. This is a real, self-improving component |
| **KV-cache compaction for sub-agent briefing** ("latent-briefing") | ⚠️ advanced | Share orchestrator state with workers via task-guided KV-cache compaction. Cutting edge |

> **Recommended research sequence:** hand-implement LoRA → train tiny transformer from scratch
> on your traces → QLoRA a 4B on Kaggle → build the gate → then try test-time LoRA.
> Each step makes the next one comprehensible, and every step produces something FRIDAY uses.

---

## 3. The gates (what makes it a *compiler* and not a *hoarder*)

A pipeline with no gates just accumulates noise. Every stage transition has one:

| Gate | Between | Question it asks | On failure |
|---|---|---|---|
| **G1 Salience** | S0 → S1 | Is this turn worth narrating at all? | Drop (small talk, acks, repeats) |
| **G2 Grounding** | S1 → S2 | Does this fact have a **provenance link** to a real trace/observation? | Reject. *No belief without a source.* |
| **G3 Contradiction** | S1 → S2 | Does it conflict with a higher-trust existing fact? | Quarantine → **ask the user** |
| **G4 Generalisation** | S2 → S3 | Did this happen ≥2 times, or was it hard enough to be worth codifying? | Skip (don't skill-ify one-offs) |
| **G5 Shadow** | S3 → promoted | Does the candidate skill beat the incumbent on the eval suite in parallel execution? | Reject, log the failure, keep as draft |
| **G6 Stability** | S2/S3 → S4 | Confidence ≥0.85, age ≥14d, accesses ≥5, `valid_to IS NULL`? | Not ready — wait |
| **G7 Regression** | S4 → promoted | Target cluster ↑≥5% **AND** held-out ↓≤1% **AND** latency not >10% worse? | Reject, keep the previous GGUF |

**G7's 1% tolerance is borrowed directly from OpenJarvis's "gate"** — they found an edit is
acceptable only if it improves the target failure cluster *without meaningful regressions
elsewhere*, default tolerance 1%.

**Every gate emits a record.** `eval/gate-log.jsonl`. Over time this becomes the most
interesting artifact in the repo: a longitudinal record of *your AI getting better*, with the
rejected candidates preserved. That's a paper.

---

## 4. Decay: the forgetting function

Memory systems that don't decay accumulate noise. FRIDAY's confidence decays on a schedule
tuned per fact *class*:

```python
HALF_LIVES_DAYS = {
    "mood":            2,      # "stressed about X" — gone fast
    "current_task":    3,
    "location":        30,
    "preference":      365,    # "prefers TypeScript" — very sticky
    "identity":        None,   # "name is Jagan" — never decays
    "relationship":    180,
    "project_state":   14,
    "environment":     90,     # "runs Windows 11 on a VivoBook"
}

def decay(fact, now):
    hl = HALF_LIVES_DAYS.get(fact.predicate_class, 90)
    if hl is None:
        return fact.confidence
    age_days = (now - fact.asserted_at).days
    # reinforcement: each access pushes the effective age back
    effective_age = age_days / (1 + 0.15 * fact.access_count)
    return fact.confidence * (0.5 ** (effective_age / hl))

# confidence < 0.15  → ARCHIVE (drop from indices, keep in Markdown as struck-through)
# confidence < 0.40  → DEMOTE  (only surfaces on direct query, never auto-injected)
```

Note the **reinforcement term**: recalling a fact makes it stickier. That's the
importance × recency × relevance scoring the ecosystem converged on — but with an explicit,
tunable half-life per *class* rather than one global number.

---

## 5. The retrieval path (read side)

Everything above is the write path, which runs **off the turn**. The read path must be fast.

```
user turn
   │
   ├─ 1. INTENT GATE (Qwen3-0.6B, ~15ms, or a fine-tuned classifier)
   │      → does this turn need memory at all?  "what time is it" → NO
   │
   ├─ 2. QUERY REWRITE (0.6B or 1.7B)
   │      conversational → search-ready. Resolve pronouns against the last 3 turns.
   │      "when does that thing expire" → "lease renewal due date Chennai flat"
   │
   ├─ 3. PARALLEL RETRIEVAL (all three at once — 100–600ms budget)
   │      ├─ FTS5 keyword      (exact terms, names, numbers — vectors are BAD at these)
   │      ├─ sqlite-vec dense  (semantic paraphrase)
   │      └─ graph walk        (subject → related entities, 1–2 hops)
   │      → k=20 candidates, deduped
   │
   ├─ 4. RERANK  (bge-reranker-base, cross-encoder, CPU)
   │      the step that turns retrieval from noise into signal
   │
   ├─ 5. HARD SCORE FLOOR  (score > 0.35, else DROP)
   │      better to return [] than noise
   │
   ├─ 6. FRESHNESS GATE
   │      drop chunks whose source file changed since embedding
   │      drop facts where retracted_at IS NOT NULL (unless the query is temporal)
   │
   └─ 7. SHIP NARROW → top-3, ~600–1,200 tokens, into the Attention Ledger's memory slot
```

**Why hybrid and not just vectors:** vectors are genuinely bad at exact tokens — names, phone
numbers, amounts, error strings. FTS5 nails those. Dense nails paraphrase. You need both, and
the cross-encoder reranker is what makes the union usable.

**Tool contract** (Law 4 — without this the model calls it every turn):
```python
@tool
def memory_search(query: str, as_of: str | None = None) -> list[Fact]:
    """Search FRIDAY's long-term memory about the user.

    Use ONLY when the turn references something not present in the current
    context: a past conversation, a personal fact, a preference, a prior
    decision, or an entity mentioned without introduction.

    Do NOT use for: general knowledge, the current task's files, or anything
    already visible in the loaded memory block.

    Pass `as_of="2025-03-01"` ONLY for explicit time-travel questions
    ("what was X last year"). Otherwise omit it — current facts are default.

    Returns at most 3 facts with provenance. An empty list is a valid and
    common answer; do not retry with rephrased queries more than once.
    """
```

---

## 6. What this gets you that nothing else does

1. **"Why do you think that?"** is always answerable — every fact has `source_refs` and
   `source_quote` pointing back to the literal trace.
2. **"You're wrong, I moved"** resolves cleanly, keeps history, and never regresses.
3. **Your data outlives the framework.** It's Markdown in git. If you abandon FRIDAY in 2028
   you can still read every fact it learned about you, and any future agent can ingest it.
4. **Cost goes down over time, not up.** Knowledge migrates from expensive-per-turn (context)
   to free-per-turn (weights). Most systems get *more* expensive as memory grows.
5. **It's a research artifact.** The gate log is a longitudinal dataset of an AI system
   improving itself on one person's life. That is genuinely publishable.

---

## 7. Build order (don't build S4 first)

```
Week 2   S0 trace writer + JSONL schema + redaction at write time
Week 2   Markdown facts parser + SQLite compiler + FTS5 + sqlite-vec
Week 3   Retrieval path (all 7 steps) ← you now have a working memory
Week 3   G2/G3 gates (grounding, contradiction) + user_edit detection via watchdog
Week 4   S1 episode compiler (local model, Dreaming phase)
Week 4   Decay + salience + the "What FRIDAY got wrong" reflection pass
Week 5   S3 skill distillation + progressive disclosure + activate_skill
Week 6   G5 shadow mode
─────────────────────────────────────────────  ← FRIDAY is now genuinely context-aware
Week 12+ S4 harvest → Kaggle QLoRA → G7 gate → promote
```

**S4 is a Phase 5 concern.** S0–S3 deliver 90% of the felt intelligence. Do not let the
exciting part (fine-tuning) make you skip the part that actually works (S0–S3).
