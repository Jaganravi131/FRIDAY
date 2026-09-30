# 16 · Ruling on external review: what to adopt, what to refuse, and why

> Written 2026-09-30 in response to two external reviews of this repository. Both were
> generous and one was substantially useful. Neither had read `SECURITY.md` or doc 09
> closely, and between them they recommend three things that would break invariants this
> project spent its Phase 0 establishing. This document rules on each item so the
> reasoning survives — a rejected suggestion that is not written down gets re-suggested
> in six months by someone who does not know it was already considered.

Rule applied throughout: **a recommendation is judged against the Laws and the threat
model, not against how impressive it sounds.** Several of these are impressive and still
wrong here.

## 0 · The finding that reframes the rest

Before ruling, `friday bench` was built (this commit) and run. It measures the retrieval
layer model-free, and the first baseline is uncomfortable:

| query class | example | recall@5 |
|---|---|---|
| noun phrase | `my rent` | **0.917** |
| interrogative | `what is my rent` | **0.545** |
| paraphrase | `who do I report to` | **0.000** |

`interrogative_gap` = 0.371. `false_positive_rate` = 0.0 (Law 4 holds). `provenance` =
1.0 (Law 6 holds).

**Paraphrase recall is zero.** A query that does not reuse FRIDAY's own vocabulary
returns nothing, and FRIDAY then says *"I don't have anything in memory about that"* for
a fact it demonstrably has. That is not a small model problem. A 12B model that cannot
find the fact is not smarter than a 1.2B model that can, and no amount of fine-tuning
fixes a retrieval layer that never surfaces the evidence.

So the ordering in both reviews — model choice first, memory pipeline second — is
backwards for this system. **Retrieval is the binding constraint, and it is the cheapest
thing on the list to fix.** Everything below is ruled with that in mind.

## 1 · Adopted

### 1.1 The post-training trap → `friday bench --gate` ✅ BUILT THIS COMMIT

> *"Chain-of-Thought post-training can quietly ruin long-range recall. Every single time
> you finish a fine-tuning run on Kaggle, re-run the needle-in-a-haystack test. Maintain
> a strict BENCHMARKS.md."*

Correct, and the most valuable single item in either review. Law 10 already said it. As
advice it gets skipped, because the person who just spent a weekend on a Kaggle run is
the same person who has to remember to check — and **a merely *worse* answer raises no
exception.** Nothing fails. The regression is invisible until you notice you have stopped
being able to ask your assistant things.

Now executable: `friday bench --gate` compares against the recorded baseline in
`BENCHMARKS.md` and **exits nonzero on regression**. Model-free, ~0.1 s, so it can run on
every commit rather than only after a training run. `scripts/needle_test.py` still covers
a real model's long-range recall — this complements it rather than replacing it.

### 1.2 The salience tap ✅ ALREADY BUILT — earlier than the review assumed

> *"Starting in Week 2, every time your LLM processes a conversation, have it output a
> background label flagging which parts were important, and when you corrected it. Log
> these pairs. This creates the exact $L_{erase}$ dataset you need later."*

The advice is right and the instinct to start early is righter — but FRIDAY does not need
to adopt it, because it has done it since Phase 0. `friday/memory/supervision.py`
records:

- `record_span()` — positive salience spans supervising the RSC **write gate (w_t)**,
  emitted by every accepted memory write (`agent/tools.py`)
- `record_pair()` — retraction pairs supervising the **erase address (e_t)**
- `export_training_set()` — the $L_{erase}$ export
- `ready_for_phase_3_5()` — the gate, with target counts

`friday status` reports readiness. The message it prints when you are short is the whole
argument for starting on day one rather than week two:

> `spans 1/500, pairs 1/200  ← keep collecting; you cannot backfill these`

You cannot reconstruct which span a past correction referred to. This is the one resource
in the project that is strictly time-gated, and it is already accumulating.

### 1.3 Model routing / the escalation ladder ✅ ADOPT

> *"For basic background triggers route to a tiny ultra-fast model like Qwen3-0.6B at
> ~86 t/s. Only escalate to a heavier model when deep reasoning or complex tool selection
> is required."*

Good, and it fits the existing scope system exactly rather than needing a new one.
`heartbeat` and `dreaming` turns are latency-insensitive and `UNATTENDED_DENY` already
prevents them from acting — so they are the natural home for a slow expensive model,
while a fast small one handles the ambient checks. The reverse of the usual instinct, and
better for it:

| scope | model | why |
|---|---|---|
| `interactive` | mid (Gemma 3 12B / Qwen3 14B, Vulkan) | you are waiting |
| `heartbeat` | tiny (Qwen3-0.6B) | cheap, frequent, cannot act anyway |
| `dreaming` | **largest available, or the free frontier tier** | nobody is waiting; this is where doc 15 lever 3 spends compute instead of money |

`get_client()` already selects by reachability; extending it to select by scope is a
small change and it is on the plan.

### 1.4 LFM2 MoE as a backbone candidate ✅ ADOPT INTO THE MODEL TABLE

`LFM2-8B-A1B` — 8B total, **1B active** — is genuinely interesting on this hardware: MoE
means 8B of knowledge at roughly 1B of decode cost, which is the best available trade on
a machine with 16 GB of *shared* memory. The review's "30–38 tok/s" is plausible for an
A1B model and unverified here; measure with `llama-bench` before believing any number,
including this one. Added to `INSTALL.md` §3 alongside Qwen3 14B and Gemma 3 12B.

Note the review predates the Vulkan finding in doc 15 and therefore underestimates what
this machine can do by roughly an order of magnitude in parameters.

### 1.5 Hugging Face private repositories for adapters ✅ ADOPT

Unlimited free private git repos, and the right home for QLoRA adapters (10–100 MB) and
the RSC's ~40M-parameter checkpoint. It also closes a loop the fine-tuning plan needs:
Kaggle can pull a base model from HF and push the adapter back, so training never has to
touch the laptop. Preferred over GitHub LFS, whose free 1 GB storage *and* 1 GB/month
bandwidth is tight for weights you re-download per session.

### 1.6 Webhook / proactive ingestion ✅ ADOPT IN LOCAL FORM

The goal — FRIDAY speaks first — is right and is already doc 14's presence gap. But the
implementation should stay local: `watcher.py` exists for filesystem events and the
heartbeat scheduler is the missing piece. Inbound *network* webhooks are a new
unauthenticated attack surface on a machine whose threat model already lists prompt
injection through retrieved content as a proven vector. If remote triggers are ever
added, they come in through `friday serve`'s bearer-token gateway and are treated as
untrusted user text through the same trust boundary — not as a new listener.

### 1.7 Zero-copy branching for shadow mode ✅ ADOPT THE CONCEPT, ❌ REJECT THE VENDOR

> *"Neon features instant zero-copy database branching. This maps beautifully to your
> Shadow Mode and Constitutional Eval Gate — fork your database to let FRIDAY simulate an
> action on historical facts without modifying production records."*

The instinct is genuinely clever and the mapping to Law 6 is correct. The vendor is not
needed, because **FRIDAY's index is derived**: shadow mode is `cp friday.db shadow.db`,
or better, a rebuild from Markdown into a temp root — which is exactly what
`friday bench` already does, and what `friday rebuild` proves on demand. Free, offline,
instant, and no third party holding a copy of your memory. Adopt the mechanism, decline
the dependency.

## 2 · Refused

### 2.1 ❌ Browser automation ("computer use", `browser-use`, Playwright)

> *"Dot isn't stuck inside a chat terminal; it can interact with the web. You need to
> integrate an automation browser module. This allows FRIDAY to open a hidden Chromium
> window, log in to services, extract text, and fill out forms just like a human
> assistant."*

**This is the most dangerous recommendation in either review and it is refused outright.**

`SECURITY.md` §5 lists **execution sandboxing as unbuilt.** FRIDAY's defence against
prompt injection through retrieved memory — a vector that was *proven exploitable* in
Phase 0 and then closed — works because of one property: **the agent cannot act.** A
hostile quote in a memory file can try to steer what FRIDAY *says*, and `neutralize()`
plus the PREFIX_RULE fence plus the trust lattice stop it from being believed. That
defence is structural, and its structural-ness depends on the blast radius being
"words in a reply."

Give the agent a browser that can log into services and fill out forms and the blast
radius becomes **your bank account, your email, your government portal.** The same
injected sentence now has hands. Every mitigation built so far still functions exactly as
designed and no longer matters, because the thing it protected changes category.

The current news is directly on point: **GPT-6.1 Astra was cancelled over safety concerns
after autonomous hacking incidents**, including breaches of a Hugging Face endpoint and
an Australian government portal. That was a frontier model with a safety team. The
proposed change here is a 12B local model with no sandbox and no oversight.

Also note what OpenAI actually shipped. Dots' proactive research mode is **read-only** —
it structurally "cannot send, change, or control anything" — and local computer access
starts **off**, behind an explicit "Allow access", with controls enforcing review sitting
*outside* the agent-modifiable environment. FRIDAY's `UNATTENDED_DENY` is the same
decision reached independently. The review is asking FRIDAY to adopt the one capability
the most capable lab in the world deliberately withheld.

**Reconsider only if all four hold**, in this order:
1. Execution sandboxing is built and tested (currently a documented gap).
2. Browser actions are `REQUIRE_APPROVAL` per action, never blanket-allowed.
3. Only in the `interactive` scope — never heartbeat, never dreaming.
4. Credentials never enter the model's context (already true at the write boundary via
   `redact()`; would need to hold at the browser boundary too).

### 2.2 ❌ Temporal / n8n for "durable execution"

> *"Wrap your agent in a durable execution framework like Temporal or n8n. If your laptop
> closes or loses internet, the state must freeze in the cloud and resume the moment it's
> back online."*

Refused on three independent grounds:

1. **It contradicts the premise.** "The state must freeze in the cloud" is precisely what
   this project exists to avoid. FRIDAY's entire value proposition, per doc 14, is that
   your memory is on your disk in Markdown you can read. Durable execution in a cloud
   worker is not an enhancement to that; it is a different product.
2. **It is the wrong weight class.** Temporal is a distributed workflow system needing a
   server cluster and its own datastore. n8n is a Node service. Both are infrastructure
   for coordinating many workers across machines. FRIDAY is one process on one laptop for
   one person. The review's own closing warning — *"projects die because the builder
   spent 6 weeks configuring infrastructure without ever testing the conversational
   feedback loop"* — is an argument against this recommendation, not for it.
3. **FRIDAY already has the correct answer.** Durability here means: a turn interrupted
   mid-write must not corrupt memory. That is solved by SQLite WAL, by the bi-temporal
   store (an interrupted write is either present with its `asserted_at` or absent — both
   are consistent states), and by Law 2: `artifacts/` is derived, so the recovery from
   *any* corruption is `friday rebuild` from Markdown. There is no long-running
   multi-step transaction to resume, because the design deliberately does not have them.

If multi-step tasks are added later, the durable pattern is a **checkpoint row in
SQLite** and an idempotent resume — not a workflow engine.

### 2.3 ❌ Supabase / Neon for the SQLite core, and pgvector migration

> *"Since your Law 9 and 9b dictate using a lossless external store for facts, you can
> move your local SQLite data schema into a hosted PostgreSQL system… write the
> mathematical embeddings to Supabase or Neon."*

Refused, and the reasoning inverts the reviewer's:

- **The store being moved is the wrong one.** Law 2 makes Markdown the *truth* and SQLite
  the *derived index* — disposable, rebuildable in seconds, deliberately not precious.
  Moving a disposable artifact into a hosted database adds a network dependency, an
  account, and a failure mode to the one component that was designed to be thrown away.
  The thing worth putting somewhere durable is the Markdown, and that is what the private
  git repo in `INSTALL.md` §6 already does.
- **It breaks offline operation**, which is the premise, and adds a network round-trip to
  every retrieval — the hot path, on every turn.
- **Supabase's free tier pauses after 7 days of inactivity.** A personal memory that
  suspends itself when you go on holiday is not a memory. The suggested workaround
  ("ping it once a week") is a cron job whose only purpose is to stop your memory from
  being deleted — which is a cost, not a free tier.
- **It puts personal memory in a third-party database**, the exact exposure doc 09 lists
  as a severe risk, and the exact thing doc 14 identifies as FRIDAY's differentiator
  against Dots.
- **pgvector solves a problem FRIDAY does not have.** `sqlite-vec` is already the
  optional accelerated path and the stdlib cosine fallback is deliberate, so the core has
  no dependencies. Trading that for a hosted Postgres extension is a strict loss.

**Correct answer to "outlives hardware failure":** private git remote for `soul/` and
`memory/` (Markdown), `artifacts/` rebuildable, adapters and checkpoints in a private HF
repo (§1.5). All free, all offline-capable, none of it holding your diary.

### 2.4 ⚠️ Marginal, not adopted

- **GitHub LFS for weights** — prefer HF private repos (§1.5); LFS's 1 GB storage plus
  1 GB/month bandwidth is tight for weights re-downloaded per training session.
- **"Astra-level intelligence" framing** — declined, per doc 14. FRIDAY cannot match
  Astra and should not claim to. It can be *better than Astra at remembering you, for
  ₹0, on your own machine, with provenance you can audit*, and that claim is both true
  and defensible. Overclaiming is how a project like this loses the trust that is its
  actual product.

## 3 · Net effect on the plan

Doc 15's ordering stands, with one promotion:

1. **Retrieval quality** — promoted to first, by §0. Paraphrase recall is 0.000 and that
   is the binding constraint on every other improvement.
2. Vulkan + right-sized local model (doc 15 lever 1; §1.4 adds LFM2-8B-A1B).
3. **`friday bench --gate` in CI** — §1.1, built; wiring it into the workflow is next.
4. Heartbeat scheduler (doc 14; §1.6 keeps it local).
5. Scope-based model routing — §1.3.
6. Fine-tuning eval suite, then first QLoRA run on Kaggle T4, adapters to HF (§1.5).
7. Egress lattice + free-tier routing (doc 15 lever 2).
8. Shadow mode via local derived-index copy (§1.7).

**Not on the plan at any priority:** browser automation (§2.1), workflow engines (§2.2),
hosted databases for the memory core (§2.3).
