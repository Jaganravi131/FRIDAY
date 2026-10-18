# 14 · Competitive analysis: GPT-6 Astra, OpenClaw, OpenAI Dots — and FRIDAY

> Written 2026-09-30. Sources: OpenAI's Dots launch at DevDay (2026-09-29), OpenAI's
> model index for `gpt-6-astra`, the OpenClaw documentation and repository, and Google
> DeepMind's Project Astra pages. Every claim about a competitor is checkable against
> those; every claim about FRIDAY is checkable against a file in this repository.

You asked whether FRIDAY can reach the level of GPT Astra, and whether it should become
an agent like OpenClaw or the Dots that OpenAI released this week — analysing them, and
fine-tuning the most accurate open model if that helps.

The analysis below does that. The verdict is at the end and it is not a comfortable one
on every axis, which is the point.

## 0 · The three things you named are three different kinds of thing

This matters, because comparing FRIDAY to "Astra" is a category error.

| | What it actually is |
|---|---|
| **GPT-6 Astra** | A **model**. OpenAI's current frontier LLM, launched 2026-09-03. 1,050,000-token context, 128K max output, $10/1M input tokens and $50/1M output (rising to $20/$75 above 272K). The reasoning engine *underneath* Dots. |
| **OpenAI Dots** | An **agent product**, launched at DevDay 2026-09-29. Always-on agents powered by Astra, each with its own cloud computer and browser, persistent memory, 4000+ app plugins, reachable from ChatGPT desktop/web/mobile, Slack, Teams and voice. Pro and Business Premium only; excluded from the EEA, UK and Switzerland on Pro. |
| **OpenClaw** | An **agent framework** — open source (MIT), ~310k stars, 1200+ contributors, now run by the OpenClaw Foundation as a 501(c)(3). A self-hosted *gateway* that connects chat apps (WhatsApp, Telegram, Discord, Slack, Signal, iMessage, Teams, Matrix, Zalo — 20+ platforms) to any model you point it at, including local ones via Ollama. Runs on macOS, Linux, Windows, Raspberry Pi. |
| **Google Project Astra** | *Not* one of the above. DeepMind's research prototype for a real-time camera-and-voice assistant. Not a released product, no launch date; its capabilities reach the public through Gemini Live. Frequently confused with GPT-6 Astra because of the name. |

So: **Astra is the engine, Dots is the car, OpenClaw is a chassis you build a car from.**
FRIDAY is competing with the car on one axis, the chassis on another, and cannot compete
with the engine at all.

## 1 · Capability, by axis

### Axis 1 — Raw model capability. FRIDAY loses, permanently, and should say so

Astra has a 1.05M-token context, frontier reasoning, and training compute measured in
billions of dollars. FRIDAY runs a 1.2B model quantized to 4 bits on a CPU.

This is not a gap that closes with effort. It closes with capital, and the project's
constraint is **₹0**. Any plan that pretends otherwise is a plan that fails in month
three.

What is worth noting is *why the gap is survivable*. FRIDAY's thesis (doc 05, doc 13) is
that on a personal assistant, **most of the perceived intelligence is memory quality and
context discipline, not reasoning horsepower**. Astra's 1.05M context does not make it
remember what you told it in March; it makes it able to hold a great deal at once.
FRIDAY's answer to "remember what you told it in March" is retrieval plus provenance,
which is a different mechanism aimed at the same felt experience — and on that specific
mechanism FRIDAY is not competing with a model, it is competing with a memory
implementation.

**Verdict: FRIDAY cannot match Astra. FRIDAY's design accepts this and routes around it.**
The honest framing is not "FRIDAY is as smart as Astra" — it is "FRIDAY remembers you
better than a frontier model that does not keep your data, and it costs nothing."

### Axis 2 — Agent architecture. FRIDAY is genuinely competitive, and ahead on trust

This is where the comparison gets interesting, because **OpenAI independently converged
on FRIDAY's Phase 0 architecture.** Not in spirit — in mechanism. Read the Dots safety
model next to `SECURITY.md`:

| OpenAI Dots | FRIDAY | Same idea? |
|---|---|---|
| Proactive research mode is **read-only** — the dot "cannot send, change, or control anything" | `UNATTENDED_DENY` — heartbeat/dreaming/eval turns cannot write, send, or control (`policy.py:178`) | **Yes, exactly** |
| **Custom Rules**: allow / require approval / block per action | `ALLOW / REQUIRE_APPROVAL / DENY` verdicts per tool per scope | **Yes** |
| Controls enforcing automatic review sit **outside** agent-modifiable environments | Law 5: enforcement lives in the data layer, not in prose; doc 09 names "rules in prose" as a **broken** fix | **Yes** |
| Passwords are **not exposed to the model**; credentials are kept away from its context | `redact()` at the write boundary — the secret becomes `[REDACTED:CREDENTIAL]` *before* it reaches any file or the database | **Yes, and FRIDAY's is at the write boundary rather than the read boundary** |
| Local computer access starts **off**; explicit "Allow access" | Tiered senses, each approved individually; `senses.allow/ask/deny` (doc 04 §7) | **Yes** |
| Sandboxing per workspace | Trust lattice + `neutralize()`; execution sandboxing **not built** | **Partly — FRIDAY is behind here** |
| Persistent memory from three sources: conversation, ChatGPT memory, private dot notes | `soul/MEMORY.md` (core block), `memory/facts/*.md` (ledger), `memory/traces/*.md` (raw) | **Yes — three sources, same split** |

That table is the single most useful thing in this document. It means FRIDAY's Phase 0
design was not a guess: **the world's best-resourced AI lab shipped the same trust
architecture in the same month**, having had every advantage in finding out what works.
FRIDAY arrived there from first principles and a threat model; they arrived there from
scale. The conclusions agree.

Where FRIDAY is **ahead** of both Dots and OpenClaw:

1. **Bi-temporal provenance.** FRIDAY knows when a fact was true *and* when it learned
   it, can answer "what did you believe on March 3rd", and treats retraction as a
   new fact rather than a deletion. Neither Dots nor OpenClaw offers this. For a personal
   assistant that runs for years, this is the difference between a memory you can trust
   and a memory you cannot argue with.
2. **`/why` is a first-class feature.** FRIDAY can show you the literal quote, source
   file and line for anything it believes. Dots has memory; FRIDAY has *auditable*
   memory. OpenClaw stores memory in a database; FRIDAY stores it in Markdown you can
   open in Notepad.
3. **Markdown as truth, indices as derived.** You can delete every database FRIDAY has
   and rebuild from plain text in seconds. There is no export step, no format lock-in,
   and no vendor. Dots' memory lives in OpenAI's account system; if that account closes,
   the memory is gone.
4. **The Attention Ledger.** FRIDAY tracks token spend per slot, enforces a hard 60%
   reserve, and keeps the prompt prefix byte-stable so caching actually works. Neither
   competitor exposes anything equivalent, because neither needs to — they can buy
   context. FRIDAY has to earn it.

Where FRIDAY is **behind**, badly:

| Gap | Dots | OpenClaw | FRIDAY |
|---|---|---|---|
| Channels | ChatGPT desktop/web/mobile, Slack, Teams, voice | **20+ chat platforms** | `friday serve` — one HTTP endpoint, no adapters |
| Skills / tools | 4000+ app plugins | **ClawHub: 700+ community skills** | 8 built-in tools, no ecosystem |
| Onboarding | A polished product flow | `openclaw onboard` | Read `INSTALL.md` and run commands |
| Personality | Custom Rules + dot notes | `~/.openclaw/soul.md` | `soul/SOUL.md` — **same idea, FRIDAY had it too** |
| Ambient agency | Always-on proactive research | **Heartbeat** (`every: 30m`, active hours) | `HEARTBEAT.md` exists; **no scheduler wired** |
| Web UI | Full product | Local web UI at `:18789` | One HTML page served by `friday serve` |
| Multi-agent | Multiple dots, delegating to each other | Sub-agents | None |
| Cloud computer | Each dot gets one | None (self-hosted) | None (by design — it is *your* computer) |

**Verdict: FRIDAY's trust and memory architecture is at or above the level of both
systems. Its surface area is roughly 5% of OpenClaw's.** That is not a defect to be
ashamed of — a 10k-line stdlib codebase cannot have 700 community skills — but it is
the difference between "an architecture that works" and "a thing people use".

### Axis 3 — Presence. This is the axis that decides whether it is real

Both reference systems treat *being reachable* as the core feature, not an accessory.
OpenClaw is literally a gateway: its value proposition is "one process, every chat
channel". Dots' pitch is "always-on agents you can reach from anywhere".

FRIDAY was laptop-only until `friday serve` landed. That is now fixed in the smallest
useful form — an authenticated HTTP gateway with a no-build web UI, an OpenAI-compatible
endpoint, loopback-by-default binding, and remote turns audited as `remote`. Exit test
#12 (phone access over Tailscale) is testable.

But note what is *still* missing: **channel adapters.** Your phone can reach FRIDAY
through a browser tab you have to open. It cannot reach it through the messaging app you
already live in, which is what makes OpenClaw feel ambient and FRIDAY feel like software.

## 2 · The fine-tuning question

You asked whether FRIDAY should take the most accurate open model and fine-tune it. The
answer is **yes, but not for the reason you might expect, and not on the model's general
intelligence.**

### What fine-tuning cannot buy

Fine-tuning a 1.2B model will not make it reason like Astra. Distillation transfers
*behaviour*, not *capability* — you can teach a small model to imitate the surface of
good reasoning, and on hard multi-step problems it will still fail. Anyone claiming a
LoRA run closes that gap is selling something.

### What fine-tuning CAN buy — and it is exactly what FRIDAY needs

FRIDAY's failure modes are not "not smart enough". They are **format and discipline**
failures, and those are precisely what supervised fine-tuning fixes best:

1. **Tool-call discipline.** Emitting well-formed tool calls with correct arguments,
   *only* when the ledger block and retrieved context warrant it, and never inventing a
   tool. This is the single highest-value target: every malformed call is a wasted turn.
2. **Provenance-aware answering.** Answering *from the retrieved block* and citing it,
   instead of answering from prior and hoping. FRIDAY's whole trust model assumes the
   model respects the fence; a model fine-tuned to respect it makes the fence load-bearing
   in behaviour and not just in prompt text.
3. **"I don't know" calibration.** The most important behaviour for an assistant that
   remembers your life: refusing to confabulate. Base models are trained to be helpful,
   which means they answer. FRIDAY needs a model that says *"I have nothing in memory
   about that"* when the retrieval block is empty. This is trainable and it is worth more
   than any capability gain.
4. **Ledger-block adherence.** Respecting slot semantics (`<recalled>` is data,
   `<identity>` is you) — the PREFIX_RULE behaviour that had to be debugged by hand.
5. **Code-switching.** You are in Tamil Nadu; a personal assistant should handle
   Tamil/English mixed speech naturally. Base models are weak here at small sizes, and
   this is a data problem, not a capability problem — the ideal LoRA target.

### The recipe that fits ₹0

You have no CUDA. Local fine-tuning is off the table — but it is not needed.

| Resource | Cost | What it gives |
|---|---|---|
| **Kaggle Notebooks** | ₹0 | 2× T4 (or P100), **30 GPU-hours/week**, 9h sessions |
| **Google Colab free** | ₹0 | 1× T4, shorter sessions, less reliable |
| Local laptop | ₹0 | Inference and evaluation only — never training |

A LoRA run on a 1.2B model needs roughly 1–4 GB VRAM at 4-bit with QLoRA. **A single
free T4 (16 GB) is comfortable.** 30 GPU-hours a week is enough for several serious
runs, and the adapters are ~10–100 MB, so shipping them to your laptop is trivial.

**Base model candidates**, in the order this project should try them:

| Model | Size | Why |
|---|---|---|
| **LFM2.5-1.2B** | 1.2B | Built for edge/CPU, first-class llama.cpp support, hybrid architecture — and it is the architecture family doc 13 already recommends. **Start here.** |
| **Qwen3-1.7B / 4B** | 1.7–4B | Strongest small-model reasoning-per-byte; 4B is the ceiling for 16 GB with anything else running |
| **Gemma 3 1B / 4B** | 1–4B | Excellent instruction-following at small sizes, permissive licence |
| **Llama 3.2 1B / 3B** | 1–3B | The safest well-documented path; huge community LoRA tooling |
| **Phi-4-mini** | 3.8B | Best reasoning at this size, but heavier |

**Training data** is the part to think hardest about, and FRIDAY has an unusual
advantage: *the system generates its own supervision signal.* Every turn already records
the ledger block, the retrieved context, the tools called, the policy verdict, and the
audit row. That is a labelled dataset of "given this context, the correct behaviour was
X" — accumulating for free, on your own machine, about your own life.

Concretely: run FRIDAY normally, then mine the audit log for turns where behaviour was
correct (accepted writes, allowed reads, honest "I don't know") and turns where it was
not (denials, malformed calls, confabulations you corrected). That is a curriculum with
no labelling cost. **This is doc 10's self-improvement loop applied to the model instead
of to the memory** — and it is the only fine-tuning plan consistent with ₹0, because the
data comes from use rather than from purchase.

Start with **synthetic + hand-written** examples (a few hundred) for the five behaviours
above, because you need *something* to fine-tune before FRIDAY has months of history.
Then let the audit-mined data take over.

**Evaluation must be built before training**, or you will not know whether a run helped:
`scripts/needle_test.py` already exists for recall; add a discipline suite measuring
tool-call validity rate, refusal-when-empty rate, fence-adherence rate, and
code-switching quality. **The refusal rate is the headline number** — a model that
confabulates less is worth more to this project than one that scores higher on a
benchmark.

### The architectural fine-tuning question

Doc 13 already ruled on the deeper version of this: if you are going to invest in
architecture rather than weights, invest in a **hybrid linear-attention backbone**
(GDN-2-style gated delta with an independently-addressed erase gate, interleaved with
sliding-window attention at roughly 3:1) — because that is what production systems
converged on, and because its constant-size recurrent state is exactly what a
prompt-cache-constrained, 16 GB, always-on agent needs.

That is a **larger** project than a LoRA run and should not be attempted first. The
order is: behaviour LoRA on an existing small model (weeks, free) → then, only if the
behaviour work proves out, architecture work (months, needs real compute).

## 3 · What FRIDAY should copy, and what it must refuse

**Copy from OpenClaw:**

- **Channel adapters.** This is the highest-leverage gap on the presence axis. One
  Telegram adapter makes FRIDAY feel ambient in a way no amount of architecture does.
  Design it as `friday/gateway/channels/` behind the existing `serve` auth, and treat
  every inbound message as *untrusted user text through the same trust boundary* — the
  boundary must not have an HTTP-only assumption baked in.
- **A skill/plugin format with a trust story.** OpenClaw's ClawHub has 700+ skills and
  scans them with VirusTotal. FRIDAY's MCP path is documented as unbuilt and as a
  supply-chain risk. If FRIDAY ever has plugins, they need the provenance discipline the
  rest of the system has — a signed skill with a declared capability set, denied by
  default.
- **The heartbeat scheduler.** `HEARTBEAT.md` exists but nothing runs it. OpenClaw's
  `every: 30m` with active hours is the right shape, and FRIDAY's `UNATTENDED_DENY`
  already makes it safe — this is nearly free to wire and it is the difference between
  an assistant and a REPL.

**Copy from Dots:**

- **Read-only proactive research as a named mode.** FRIDAY has the mechanism
  (`UNATTENDED_DENY`) but not the *product concept* — "the agent may go and find things
  out for you, and structurally cannot act on them". That is a feature worth naming and
  surfacing.
- **Per-workspace isolation.** FRIDAY has one root. Multiple roots with separate trust
  lattices would be the multi-agent path, if it is ever needed.

**Refuse to copy:**

- **Cloud computers.** FRIDAY's entire premise is that your memory is on your disk, in
  your Markdown, in your git repo. A dot's cloud computer is a convenience that costs
  you custody of your own life. This is not a limitation to apologise for; it is the
  product.
- **Plugin counts as a goal.** 4000 plugins is a distribution advantage, not a quality
  one, and each is an attack surface. FRIDAY should have eight excellent tools and a
  trust model that makes the ninth safe to add.
- **Always-on by default.** Dots requires 18+, excludes three regions, and had GPT-6.1
  Astra *cancelled over safety concerns after autonomous hacking incidents*. FRIDAY's
  tiered-sense model — every new capability explicitly approved — is the correct response
  to that news, not a slower version of it.

## 4 · The verdict

**Can FRIDAY reach the level of GPT Astra?**
No, and it never will. Astra is a frontier model with a 1.05M context and billions in
compute behind it. FRIDAY is a 1.2B model on a CPU. Anyone who tells you otherwise is
selling you something. **The right response is to stop competing on that axis** and
compete where the constraint is an advantage.

**Can it be an agent like OpenClaw or the Dots?**
On architecture, **it already is** — and the Dots launch is the strongest possible
external validation of FRIDAY's Phase 0, because OpenAI converged on the same
read-only-when-unattended, rules-not-prose, three-source-memory, credentials-never-in-
context design in the same month, with vastly more resources. On *trust and memory*,
FRIDAY is **ahead** of both: bi-temporal provenance, `/why`, retraction-as-fact,
Markdown-as-truth, and a token ledger none of them exposes. On *surface area*, FRIDAY is
**far behind**: one HTTP gateway against 20+ channels, eight tools against 700 skills, a
`HEARTBEAT.md` nobody runs against a real scheduler.

**Should you fine-tune?**
Yes — but for **behaviour, not capability**: tool-call discipline, "I don't know"
calibration, fence adherence, provenance-aware answering, Tamil/English code-switching.
Free Kaggle T4 hours are enough for QLoRA on a 1.2B model, the adapter ships to your
laptop in megabytes, and FRIDAY's own audit log is a self-generating training set. Build
the evaluation suite first; make **refusal rate** the headline metric.

**The position to claim, stated plainly:**

> *Dots cannot show you why it believes something. OpenClaw's memory is a database.
> FRIDAY's memory is Markdown you can read, edit and audit, every claim traceable to your
> own words, on your own machine, for ₹0.*

That is not a consolation prize for losing the model race. It is a different product,
and it is the one that survives the failure modes the others have already demonstrated —
account closure, region exclusion, a model line cancelled for safety reasons, and a
memory you are not allowed to inspect.

## 5 · Ordered next steps

Ranked by (value ÷ effort), all consistent with ₹0:

1. **Wire the heartbeat scheduler.** `HEARTBEAT.md` already exists and `UNATTENDED_DENY`
   already makes it safe. Smallest change with the largest shift in felt behaviour.
2. **Build the fine-tuning evaluation suite.** Refusal rate, tool-call validity, fence
   adherence, recall. Must exist *before* any training run or the runs are unmeasurable.
3. **First QLoRA run on LFM2.5-1.2B**, targeting behaviours 1–3 from §2, on a few
   hundred hand-written examples. Free T4, one weekend.
4. **One Telegram channel adapter** behind the existing gateway auth, with inbound
   messages treated as untrusted through the same trust boundary. This is the presence
   gap, and one adapter proves the abstraction.
5. **Mine the audit log into training data.** The self-improvement loop, applied to the
   model. Depends on 2 and 3 existing.
6. **Then, and only then, the architecture work** from doc 13 — GDN-2 + SWA hybrid,
   custom erase gate. Months, not weeks, and it needs the behaviour work to have proven
   the evaluation harness first.
