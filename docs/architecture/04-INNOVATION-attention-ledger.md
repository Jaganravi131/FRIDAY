# 04 — INNOVATION #4: The Attention Ledger

> The reason FRIDAY's answers will be **clear** instead of vaguely plausible.
>
> Context engineering is not "write a good prompt." It is **deciding which tokens the model sees
> on every single inference call** — and treating that as a *budgeted, scheduled, measurable,
> cached* resource.

---

## 1. The problem, stated precisely

A naive agent does this every turn:
```
system_prompt + ALL_tools + top_20_memories + full_history + user_message  →  LLM
```

Failure modes, all observed in production:
| Symptom | Cause |
|---|---|
| Degrades after ~20 turns | History grows linearly; cost grows **quadratically** (every step re-reads everything before it) |
| Contradicts what you just said | Stale memories from 3 sessions ago outrank the current turn |
| Dithers between similar tools | 30 tool definitions in context, no clear contract on when to use which |
| "Panic compaction" at the worst moment | Window ran to 99%; the next big tool result forced a rewrite mid-task |
| Answers get *worse* as you add RAG | Raw chunks dumped unfiltered; accuracy loss begins ~**50K tokens of even genuinely relevant** info |
| Bill explodes | Prompt cache misses because the prefix changes every turn |
| Burned through the window on small talk | No budget allocation; every slot competes freely |

**The Attention Ledger is FRIDAY's answer: context assembly becomes a deterministic, budgeted
compilation step — not string concatenation.**

---

## 2. The model: a ledger, literally

Every token that enters the context window is a **debit** against a **slot** with a **budget**.
The Ledger is the scheduler. It runs *before* every LLM call, in **<5 ms**, with **no model
invocation**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ATTENTION LEDGER — turn #42, budget 8,192 tokens (window 32,768)        │
├──────────────────┬─────────┬─────────┬──────────┬────────────────────────┤
│ SLOT             │ BUDGET  │ SPENT   │ CACHE    │ SOURCE                 │
├──────────────────┼─────────┼─────────┼──────────┼────────────────────────┤
│ identity         │   1,024 │     912 │ ♻ stable │ SOUL.md + AGENTS.md    │
│ user             │     512 │     248 │ ♻ stable │ USER.md                │
│ core_memory      │     640 │     401 │ ♻ stable │ MEMORY.md              │
│ senses           │     256 │     180 │ ♻ stable │ Sense Registry         │
│ skill_index      │     512 │     366 │ ♻ stable │ skills-index.md        │
├──────────────────┼─────────┼─────────┼──────────┼────────────────────────┤
│  ── CACHE BREAKPOINT ── (everything above is byte-identical turn to turn)│
├──────────────────┼─────────┼─────────┼──────────┼────────────────────────┤
│ recalled_memory  │   1,200 │     842 │ ✗ JIT    │ top-3 reranked facts   │
│ retrieved_docs   │   3,000 │   2,140 │ ✗ JIT    │ 3 chunks, score>0.35   │
│ transcript       │   2,560 │   2,551 │ ~ append │ last 9 turns, capped   │
│ scratchpad       │     768 │     402 │ ✗ JIT    │ pointers, not payloads │
├──────────────────┼─────────┼─────────┼──────────┼────────────────────────┤
│ RESERVED (never spent — the model needs room to think)                   │
│ reserve          │   2,000 │       0 │          │                        │
├──────────────────┼─────────┼─────────┼──────────┼────────────────────────┤
│ turn_total       │  12,472 │   8,042 │ cache 62%│ → L1 qwen3:4b          │
└──────────────────┴─────────┴─────────┴──────────┴────────────────────────┘
   ledger_hash: a91f3c…   (recorded in the trace → correlate quality w/ starvation)
```

Every turn writes this record. That's what makes context a **measurable engineering quantity**
rather than a vibe.

---

## 3. The five laws of the Ledger

### Law A — Split the window into a CACHED prefix and a JIT zone

This is the single highest-value rule and almost nobody does it.

```
┌─────────────────────────────────────────┐
│  STABLE PREFIX  (byte-identical)        │  ← prompt cache HITS here
│  identity · rules · user · core_memory  │     ~1,750 tokens
│  senses · skill_index                   │     changes ≤ 1×/day
├════════ CACHE BREAKPOINT ═══════════════┤  ← explicit breakpoint marker
│  JIT ZONE  (varies every turn)          │  ← never cached, and that's fine
│  recalled_memory · retrieved_docs       │     budgeted, reranked, floored
│  transcript (append-only) · scratchpad  │
├─────────────────────────────────────────┤
│  RESERVED HEADROOM  (never spent)       │  ← ≥2,000 tokens
└─────────────────────────────────────────┘
```

**Why it matters financially and qualitatively:**
- A **stable, deterministic cap** makes the prefix shorter but leaves it **byte-identical** from
  turn to turn → cache hits. A **summary** makes the prefix shorter by making it *different* →
  cache misses on every turn.
- Measured: summarising history dropped in-session recall from **92% → ~33%** *and* cost ~2× more
  than keeping everything.
- Measured: **capping each tool output at a fixed size** cut cost/turn by **38%**, won **14 of 15**
  trajectories, and left memory-probe accuracy **identical**.

> **The rule: shrink the prefix, don't rewrite it.**

**Practical consequence:** `SOUL.md`, `AGENTS.md`, `USER.md`, `MEMORY.md` may only be edited by
the **Dreaming phase** (nightly), never mid-session. If FRIDAY learns something at 2pm, it goes
into JIT-recalled memory today and gets promoted into `MEMORY.md` at 3am. That one constraint is
what keeps your cache hit rate above 85%.

---

### Law B — Cap, don't summarise (the compaction ladder)

When a slot overflows, walk this ladder **top to bottom** and stop at the first rung that fits.
Each rung is cheaper and less lossy than the one below.

```
RUNG 0  Do nothing.                    (default — most turns fit)
RUNG 1  CAP tool output at N tokens    ← free, no model call, −38% cost, recall unchanged
RUNG 2  TRUNCATE with a marker         ← "[…4,213 tokens elided: <path>#L88-L402…]"
RUNG 3  STRUCTURAL DEDUP               ← collapse repeated content, canonicalise paths
RUNG 4  OFFLOAD to file + POINTER      ← fully reversible, nothing lost
RUNG 5  EVICT oldest transcript turns  ← archive to disk, keep the pointer
RUNG 6  SUMMARISE (last resort)        ← only when you can NAME the constraint
```

**Rung 4 is the best one and it's underrated.** Write the chunk to a file, cross-link with
pointers, keep **one index file** that maps them all, and let the agent read only the index.
It searches its way back to details per task — so a complex question pulls more and a simple one
pulls less.

> **Context scales with task complexity instead of with session length.**

**Rung 6 requires a named constraint.** Only summarise when one of these is true and you've
measured it:
1. The context genuinely does not fit the window (local-model territory → *retrieve*, don't stuff)
2. Cached-input cost makes resending expensive (check the ~$0.55/M threshold against your model)
3. Quality **measurably** degrades — and you've measured *where*

**When you do summarise:** trigger at **~70–75% of the window, not 95–98%.** Compacting at 98%
leaves the model "context-anxious" with no output tokens to write a good summary. And tune the
compaction prompt by **maximising recall first** (capture everything), *then* improving precision
(remove superfluous content) — in that order, on complex real traces.

---

### Law C — Just-in-time, with a contract on every retrieval tool

**Pre-packing** asks: *"what might be relevant in some possible future turn?"* — unanswerable.
**JIT** asks: *"what is the minimum the model needs for this sub-step?"* — answerable.

The agent holds **lightweight identifiers** (file paths, stored queries, entity IDs, URLs) and
loads data at runtime. Each interaction yields context that informs the next decision: file sizes
hint at complexity, naming conventions hint at purpose, timestamps proxy for relevance. It builds
understanding **layer by layer**.

**Hybrid is correct** (Claude Code does exactly this): a small stable block up front
(`CLAUDE.md`-style), plus `glob`/`grep`-style primitives for JIT navigation. That bypasses stale
indexing and complex syntax trees entirely.

**The contract is in the docstring.** This is the detail that decides whether you have JIT or
pre-packing-with-extra-steps:

```python
@tool
def read_fact(subject: str, predicate: str) -> str:
    """Read one specific fact about the user.

    Use ONLY when a turn references a concrete attribute you do not already
    have (a name, a number, a date, a preference) AND it is not in the
    loaded core_memory block.

    Do NOT use to browse. Do NOT call this more than twice per turn.
    For open-ended recall use memory_search instead.
    """
```

Without *"Use ONLY when…"* and *"Do NOT…"*, the model calls the retriever on every turn
"just to be sure" — and your JIT system quietly becomes pre-packing.

---

### Law D — The four-part retrieval pipeline (skipping any part = rot)

```
1. INTENT GATE        does this turn need retrieval at all?
                      (Qwen3-0.6B, ~15 ms, or a fine-tuned classifier)
                      "what time is it" → NO. Saves the whole pipeline.
        ↓
2. QUERY REWRITE      conversational → search-ready; resolve pronouns
                      against the last 3 turns
                      "when does that expire" → "lease renewal due date Chennai flat"
        ↓
3. RETRIEVE WIDE      parallel: FTS5 keyword ‖ sqlite-vec dense ‖ graph walk
                      k=20 candidates, dedupe
        ↓
4. RERANK             bge-reranker-base cross-encoder (CPU)
                      THIS is what turns retrieval into signal
        ↓
5. HARD FLOOR         score > 0.35 else DROP → return [] rather than noise
        ↓
6. FRESHNESS GATE     drop chunks whose source file changed since embedding
                      drop retracted facts (unless the query is explicitly temporal)
        ↓
7. SHIP NARROW        top-3 → recalled_memory slot (~600–1,200 tokens)
```

**Latency budget:** vector-only 10–50 ms · graph traversal 50–150 ms · multi-strategy parallel
100–600 ms · **LLM synthesis 800–3,000 ms ← never on the turn path.**

**For code specifically, don't use embeddings.** Structural retrieval wins decisively: returning
a symbol's actual definition plus its call sites lifted **precision@5 from 0.14 → 0.48**
(Sourcegraph). Use tree-sitter/LSP/ctags for code, embeddings for prose.

---

### Law E — Isolate sub-tasks; report summaries upward

One agent, one context — **until** a sub-task would pollute it. Then spawn a sub-context:

```
main context (clean)
   │
   ├── spawn: "research 32GB SODIMM prices in India"  → own 8K window
   │            burns 6,000 tokens on 14 web pages
   │            returns:  ──────────────►  "Corsair 3200MHz ₹5,200 / Crucial ₹4,800 /
   │                                         Kingston ₹5,600. Amazon+Flipkart in stock.
   │                                         VivoBook 15 (X512) has 1 free SODIMM slot,
   │                                         max 32GB total."   ← 62 tokens
   │
   └── main context is now 62 tokens dirtier, not 6,000
```

The four operations, mapped:
| Operation | Mechanism in FRIDAY |
|---|---|
| **Write** | S0–S3 Memory Compiler, scratchpad files, `HEARTBEAT.md` open loops |
| **Select** | Law C/D — JIT retrieval, hybrid, reranked, floored |
| **Compress** | Law B — the rung ladder, cap-first |
| **Isolate** | Law E — sub-contexts returning distilled summaries |

**When is a sub-agent justified?** Only for genuinely parallel or genuinely polluting work.
Not for "multi-agent architecture" aesthetics. A personal assistant with shared memory is faster
and cheaper as **one agent with isolated sub-contexts**.

---

## 4. Tool surface management (progressive disclosure)

30 tool definitions ≈ 3,000–5,000 tokens of *fixed* cost on every turn. On a 4B local model with
an 8K working budget, that is catastrophic.

**Three levels of disclosure:**

```
LEVEL 0 — CORE (always visible, ≤6 tools, ~400 tokens)
  memory_search · memory_write · activate_skill · note · ask_user · done

LEVEL 1 — SKILL-GATED (only name + when_to_use visible, ~30 tokens each)
  skills-index.md:
    lease-chase      — chase broker for pending docs, unanswered >24h
    git-workflow     — branch/commit/PR in the FRIDAY repo or user repos
    web-research     — multi-source research with citations
    calendar-triage  — read/schedule/decline meetings
    code-explain     — structural code retrieval + explanation

  → model calls activate_skill("web-research") → that skill's 4-8 tools
    are injected for the remainder of the turn only

LEVEL 2 — MCP SERVERS (never listed; discovered on demand via server/discover)
  screenpipe · filesystem · github · gmail · homeassistant · …
```

This is `roiguri/jarvis-agent`'s **progressive tool disclosure** + Hermes's **Skill Bundles**
(a bundle groups several skills under one slash command, so a "write code" bundle loads code
review + testing + PR workflow together).

**Diagnostic:** if the model **dithers between similar tools**, the fix is not a better prompt —
it's **cut the tool set and load groups dynamically.**

---

## 5. Reference implementation

```python
# core/ledger/ledger.py
from dataclasses import dataclass, field
from typing import Literal

SlotName = Literal[
    "identity", "user", "core_memory", "senses", "skill_index",   # stable prefix
    "recalled_memory", "retrieved_docs", "transcript", "scratchpad",  # JIT zone
]

@dataclass(frozen=True)
class Slot:
    name: SlotName
    budget: int
    cached: bool            # True → part of the stable prefix
    required: bool          # True → turn is invalid without it
    overflow: Literal["cap", "truncate", "offload", "evict", "summarise", "reject"]

# ── THE BUDGET. Tune this, don't hardcode it in string assembly. ──────────────
LEDGER_SPEC: tuple[Slot, ...] = (
    Slot("identity",        1024, cached=True,  required=True,  overflow="reject"),
    Slot("user",             512, cached=True,  required=True,  overflow="reject"),
    Slot("core_memory",      640, cached=True,  required=True,  overflow="reject"),
    Slot("senses",           256, cached=True,  required=False, overflow="truncate"),
    Slot("skill_index",      512, cached=True,  required=False, overflow="cap"),
    # ── cache breakpoint ──
    Slot("recalled_memory", 1200, cached=False, required=False, overflow="cap"),
    Slot("retrieved_docs",  3000, cached=False, required=False, overflow="offload"),
    Slot("transcript",      2560, cached=False, required=True,  overflow="evict"),
    Slot("scratchpad",       768, cached=False, required=False, overflow="offload"),
)
RESERVE = 2000          # never spent. the model needs room to think.
COMPACTION_THRESHOLD = 0.75   # of working budget — NOT 0.98


@dataclass
class LedgerEntry:
    text: str
    tokens: int
    source: str           # "SOUL.md", "fact:f_041", "file:~/x.py#L12-L40"
    score: float | None = None
    elided: int = 0       # tokens removed by cap/truncate
    pointer: str | None = None   # set when offloaded to a file


@dataclass
class CompiledContext:
    blocks: list[LedgerEntry]
    prefix_hash: str           # stable-prefix hash → cache-hit predictor
    ledger_hash: str           # full hash → recorded in the S0 trace
    spent: dict[SlotName, int]
    budget: dict[SlotName, int]
    overflow_events: list[tuple[SlotName, str]]   # which rung fired
    est_cache_hit: bool


class AttentionLedger:
    """Deterministic context compiler. No model calls. Runs in <5ms."""

    def __init__(self, spec=LEDGER_SPEC, window: int = 32_768, reserve: int = RESERVE):
        self.spec, self.window, self.reserve = spec, window, reserve
        self.working = window - reserve

    def compile(self, candidates: dict[SlotName, list[LedgerEntry]]) -> CompiledContext:
        blocks: list[LedgerEntry] = []
        spent, budget, overflows = {}, {}, []

        for slot in self.spec:                      # order matters: cached slots first
            budget[slot.name] = slot.budget
            entries = candidates.get(slot.name, [])
            room = slot.budget
            out: list[LedgerEntry] = []

            for e in entries:                       # entries arrive pre-sorted by score
                if e.tokens <= room:
                    out.append(e); room -= e.tokens; continue

                # ── walk the compaction ladder for THIS slot ──
                e2 = self._apply_rung(slot, e, room)
                if e2 is None:
                    overflows.append((slot.name, "dropped")); continue
                if e2 is not _REJECT:
                    out.append(e2); room -= e2.tokens
                    if e2.elided: overflows.append((slot.name, slot.overflow))

            if slot.required and not out:
                raise ContextStarvation(slot.name)  # fail LOUD, never silently degrade

            blocks.extend(out)
            spent[slot.name] = slot.budget - room

        self._guard_total(spent, overflows)
        return self._assemble(blocks, spent, budget, overflows)

    def _apply_rung(self, slot, e, room):
        """Cap → truncate → offload → evict → summarise. Never skip rungs."""
        if slot.overflow in ("cap", "truncate"):
            head = _take_tokens(e.text, room)
            return replace(e, text=head + _elision_marker(e, room),
                           tokens=room, elided=e.tokens - room)
        if slot.overflow == "offload":
            path = spill_to_file(e.text, e.source)      # artifacts/scratch/…
            ptr = f"[offloaded {e.tokens} tok → {path}]"
            return replace(e, text=ptr, tokens=_count(ptr),
                           elided=e.tokens, pointer=path)
        if slot.overflow == "evict":
            return _REJECT                                # caller archives to disk
        return None

    def _guard_total(self, spent, overflows):
        total = sum(spent.values())
        if total > self.working * COMPACTION_THRESHOLD:
            overflows.append(("_total", f"{total}/{self.working} — compact NOW, not later"))

    def _assemble(self, blocks, spent, budget, overflows):
        prefix = [b for b, s in zip(blocks, self.spec) if s.cached]
        prefix_hash = _hash_stable(prefix)     # byte-identical → cache hit
        return CompiledContext(
            blocks=blocks,
            prefix_hash=prefix_hash,
            ledger_hash=_hash_all(blocks),
            spent=spent, budget=budget,
            overflow_events=overflows,
            est_cache_hit=(prefix_hash == self._last_prefix_hash),
        )
```

**`ContextStarvation` must be a loud failure.** An agent that silently drops its identity block
produces garbage that *looks* fine. Fail, log it, and let the ladder escalate.

---

## 6. Telemetry (if you don't measure it, you're guessing)

Every turn emits:

```jsonl
{"turn":42,"ledger_hash":"a91f3c","prefix_hash":"77b2e1",
 "cache_hit":true,"prefix_stable_turns":41,
 "spent":{"identity":912,"user":248,"core_memory":401,"senses":180,"skill_index":366,
          "recalled_memory":842,"retrieved_docs":2140,"transcript":2551,"scratchpad":402},
 "utilisation":{"identity":0.89,"recalled_memory":0.70,"retrieved_docs":0.71,"transcript":1.00},
 "overflow_events":[["transcript","evict"]],
 "retrieval":{"intent_gate":"needed","candidates":20,"after_rerank":7,"shipped":3,
              "top_score":0.71,"floor_dropped":4,"latency_ms":184},
 "tokens_in":8042,"tokens_out":187,"ttft_ms":312,"total_ms":2410,
 "tier":"L1","model":"qwen3:4b","cost_usd":0.0}
```

**The four dashboards that matter:**

| Chart | What a bad trend means |
|---|---|
| **Cache hit rate** over turns | <85% → your prefix is mutating. Find what's changing it (usually a timestamp or a "current time" line sneaking into `SOUL.md`). **Never put the current time in the cached prefix.** |
| **Tokens/turn** over a 200-turn session | Growing → a slot is leaking. Usually `transcript` or `retrieved_docs`. |
| **Slot utilisation** heatmap | A slot pinned at 1.0 constantly → under-budgeted or over-retrieving. A slot at 0.1 → delete it, it's dead weight. |
| **Retrieval precision@3** (from your eval suite) | <0.7 → your reranker or your floor is misconfigured. |

**Correlate `ledger_hash` against user feedback** (barge-in, rephrase, correction — all captured
in S0). Over a few hundred turns you'll see *exactly* which context shapes produce bad answers.
That dataset is what makes GEPA's prompt evolution work — it's the feedback function.

---

## 7. Anti-patterns (the "additive pseudo-techniques")

| Don't | Why |
|---|---|
| **Max out the context window** | Accuracy loss begins ~50K tokens of *relevant* info, regardless of a 200K–1M rating. Budget to a fraction of the ceiling. |
| **RAG your whole repo / whole drive** | Pre-packing at scale. Retrieve JIT. For code, use structural retrieval. |
| **Compact on a timer** | Compact on a *constraint*, at 70–75%, cap-first. |
| **Put a timestamp or "today's date" in the cached prefix** | Kills every cache hit. Put the time in the JIT zone or pass it as a tool result. |
| **Inject top-20 memories every turn** | Within a week the agent acts on 3-session-old memories and contradicts what you just said. Write selectively, recall JIT, timestamp everything, prefer *recent + verified* over *older + popular*. |
| **Let `AGENTS.md` grow** | Prepended every turn — every line is a recurring tax. Durable project-wide rules only; layer the rest JIT. |
| **Store full documents in facts** | Memories are short. A title + one-line body is plenty. Embed for *recall*, not for full-text retrieval. |
| **Rely on compaction to capture important context** | Fragile, retroactive extraction. Write durable facts to memory **on write**, at the moment they're established. External memory is the source of truth — never the compaction summary. |
| **Let the reserve be spent** | An agent with no headroom panics at exactly the wrong moment. |

---

## 8. What "clear" actually means, operationally

You said the main objective is a **clear** context-aware assistant. Here's what that decomposes
into, and which mechanism delivers it:

| Felt quality | Mechanism |
|---|---|
| "It knows what I'm talking about" | Query rewrite + pronoun resolution + hybrid retrieval + rerank |
| "It remembers what I told it" | Bi-temporal facts with provenance ([02](./02-INNOVATION-memory-compiler.md)) |
| "It doesn't confuse old info with new" | `valid_to` / `retracted_at` + freshness gate |
| "It doesn't ramble" | `SOUL.md` register + tier-appropriate model + spoken-mode markdown ban |
| "It's not generic" | JIT recall of *your* facts instead of top-20 memory soup |
| "It doesn't contradict itself mid-session" | Stable cached prefix + append-only transcript + no mid-session summarisation |
| "It's fast" | Intent gate skips retrieval; preemptive generation; L0/L1 local tiers |
| "It gets better" | The gate log + GEPA + S3/S4 ([06](./06-INNOVATION-self-improvement-loop.md)) |

**Clarity is not a prompt adjective. It's the emergent property of a budgeted context, a
reranked retrieval path, and a temporal memory model.**
