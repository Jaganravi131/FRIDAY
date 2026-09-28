# 08 — Proactivity: The Heartbeat & Circadian Rhythm

> **Law 1: Proactivity is a *heartbeat*, not a cron job.**
>
> A scheduler that runs a script is automation. A scheduler that runs an **agentic turn** which
> decides *whether to speak* is a companion.

---

## 1. Why this is the feature that makes FRIDAY feel alive

Every other capability in this blueprint — memory, voice, vision, context — makes FRIDAY
*good*. Proactivity is what makes it feel *present*. The difference:

| Reactive | Proactive |
|---|---|
| You: "what's my rent?" → FRIDAY: "₹28,000" | FRIDAY (Oct 8, 09:00): "Renewal's due on the 15th and Ramesh still hasn't sent the draft. Want me to chase him at 10?" |
| You: "summarise my day" | FRIDAY (07:30): briefing already compiled overnight, waiting |
| You: "is the server up?" | FRIDAY (02:14, only because it's actually down): "build worker's offline. Restarting it — say stop if you want to look first." |

The mechanism is genuinely simple. The **discipline** is the hard part.

---

## 2. The Heartbeat

### The loop

```
        every 15–30 min while you're awake
                    │
        ┌───────────▼────────────────────────────────────┐
        │  HEARTBEAT TICK  (scope: heartbeat)            │
        │                                                │
        │  reads:  HEARTBEAT.md  (the checklist)         │
        │          today's chat summary                  │
        │          what the interactive scope already    │
        │            told you  ← prevents duplication    │
        │          open loops + their due times          │
        │                                                │
        │  prompt: terse, operational, NOT conversational│
        │  tools:  a WHITELIST (read-mostly, no shell)   │
        │                                                │
        │  must return a STRUCTURED tick-ack:            │
        └───────────┬────────────────────────────────────┘
                    │
        ┌───────────▼────────────────────┐
        │  {"action": "notify" | "silent"|│
        │   "act",                        │
        │   "urgency": "low"|"normal"|    │
        │              "high",            │
        │   "message": "...",             │
        │   "surface": "mobile",          │
        │   "reason": "...",              │
        │   "open_loops_touched": [...],  │
        │   "next_check_hint": "..."}     │
        └───────────┬────────────────────┘
                    │
     ┌──────────────┴──────────────┬─────────────────────┐
     │                             │                     │
 "silent"                      "notify"               "act"
     │                             │                     │
 ▼   ▼                             ▼                     ▼
HEARTBEAT_OK                 route via the          irreversible?
gateway SUPPRESSES           Presence Fabric's      → CONFIRMATION GATE
it. Never delivered.         ROUTING_POLICY         → else execute + audit
Logged only.                                        → write open loop
```

### `HEARTBEAT_OK` is the most important token in the system

OpenClaw's insight, and it is *the* thing that decides whether you keep proactivity on:

> If nothing needs doing, it replies `HEARTBEAT_OK`, **which the Gateway suppresses and never
> delivers to you.**

Your eval suite must enforce this (`pro-001`): **100 heartbeats with nothing actionable →
0 messages sent, `heartbeat_ok_rate ≥ 0.95`.**

**Target suppression rate: >80%.** A heartbeat that speaks more than 20% of the time is broken.
It should be *boring* almost always. That's what makes the 20% land.

### `HEARTBEAT.md` — the checklist

```markdown
<!-- soul/HEARTBEAT.md -->
# FRIDAY's standing watch

## Rules for every tick
- Default to `silent`. Speaking is the exception and must be justified in `reason`.
- NEVER repeat something the interactive scope already told Jagan today.
- NEVER speak between 22:30 and 07:00 unless urgency == "high".
- NEVER send more than 2 proactive messages per hour, or 8 per day.
  If you have more than that, you are being annoying. Batch them.
- If a check fails 3 times in a row, STOP checking it and tell Jagan once.
- Anything irreversible → confirmation gate. No exceptions.

## Standing checks
- [ ] **Lease/broker** — open loop `ol_044`. Chase only if >24h unanswered AND
      10:00–19:00 AND not Sunday. (skill: lease-chase)
- [ ] **Calendar conflicts** — next 48h. Notify only on a NEW conflict, and only
      if it's <12h away. Don't re-notify a conflict already mentioned.
- [ ] **Inbox triage** — flag only: (a) contains a deadline within 72h, or
      (b) from the 5 VIP senders in USER.md, or (c) contains an invoice/receipt.
      Everything else is silent. Do NOT summarise the inbox.
- [ ] **Build worker / home server** — every tick. Notify only on state CHANGE
      (up→down or down→up), never on state.
- [ ] **Price watch** — the 32GB SODIMM, `ol_051`. Notify if <₹4,200. Once.
- [ ] **Weather** — notify only if rain >70% AND Jagan has an outdoor calendar
      event in the next 12h. Otherwise silent.

## Do not watch (explicitly, so the model doesn't invent checks)
- News. Never proactively.
- Social media. Never.
- Anything requiring shell access.
- Any account marked `sensitive: true` in the Sense Registry.

## Annoyance budget  ← the mechanism that keeps this honest
- today: 2 / 8 used
- this hour: 0 / 2 used
- dismissed_without_reading (7d): 1  ← if this exceeds 30%, HALVE the checks
- marked_annoying (7d): 0
```

**The "Do not watch" section is as important as the watch list.** An LLM given a checklist will
*invent* additional checks — that's what LLMs do. An explicit negative list, plus the annoyance
budget, is what stops FRIDAY from becoming a notification spam engine.

**The annoyance budget is a real feedback controller**, not decoration:
```python
dismiss_rate = dismissed_without_reading / total_proactive_sent   # rolling 7d
if dismiss_rate > 0.30:
    # FRIDAY is being ignored → it's noise. Automatically halve the tick rate
    # and prune the lowest-value checks. Tell the user once.
    heartbeat_interval_minutes *= 2
    disable_lowest_value_checks(n=2)
    notify_once("I've been pinging you too much. Cut back to the important stuff.")
```

---

## 3. The Circadian Rhythm

A heartbeat is reactive-on-a-timer. The circadian rhythm is **when the heavy thinking happens.**

```
03:00 ─ 04:30   DREAMING  (the consolidation phase)
                ┌──────────────────────────────────────────────────────┐
                │ 1. COMPILE   S0 traces → S1 episode for the day      │
                │              (local Qwen3-4B — summarisation, cheap) │
                │ 2. REFLECT   "What FRIDAY got wrong" pass over        │
                │              implicit feedback signals:              │
                │                barge-ins · immediate rephrases ·     │
                │                "no I meant" · unnecessary            │
                │                escalations · missed escalations      │
                │ 3. EXTRACT   S1 → S2 facts, through G2 (grounding)   │
                │              and G3 (contradiction)                  │
                │ 4. RECONCILE invalidate superseded facts (bi-temporal│
                │              retraction, never deletion)             │
                │ 5. DECAY     apply half-lives per predicate_class    │
                │              archive <0.15, demote <0.40             │
                │ 6. DEDUPE    merge redundant facts, cluster near-dups│
                │ 7. DISTIL    S3 skill proposals (G4: ≥2 occurrences  │
                │              or hard enough to codify)               │
                │ 8. PROMOTE   propose MEMORY.md swaps (salience ×     │
                │              confidence × access_count)              │
                │ 9. EVOLVE    GEPA on prompts vs the LOCKED suite     │
                │              (train/val splits; teacher = cloud,     │
                │               search-time only)                      │
                │10. PRECOMPILE tomorrow's stable prefix — so the      │
                │    cache is warm at 07:00 and the first turn of the  │
                │    day is fast                                       │
                │11. PRE-FETCH morning briefing data (calendar,        │
                │    weather, overnight alerts, inbox triage) so 07:00 │
                │    is a render, not a computation                    │
                │12. COMMIT     git commit + push. The diff IS the      │
                │    human-reviewable record of what FRIDAY learned    │
                └──────────────────────────────────────────────────────┘

04:30 ─ 05:00   SELF-TEST   run the eval suite against the new state.
                            If held-out regressed >1% → git revert the
                            Dreaming commit. FRIDAY wakes up as yesterday's
                            FRIDAY. You get a note: "tried to improve
                            overnight, made it worse, rolled back."

07:00           BRIEFING    rendered from pre-compiled data. Not generated.
                            This is why it's instant.

07:00 ─ 22:30   HEARTBEAT   every 15–30 min (adaptive; backs off if ignored)

22:30 ─ 07:00   QUIET       heartbeats suppressed except urgency=high.
                            Dreaming prep: stage tomorrow's checks.

WEEKLY (Sun)    DISTILL     harvest stable S2/S3 → synthesise training data
                            → Kaggle free T4 (quota resets Sunday UTC — this
                            is deliberate) → QLoRA → GGUF → G7 gate → promote
                            → prune retired skills → review the gate log

MONTHLY         AUDIT       you read: gate-log summary, memory diff stats,
                            annoyance metrics, cost/turn trend, denied-access
                            attempts. Prune facts you disagree with by hand.
```

### Why "Dreaming" is the right frame, not just "nightly cron"

Letta's **sleep-time compute** is the principle: a background agent reorganises memory while the
primary agent is idle, so **consolidation cost never lands in the user-facing latency budget.**
Most memory systems do their thinking on the critical path — that's why they're slow and why
ingestion (500–2,000 ms) pollutes turn latency.

**FRIDAY's rule: nothing expensive ever happens on a turn.**
| Operation | Latency | Where it runs |
|---|---|---|
| Memory ingestion / fact extraction | 500–2,000 ms | **Dreaming** |
| LLM-synthesis retrieval ("reflect") | 800–3,000 ms | **Dreaming** |
| GEPA prompt evolution | minutes–hours | **Dreaming** |
| QLoRA training | minutes | **Weekly, Kaggle** |
| Retrieval (vector+FTS+graph, reranked) | 100–600 ms | turn — **parallelised** |
| Intent gate | ~15–40 ms | turn |
| Ledger compile | <5 ms, no model | turn |

There's a second, softer reason the frame is right: the nightly reflection pass — scanning the
day, identifying recurring patterns, merging redundant facts, surfacing meta-insights — means
FRIDAY **doesn't just remember "you told me X"; it learns "your investment philosophy is Y."**
That's the difference between a log and a model of a person. Practitioners report that over a
3-month session this makes memory *cleaner*, whereas naive RAG gets *noisier* with redundant data.

---

## 4. Two scopes, one memory (the anti-duplication mechanism)

This is `roiguri/jarvis-agent`'s design and it fixes the #2 project-killer.

| | interactive scope | heartbeat scope |
|---|---|---|
| Prompt | conversational, warm, expansive | **terse, operational** |
| Sees | full JIT memory + today's proactive notifications | today's chat summary + `HEARTBEAT.md` |
| Tools | full registry (with confirmation gate) | **read-mostly whitelist, no shell** |
| May message you | yes, in reply | only via a `notify` tick-ack |
| Awareness | reads today's notifications → doesn't re-answer | reads today's chat → doesn't repeat |

**Awareness flows both ways.** Without it you get the bug where FRIDAY tells you at 09:00 that
the lease is due, you discuss it at 09:05, and FRIDAY tells you *again* at 09:30.

Implementation is trivial — both scopes write to the same `sessions`/`turns` tables with a `scope`
column, and each prompt includes a compact digest of the other's activity today.

---

## 5. The confirmation gate

Proactivity + capability = risk. Every irreversible action routes through a channel-agnostic
confirmation, regardless of which surface you're on.

```python
RISK_TIERS = {
    "tier_0_read":     {"examples": ["read calendar", "search memory", "check server"],
                        "policy": "execute, log"},
    "tier_1_reversible": {"examples": ["create draft", "add open loop", "write note"],
                        "policy": "execute, log, show in the daily diff"},
    "tier_2_external": {"examples": ["send message", "post comment", "email"],
                        "policy": "ALWAYS show the exact payload → confirm → send"},
    "tier_3_destructive": {"examples": ["delete file", "git push --force", "cancel booking",
                                          "modify a fact marked user_edit"],
                        "policy": "confirm + type-the-word + 10s cooldown"},
    "tier_4_financial":  {"examples": ["any payment", "any order", "any transfer"],
                        "policy": "NEVER autonomous. Human executes. FRIDAY may only prepare."},
}
```

Rules:
- **Confirmations are channel-agnostic.** An inline button on Telegram, a modal in the PWA, a
  spoken "yes/no" on the voice node — same underlying `confirmation_required` event.
- **Show the exact payload.** Never "send a message to Ramesh?" — always the literal text.
- **Tier 4 is never autonomous.** Not because FRIDAY can't, but because a single
  prompt-injection-triggered payment ends the project. This is a bright line.
- **A `rm`, `sudo`, `curl`, or force-push pattern in any shell command → always confirm.**
- **Every gate event writes to `audit`.** You should be able to answer "what did FRIDAY do while
  I was asleep?" with one query.

---

## 6. What to watch first (Phase 4 build order)

Do not build all of `HEARTBEAT.md` at once. Add checks one at a time and measure the
useful:annoying ratio after each.

```
Week 9   Heartbeat loop + tick-ack schema + HEARTBEAT_OK suppression + audit logging
         → run with an EMPTY checklist for 3 days. Verify it stays silent.
         → this is the test that matters. If it speaks with nothing to say, stop and fix it.

Week 9   Scope separation (interactive vs heartbeat) + bidirectional awareness

Week 10  ONE check: calendar conflicts. Measure for a week.
         Target: ≥3 useful notifications, 0 annoying.

Week 10  Confirmation gate (tiers 0–3). Adversarial-test it.

Week 11  Morning briefing, pre-compiled overnight. It must render in <2s at 07:00.

Week 11  Dreaming phase (steps 1–8). git commit + push. Read the diff every morning
         for two weeks — this is how you catch bad learning early.

Week 12  Annoyance budget controller + self-test/rollback at 04:30.

Week 13+ Add checks ONE AT A TIME. Two weeks each. Prune anything with a >30% dismiss rate.
```

**The measure that decides success:** after one week, count
`useful_initiations : annoying_initiations`. Target **>5:1**. If you can't tell which were
annoying, add a one-tap "that was noise" button on every proactive message and use it honestly.

---

## 7. The always-on problem (be honest about this)

**A proactive assistant that only exists when your laptop is open is not proactive.**

A VivoBook sleeps. When it sleeps, there is no heartbeat, no briefing, no alerts. Options:

| Option | Cost | Trade-off |
|---|---|---|
| Disable laptop sleep on AC, lid closed | ₹0 | Thermal wear, fan noise, battery degradation, not actually always-on (you carry it around) |
| **₹800/month VPS (8 vCPU / 32 GB)** | ~₹9.6k/yr | ⭐ **Best.** Gateway + heartbeat + memory + Dreaming live there; the VivoBook is a *client* and heavy-inference node. FRIDAY is genuinely always-on. 32 GB also unlocks the 30B MoE on the server |
| Used mini-PC / old laptop at home | ₹8–15k once | Always-on, private, no recurring cost. Needs power + network reliability |
| Raspberry Pi 5 | ₹6–8k | Runs gateway + heartbeat + tiny models. Whisper is too slow (~3–4 s) but speech-to-phrase and wake-word work. Great *node*, weak *server* |
| Mac Mini M4 24 GB | ~₹75k | The premium appliance: <1 s Whisper, <1 s 8B generation, 24 GB unified memory all usable for models, silent, always-on |

**Recommended path:** VivoBook for Phase 0–3 (development) → **₹800 VPS from Phase 4 onward**
(proactivity needs it) → Mac Mini or a used RTX 3090 box later if you want heavy local inference.

The architecture doesn't change at any of these steps. That's the point of the Presence Fabric:
the server moves, the surfaces just reconnect.
