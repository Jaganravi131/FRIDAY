# 21 — External system notes: NousResearch `hermes-agent`

**2026-10-18. The user asked:** analyse the repo, and replicate whatever of its
architecture or code is useful for FRIDAY. Same protocol as doc 16 / T-017
(the OpenClaw reconciliation): verify against source, adopt small, document
every rejection with the law it breaks, flag overrides as user decisions.

Source: `github.com/nousresearch/hermes-agent` (MIT, © 2025 Nous Research),
reviewed at the shallow clone of 2026-10-18. It is a self-improving personal
assistant: a Telegram/terminal agent whose background loop re-reads its own
transcripts, saves what it missed to a layered memory (USER.md / MEMORY.md /
lessons / skills), and lets the user edit everything by hand. The repo is
large (20k+ files: plugins, providers, Docker, a TUI, a gateway, multi-OS
setup scripts) — most of it is surface FRIDAY deliberately does not have.

What follows is the part worth keeping.

---

## 1. What Hermes actually got right (the three findings)

### 1.1 The background review loop — `agent/background_review.py`

Their best idea, and the one their README is really about: **an agent that only
writes memory mid-task misses everything the task was not about.** Narratives,
mid-sentence facts, user preferences dropped while debugging something else —
the live loop's attention is on the task, not on memory hygiene. So a separate
review pass re-reads the transcript and saves what was missed.

Their prompt discipline is genuinely good (MIT, adapted with attribution in
`friday/memory/review.py`):

* *"save it once, in the right store"* — one fact, ONE store; USER.md is who
  you are, MEMORY.md is your environment (their routing rule).
* **"'Nothing to save.' is a real option"** — the review is told that finding
  nothing is a fine answer, which is what keeps it from inventing work.
* **Do-not-capture list:** environment-dependent failures, negative tool
  claims ("X doesn't work"), transient errors, one-off narratives, and — the
  sharpest line — *unresolved failures dressed as workflow*. An agent that
  writes "step 3: retry the flaky call twice" has memorised its own bug.

**What FRIDAY cannot adopt:** their honesty is enforced by prompt text. A
review that hallucinates a quote saves the hallucination.

### 1.2 The lesson/skill layer — procedure-first, to the user's specs

Hermes treats a **skill** as a class-of-task (renewing a lease, booking a
specific airline) written as a procedure **to the user's standards**, not a
generic capability. The writing rules that transfer:

* procedure first, then pitfalls as *generalizable imperatives with the WHY*
  ("always book 3 days ahead — the price jumps inside 72h"), never as
  narrative ("last time it was expensive");
* **search before append** — look for the existing skill first; a duplicate is
  a routing hazard;
* **fix in place, never "UPDATE:"** — a skill that appends errata to itself
  decays into an argument with itself;
* `references/` for depth, body stays lean — progressive disclosure again;
* the skill review is told to **"Be ACTIVE"**: user corrections are first-class
  signals, and a skill that was consulted but wrong gets patched *now*, not
  noted for later.

### 1.3 The skill format — agentskills.io, progressive disclosure

`SKILL.md` frontmatter (`name` / `description` / `version` / **`when_to_use`**)
with the body as markdown. The index ships only name + when_to_use (~30
tok/skill); the body loads on activation. This is the same shape as doc 03's
S3 design — Hermes is the field evidence that the shape works, including the
detail that **`when_to_use` is the whole routing surface**: a skill that
cannot say when it applies can never be routed to.

---

## 2. What FRIDAY adopted (T-022, T-023, T-024)

| Hermes idea | FRIDAY implementation | The FRIDAY difference |
|---|---|---|
| SKILL.md format + index | `friday/memory/skills.py`, `skills` table, STABLE ledger slot, `friday skills` CLI (T-022) | Validation is load-bearing (bad skill → reported, never half-registered); the index is an **injection surface**, so a hostile `when_to_use` is withheld (same boundary as core_memory); re-register preserves `uses` |
| Background review | `friday/memory/review.py`, `friday review` (T-023), scope=`dreaming` | Honesty lives in the **data layer**: the review runs on the same session, so a write whose `source_quote` is not in the reviewed turns is **denied by the policy engine**, not discouraged by prompt text |
| **Skill review** (`_SKILL_REVIEW_PROMPT` — "Be ACTIVE", signals, preference order) | `friday/memory/distill.py` + `friday dream` (T-024) | See §2.1 below — three gates Hermes does not have |
| **Read-before-write** (their `skill_manage` refuses a patch without a fresh `skill_view`) | enforced by construction (§2.1 gate 2) | a flag can be forgotten; a skill body that was never in the prompt cannot be patched |
| **Write-origin provenance** (ContextVar: review-fork writes vs user-directed writes) | structural: dreams write only under `dreams/`, promotion is a separate user-run command | the origin distinction is a directory, not a runtime flag |
| **Usage telemetry** (`.usage.json` sidecar: active→stale→archived, pins) | `record_activations()` on the `uses`/`last_used` columns the index already orders by (T-024) | no sidecar to desync; bumps only ACTIVE skills; best-effort — telemetry never fails a turn |
| Prompt fragments | adapted with attribution in the module docstrings | — |
| Skills as user-owned | user hand-edits `skills/<name>/SKILL.md`; the watcher re-registers on the next compile (same rule as `soul/`) | Hermes's edit-by-chat is *not* adopted — see §3 |

Both halves shipped with the bench at ratchet (recall@1 0.963) — an empty
skill set compiles the byte-identical prompt it always did.

### 2.1 The lesson layer (T-024) — the loop's second half

Their skill review is where their self-improvement actually lives: most
sessions produce at least one skill update, user corrections are FIRST-CLASS
skill signals, and a consulted-but-wrong skill gets patched NOW. FRIDAY
adopts the whole prompt discipline (procedure first; a pitfall is a
generalizable imperative + one clause of WHY, never an incident narrative;
the same lesson learned twice is ONE rule; fix in place, never "UPDATE:";
do-not-capture) — and adds three gates their architecture has no place to put:

1. **Provenance.** A proposal carries `source_quotes`, and every one is
   checked with `policy.quote_is_user_authorised` against the turns being
   reviewed — the exact function that gates `memory_write`. A distilled
   "the user prefers X" the user never said is denied, audited
   (`dream.skill_denied`), and writes nothing.
2. **Read-before-write by construction.** The distill prompt embeds only the
   N most-used skill bodies (char-budgeted); a patch is accepted only for a
   skill whose body WAS embedded. Their guard is a session flag set by a
   `skill_view` call; FRIDAY's is the prompt itself.
3. **The stable-prefix law.** A dream writes under `dreams/` and only there —
   never `skills/`. The draft is not registered, not in FTS, not in the
   ledger; the prompt is byte-identical. It goes live only via
   `friday skills --promote NAME` (re-validated, `PROPOSAL.json` kept beside
   the skill as the durable why, audited `memory.skill_promoted`, actor
   **user**). Hermes's review fork edits the live library directly; FRIDAY's
   user holds the diff — which is what doc 06's governance ladder demands of
   the manual rung.

`friday dream` runs both seeds in one tick (fact review, then distillation).
Scheduling it (their trigger is session-end, in the background) is still a
user decision: Q2 — the default tier is a 2.6B local model and dreaming is a
model job. Until then it is a command, not a daemon.

### 2.2 Not adopted, Hermes-side self-improvement machinery

The **background curator** (consolidation at scale, auto-stale/archive by
usage) is real engineering but premature: FRIDAY's library is user-gated by
construction, so sprawl needs a library first. Deferred with the doc 06
G-gates, which are the blueprint's own version of the same lifecycle. The
**skill hub** (install skills from ClawHub/GitHub) is a supply-chain surface
this project's SECURITY.md posture rejects outright (same law as the
self-update rejection in T-017).

## 3. What FRIDAY rejected, with the law each breaks

| Hermes subsystem | Law it breaks |
|---|---|
| Plugin catalog, provider ecosystem, MCP toolbox (`plugins/`, `providers/`, `toolsets.py`) | The three-tool ceiling is a measured budget decision (tools.py docstring, doc 04); tools are added with audit + policy, not volume |
| Subagents / background agent workers | Multi-agent coordination is T-009 and goes through the board; a second agent is a second trust domain |
| Terminal backends, Docker, `docker-compose*.yml`, Nix, setup scripts per-OS | ₹0 budget, stdlib-only core, local-first (INSTALL.md); the target is one 16 GB laptop, not a fleet |
| Gateway platform + TUI + web UI | FRIDAY's UI surfaces (CLI, offline web UI, phone via Tailscale) are already specified offline and CDN-free — pinned by test |
| **Honcho** (external "user graph" service) | Local-first: the user model IS the fact store + graph (doc 03); shipping the user's model to a third party contradicts the trust architecture (doc 09) |
| Trajectory compressor (`trajectory_compressor.py`) | Traces are the audit trail; lossy compression breaks the provenance FRIDAY's quote-gate is built on |
| Two-file MemoryStore (markdown + JSON sidecar) | FRIDAY's stores are typed and bi-temporal (doc 03 §schema); a JSON shadow of markdown loses the invariant and re-creates the two-sources-of-truth problem USER.md/MEMORY.md budgets exist to prevent |
| CJK FTS, WAL repair, Windows ACLs (`hermes_dir_acl.py`) | Real engineering, wrong scope: no CJK corpus, no corruption incidents, not the target OS |
| Skills edited by chat (agent rewrites its own skills mid-session) | The stable-prefix rule: skills change *between* sessions, with the user's hands on the diff (doc 06 governance ladder — the agent doesn't own the suite, and it doesn't own the skills either) |

None of these are user decisions yet. If the user wants one of them (Honcho
and skill-edit-by-chat are the two plausibles), that is an override to record
on the board, not a refactor to slip in.

## 4. Net

Hermes validated two FRIDAY designs (S3 skills, dreaming-scope review) and
contributed the one thing FRIDAY's docs were thin on: **what the review and
the skill-writer should refuse to capture.** That list now lives in
`friday/memory/review.py`'s prompt with attribution. The replication is
T-022 + T-023; the rejection table above is the rest of the answer.
