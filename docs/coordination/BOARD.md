# BOARD — shared work queue

Read `AGENTS.md` first (lanes, gates, the loop). Every change names an item here.
Statuses: `open` · `in progress (lane, branch)` · `blocked (why)` · `done (numbers)` ·
`partial (what remains)` · `QUESTION`.

Last updated: 2026-10-18 (review round 2: T-018…T-021; post-refresh restore), by the agent on branch `arena/ad9dff8f-friday`.

## Goal

Make FRIDAY a measurably intelligent, personal, and novel assistant, within a ₹0 budget
and a 16 GB laptop. The model's raw intelligence comes from the user's machine (Lane L).
Everything below is about the system around the model, plus the evaluation that proves it.

---

## Done

| ID | Lane | Item | Result |
|---|---|---|---|
| T-000 | Q | Bring the full project onto the session branch (fast-forward from `arena/01a0e981-friday`, 34 commits) | `main` and the session branch had only the README before this |
| T-001 | R | **Concept lexicon** for paraphrase recall — `friday/retrieval/concepts.py`, doc 20 | Committed bench paraphrase recall@5 0.00 → 0.75 (n = 4); overall 0.815 → 0.926; false positives 0.0. Held-out probe 2: 0.25 → 0.50, FPR 0.0. Probe 1 is contaminated (debugging set). Gate ratcheted. |
| T-001a | R | Regression found by the invariance test: a set-collapse dropped a query term | Fixed; `tests/test_retrieval.py` green |
| T-001b | E | Test that pinned the paraphrase defect replaced by a ratchet | `tests/test_delivery.py` asserts paraphrase ≥ 0.5 |
| T-006 | Q | Stale status and counts | README no longer says "not started"; CONTRIBUTING counts corrected to ~670 |
| T-007 | E + G | **Grounded-answer harness** — `friday/grounding.py`, `tests/test_grounding.py` | Mock: grounded accuracy 8/8, fabricated 0/5, misdirected 0. The ratchet is backed by two deliberately bad clients, both caught. **The mock number is a plumbing check, not a measure of intelligence.** Real-model run is pending (T-003 / Q2). **CI found a Windows-only bug in this item:** an unclosed sqlite handle made temp cleanup fail (`WinError 32`) on all three Windows jobs. Fixed, and a Linux guard now checks for the leaked handle, so the bug class is caught on Linux too (verified: the guard fails when the close is removed). |
| T-005 (part) | Q | Lint defects that can hide bugs | F841 (unused variables) 9 → 0; F401 (unused imports) 37 → 0 outside `__init__.py`. See T-005 below for what remains. |
| T-005a | E | Two tests that could not fail | `test_dropped_stale_is_counted` asserted `>= 0`, now asserts the value 1 and that the stale fact is withheld. `test_only_the_edited_file_is_rederived` in `test_watcher` computed a snapshot and never compared it; now it does. |
| T-005b | E | Missing assertion in an RSC loss test | `test_salience_class_balances` built a second error case and never compared it; now it does. Missing a rare positive costs more than one negative error, 2.31 vs 0.27. |
| T-005c | R/S | Dead code (`concat_heads`, unused `sl`, `k`, `rng`) | Removed; behaviour unchanged, full suite green |

---

## Current repair pass (2026-10-14)

### T-013 · Installed-package completeness (Lane Q + G) — `done`
Reviewed both Arena branches; `5319953` contains the older branch and is the imported
baseline on `arena/ad9dff8f-friday`. Baseline: 674 collected, one expected failure.
The first clean-wheel probe failed: `agent/loop.py`, `memory/compiler.py`, `store/db.py`
and `store/schema.sql` were all absent. Package discovery now includes all FRIDAY
subpackages and the schema. Needle/ablation implementations ship in `friday/commands/`,
with checkout launchers retained; CLI invocations no longer modify `sys.path` or load
scripts from the data root. Dev extras include the test plugin and build frontend.
Acceptance passed: sdist → wheel → offline, dependency-free install into a fresh venv
outside the checkout; console entrypoint, seed, build, correct mock rent, dataset-only
needle, and quick ablation. The check is added to Linux/Windows CI; only Linux was run
locally. Both standalone launchers pass. No evaluator queries or research math changed.

### T-014 · Gateway request validation (Lane G + S) — `done`
Attack-first tests reproduced 47 failures initially; a later oversized-integer probe
also reproduced a decoder crash. Added 59 gateway cases. Non-object JSON, invalid
UTF-8, malformed chat messages/text fields, deep nesting, and integer digit-limit
errors now return JSON 400 without calling the agent; oversized bodies return 413.
Invalid credentials return 401 even with non-ASCII header values. Empty configured
tokens fail closed, and blank persisted token files are regenerated. Valid bearer
and x-api-key clients still work. HTTP input requirements are documented in INSTALL.
Branch: `arena/ad9dff8f-friday`.

### T-015 · Mock loses the question after a tool result (Lane G) — `done`
T-013's clean-install probe asked the seeded monthly rent; the reply was instead the
lease renewal date. `_last_user` read the portable `[tool result]` user-role envelope
as the question. Three regression cases failed before the fix. The client now preserves
the original question and recognizes when a tool has already answered the current turn,
without repeating writes/searches. Six focused cases pass, including native tool roles,
a new turn after an observation, and the real seeded agent roundtrip. The unchanged
grounding harness still passes (8/8 answerable, 0/5 fabricated). No evaluator or
retrieval ranking was changed. Branch: `arena/ad9dff8f-friday`.

### Repair-pass validation

Linux, Python 3.11, stdlib-only core; no model weights installed:
- Full suite: **739 passed, 1 xfailed**, up from 674 collected (**66 new cases**).
- Phase 0: **11 PASS, 0 FAIL, 1 SKIP** (real-model idle/unload test remains manual).
- Retrieval gate: **PASS**, recall@5 **0.9259**, paraphrase recall@5 **0.75** (n=4),
  false positives **0.0**; baseline and evaluator unchanged.
- sdist → wheel → isolated offline install: **PASS**; checkout launchers also pass.
- Pyflakes (`ruff check --select F`) on changed Python files and `git diff --check`: **PASS**.
  Broader existing style and catch-all-exception lint debt remains (T-005).
- Windows CI has been configured for the wheel check but not run in this sandbox.

See [the review summary](2026-10-14-review.md) for scope and remaining limitations.

---

## Open — ordered by leverage

### T-002 · Blind holdout set for retrieval (Lane E)  — `partial (runner ready, queries awaited)`
Every number in doc 20 is weaker than it looks, because the lexicon was tuned on the
same agent's probes. Write **40+ paraphrase queries and 10+ unanswerable queries**, with
the expected predicate, **in a file no retrieval agent edits**. Write them *before*
looking at the current results. Report recall@5 before and after the next change.
*Not done in this session on purpose:* an agent that wrote the holdout and then graded
the change would be the same agent twice. It needs a different agent or the user.
*Acceptance:* a holdout that Lane R can run but not read or edit, plus its first number.

**2026-10-16 — the instrument half is built** (`friday holdout`, T-002 runner):
- `python -m friday holdout <file>` scores any holdout JSON with **`bench.score_cases`,
  the same instrument as the committed bench** (extracted in this change; bench
  behaviour identical, its restore test now also passes on the exception path).
- Built so a retrieval agent can run it without tuning against it: class aggregates
  only, no per-query output, and validation errors name the case index + expected
  predicate, never the query text. A below-minimum set is flagged as a smoke test.
- Format and author protocol: `eval/holdout/README.md` (who may write, who may run,
  the 13 valid expected predicates, and the run-on-two-commits comparison).
- Shipped in the wheel: `scripts/check_wheel.py` now smoke-runs the installed
  `holdout` and asserts the no-leak rule holds there too. `friday bench --json` also
  no longer prints build chatter in front of the JSON.

**Still needed (the human/other-agent half): the queries.** Whoever writes them must
not have seen retrieval output — the user in an afternoon, or a second agent pointed
at `eval/holdout/README.md` and nothing else. Until the file exists and its first
number is recorded here, the doc 20 numbers stay unvalidated and T-012 stays open.

### T-003 · Semantic embedder comparison on the user's machine (Lane L, user runs)  — `blocked (weights)`
Run `pip install -e ".[embed]"` then `python -m friday bench --record` with a real
embedder. Keep the hashing baseline as a row beside it. Doc 20 §6 explains why the
comparison is the result worth keeping. *Blocked here:* model weights cannot be downloaded
in this sandbox. Agents may prepare the script; the user runs it.

### T-004 · Lexicon coverage gaps (Lane R)  — `done (with two gaps left open on purpose)`
From doc 20 §5: possessive `'s` handling, a "what time" intent cue, and the words
*program*, *usually*, *have*. **Each word gets a test first**, a benign negative case, and
a check that no existing bench number drops. Do not add a word just because a probe missed it.

Outcome (2026-10-16, doc 20 §5b has the numbers): the thing-scoped "what time" cue,
`usually`/`usual` as filler, and the cluster-conditional possessive strip shipped; each
is pinned next to its Law 4 negative in `tests/test_lexicon_gaps.py` (14 cases + 1
strict xfail). `program` was **rejected by its own benign negative** — in the editor
cluster it scores "what is my training program" 0.589 against `editor: neovim`, a shipped
wrong answer; it stays out until the embedder tier can judge meaning. The process also
found a real defect: `CONTENT_STOPWORDS` holds surface forms but the reranker subtracts
stems, so `have`/`hav`, `were`/`wer`, `really`/`real` and more were never actually
stopped — fixed with a stem-normalised view of the one canonical list. Bench:
recall@1/@5/MRR 0.9259 → **0.963**, noun recall@5 → **1.0**, gap → **0.0**, false
positives **0.0**, baseline re-recorded. Two gaps remain open on purpose: "flat's owner"
(strict xfail; the cluster merge or alias that would close it ships the landlord fact at
0.739 for "my rent") and "drink in the morning" (`morning` is real content, not filler —
that one is the embedder's job, T-003).

### T-005 · Remaining lint debt (Lane Q)  — `partial`
F841 and F401 are done (above). **S110 is done for `friday/`**: 19 sites now call
`friday.util.swallowed(where, exc, warn=…)`, which keeps a bounded record and logs (DEBUG,
or WARNING for audit writes). `tests/test_swallowed.py` fails on any new bare
`except Exception: pass` in the package. Two sites in `scripts/` are out of scope and remain.
Still open: `BLE001` blind `except Exception` (59 sites, unchanged; the new handlers still
catch broadly on purpose). Fix a site only with a test that would catch a regression in
what it swallows. Style-only rules (`I001`, `UP035`, `FURB167`) are low priority.

### T-012 · Two parallel fixes for the same paraphrase defect  — `QUESTION` (decision needed)
Branch `friday-dev` (commit `d188667`, same GitHub account, not created by this session)
adds `friday/retrieval/synonyms.py`, a curated human-word → store-vocabulary map with six
curation rules. It does what T-001 does, by a different route. Both touch `pipeline.py`,
`rerankers.py`, `bench.py`, `test_delivery.py` and `BENCHMARKS.md`, so they conflict.

Measured on the same held-out probes (run once each, scratch copy of `friday-dev`):

| Implementation | Committed bench paraphrase | Committed bench overall | Probe 1 (contaminated for T-001) | Probe 2 |
|---|---|---|---|---|
| `friday-dev` synonym bridges | 1.00 (n = 4) | 0.963 | 0.25 | 0.42 |
| This branch, concept lexicon | 0.75 (n = 4) | 0.926 | 0.58 | 0.50 |

Neither is shown to generalise better. The committed set is too small to decide, and probe 1
is biased toward this branch. The blind set (T-002) is the thing that can decide it.
*Not done:* merging either branch. Each is someone else's decision to make. A combined
version (their curation rules applied to our cluster approach) is possible and is a
reasonable next step if the blind set is in place.

### T-008 · Phase 0 SKIP conditions  — `done`
CI's comment said conditions 11 and 2 skip. The script reports one skip (11). Condition 2 is
automated and passes. The CI comment is now corrected to match the script.

### T-004 learning · possessive residue  — `reverted`
See doc 20 §5a. Dropping the stray `s` from reranker query terms raised bench false positives
from 0.0 to 0.2. The gate blocked it. Needs a tokeniser fix that is checked against the gate.

### T-009 · Multi-agent coordination — start using the board (Lane Q)  — `partial`
`AGENTS.md` and this board are the first version, used by this session. Still to prove:
two different agents closing items through the board as the only handoff channel.

### T-010 · The erase-gate ablation measures nothing (RSC)  — `QUESTION` (spec needed)
`friday/rsc/operators.py`, `EraseThenDelta.step`: the `erase_rank_one=False` branch is a copy
of the `True` branch, so the flag changes no output. Confirmed by running both (`S` identical).
Any ablation that toggles the flag reports a null result by construction.

**Why it is not fixed.** Doc 13 writes the erase step as `(I − γ_t e_t ê_tᵀ)` and never defines
`ê_t`. The code's True branch is exactly the case `ê = e`, which I checked by hand. So the
False branch has no specification to implement. Two candidates, and the choice is a research call:
* **(a) `ê_t = q_t`**, the query-driven second erase direction (arXiv 2608.13668, cited in doc 13
  §2.3). It fits FRIDAY's case: "what's my rent?" erases what is stale about rent.
* **(b) a separate learned read address**, a second projection from the same `z_t`.

*Pinned:* `tests/test_rsc.py::test_erase_rank_one_flag_changes_the_operator` is a strict xfail.
The reason string now names both candidates.
*Acceptance:* the owner of doc 13 picks (a) or (b) and writes the definition; then the operator
implements it, the xfail is removed, and the ablation is re-run.

---

### T-016 · Proactive heartbeat, Week-9 layer (Lane Q + G)  — `done (deterministic layer)`
2026-10-16: the gap the "OpenClaw" comparison actually found — FRIDAY was purely
reactive, and `soul/HEARTBEAT.md` was seeded with rules no engine consumed.
Built per doc 08's build order (loop + tick-ack schema + HEARTBEAT_OK
suppression + audit logging, then ONE check — not the whole checklist):

- `friday/agent/heartbeat.py`: `beat()` runs read-only deterministic checks and
  returns doc 08's tick-ack. Silent is the default and the norm; a silent tick
  is audit-logged and never delivered. `friday heartbeat` runs one tick (exit 0
  always — silent is success, which is the whole design).
- The ONE check is the deadline watch: a live fact whose object carries a date
  within 30 days (7-day overdue grace) earns ONE high/normal notification,
  deduped forever via the audit trail. Junk objects, schedules, validity ranges
  and retracted facts stay quiet — pinned by tests.
- Enforced in code, each pinned: quiet hours 22:30–08:00 unless high urgency;
  annoyance budget 2/hour and 8/day; one message per tick (the rest batch to
  the next tick); a check that raises is silence, not a crash.
- Live smoke: the seeded `lease_renewal_due 2026-10-15` earns one HIGH word the
  day after, then HEARTBEAT_OK forever; `friday audit --today` shows both ticks.
- 14 tests in `tests/test_heartbeat.py`; the seeded HEARTBEAT.md now documents
  exactly what the engine enforces, and marks the Dreaming-phase checks as
  pending so nobody trusts them yet.

Not built on purpose: the agentic tick (a model reading HEARTBEAT.md in
`scope=heartbeat` — the loop and unattended-scope policy denies already exist),
the scheduler (doc 08 §7 wants an always-on box for that), and any further
check until this one has a week of useful:annoying measurement.

### T-017 · "OpenClaw parity" plan — reconciled against the constitution  — `recorded`
2026-10-16: the user shared an external assistant's gap analysis and upgrade
path for FRIDAY. Much of it described a system FRIDAY is not; the rest
contradicts the project's own laws. Recorded here so it is not re-litigated:

**Already built (the comparison's "gaps" that are not gaps):** hybrid memory is
the bi-temporal fact store + episodes + FTS/vector/graph recall (doc 03, 04);
RAG is the retrieval pipeline feeding `memory_search`; fine-tuning is
`friday tune` (supervision export → Colab/Kaggle QLoRA runbook, doc 15);
quantised local models are the plan (INSTALL.md model table, GGUF/llama.cpp);
the UI is CLI + gateway + offline web UI + phone via Tailscale; security is
local-first with token auth (T-014), an audit trail, tiered consent and doctor
secret scans; testing is 779 tests + the bench gate + Phase 0 + the wheel probe.

**Rejected, with the law each breaks:** Weaviate/ChromaDB in Docker (₹0 budget,
16 GB laptop, stdlib-only core — the built-in vector index + optional
sentence-transformers tier is the sanctioned path, T-003); LangChain/LangGraph
(the hand-rolled loop exists because a 2.6B model cannot afford the framework
tax — loop.py docstring); "100+ tools" (three tools is a measured ceiling on an
8K budget, tools.py docstring; tools are added with audit+policy, not volume);
AutoGen GroupChat (multi-agent coordination is T-009, through this board);
Streamlit/Gradio (the web UI must stay offline and CDN-free — pinned by test);
TensorRT (the target laptop has an AMD iGPU; Vulkan/llama.cpp is the plan);
Whisper voice (real, but doc 07's design, blocked on hardware like T-003);
`pull_and_install_updates()` self-update (a supply-chain hole that contradicts
SECURITY.md's posture — updates are a human decision on this project).

**Accepted, acted on:** proactivity was the fair hit → T-016 above. That is the
one place the plan changed the codebase this session.

**Still genuinely open (unchanged):** Q1–Q4 below, T-002's queries, T-003's
embedder run, T-010's spec, always-on hosting (doc 08 §7's VPS option).

### T-018 · External review round 2: permissions before confirmation; scope follows the real arg  — `done`
2026-10-18: a second external review ("GPT-6 Astra", user-pasted) listed 7
code-level findings. Two were real permission bugs, verified against source
before fixing (the round-1 lesson: verify first):

* **Confirmation bypassed permission checks (finding 2).** `check()` answered
  `confirmation_required` from `ALWAYS_CONFIRM` BEFORE the sense
  registry/scope checks, and the loop treats that verdict as "checked, awaiting
  the human" and invokes directly on "yes" — so a confirmed-but-ungranted,
  disabled, or out-of-scope `email_send` EXECUTED. The gate now runs LAST,
  after unattended-deny, registered/enabled/scope. The old order was even
  pinned by a test that argued confirmation-first gave "the more useful
  message" — that test is rewritten with the bypass documented as why the
  argument lost. Doc 09 §2 row updated.
* **Scope matcher ignored `url`/`to` and passed empty targets (finding 3).**
  `_scope_allows` read only target/app/path: `web_read(url=…)` and
  `email_send(to=…)` presented an empty target, and an empty target SKIPPED the
  allow list entirely — every scoped web/mail grant was unscoped for exactly
  the tools where scope matters most. Now: destination args include url/to
  (lists joined); a scoped grant with no destination fails closed; host-like
  entries match HOSTS (parsed from URLs/emails, credentials-safe, dot-suffix
  subdomains) not substrings — `gmail.com.attacker.example` no longer matches
  `gmail.com`; with lists, EVERY recipient must be in scope. Word entries keep
  substring semantics for natural-language targets (screen windows) — pinned
  tests unchanged.

Acceptance tests: confirmed-yes × {ungranted, disabled, out-of-scope} → tool
uncalled; url/to shapes tested against the real connectors; lookalike-domain
and userinfo-URL denials; multi-recipient leak denial.

### T-019 · Review round 2: the conversation survives the process  — `done`
Finding 1, verified: `_persist` wrote ONLY the assistant row to `turns`; user
words existed only in the gitignored trace JSONL. `_recent_transcript` reads
`turns`, so after a restart FRIDAY replayed its own answers with the user's
halves — corrections, constraints, preferences — missing from all of them. And
`idx` was set from `turn_idx`, the tool-loop counter that resets every run, so
cross-run rows collided. (Latent proof of intent: `friday why --last` has
always joined `turns` on `t.role='user'` — a row that was never written.)

Fix: both halves of every exchange persist — the user row at turn start, the
assistant row at `_persist` — sharing a session-scoped exchange number
(`MAX(idx)+1`), with a deterministic replay order (`ORDER BY idx DESC, role
ASC` + reverse). Tool results stay out of `turns` by design: the ledger
carries recent context within a run, traces carry the raw record, and replaying
old tool output as conversation is noise, not memory.

Acceptance test: the review's ten-turn conversation (including a rent
correction) survives a brand-new Agent over the same DB, and the answer comes
from the corrected fact.

### T-020 · Review round 2: evidence pipeline — quotes that prove, spans that resolve  — `done`
Findings 4, 5A, 5B — all verified:

* **Stated-without-quote passed as user authority (4).**
  `quote_is_user_authorised("")` returned True ("no claim made"). Any
  `memory_write` that omitted the quote was treated as user-authorised. Now:
  an empty quote is NOT authorised (distinct reason: "quotes nothing"); a
  quote with no distinctive words is not authorised; and matching is
  per-sentence with NEGATION and PERSON agreement — "I no longer work at X" is
  not authorised by "I work at X", and "I work at X" is not authorised by "my
  friend works at X". The tool schema now tells the model a stated write
  without a quote will be asked back.
* **Spans anchored to a filename invented from an id (5A).** facts.py built
  `trace_file = {trace_id[:10]}.jsonl` — an id prefix, not a date; the writer
  names day files by timestamp. Every trace-anchored supervision span pointed
  at a file that never existed, `span_text` returned "", and RSC negative
  mining silently produced nothing. The writer's actual path now flows
  Agent → State.trace_path → ToolContext.trace_file → Fact.trace_file →
  SpanLabel; `friday write` passes it too. Unanchored facts degrade to inline
  spans (quote) or no span — never a broken one.
* **Byte offsets mixed with char offsets (5B).** `append()` used `st_size`
  (bytes) as start and added a char count for end; the reader slices decoded
  text. ASCII days coincided; the first Tamil/emoji line drifted every
  subsequent span into the middle of the wrong line. Offsets are now one unit
  — characters — with the day file read once per append (day-bounded files;
  anchor correctness over a stat()).

Acceptance tests: exact span round-trip for English/Tamil/emoji across many
appends; the "work at / no longer work at / friend works at" triple; span
row's file exists and resolves to the turn's line.

### T-021 · Review round 2: named outcomes; tool results are not user turns  — `done`
Findings 6, 7 — both verified:

* **`ok` was "denied" not in output (6).** Errors and user-declined
  confirmations read as successes — the behavioural-signal analysis was
  training on inverted labels, and max-turns exhaustion had no label at all
  (it looked like a successful turn that said nothing). Now every tool_log
  entry carries its tool name and an explicit `outcome`
  (succeeded|denied|cancelled|error, `loop._tool_outcome`), the trace's
  `tool_calls` carry `outcome` + derived `ok`, the Trace itself gains
  `outcome` (succeeded|budget_exhausted), and exhaustion sets
  `State.exhausted` with an honest final message.
* **Tool results promoted to user-role messages (7).** `_messages` folded tool
  output into user-role "[tool result]" envelopes for every server. They now
  keep the `tool` role (with the tool's name); the envelope is a FALLBACK the
  client applies only when a server actually rejects the role (400 → sticky
  `native_tool_messages=False`, same degrade-and-retry pattern as tool
  definitions — which also fixes a latent infinite-retry when a server 400s
  the prompt-embedded form too). `_last_user_turn` already handled both
  forms (T-015 parametrization).

Acceptance tests: assembled messages contain native tool-role entries and
never a `[tool result]` user turn; cancelled/succeeded/exhausted labels read
back from the trace.

**Round-2 scoreboard:** 7/7 findings verified real (unlike round 1, where most
"gaps" were already built). All fixed with acceptance tests; suite 799
passed / 2 xfailed after.

**Process note (both rounds):** the sandbox refreshed between sessions and git
history reset to the initial commit while the working tree persisted; the
T-001…T-017 tree is recommitted (restore commit below/above), and this
session's fixes land as per-ID commits on top.

## Questions for the user

* **Q1 · Which embedder should the "top-level" bar be measured against?** The default
  hashing embedder has no semantics. The optional tier is `sentence-transformers`, and
  the model must be downloaded on the user's machine. Which model fits the 16 GB laptop
  (doc 05 lists the candidates)?
* **Q2 · Is a generation model on the laptop (Lane A) available now?** If yes, T-007 can
  run against a real model. The metric is only a measure of intelligence once it is run
  that way: the mock score of 1.0 proves plumbing, nothing more.
* **Q3 · Which direction is the "novel" bet?** The blueprint (doc 00) lists the
  innovations. The retrieval-supervised recurrent state (doc 13 §4.1, the RSC) is the most
  distinctive, but it is the furthest from working, and T-010 shows its ablation harness
  has not yet been checked end to end. The concept lexicon and the grounded-answer metric
  are the nearest. Pick the one to fund first.
* **Q4 · Who writes the blind holdout (T-002)?** It must not be the agent that changes the
  retrieval code. If another agent is available, point it at T-002. If not, the user can
  write 40 paraphrases in an afternoon.
