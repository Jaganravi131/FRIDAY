# BOARD — shared work queue

Read `AGENTS.md` first (lanes, gates, the loop). Every change names an item here.
Statuses: `open` · `in progress (lane, branch)` · `blocked (why)` · `done (numbers)` ·
`QUESTION`.

Last updated: 2026-10-13, by the agent on branch `arena/aee71f50-friday`.

## Goal

Make FRIDAY a measurably intelligent, personal, and novel assistant, within a ₹0 budget
and a 16 GB laptop. The model's raw intelligence comes from the user's machine (Lane L).
Everything below is about the system around the model, plus the evaluation that proves it.

---

## Done

| ID | Lane | Item | Result |
|---|---|---|---|
| T-000 | Q | Bring the full project onto the session branch (fast-forward from `arena/01a0e981-friday`, 34 commits) | `main` and the session branch had only the README before this |
| T-001 | R | **Concept lexicon** for paraphrase recall — `friday/retrieval/concepts.py`, doc 20 | Committed bench paraphrase recall@5 0.00 → 0.75 (n = 4); overall 0.815 → 0.926; false positives 0.0. Held-out probe 2: 0.25 → 0.50, false positives 0.0. Probe 1 is contaminated (debugging set). Gate ratcheted. |
| T-001a | R | Found and fixed a regression the invariance test caught (set-collapse of two query terms into one concept) | `tests/test_retrieval.py` green |
| T-001b | E | Replaced a test that pinned the paraphrase defect with a ratchet | `tests/test_delivery.py` asserts paraphrase ≥ 0.5 |

---

## Open — ordered by leverage

### T-002 · Blind holdout set for retrieval (Lane E)  — `open`, **highest priority**
Every number in doc 20 is weaker than it looks, because the lexicon was tuned on the
same agent's probes. Write **40+ paraphrase queries and 10+ unanswerable queries**, with
the expected predicate, **in a file no retrieval agent edits**. Write them *before*
looking at the current results. Report recall@5 before and after the next change.
*Acceptance:* a holdout that Lane R can run but not read or edit, plus its first number.

### T-003 · Semantic embedder comparison on the user's machine (Lane L, user runs)  — `blocked (weights)`
Run `pip install -e ".[embed]"` then `python -m friday bench --record` with a real
embedder. Keep the hashing baseline as a row beside it. Doc 20 §6 explains why the
comparison is the result worth keeping. *Blocked here:* model weights cannot be downloaded
in this sandbox. Agents may prepare the script; the user runs it.

### T-004 · Lexicon coverage gaps (Lane R)  — `open`
From doc 20 §5: possessive `'s` handling, a "what time" intent cue, and the words
*program*, *usually*, *have*. **Each word gets a test first**, a benign negative case, and
a check that no existing bench number drops. Do not add a word just because a probe missed it.

### T-005 · Real defects hiding in lint output (Lane Q)  — `open`
`ruff check .` reports 332 findings. Most are style. The ones to fix first are the ones
that can hide a bug: `F841` (unused variable, which can mean a value was computed and
dropped), `F401` (unused import, which can mean a missing call), and `S110` (try/except
pass, which swallows an error). *Acceptance:* each fix comes with the test that would
have caught the bug, or a note that it is only style.

### T-006 · README and status drift (Lane Q)  — `open`
`README.md` still says "architecture complete, implementation not started". The code
says Phase 0 works. Also: `CONTRIBUTING.md` says "~455 tests", the suite has 642+. Fix
the counts in the same change that adds tests, so they stop drifting.

### T-007 · Grounded-answer evaluation for the agent loop (Lane E + G)  — `open`
The Phase 0 gate proves the agent *degrades correctly*. It does not prove an answer is
*grounded*: that every claim in a reply is backed by a retrieved fact, and that a reply
with no backing says "I don't know". Define the metric first (fraction of factual claims
citing a shipped fact; refusal rate on unanswerable queries), run it with the mock model,
and record it in `BENCHMARKS.md`. *Acceptance:* the metric is recorded, and the gate fails
on regression.

### T-008 · Phase 0 SKIP conditions (user runs)  — `blocked (hardware)`
Condition 11 (RSS after an hour idle, with a real model loaded) and condition 2 need the
user's machine. Note: CI's comment says two conditions skip, but the script reports one.
Reconcile the comment with the script in the same change.

### T-009 · Multi-agent coordination — start using the board (Lane Q)  — `in progress`
This file and `AGENTS.md` are the first version. The next agent to pick up work should
claim an item here before editing code, and should write its own result into the Done
table. *Acceptance:* two items closed by two different agents, with the board the only
handoff channel.

---

## Questions for the user

* **Q1 · Which embedder should the "top-level" bar be measured against?** The default
  hashing embedder has no semantics. The optional tier is `sentence-transformers`, and
  the model must be downloaded on the user's machine. Which model fits the 16 GB laptop
  (doc 05 lists the candidates)?
* **Q2 · Is a generation model on the laptop (Lane A) available now?** If yes, T-007
  can be run against a real model, not just the mock. That would turn the grounded-answer
  metric from a harness into a measurement of intelligence.
* **Q3 · Which direction is the "novel" bet?** The blueprint (doc 00) lists the innovations. The
  retrieval-supervised recurrent state (doc 13 §4.1, the RSC) is the most distinctive, but
  it is the furthest from working. The concept lexicon and the grounded-answer metric are
  the nearest. Pick the one to fund first.
