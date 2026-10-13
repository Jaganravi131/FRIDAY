# BOARD — shared work queue

Read `AGENTS.md` first (lanes, gates, the loop). Every change names an item here.
Statuses: `open` · `in progress (lane, branch)` · `blocked (why)` · `done (numbers)` ·
`partial (what remains)` · `QUESTION`.

Last updated: 2026-10-13 (second session), by the agent on branch `arena/aee71f50-friday`.

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

## Open — ordered by leverage

### T-002 · Blind holdout set for retrieval (Lane E)  — `open`, **highest priority**
Every number in doc 20 is weaker than it looks, because the lexicon was tuned on the
same agent's probes. Write **40+ paraphrase queries and 10+ unanswerable queries**, with
the expected predicate, **in a file no retrieval agent edits**. Write them *before*
looking at the current results. Report recall@5 before and after the next change.
*Not done in this session on purpose:* an agent that wrote the holdout and then graded
the change would be the same agent twice. It needs a different agent or the user.
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
