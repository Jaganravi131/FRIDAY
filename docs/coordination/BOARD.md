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
| T-007 | E + G | **Grounded-answer harness** — `friday/grounding.py`, `tests/test_grounding.py` | Mock: grounded accuracy 8/8, fabricated 0/5, misdirected 0. The ratchet is backed by two deliberately bad clients, both caught. **The mock number is a plumbing check, not a measure of intelligence.** Real-model run is pending (T-003 / Q2). |
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
F841 and F401 are done (above). Still open, in order: `S110` try/except/pass (21 sites:
each should log, or carry a comment saying why silence is correct) and `BLE001` blind
`except Exception` (59 sites). Fix a site only with a test that would catch a regression
in what it swallows. Style-only rules (`I001`, `UP035`, `FURB167`) are low priority.

### T-008 · Phase 0 SKIP conditions (user runs)  — `blocked (hardware)`
Condition 11 (RSS after an hour idle, with a real model loaded) and condition 2 need the
user's machine. CI's comment says two conditions skip, but the script reports one.
Reconcile the comment with the script in the same change.

### T-009 · Multi-agent coordination — start using the board (Lane Q)  — `partial`
`AGENTS.md` and this board are the first version, used by this session. Still to prove:
two different agents closing items through the board as the only handoff channel.

### T-010 · The erase-gate ablation measures nothing (Lane S/G, RSC)  — `open`, **found this session**
`friday/rsc/operators.py`, `EraseThenDelta.step`: the `erase_rank_one=False` branch is a
copy of the `True` branch, so the flag changes no output. Confirmed by running both
(`S` identical). Any ablation that toggles the flag reports zero difference, which means a
published erase-gate ablation would be a null result by construction.
*Pinned, not fixed:* `tests/test_rsc.py::test_erase_rank_one_flag_changes_the_operator` is
a **strict xfail**. The fix needs the intended alternative, and doc 13 does not spell it out
clearly enough to implement it correctly. Once implemented, the strict xfail fails, which is
the signal to remove the marker.
*Acceptance:* the alternative is specified in doc 13, implemented, and the xfail removed.

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
