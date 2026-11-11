# AGENTS.md — how to work on FRIDAY when more than one agent is in the repo

This file is the operating manual for **any** agent, human or AI, changing this
repository. Read it, then `CONTRIBUTING.md` (the laws and the non-obvious rules), then
`SECURITY.md`. Where this file and those disagree, the laws win and this file is wrong:
fix it in the same change.

The shared work queue is **`docs/coordination/BOARD.md`**. If it is not on the board, do
not start it.

---

## 1 · The goal, stated honestly

FRIDAY aims to be a top-level personal assistant that is **accurate, personal, and
novel**. The honest limits, which every agent must keep in mind:

* **The language model is not in this repository, and cannot be.** Weights cannot be
  downloaded from the sandbox, and the ₹0 budget rules out cloud models. Model
  intelligence comes from the user's machine (Lane A) and Colab (Lane B). The model
  track is documented in `docs/architecture/19-model-track-execution.md`.
* **What agents *can* raise here is the system around the model:** retrieval, memory,
  the agent loop, safety, evaluation, and coordination. Each of those is measurable
  without weights, and each is gated.
* **"Top-level" is claimed only with numbers.** Anything called better must move a gate
  in `BENCHMARKS.md` or a row on the board. Adjectives do not count.

## 2 · Lanes (who owns what)

Each lane is a role. One agent may hold several roles over time, but **a change touches
one lane at a time**. A change that crosses lanes must say so on the board.

| Lane | Owns | Gate it must keep green | May NOT |
|---|---|---|---|
| **R · Retrieval** | `friday/retrieval/`, `friday/store/db.py` (FTS), `friday/memory/facts.py` aliases | `friday bench --gate`, `tests/test_retrieval.py`, `tests/test_concepts.py` | edit the bench corpus or query sets (that is Lane E) |
| **M · Memory & Ledger** | `friday/memory/`, `friday/ledger/`, `friday/store/` | `scripts/phase0_exit_test.py` (11 PASS), `tests/test_bi_temporal.py`, `tests/test_ledger.py` | make the database the only copy of anything (Markdown is truth) |
| **G · Agent, tools & connectors** | `friday/agent/`, `friday/connectors/`, `friday/cli.py`, `friday/serve.py` | `tests/test_policy_and_agent.py`, `tests/test_connectors.py`, `tests/test_cli.py` | add a write path without a confirm gate and an audit row |
| **S · Security & privacy (review)** | `friday/security/`, `friday/agent/policy.py`, `SECURITY.md` | `tests/test_security.py`, `tests/test_delivery.py` | — S has **veto** on any change that moves a check from code into a prompt (Law 5) |
| **E · Evaluation (holdout keeper)** | `friday/bench.py` corpus and query sets, `BENCHMARKS.md`, `eval/` | `friday bench --gate` (it compares to the recorded baseline) | be the same agent that tunes the code being measured, for a claim it then makes |
| **Q · Quality & docs** | `README.md`, `docs/`, lint debt, CI | `scripts/ci_pytest.py`, `ruff check` on touched files | change behaviour while claiming a refactor |
| **L · Model track (human-run)** | `friday/tune/`, `friday/rsc/`, `scripts/needle_test.py` | `scripts/tune_verify.py`, `scripts/needle_test.py` after every training run | pretend a model ran here: GPU and weight steps happen on the user's hardware |

**Law 6 applies to agents, not just to code:** an agent must not change the evaluator
that scores its own change. Lane R may not edit `friday/bench.py` queries to make a
result pass. A new query goes in through Lane E, written **before** the code it tests.

## 3 · The loop every change follows

1. **Claim** a board item (`T-###`). Set it to `in progress` with your lane and branch.
2. **Write the failing test or the probe first.** For retrieval, that means a
   held-out probe that is not used for tuning. For security, an attack-first test.
3. **Change the code** in your lane only.
4. **Run the gates** (section 4). Record any number that moved.
5. **Commit** with the board ID first: `T-001: concept lexicon — coverage over concepts`.
6. **Update the board**: status, the numbers, and anything left open.

## 4 · The gates (run all of these before you say "done")

```bash
pip install pytest pytest-subtests ruff          # the dev extras; the core needs nothing
python scripts/ci_pytest.py                      # the full suite (642+ tests), exit 0
PYTHONUTF8=1 python scripts/phase0_exit_test.py  # 11 PASS · 0 FAIL · 1 SKIP is the baseline
python -m friday bench --gate                    # retrieval regression gate, exit 0
ruff check <files you touched>                   # touched files only; the repo has known debt
```

`BENCHMARKS.md` is written only by `python -m friday bench --record`. **Never edit its
numbers by hand.** Record only after the gate passes and you have explained the change.

## 5 · Rules that agents break most often

* **Adding a rule to this file instead of a check in code.** A line in a prompt is a
  suggestion to a model that may be reading an attacker's text. If it must hold, enforce
  it in `policy.py` or the data layer (Law 5).
* **Tuning to the benchmark.** Four queries cannot measure a paraphrase fix. Report the
  number *and* say how many queries it rests on. A held-out set is the only evidence that
  generalises.
* **Hard-coding the answer.** A lexicon entry that exists only because a bench query
  failed is overfitting. Ask whether the word belongs in the language, not the corpus.
* **Removing a token that was acting as a penalty.** A stray or uncovered term lowers
  scores. Removing it raises every score, including the unanswerable ones. Dropping
  `s` from `flat's` raised bench false positives from 0 to 0.2 (doc 20 §5a). Any change to
  term counting must run the false-positive gate, not just the recall numbers.
* **Lowering a floor to buy recall.** Law 4: an empty list beats noise. Any change that
  raises `false_positive_rate` above 0 is rejected, however good the recall looks.
* **Wrapping a crash in a fallback that hides it.** Degrade loudly (see `SECURITY.md`).
* **Assuming `with sqlite3.connect(...)` closes the connection.** It does not: the context
  manager only commits. The open handle makes temp-directory cleanup fail on Windows
  (`WinError 32`), while Linux passes silently, so the suite is green locally and red in
  CI. Close the connection in a `finally`, as `friday/bench.py` does. The Linux guard
  `test_the_harness_leaves_no_open_handle_on_its_temp_root` checks for this.
* **Using `hash()` for anything reproducible.** Use `friday.util._stable_seed`.
* **Committing user data.** `memory/`, `soul/`, `artifacts/` and `capture/` stay out of
  git. CI checks this; do not work around the check.

## 6 · Git and branches

* Work only on the branch your session names; do not copy a branch name from an
  earlier session's notes. Do not create or push to other branches.
* Commit small and often, one board ID per commit. Do not rewrite history on a branch
  someone else may have pulled.
* CI runs on every push (`ubuntu-latest` and `windows-latest`, Python 3.11–3.13). A red
  Windows job is a real failure: the Windows machine is the primary deployment target.

## 7 · When two agents disagree

Put the disagreement on the board as a `QUESTION` item with the evidence (a gate
number, a failing test, a file reference). Do not resolve it by quietly reverting the
other agent's change. A question that touches a law is decided by the laws; a question
about a number is decided by a measurement that both agents can re-run.
