# Contributing

FRIDAY is a personal, context-aware AI agent: local-first, ₹0 cloud budget, Markdown as
truth, SQLite as a derived index. It is designed to run on a 16 GB Windows laptop with
no CUDA, and to be readable by the one person whose memory it holds.

This document is about **how to change it without breaking the things that are easy to
break silently**. Read `SECURITY.md` too — several invariants there are security
invariants, and they fail in the permissive direction.

---

## Start here

```bash
pip install pytest pytest-subtests        # that is the whole dependency list
python -m pytest tests/ -q                # ~455 tests, ~8 seconds
python scripts/phase0_exit_test.py        # the 12-condition Phase 0 gate
```

No numpy, no torch, no virtualenv required for the core. `friday/rsc/` gates torch
behind an import check and its tests skip when it is absent — that is deliberate, not
an oversight.

To actually talk to it, with no model downloaded:

```bash
export FRIDAY_ROOT=/tmp/friday-scratch     # never point this at the repository
python -m friday seed && python -m friday build
python -m friday ask "what is my monthly rent" --mock
```

The ledger block it prints is the single most useful thing in the repository. Leave it
on.

---

## The laws

These are in `docs/architecture/00-BLUEPRINT.md` with their reasoning. They are not
style preferences; each one is a failure that was observed or calculated.

| Law | Rule | Enforced by |
|---|---|---|
| 1 | Proactivity is a heartbeat, not a cron job | `friday/agent/` scopes |
| 2 | Do not summarise conversation history by default — cap, offload, evict first | `ledger/ladder.py`, the 6-rung ladder |
| 2b | Never ask a lossy state to be a database | `eval/`, the locked suite |
| 2c | Recall is a property of a checkpoint, not an architecture | `scripts/needle_test.py` after every training run |
| 3 | Just-in-time retrieval beats pre-packing | pointers, `read_artifact` |
| 4 | Retrieval without a reranker is noise | `retrieval/rerankers.py` |
| 5 | Enforce permissions in the data layer, never in the prompt | `agent/policy.py` |
| 6 | The agent must not be able to edit its own evaluator | `eval/` is read-only to the agent |
| 7 | Memory files are human-readable and human-editable | `memory/watcher.py`, confidence `user_edit`=1.0 |
| 8 | One language for the brain, one process for the truth | stdlib + sqlite3, no second store |

**Law 5 is the one new contributors break most often**, by adding a rule to `AGENTS.md`
instead of a check in `policy.py`. A prompt line is a suggestion to a model that is
currently reading an attacker's text. If it must happen, it must happen in code.

---

## Non-obvious rules that will bite you

**Markdown is truth; SQLite is derived.** `rm -rf artifacts && python -m friday build`
must restore a correct system from the `.md` files alone. If your change makes the
database the only copy of something, it is wrong. Exit test #1 checks exactly this.

**The core is stdlib + sqlite3.** Adding numpy or torch to anything outside
`friday/rsc/` breaks the deployment target. CI installs neither on purpose, so a
transitive requirement fails the build rather than passing locally.

**The stable prefix is byte-identical across turns.** `identity` and `senses` form the
cached prefix. Anything that varies per turn — a clock, a random id, a timestamp, a
ticking counter — must go in a JIT slot. Putting one in the prefix invalidates the
provider's prompt cache on every turn and undoes Law 2. Exit test #7 measures
`cache_hit_rate >= 0.85` over 50 turns; that is the test that catches this.

**Retraction is not deletion.** A superseded fact gets `retracted_at`, not `DELETE`.
"What did I used to think?" has to keep working. Bi-temporal means two axes:
`valid_from`/`valid_to` (the world) and `asserted_at`/`retracted_at` (FRIDAY's belief).

**One predicate per concept.** Add natural synonyms to `CANONICAL_PREDICATES` in
`config.py`. Without that, `monthly_rent` and `lease_amount_monthly` become two live
facts about the same thing and the hierarchy starts arbitrating between them.

**`eval/` is read-only to the agent** (Law 6). Anything that lets a run modify its own
score invalidates every result.

**Reporting paths must degrade, never crash.** `as_dicts()`, `summary()`, `printout()`,
`redact()`, `detect()` and `_audit()` all run on partial or malformed data by design.
A `KeyError` in a renderer takes down a turn; use the `_cell()` helper pattern.

**Never use built-in `hash()` for anything reproducible.** It is randomized per process.
Use `friday/util._stable_seed()` (blake2b). This shipped once and made an ablation
ladder unreproducible across runs.

---

## Testing discipline

**Tests import modules at collection time.** Deleting and re-importing `friday.paths`
creates a *second* module object, and whichever one a previously-imported module bound
is the one that keeps writing to the real repository. The doctrine: never delete and
re-import. Mutate the one module in place via `paths.rebind(tmp_path)` from the autouse
`isolated_paths` fixture in `conftest.py`. Every test gets an isolated root whether or
not it asked for one.

**Write the failing test before the fix.** Every significant bug in this repository was
found by a test that failed, not by reading code — including three that a `str.replace`
patch had silently not applied. See `SECURITY.md` §4.

**A test that asserts the good case is a test that passes while the hole is open.**
Security tests here are written attack-first and named for the vector they close.

**Parametrize the false positives too.** `test_ordinary_memory_content_is_not_flagged`
and `test_ordinary_content_is_not_redacted` exist because a control that cries wolf gets
disabled. If you add a detector pattern, add a benign case that must *not* match.

---

## Where things live

```
friday/
  agent/       the loop, the tool contract, the policy engine (Law 5)
  ledger/      the Attention Ledger: slots, the 6-rung compaction ladder
  memory/      Markdown <-> SQLite: compiler, decay, facts, traces, watcher, provenance
  retrieval/   embedders, rerankers, the hybrid FTS+vec pipeline
  security/    the trust boundary: trust, injection, redact   <- read SECURITY.md first
  store/       schema.sql, db helpers (per-statement apply; executescript cannot run it)
  rsc/         the Phase 3.5 operator prototypes (torch-gated)
  llm/         client selection, the mock client
docs/architecture/   14 documents; 00-BLUEPRINT is the index
scripts/       phase0_exit_test.py, needle_test.py, rsc_ablation.py
eval/          the locked suite (Law 6)
tests/         ~455 tests
```

`scripts/` is code that ships with the package. `$FRIDAY_ROOT` is user **data**. They
are not the same directory and must never be conflated — `_scripts_dir()` resolves
relative to the package for this reason.

---

## Making a change

1. Read the relevant doc in `docs/architecture/`. If your change contradicts a doc,
   **update the doc in the same commit**. A doc that quietly disagrees with the code is
   worse than no doc: the next person implements the doc. (This has happened — the
   identity slot budget in doc 04 was stale by 512 tokens, and every fresh install
   shipped with its own personality elided.)
2. Write the test first. Watch it fail for the reason you expect.
3. Make the change.
4. `python -m pytest tests/ -q` — all green, no skips you did not intend.
5. `python scripts/phase0_exit_test.py` — 10 PASS, 0 FAIL.
6. Run the CLI without `--quiet` and look at the ledger block.
7. Commit with the *reasoning*, not just the change. The comments in this repository
   explain why, including the failures that motivated them; keep doing that.

If you add a slot, a tool, a predicate, a sense or a compaction rung, expect to touch
more than one file: `config.py` (predicates, senses), `ledger/slots.py` (`SLOT_ORDER`),
`security/trust.py` (`PROTECTED_TAGS`), `agent/policy.py` (`TOOL_SENSE`,
`ALWAYS_CONFIRM`, `UNATTENDED_DENY`), and `docs/architecture/04`.

---

## Questions worth asking before you open a PR

- Does this make the database the only copy of something?
- Does this put anything per-turn into the stable prefix?
- Does this enforce a rule in a prompt instead of in code?
- Does this delete rather than retract?
- Does this fail silently, or does it tell the user?
- Would a user notice if this broke? If not, is there a test that would?
