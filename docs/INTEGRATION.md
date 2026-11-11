# Integration map: five branches, two products, one fork

> Measured on 2026-10-19 against `origin/*`. Reproduce with `scripts/branch_audit.sh`.
> This document exists because the repository contains **five branch tips that all share one
> merge-base** — `main` at `84be0d5` — and no branch is a git descendant of any other.

---

## 1. Why they look unrelated, and why they aren't

Every branch was cut from the same one-line README. There is **no ancestry** between them:

```text
                         84be0d5  main  (README only)
                              │
   ┌──────────────┬───────────┼────────────┬─────────────────┐
   │              │           │            │                 │
 95e7eb54      01a0e981    friday-dev   aee71f50        ad9dff8f
 (v0.1 HUD)    (33)        (34)         (40)            (5, rebuilt)
   │              │           │            │                 │
   │              └───────────┴────────────┴─────────────────┘
   │                     CONTENT lineage (each a superset)
   │
   └── orthogonal lineage — 11 shared paths, all different content
```

**By content**, the four mature branches form a chain of successive supersets:

| Branch | Files | Python LOC | Tests | Commits | Content relationship |
|---|---:|---:|---:|---:|---|
| `arena/01a0e981-friday` | 117 | 26,161 | 21 files | 33 | base of the mature line |
| `friday-dev` | 119 | 26,715 | 22 files | 34 | `01a0e981` + 1 (synonym fork) |
| `arena/aee71f50-friday` | 127 | 27,314 | 25 files | 40 | superset, +holdout harness |
| `arena/ad9dff8f-friday` | **147** | **31,362** | **34 files** | 5 (rebuilt) | **superset of all three** |
| `arena/95e7eb54-friday` (v0.1) | 50 | 5,172 | 7 files | 1 | **separate product** |

Containment, verified by tree diff:

- `ad9dff8f` ⊇ `01a0e981` — **0 files unique** to `01a0e981`
- `ad9dff8f` ⊇ `aee71f50` — 1 file unique to `aee71f50` (`friday/holdout.py`, since relocated to `friday/commands/`)
- `ad9dff8f` ⊇ `friday-dev` — 2 files unique to `friday-dev` (`friday/retrieval/synonyms.py`, `tests/test_synonyms.py`)
- `95e7eb54` ⟂ `ad9dff8f` — 11 shared paths, **39 files only in v0.1, 136 only in the mature line**

`ad9dff8f`'s history was **rebuilt** (`8160f0e restore: rebuild the tracked tree after a second sandbox
refresh`), which is why it shows only 5 commits ahead. Its *tree* is the largest, but git cannot
fast-forward it onto the others.

---

## 2. Who is better at what

Not a competition — a division of labour. Every row is a real difference, not a preference.

| Dimension | Mature line (`ad9dff8f`) | v0.1 (`95e7eb54`) | Winner |
|---|---|---|---|
| **Memory architecture** | Compiler: episodes → facts → skills → weights, bi-temporal, ledger, distill/gate/promote | SQLite + FTS5 sessions and facts | **Mature** |
| **Retrieval quality** | Concept lexicon, recall@5 **0.963**, paraphrase **0.75**, FP **0.0**, ratcheted gate | Keyword FTS5 | **Mature** |
| **Evaluation** | Bench, blind-holdout harness, grounded-answer harness (fabrication 0/5) | None | **Mature** |
| **Safety engineering** | Secret scanning, retrieved-data trust boundary, injection fencing, audited gateway, bearer always, loopback default, no CORS wildcard | Risk tiers, approval gate, workspace jail, SSRF guard, optional token, **CORS `*`** | **Mature** |
| **Connectors** | Gmail/IMAP, CalDAV, GitHub REST, SMTP (confirm-then-send) | None | **Mature** |
| **Proactive behaviour** | Heartbeat: deadlines, quiet hours, annoyance budget, audit | None | **Mature** |
| **Test depth** | **839 pass, 2 xfail** | 70 pass | **Mature** |
| **Self-improvement** | Dreaming/lesson layer, skills, RSC training track | None | **Mature** |
| **Streaming interface** | One inline HTML form, `/api/ask` — **no SSE, no streaming** | HUD: SSE tokens, live tool timeline, inline approvals, telemetry, voice, mobile | **v0.1** |
| **Provider breadth** | OpenAI-compatible client only | OpenAI-compatible + **Anthropic** + offline simulator | **v0.1** |
| **Tool ergonomics** | memory_*, shell, python, web_read, calendar, email, github | 16 tools incl. `edit_file`, `grep_files`, `math`, `clock`, plugins | **v0.1** |
| **Approval UX** | Policy-level confirm-then-send | Generic risk-tier gate with a blocking browser round-trip | **v0.1** |
| **Packaging / deploy** | stdlib-only, wheel checks, CI (Linux + Windows) | Docker, compose, Makefile, CI 3.11/3.12 | Tie — different needs |

**The split is clean:** the mature line is the **brain and conscience**; v0.1 is the **face and hands**.

---

## 3. `friday-dev` is not drift — it is a decision

`friday-dev` exists to fix the *same defect* as the mature line's T-001, by a different route:

| | `friday-dev` | `ad9dff8f` (T-001) |
|---|---|---|
| Mechanism | curated synonym bridges (`synonyms.py`) | concept lexicon (`concepts.py`) |
| Committed paraphrase | **1.00** (n=4) | 0.75 (n=4) |
| Committed overall | 0.963 | 0.926 |
| Held-out probe 1 (biased) | 0.25 | **0.58** |
| Held-out probe 2 | 0.42 | **0.50** |

Neither is proven to generalise. The board logged this as **T-012 `QUESTION`** and refused to merge
either — correct behaviour. It cannot be settled by argument, only by the blind holdout (T-002).

---

## 4. A git merge is the wrong tool

`git merge-tree` on `ad9dff8f` + `95e7eb54` → **11 conflicted files** — ten `add/add` and one
content conflict, all at the package roots:

```text
.github/workflows/ci.yml   .gitignore   README.md   pyproject.toml
friday/__init__.py         friday/__main__.py         friday/cli.py
friday/config.py           friday/llm/__init__.py     tests/conftest.py
tests/test_cli.py
```

Both trees claim the top-level `friday` package: 111 modules vs 30. A textual merge produces a
chimera, not a product. **This is a graft with a defined seam, not a merge.**

---

## 5. The union: three seams

```text
                 ┌──────────────────────────────┐
                 │  Face  — v0.1 static HUD     │  static/, SSE protocol
                 │  approvals · timeline · voice│
                 └──────────────┬───────────────┘
                                │ /api/chat · /api/approvals
                 ┌──────────────▼───────────────┐
                 │  Gateway — mature serve.py   │  bearer auth, audit,
                 │  loopback default, no CORS * │  /v1/chat/completions
                 └──────────────┬───────────────┘
                                │ Agent.run()
                 ┌──────────────▼───────────────┐
                 │  Core — mature agent loop    │  memory compiler, ledger,
                 │  retrieval, heartbeat, evals │  connectors, safety
                 └──────────────────────────────┘
   v0.1 contributes: HUD, SSE transport, Anthropic adapter, approval UI,
                     file-editing tools, Docker/compose, plugin loader
```

**Seam 1 — transport.** Add SSE to the mature gateway: emit `token`, `tool_call`,
`tool_result`, `approval_required`, `done` frames. v0.1's `engine.py` event schema is the contract;
`static/app.js` already speaks it.

**Seam 2 — providers.** Port `friday/llm/anthropic.py` from v0.1 as a second `Client`
implementation beside `OpenAICompatibleClient`. Small, isolated, no core changes.

**Seam 3 — tools.** v0.1's `edit_file`, `grep_files`, `math`, `clock`, `list_files` register into
the mature tool layer (`friday/agent/tools.py`); the mature `memory_*` tools stay authoritative.
v0.1's own SQLite store is **not** migrated — it becomes a session cache, not a second source of truth.

### What is explicitly *not* ported

- v0.1's `friday/engine.py` agent loop — the mature loop is better (compiler, ledger, safety)
- v0.1's `memory.py` as primary store — mature memory wins on every measured axis
- v0.1's CORS `*` default — the mature gateway's refusal is correct

---

## 6. Sequence

| # | Step | Owner | Why now |
|---|---|---|---|
| 1 | Merge one line to `main` — either the mature line or v0.1 | **You** | `main` is still a one-line README; work is invisible |
| 2 | Open `ad9dff8f → main` as the project of record | You/me | largest, newest, 839 tests, superset |
| 3 | Land SSE on the mature gateway (Seam 1) | me | unlocks the HUD |
| 4 | Port the HUD + Anthropic adapter (Seams 1–2) | me | product face |
| 5 | Register file tools (Seam 3) | me | operator ergonomics |
| 6 | Write the blind holdout — 40 paraphrase + 10 unanswerable | **You** | decides T-012; no agent may grade itself |
| 7 | Resolve T-012, archive the superseded branches | You/me | collapse 5 branches to 2 |

Steps 1, 2 and 6 need a human. **Everything else I can execute.**

---

## 7. The one decision

> **Which tree becomes the trunk?**

The recommendation is unambiguous: **`arena/ad9dff8f-friday` is the project**, and v0.1 is its
interface layer. The mature line has 6× the code, 12× the tests, the safety model, the evidence, and
the research thesis. v0.1 has the only thing a human can actually *use* — and that is a seam, not a
rival.

Merging v0.1 into `main` first would make the repository *look* finished while orphaning 31k lines
of the real work. That is the failure mode this document exists to prevent.
