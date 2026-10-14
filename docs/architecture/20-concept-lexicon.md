# 20 — The concept lexicon: closing the paraphrase gap without a model

> Written 2026-10-13. Implements the first lever in the retrieval track of
> `docs/coordination/BOARD.md` (T-001). Code: `friday/retrieval/concepts.py`.

## 1 · The problem, measured

A fact is stored in the vocabulary of the *record*: `manager_name: Priya Raman`.
A person asks in the vocabulary of *speech*: "who do I report to". The two share no
word, so every lexical recall path misses, and the lexical reranker (which needs literal
coverage of the query's content words) would discard anything recall did find.

`friday bench` measured it before this change:

| Probe | Queries | Paraphrase recall@5 |
|---|---|---|
| Committed bench (`bench.py`) | 4 | **0.00** |
| Held-out probe 2 (written after the lexicon was frozen) | 12 | **0.25** |

Lowering the score floor does not fix this. It lifts paraphrase recall to 0.25 and buys
one false positive, and then plateaus (recorded in `BENCHMARKS.md`).

## 2 · Why a lexicon, not a model

Retrieval must work when no model is reachable: that is the ₹0 and Law 4 constraint, and
the reason the core is stdlib-only. A semantic embedder is still the better long-run
answer (doc 15, lever 4), and it is an optional tier. The lexicon is the **floor the
embedder has to beat**, and it is cheap enough to ship in the default install.

## 3 · The design

Two changes, both driven by the same clusters:

1. **Recall widens.** `search()` appends the cluster-mates of each query term to the FTS
   query. FTS is OR-matched, so this only adds candidates and removes none.
2. **Coverage counts concepts.** `LexicalReranker` maps both query terms and candidate
   terms to a concept key (`c:<cluster>`, or the stem when there is no cluster). "report"
   and "manager" are then the same term.

The reranker still scores against the user's own words. Expansion only decides what is
*considered*, never what is *shipped*. The floor, the false-positive gate and the
provenance gate are unchanged.

### The rules, each pinned by `tests/test_concepts.py`

* **One cluster per word.** A word in two clusters would make the mapping
  order-dependent. Import raises on a stem collision.
* **No cluster is named after a predicate.** The lexicon encodes language, not the bench
  corpus. (This test caught `editor`, renamed `code_editor`.)
* **Generic and ambiguous words stay out.** `pay`, `group`, `type`, `take`, `write`.
  Each would map ordinary questions to the wrong fact, which is a false positive by
  another name.
* **Near-neighbours that name a different thing stay separate.** `dentist` is not
  `doctor`, so "who is my dentist" returns nothing. The lexicon has no entry for it.

### A bug the design had, and the invariance test that caught it

Collapsing query terms to a *set* of concepts dropped a term: "rent" and "monthly" both
map to `c:rent`, the set kept one, coverage fell from 2/3 to 1/2, and the question fell
below the 0.45 floor. The fix keeps duplicate hits on the query side for coverage and
uses sets only for overlap. `tests/test_retrieval.py::test_asking_a_question_...` is the
test that found it, and it is the reason this is recorded here, not just in the diff.

## 4 · Results

| Measure | Before | After | Notes |
|---|---|---|---|
| Committed bench recall@5 | 0.815 | **0.926** | gate ratcheted with `bench --record` |
| Committed bench paraphrase recall@5 | 0.00 | **0.75** | n = 4, so one query moves it by 25 points |
| Committed bench false positives | 0.0 | **0.0** | must stay 0 |
| Held-out probe 2 recall@5 | 0.25 | **0.50** | fresh, run once; see caveat |
| Held-out probe 2 false positives | 0.0 | **0.0** | 3 unanswerable queries |
| Phase 0 exit gate | 11 PASS | **11 PASS** | unchanged |

### How much to trust these numbers

* **The committed paraphrase set has 4 queries.** 0.75 is three of four. Treat it as a
  smoke test, not a measurement.
* **Probe 1 is not held out.** It was written first, then used to debug the lexicon. Its
  numbers (0.17 → 0.58) are an upper bound on the effect, not an estimate.
* **Probe 2 is nearly held out.** It was run once, after the lexicon was frozen, and
  nothing was tuned against it afterwards. But two of its words (*appraisal*, *tongue*)
  were already in the lexicon, so it is an optimistic estimate too.
* **The real generalisation number needs a blind set.** One that a different agent writes,
  before the change, and that the retrieval agent cannot read. That is board item T-002.

## 5 · What it does not cover, and what now closes three of the four

Probe 2's misses were four mechanisms, not four ranking mistakes. Each now has a
test first (`tests/test_concepts.py`) and a fix in the layer where the mechanism lives:

* **Possessive `'s`** — *fixed.* `util.strip_possessives` removes the clitic on the
  query and candidate sides before tokenising, so `flat's` → `flat`, `what's` → `what`.
  The stray `s` no longer counts as an uncovered term. It exposed a leak (§5a), which
  needed the next fix.
* **Generic field words** — *fixed alongside.* `name` / `number` / `date` name a kind of
  field, not a subject; every `*_name` predicate carries `name`. The reranker now drops
  them from coverage when the query has something more specific, and keeps them when
  it has nothing else ("what is my name" still works). Pinned on a family of five
  unanswerable subjects, not one case.
* **Frequency adverbs** — *fixed.* `usually`, `normally`, `generally`, `often`,
  `sometimes` say how often, not what. They were failing twice: they were not in
  `CONTENT_STOPWORDS`, and the reranker checked stopwords *after* stemming, so
  "usually" → "usual" never matched. The reranker now filters on the surface word,
  matching `fts_terms`, as its own docstring already required.
* **Clock time** — *fixed.* `concepts.text_concepts` gives any candidate holding a
  clock time (09:30, 10am) the `time` concept, so "what time is the scrum" is covered
  by a fact that stores a day and a time. Decimals and bare numbers are not clock times.

What is still open:

* `program` / `programme` is in the editor cluster, with a benign negative pinning the
  TV reading ("what program is on TV tonight" must not clear the floor on it alone).
* None of this is held-out evidence. The cases came from probe 2's misses, so they show
  the mechanism is fixed, not that retrieval generalises. That needs the blind set (T-002).

Result on the committed bench: `my standup time` moved from a miss to a hit, so
recall@1/@5 and MRR rose 0.9259 → 0.963. Paraphrase recall@5 is unchanged at 0.75 (n=4).
False positives stayed 0.0. Exactly one query moved; the per-query diff is the check.

Each gap is a lexicon or tokenisation gap. None is solved by lowering the floor.

## 5a · A fix that was tried and reverted

*Possessive residue.* "who is my flat's owner" tokenises to `flat`, `s`, `owner`. The stray
`s` counted as an uncovered query term, and the score was 0.35 against 0.83 for "who is my
landlord". Dropping single-character tokens in the reranker (as `fts_terms` already does)
lifted that case, and **raised `false_positive_rate` from 0.0 to 0.2** on the bench. The
`s` was pulling down the scores of unanswerable questions. Their scores were too high, and
the floor was doing the job the residue did by accident. The change was reverted. The
gate caught it, as designed.

The correct fix is a tokeniser change that handles possessives in both layers, measured
against the false-positive gate. Do not drop a token to raise a score without that check.

**Update.** That tokeniser fix has now landed (`strip_possessives`), and it did raise the
false positive exactly as warned: "what is my sister's name" cleared the floor for the
manager fact, because `name` matches every `*_name` predicate. The fix was the generic
field-word rule (§5), not a higher floor. Gate stays at 0.0. The lesson survives: a token
that looks like noise may be load-bearing, and the gate is how you find out which.

## 6 · Next levers, in order

1. **Blind holdout (T-002).** Without it the numbers above cannot be trusted past the
   first cycle.
2. **Blind validation of the §5 fixes.** They repair known mechanisms; only T-002 can say
   whether that is the same as getting better. Do not add another lexicon word until it exists.
3. **Semantic embedder on the user's machine (T-003).** Run `friday bench` with the
   `embed` extra and record a second baseline. This is the lever the lexicon is a
   stand-in for, and the comparison between the two is itself the result worth keeping.
