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

## 5 · What it does not cover (the honest gap)

Probe 2's misses are vocabulary the lexicon lacks, not ranking mistakes:

* "what do I usually have to **drink**" — `drink` is in the cluster, but the fact is
  `coffee_order`, and `usually` / `have` do not reach it.
* "what program do I **type my code** in" — `program` is not a cluster word here.
* "who is my **flat's owner**" — the possessive `flat's` does not stem to `flat`.
* "what time is the **scrum**" — the question asks for a time, and `standup_day` stores
  a day and a time together.

Each is a lexicon or tokenisation gap. None is solved by lowering the floor, and each
should be handled by the owner of the retrieval lane, with a test added first.

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

## 5b · The gaps, closed where they could be closed (2026-10-16, T-004)

Board item T-004 took the four §5 gaps with the rule *test first, benign negative, no
bench number may drop*. Two of the four closed, one was rejected by its own negative,
and the process found a defect nobody had listed. Scores below were measured on this
corpus, floor 0.45, before each change.

**Shipped — the "what time" cue, thing-scoped.** `standup_day: Tuesday 09:30 IST` never
contains the word "time", so "my standup time" (the bench's one noun-class miss, 0.380)
and "what time is the scrum" (0.380) were floored no matter how good recall was. Weekday
stems and clock times (`\d{1,2}:\d{2}`) now carry one concept key, `c:temporal`, on the
fact side, and the query stem `tim` maps to it — **only when the query names something
else**. A bare `time` is a mass noun: "how much time do I have" would otherwise be
answered by any schedule fact at coverage 1/1. The cue lives in the reranker, not the
recall layers, because the miss is a coverage miss: recall *finds* the fact, the floor
drops it.

**Shipped — `usually`/`usual` as filler, and a stopword leak they exposed.** "usually"
says how often, never which fact — the same job "just" and "really" already do — so both
surface forms joined `CONTENT_STOPWORDS` (both, because `fts_terms` compares raw words
while the reranker compares stems). Measuring why "what do I usually have to drink"
still missed then found the real defect: **`have` was never filler in the reranker at
all.** `CONTENT_STOPWORDS` holds surface forms, the scorer subtracts stems, and
`stem("have") == "hav"` — so `have`, `were`, `some`, `really`, `please`, `actually`
and friends all counted as content terms (§5's "usually / have do not reach it" was half
right: `usually` reached it as `usual`, `have` reached it as `hav`). The scorer now
subtracts a stem-normalised view derived from the one canonical list, pinned by an
equality test ("what do I have to drink" scores exactly what "what do I drink" scores).

**Shipped — the possessive strip, cluster-conditional.** `strip_possessives` removes
the clitic only for lexicon vocabulary ("landlord's" → "landlord"), where stripping is
pure morphology. Unknown possessives ("sister's") keep the residue, and that asymmetry
is the guard: a blind strip scores "what is my sister name" 0.429 against the manager
fact — under the floor by 0.021, over it once the ×1.06 multi-path nudge applies, which
is precisely the §5a 0.2 false-positive rate. Irregular plurals ("bosses'", where
`stem(bosses) == "boss"` but `stem(boss) == "bos"`) also keep the residue: conservative
is the safe direction.

**Rejected — `program`.** Mapping it into the editor cluster lifts "what program do I
type my code in", but the same mapping scores "what is my training program" **0.589**
against `editor: neovim` — a wrong answer shipped for an unrelated question. `program`
names TV schedules and training plans as readily as editors; ambiguous words stay out.
The gap it leaves (0.326) stays open for the embedder tier (T-003) to close by meaning.

**Still open — "flat's owner" and "drink in the morning".** With the conditional strip,
"who is my flat's owner" reaches 0.43: `flat` maps to the rent concept, which the
landlord fact does not carry. The two ways to close it were measured and both ship wrong
answers: merging the rent and landlord clusters, or adding a "flat owner" alias to
`landlord_name`, puts the landlord fact at **0.739 for "my rent"** — noise on every
rent question. Pinned as a strict xfail (`test_flat_owner_probe_reaches_its_fact`) so
closing it is a decision, not a drift. "what do I drink in the morning" (0.403) stays
missed on purpose: `morning` is real content — a daily log can be *about* a morning —
and stopwording it to lift one bench query is exactly what this lexicon exists not to
do. That one is the embedder's job too.

| Measure | Before | After |
|---|---|---|
| recall@1 / recall@5 / MRR | 0.9259 | **0.963** |
| noun recall@5 | 0.9167 | **1.0** |
| interrogative recall@5 | 1.0 | 1.0 |
| paraphrase recall@5 | 0.75 | 0.75 (one documented miss remains) |
| false positives | 0.0 | **0.0** |
| interrogative gap | −0.0833 | **0.0** |

Tests: `tests/test_lexicon_gaps.py` — 14 cases plus the strict xfail, every fix pinned
next to its Law 4 negative.

## 6 · Next levers, in order

1. **Blind holdout (T-002).** Without it the numbers above cannot be trusted past the
   first cycle.
2. ~~**Possessive and intent handling**~~ — done 2026-10-16, see §5b for what shipped,
   what was rejected, and what stays open on purpose.
3. **Semantic embedder on the user's machine (T-003).** Run `friday bench` with the
   `embed` extra and record a second baseline. This is the lever the lexicon is a
   stand-in for, and the comparison between the two is itself the result worth keeping.
