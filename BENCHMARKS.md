# BENCHMARKS

Recorded by `python -m friday bench --record`. **Do not edit the numbers by
hand** — the gate in `friday/bench.py` parses this table and compares against
it, so a hand-edited baseline is a gate that no longer measures anything.

## Why this file exists

Law 10: post-training can quietly ruin long-range recall, so every fine-tuning
run must re-run the needle test. As advice that gets skipped, because a merely
*worse* answer raises no exception. `friday bench --gate` makes it a nonzero
exit instead. `scripts/needle_test.py` still measures a real model's long-range
recall; this measures the retrieval layer, model-free, in about a second, so it
can run on every commit.

## Baseline

- recorded: 2026-10-01T03:58:15Z
- embedder: `hashing` · reranker: `lexical` · floor: 0.45
- corpus: 13 facts · 27 queries · 5 unanswerable · 0.072s

| metric | value |
|---|---|
| `recall_at_1` | 0.8148 |
| `recall_at_5` | 0.8148 |
| `mrr` | 0.8148 |
| `provenance_complete` | 1.0 |
| `interrogative_gap` | -0.0833 |
| `false_positive_rate` | 0.0 |

### Recall by query class

| class | recall@5 | n |
|---|---|---|
| interrogative | 1.0 | 11 |
| noun | 0.9167 | 12 |
| paraphrase | 0.0 | 4 |

## Reading these numbers

- **`interrogative_gap` is recall on "my manager" minus recall on "who is my
  manager"** — the same fact, asked the way a person asks it. Anything above
  0 means FRIDAY answers "I don't have anything in memory about that" for
  something it does have, so the user cannot predict which phrasings work.
  **This defect is CLOSED.** It was three stopword lists that disagreed: the
  one the reranker scored against was missing `when`, `where`, `why`, `how`,
  `which` and `who`, and because coverage divides by the number of query
  words, one unstripped question word halved the score of a perfect match and
  pushed it under the floor. Unifying them into `db.CONTENT_STOPWORDS` moved
  interrogative recall@5 0.545 -> 1.000 and overall 0.630 -> 0.815. The gate
  keeps it closed; `tests/test_retrieval.py` pins the invariance itself.
- **`paraphrase` is now the known defect, and it cannot be bought with a
  threshold.** Measured: lowering the floor 0.45 -> 0.25 lifts paraphrase to
  0.25, costs one false positive out of five unanswerable queries, and then
  plateaus at 0.25 however low the floor goes. Law 4 and the
  `false_positive_rate` gate both forbid that trade, so the only route is
  better SIGNAL — lever 4 in doc 15 (a real semantic embedder or
  cross-encoder, in the optional dependency tier). A hashing embedder has no
  semantics to find "who do I report to" inside a fact about a manager.
- **`false_positive_rate` must stay at 0.** Law 4: an empty list beats noise.
  A retrieval layer that returns something for an unanswerable query is
  manufacturing raw material for a confabulation, and this is the metric that
  guards the refusal behaviour doc 15 makes its fine-tuning headline.
- **`provenance_complete` must stay at 1.0.** A shipped fact without a source
  quote cannot be shown by `/why`, and `/why` is what the whole trust model
  rests on.
- Scores are environment-dependent: with numpy the embedder is different than
  without it. Re-record the baseline when the environment changes, and say so
  in the commit — a baseline nobody trusts is a gate nobody keeps.
