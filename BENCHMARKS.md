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

- recorded: 2026-10-16T06:16:53Z
- embedder: `hashing` · reranker: `lexical` · floor: 0.45
- corpus: 13 facts · 27 queries · 5 unanswerable · 0.096s

| metric | value |
|---|---|
| `recall_at_1` | 0.963 |
| `recall_at_5` | 0.963 |
| `mrr` | 0.963 |
| `provenance_complete` | 1.0 |
| `interrogative_gap` | 0.0 |
| `false_positive_rate` | 0.0 |

### Recall by query class

| class | recall@5 | n |
|---|---|---|
| interrogative | 1.0 | 11 |
| noun | 1.0 | 12 |
| paraphrase | 0.75 | 4 |

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
- **`paraphrase` was the known defect; a better SIGNAL now closes part of it.**
  Lowering the floor cannot buy it (0.45 -> 0.25 lifts paraphrase to 0.25 and
  costs a false positive). The concept lexicon (`friday/retrieval/concepts.py`,
  doc 20) maps "report" and "manager" to one concept, so "who do I report to"
  reaches a manager fact with no model. It is a hand-written floor, not a
  semantic embedder: a real embedder (doc 15, lever 4) is still the target for
  the vocabulary the lexicon does not cover.
- **2026-10-16 ratchet (T-004, doc 20 §5b):** recall@1/@5/MRR 0.9259 -> 0.963,
  noun recall@5 0.9167 -> 1.0, gap -0.0833 -> 0.0. The lift is the thing-scoped
  "what time" cue — "my standup time" was the one noun miss, floored at 0.380
  because the fact stores "Tuesday 09:30" and never the word "time". Paraphrase
  stays 0.75 with one documented miss ("what do I drink in the morning": the
  residue word `morning` is real content, and stopwording it to lift one query
  is what this lexicon exists not to do). False positives stayed 0.0 through
  both changes — that, not the recall lift, is why they shipped.
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
