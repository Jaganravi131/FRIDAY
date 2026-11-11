# The blind holdout (T-002) — protocol and file format

> Law 6: FRIDAY may edit itself. It may never edit its examiner.

## Why this directory exists

Every paraphrase number in `BENCHMARKS.md` is weaker than it looks: the concept
lexicon (doc 20) was tuned by the same agent that wrote the committed bench
queries. A held-out probe written *after* freezing the lexicon helped, but two of
its words were already in the lexicon, and it was still written by the same
agent. The blind holdout is the fix: **40+ paraphrase queries and 10+
unanswerable queries, written by someone who has not looked at the retrieval
code's results, before the next retrieval change is made.**

## Who may do what

| Role | May do | Must not do |
|---|---|---|
| Holdout author (the user, or an agent that has NOT touched `friday/retrieval/`) | Write the queries, choose expected predicates | Look at retrieval results while writing |
| Retrieval agent (Lane R) | `python -m friday holdout <file>` | Read, edit, or write the holdout file; tune against per-query results |
| Anyone | Record aggregate numbers on the board | Publish the queries while the holdout is in use |

The runner itself enforces the retrieval agent's side as far as code can: it
prints **class aggregates only**, and validation errors name the case *index*
and the expected predicate, never the query text. The rest is process — the
file's author decides where it lives (outside the repository is fine; pass any
path).

## Writing the holdout

Write it in one sitting, from memory of your own life-questions, BEFORE looking
at any retrieval output. For each answerable query, pick the `expected`
predicate from the bench corpus:

`manager_name, team_name, standup_day, review_cycle, lease_amount_monthly,
lease_end_date, landlord_name, blood_group, medication, doctor_name,
coffee_order, editor, language_mix`

Paraphrase is the point: ask the way you would say it out loud, not the way
FRIDAY stores it ("who do I report to", not "my manager"). For the unanswerable
set, ask real questions whose answers are NOT any of the thirteen facts above
(a dentist, a sister, a flight — things you have never told FRIDAY).

## File format

JSON, one object:

```json
{
  "suite": "blind-holdout-v1",
  "author": "your name, and a line saying you wrote it blind",
  "cases": [
    {"query": "REPLACE WITH YOUR OWN PHRASING", "expected": "manager_name"},
    {"query": "REPLACE WITH YOUR OWN PHRASING", "expected": "coffee_order", "class": "paraphrase"},
    {"query": "REPLACE WITH AN UNANSWERABLE QUESTION", "expected": null}
  ]
}
```

* `expected: null` marks an unanswerable query — anything FRIDAY returns for it
  is a false positive (Law 4).
* `class` is optional and defaults to `"paraphrase"`; use it if you want to mix
  noun-phrase or interrogative phrasings and see them broken out.

## Running it

```
python -m friday holdout /path/to/blind-v1.json
python -m friday holdout /path/to/blind-v1.json --json
```

The first number has no baseline — a low score is a measurement, not a failure.
To measure a change: check out the commit *before* the change, run the holdout,
record the aggregates on the board; then check out the change, run again, record
both. **Report recall@5 before and after, and the false-positive rate.** Once a
holdout has been used to grade a change, it is spent — write the next one blind
again.
