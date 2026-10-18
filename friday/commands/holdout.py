"""`friday holdout` — run a BLIND retrieval holdout, and report aggregates only.

WHY THIS EXISTS (board item T-002, doc 20 §4)
---------------------------------------------
Every number in the committed bench is weaker than it looks, because the concept
lexicon was tuned by the same agent that wrote the bench queries. The blind
holdout breaks that loop: it is written by someone who has NOT seen the
retrieval code's behaviour — a different agent, or the user — and it is written
BEFORE the next change is made.

This command is the instrument. Its design rule is that a retrieval agent may
RUN the holdout without being able to tune against it:

* per-query results are never printed — class aggregates only;
* validation errors name the case index and the expected predicate, never the
  query text;
* `bench.score_cases` is the shared scorer, so the committed set and the blind
  set are measured by exactly the same code.

Law 6: FRIDAY may edit itself, but never its examiner. The holdout file IS the
examiner — the agent that changes retrieval code must not read, edit, or write
it. See `eval/holdout/README.md` for the protocol and the file format.

Exit codes: 0 the run itself (a low score is a MEASUREMENT, not a failure —
the first number has no baseline to fail against), 2 the file could not be
loaded or is malformed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The T-002 minimum: below this the number is a smoke test, not a measurement.
MIN_PARAPHRASE = 40
MIN_UNANSWERABLE = 10


def _load(path: Path) -> tuple[str, str, list[tuple[str, str | None, str]]]:
    """Read and validate a holdout file. Raises ValueError with messages that
    never contain a query's text — the phrasings belong to the holdout's author,
    and an error trace is not a place to leak them to whoever is running it."""
    from ..bench import CORPUS

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e.strerror or e}") from e
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a JSON object")

    entries = data.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'cases' must be a non-empty list")

    known = {p for facts in CORPUS.values() for p, _, _ in facts}
    cases: list[tuple[str, str | None, str]] = []
    for i, entry in enumerate(entries):
        where = f"{path}: case {i}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be an object")
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{where}: 'query' must be a non-empty string")
        expected = entry.get("expected")
        if expected is None:
            cases.append((query, None, "unanswerable"))
            continue
        if not isinstance(expected, str) or expected not in known:
            # Names the predicate, not the query: a typo in an expected
            # predicate is the author's mistake to fix, and is safe to show.
            raise ValueError(
                f"{where}: expected predicate {expected!r} is not in the bench "
                f"corpus — the author's typo, not a retrieval result"
            )
        cls = entry.get("class", "paraphrase")
        if not isinstance(cls, str) or not cls.strip():
            raise ValueError(f"{where}: 'class' must be a non-empty string")
        cases.append((query, expected, cls))

    suite = str(data.get("suite") or path.name)
    author = str(data.get("author") or "unknown")
    return suite, author, cases


def run(path: Path, *, as_json: bool = False) -> int:
    """Score the holdout and print aggregates. Never per-query results."""
    from ..bench import _write_corpus, score_cases, temporary_root

    suite, author, cases = _load(path)
    answerable = [c for c in cases if c[1] is not None]
    unanswerable = [c for c in cases if c[1] is None]

    with temporary_root() as root:
        import io
        from contextlib import redirect_stdout

        from friday import paths
        from friday.cli import main as cli
        from friday.store import db

        facts = _write_corpus(root)
        # Build chatter stays out of the report: `--json` must be parseable on
        # its own, and the text report prints its own corpus line.
        with redirect_stdout(io.StringIO()):
            cli(["build"])
        conn = db.connect(paths.DB_PATH)
        try:
            res = score_cases(conn, cases)   # verbose stays off, on purpose
            res.facts = facts
        finally:
            conn.close()

    if as_json:
        out = res.as_dict()
        out.update({"suite": suite, "author": author})
        print(json.dumps(out, indent=2))
        return 0

    print(f"FRIDAY blind holdout — {suite}")
    print("=" * 62)
    print(f"  author            {author}")
    print(f"  cases             {len(answerable)} answerable · "
          f"{len(unanswerable)} unanswerable · {res.facts}-fact corpus")
    print(f"  embedder/reranker {res.embedder} / {res.reranker} "
          f"(floor {res.floor})")
    print(f"  recall@1          {res.recall_at_1}")
    print(f"  recall@5          {res.recall_at_5}")
    print(f"  MRR               {res.mrr}")
    print(f"  provenance        {res.provenance_complete}")
    print(f"  false positives   {res.false_positive_rate}  "
          f"(must be 0.0 — Law 4)")
    for cls, d in res.by_class.items():
        print(f"    · {cls:14} recall@5 {d['recall_at_5']}  (n={d['n']})")
    paraphrase_n = res.by_class.get("paraphrase", {}).get("n", 0)
    if paraphrase_n < MIN_PARAPHRASE or len(unanswerable) < MIN_UNANSWERABLE:
        print(f"  ⚠ below the T-002 minimum ({MIN_PARAPHRASE} paraphrase, "
              f"{MIN_UNANSWERABLE} unanswerable): treat as a smoke test")
    print("  aggregates only — per-query results are deliberately not reported.")
    print("    Run on two commits and compare these numbers to measure a change.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="friday holdout",
        description="Run a BLIND retrieval holdout. Aggregates only — the "
                    "holdout's queries stay unread by whoever runs it. "
                    "Format and protocol: eval/holdout/README.md",
    )
    ap.add_argument("file", help="path to the holdout JSON file")
    ap.add_argument("--json", action="store_true", help="machine-readable aggregates")
    args = ap.parse_args(argv)

    try:
        return run(Path(args.file), as_json=args.json)
    except ValueError as e:
        print(f"holdout: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - convenience launcher
    raise SystemExit(main())
