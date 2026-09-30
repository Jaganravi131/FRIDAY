"""`friday bench` — the regression gate for the parts of FRIDAY that can silently rot.

WHY THIS EXISTS
---------------
Law 10 says post-training can quietly ruin long-range recall, and that every
fine-tuning run must re-run the needle test. That is correct advice and, as advice, it
will be skipped — because the person who just spent a weekend on a Kaggle run is the
same person who has to remember to check, and the failure it prevents is invisible. An
answer that is merely *worse* does not raise an exception.

So this turns the advice into a gate: a deterministic benchmark over the retrieval layer
that records a baseline into `BENCHMARKS.md` and exits nonzero when the current run is
worse than the recorded one. Same discipline as the test suite, applied to quality
rather than correctness.

WHAT IT MEASURES, AND WHY THOSE
-------------------------------
Deliberately **model-free**. It exercises Markdown → `build` → index → retrieval, using
whatever embedder and reranker are configured. A benchmark that needs a 12B model to run
is a benchmark nobody runs; this one takes about a second and runs in CI.

  * **recall@1 / recall@5 / MRR** — did the right fact come back, and how high?
  * **interrogative gap** — the difference between noun-phrase queries ("my manager")
    and the same question asked the way a person actually asks it ("who is my
    manager"). This is a real defect that was found by hand while testing the gateway:
    the hashing embedder dilutes on question words, so the interrogative form falls
    below the score floor and FRIDAY answers "I don't have anything in memory about
    that" for a fact it plainly has. Tracking the gap is what makes it fixable rather
    than merely noticed.
  * **paraphrase recall** — "who do I report to" for a `manager_name` fact. Measures
    whether retrieval depends on the user echoing FRIDAY's own vocabulary.
  * **false-positive rate on unanswerable queries** — Law 4 says an empty list beats
    noise, and Law 6 says never fabricate. A retrieval layer that returns *something*
    for a question whose answer is not in memory is manufacturing the raw material for
    a confabulation. This is the metric that guards the refusal behaviour the
    fine-tuning plan in doc 15 makes its headline.
  * **provenance completeness** — every shipped fact must carry a source quote. A
    retrieval result without provenance cannot be shown by `/why`, which is the feature
    the whole trust model rests on.

This does NOT replace `scripts/needle_test.py`, which measures a real model's long-range
recall and is the test Law 10 names. It complements it: needle needs a model and minutes
of wall clock, this needs neither and can therefore run on every commit.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: How much worse than the baseline is tolerated before the gate fails. Retrieval scores
#: are floats and the lexical reranker is deterministic, so in principle zero tolerance
#: would work — but embedder choice is environment-dependent (numpy present or not), and
#: a gate that fails on an unrelated machine gets disabled, which is worse than a gate
#: with a small tolerance.
TOLERANCE = 0.02

BENCH_FILE = "BENCHMARKS.md"

# ── the corpus ─────────────────────────────────────────────────────────────────
# Fixed and synthetic. Never the user's data: a benchmark that reads real memory is a
# benchmark whose scores move when your life does, and which cannot be compared across
# machines or commits.

#: domain -> [(predicate, value, quote)]
CORPUS: dict[str, list[tuple[str, str, str]]] = {
    "work": [
        ("manager_name", "Priya Raman", "my manager is Priya Raman"),
        ("team_name", "infra platform", "I work on the infra platform team"),
        ("standup_day", "Tuesday 09:30 IST", "standup is Tuesday at half past nine"),
        ("review_cycle", "quarterly", "our performance review is quarterly"),
    ],
    "housing": [
        ("lease_amount_monthly", "18000 INR", "my rent is 18000 a month"),
        ("lease_end_date", "2027-03-31", "my lease ends on the last day of March 2027"),
        ("landlord_name", "Suresh Kumar", "my landlord is Suresh Kumar"),
    ],
    "health": [
        ("blood_group", "O positive", "my blood group is O positive"),
        ("medication", "vitamin D3 weekly", "I take vitamin D3 once a week"),
        ("doctor_name", "Dr Anita Iyer", "my doctor is Dr Anita Iyer"),
    ],
    "preferences": [
        ("coffee_order", "filter coffee, no sugar", "I take filter coffee with no sugar"),
        ("editor", "neovim", "I use neovim as my editor"),
        ("language_mix", "Tamil and English", "I mostly speak Tamil mixed with English"),
    ],
}

#: (query, expected predicate, query class)
QUERIES: list[tuple[str, str, str]] = [
    # noun phrase — the easy case, and the one that works today
    ("my manager", "manager_name", "noun"),
    ("my rent", "lease_amount_monthly", "noun"),
    ("my blood group", "blood_group", "noun"),
    ("my team", "team_name", "noun"),
    ("my landlord", "landlord_name", "noun"),
    ("my coffee order", "coffee_order", "noun"),
    ("my editor", "editor", "noun"),
    ("my lease end date", "lease_end_date", "noun"),
    ("my medication", "medication", "noun"),
    ("my doctor", "doctor_name", "noun"),
    ("my standup time", "standup_day", "noun"),
    ("my review cycle", "review_cycle", "noun"),
    # ⭐ interrogative — how a person actually asks. This is the gap.
    ("who is my manager", "manager_name", "interrogative"),
    ("what is my rent", "lease_amount_monthly", "interrogative"),
    ("what is my blood group", "blood_group", "interrogative"),
    ("which team am I on", "team_name", "interrogative"),
    ("who is my landlord", "landlord_name", "interrogative"),
    ("how do I take my coffee", "coffee_order", "interrogative"),
    ("what editor do I use", "editor", "interrogative"),
    ("when does my lease end", "lease_end_date", "interrogative"),
    ("what medication do I take", "medication", "interrogative"),
    ("who is my doctor", "doctor_name", "interrogative"),
    ("when is my standup", "standup_day", "interrogative"),
    # paraphrase — must not require the user to echo FRIDAY's vocabulary
    ("who do I report to", "manager_name", "paraphrase"),
    ("how much do I pay for the flat each month", "lease_amount_monthly", "paraphrase"),
    ("what do I drink in the morning", "coffee_order", "paraphrase"),
    ("where do I write my code", "editor", "paraphrase"),
]

#: Queries whose answer is NOT in the corpus. Anything returned for these is a false
#: positive — the raw material for a confabulation.
UNANSWERABLE = [
    "what is my sister's name",
    "when is my flight to Delhi",
    "what is my car registration number",
    "who is my dentist",
    "what is my gym membership number",
]


@dataclass
class Result:
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    by_class: dict[str, dict[str, float]] = field(default_factory=dict)
    interrogative_gap: float = 0.0
    false_positive_rate: float = 0.0
    provenance_complete: float = 1.0
    queries: int = 0
    unanswerable: int = 0
    facts: int = 0
    seconds: float = 0.0
    embedder: str = ""
    reranker: str = ""
    floor: float = 0.0

    #: The metrics the GATE compares. Everything else is informational.
    GATED = ("recall_at_1", "recall_at_5", "mrr", "provenance_complete")

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["interrogative_gap"] = round(self.interrogative_gap, 4)
        d["false_positive_rate"] = round(self.false_positive_rate, 4)
        return d


def _write_corpus(root: Path) -> int:
    """Write the corpus as real Markdown fact files, in the on-disk format."""
    n = 0
    for domain, facts in CORPUS.items():
        lines = ["---", f"domain: {domain}", "version: 1", "---", "",
                 f"# {domain.title()}", "", "## Facts", ""]
        for predicate, value, quote in facts:
            lines.append(f"- {predicate}: **{value}** [bench · user_edit · conf 1.00]")
            lines.append(f'  <!-- src:  "{quote}" -->')
            lines.append("")
            n += 1
        p = root / "memory" / "facts" / f"{domain}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
    return n


def _predicate_of(cand) -> str:
    row = getattr(cand, "row", None)
    if row is not None:
        try:
            return str(row["predicate"])
        except Exception:
            pass
    m = re.match(r"\s*-?\s*([a-z_][a-z0-9_]*)\s*:", getattr(cand, "body", "") or "")
    return m.group(1) if m else ""


def run(*, verbose: bool = False) -> Result:
    """Build a throwaway root, index the corpus, and score the query set.

    ⚠️ Restores `FRIDAY_ROOT` and the `paths` module before returning. It did not
    originally, and the consequence was severe rather than cosmetic: after a bench run
    the whole process believed its memory root was `/tmp/friday-bench-XXXX` — a directory
    the `TemporaryDirectory` context manager had already deleted. Every later operation
    in that process would then read from (or silently recreate) an empty root. A
    benchmark that destroys the caller's environment is worse than no benchmark, because
    it runs on every commit.
    """
    from friday import paths
    from friday.cli import main as cli
    from friday.retrieval.embedders import get_embedder
    from friday.retrieval.pipeline import search
    from friday.retrieval.rerankers import get_reranker
    from friday.store import db

    res = Result()
    saved_env = os.environ.get("FRIDAY_ROOT")
    saved_root = paths.ROOT
    with tempfile.TemporaryDirectory(prefix="friday-bench-") as td:
        root = Path(td)
        # Set the env BEFORE rebind: paths reads it at import, and every module holds a
        # reference to the one module object, so rebinding in place reaches all of them.
        os.environ["FRIDAY_ROOT"] = str(root)
        paths.rebind(root)
        paths.ensure_layout()
        (root / "soul").mkdir(exist_ok=True)
        (root / "soul" / "SOUL.md").write_text("# SOUL\nbench fixture\n", encoding="utf-8")

        res.facts = _write_corpus(root)
        cli(["build"])

        conn = db.connect(paths.DB_PATH)
        emb, rr = get_embedder(), get_reranker()
        res.embedder = getattr(emb, "name", type(emb).__name__)
        res.reranker = getattr(rr, "name", type(rr).__name__)
        res.floor = float(getattr(rr, "recommended_floor", 0.0))

        t0 = time.perf_counter()
        hits1 = hits5 = 0
        rr_sum = 0.0
        per_class: dict[str, list[float]] = {}
        quoted = shipped = 0

        for query, expected, cls in QUERIES:
            out = search(conn, query)
            preds = [_predicate_of(c) for c in out.kept[:5]]
            res.queries += 1
            rank = preds.index(expected) + 1 if expected in preds else 0
            h1 = 1.0 if rank == 1 else 0.0
            h5 = 1.0 if rank else 0.0
            hits1 += h1
            hits5 += h5
            rr_sum += (1.0 / rank) if rank else 0.0
            per_class.setdefault(cls, []).append(h5)
            if verbose:
                print(f"  {'✓' if rank else '✗'} {query!r:46} -> {preds[:3] or '[]'}")
            for c in out.kept[:5]:
                shipped += 1
                if (getattr(c.row, "keys", None) and "source_quote" in c.row.keys()
                        and c.row["source_quote"]):
                    quoted += 1

        for query in UNANSWERABLE:
            out = search(conn, query)
            res.unanswerable += 1
            if out.kept:
                res.false_positive_rate += 1.0
                if verbose:
                    print(f"  ✗ UNANSWERABLE {query!r} returned "
                          f"{[_predicate_of(c) for c in out.kept[:3]]}")

        res.seconds = round(time.perf_counter() - t0, 3)

    # Restore BEFORE computing the summary: nothing below needs the temp root, and a
    # `finally` here would leave the caller pointed at a deleted directory if any of the
    # scoring raised.
    if saved_env is None:
        os.environ.pop("FRIDAY_ROOT", None)
    else:
        os.environ["FRIDAY_ROOT"] = saved_env
    paths.rebind(saved_root)

    n = max(res.queries, 1)
    res.recall_at_1 = round(hits1 / n, 4)
    res.recall_at_5 = round(hits5 / n, 4)
    res.mrr = round(rr_sum / n, 4)
    res.by_class = {k: {"recall_at_5": round(sum(v) / len(v), 4), "n": len(v)}
                    for k, v in sorted(per_class.items())}
    noun = res.by_class.get("noun", {}).get("recall_at_5", 0.0)
    inter = res.by_class.get("interrogative", {}).get("recall_at_5", 0.0)
    res.interrogative_gap = round(noun - inter, 4)
    res.false_positive_rate = round(res.false_positive_rate / max(res.unanswerable, 1), 4)
    res.provenance_complete = round(quoted / shipped, 4) if shipped else 1.0
    return res


# ── the baseline file ──────────────────────────────────────────────────────────

def render_md(res: Result, *, when: str = "") -> str:
    lines = [
        "# BENCHMARKS",
        "",
        "Recorded by `python -m friday bench --record`. **Do not edit the numbers by",
        "hand** — the gate in `friday/bench.py` parses this table and compares against",
        "it, so a hand-edited baseline is a gate that no longer measures anything.",
        "",
        "## Why this file exists",
        "",
        "Law 10: post-training can quietly ruin long-range recall, so every fine-tuning",
        "run must re-run the needle test. As advice that gets skipped, because a merely",
        "*worse* answer raises no exception. `friday bench --gate` makes it a nonzero",
        "exit instead. `scripts/needle_test.py` still measures a real model's long-range",
        "recall; this measures the retrieval layer, model-free, in about a second, so it",
        "can run on every commit.",
        "",
        "## Baseline",
        "",
        f"- recorded: {when or 'unknown'}",
        f"- embedder: `{res.embedder}` · reranker: `{res.reranker}` · "
        f"floor: {res.floor}",
        f"- corpus: {res.facts} facts · {res.queries} queries · "
        f"{res.unanswerable} unanswerable · {res.seconds}s",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in ("recall_at_1", "recall_at_5", "mrr", "provenance_complete",
              "interrogative_gap", "false_positive_rate"):
        lines.append(f"| `{k}` | {getattr(res, k)} |")
    lines += ["", "### Recall by query class", "", "| class | recall@5 | n |", "|---|---|---|"]
    for cls, d in res.by_class.items():
        lines.append(f"| {cls} | {d['recall_at_5']} | {d['n']} |")
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- **`interrogative_gap` is a defect, not a curiosity.** It is recall on \"my",
        "  manager\" minus recall on \"who is my manager\" — the same fact, asked the way",
        "  a person asks it. Anything above 0 means FRIDAY answers \"I don't have",
        "  anything in memory about that\" for something it does have. Fixing it is",
        "  lever 4 in doc 15 (hybrid RRF fusion + query rewriting).",
        "- **`false_positive_rate` must stay at 0.** Law 4: an empty list beats noise.",
        "  A retrieval layer that returns something for an unanswerable query is",
        "  manufacturing raw material for a confabulation, and this is the metric that",
        "  guards the refusal behaviour doc 15 makes its fine-tuning headline.",
        "- **`provenance_complete` must stay at 1.0.** A shipped fact without a source",
        "  quote cannot be shown by `/why`, and `/why` is what the whole trust model",
        "  rests on.",
        "- Scores are environment-dependent: with numpy the embedder is different than",
        "  without it. Re-record the baseline when the environment changes, and say so",
        "  in the commit — a baseline nobody trusts is a gate nobody keeps.",
        "",
    ]
    return "\n".join(lines)


_TABLE = re.compile(r"^\|\s*`([a-z_0-9]+)`\s*\|\s*([0-9.]+)\s*\|", re.M)


def read_baseline(path: Path) -> dict[str, float]:
    """Parse the baseline table back out of BENCHMARKS.md."""
    if not path.exists():
        return {}
    return {k: float(v) for k, v in _TABLE.findall(path.read_text(encoding="utf-8"))}


def gate(res: Result, path: Path, tolerance: float = TOLERANCE) -> tuple[bool, list[str]]:
    """Compare against the recorded baseline. Returns (ok, messages).

    Only the metrics where *higher is better* are gated. `interrogative_gap` and
    `false_positive_rate` are lower-is-better, and gating those needs a separate
    direction — so they are reported and asserted individually instead.
    """
    base = read_baseline(path)
    msgs: list[str] = []
    if not base:
        return False, [f"no baseline in {path} — run `python -m friday bench --record`"]

    ok = True
    for k in Result.GATED:
        if k not in base:
            continue
        delta = getattr(res, k) - base[k]
        if delta < -tolerance:
            ok = False
            msgs.append(f"REGRESSION {k}: {getattr(res, k)} < baseline {base[k]} "
                        f"({delta:+.4f}, tolerance {tolerance})")
        else:
            msgs.append(f"ok {k}: {getattr(res, k)} (baseline {base[k]}, {delta:+.4f})")

    # Lower-is-better, and both have a hard floor rather than a tolerance.
    if res.false_positive_rate > base.get("false_positive_rate", 0.0):
        ok = False
        msgs.append(f"REGRESSION false_positive_rate: {res.false_positive_rate} — Law 4 "
                    f"says an empty list beats noise; this must not rise")
    if res.interrogative_gap > base.get("interrogative_gap", 1.0) + tolerance:
        ok = False
        msgs.append(f"REGRESSION interrogative_gap: {res.interrogative_gap} — natural "
                    f"questions are retrieving worse than noun phrases")
    return ok, msgs


def main(argv: list[str] | None = None) -> int:
    import argparse
    import datetime as _dt

    ap = argparse.ArgumentParser(prog="friday bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true",
                    help=f"write the current run into {BENCH_FILE} as the new baseline")
    ap.add_argument("--gate", action="store_true",
                    help="compare against the recorded baseline; exit 1 on regression")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-query results")
    ap.add_argument("--file", default=None, help=f"path to {BENCH_FILE}")
    args = ap.parse_args(argv)

    from friday import paths

    path = Path(args.file) if args.file else (
        Path(__file__).resolve().parent.parent / BENCH_FILE)

    # run() owns environment restoration, so a programmatic caller is as safe as the CLI.
    res = run(verbose=args.verbose)

    if args.json:
        print(json.dumps(res.as_dict(), indent=2))
    else:
        print(f"FRIDAY bench — retrieval regression gate")
        print("=" * 62)
        print(f"  corpus            {res.facts} facts, {res.queries} queries, "
              f"{res.unanswerable} unanswerable")
        print(f"  embedder/reranker {res.embedder} / {res.reranker} (floor {res.floor})")
        print(f"  recall@1          {res.recall_at_1}")
        print(f"  recall@5          {res.recall_at_5}")
        print(f"  MRR               {res.mrr}")
        print(f"  provenance        {res.provenance_complete}  (must be 1.0)")
        print(f"  false positives   {res.false_positive_rate}  (must be 0.0)")
        print(f"  interrogative gap {res.interrogative_gap}  (0.0 = natural questions "
              f"retrieve as well as noun phrases)")
        for cls, d in res.by_class.items():
            print(f"    · {cls:14} recall@5 {d['recall_at_5']}  (n={d['n']})")
        print(f"  time              {res.seconds}s")

    if args.record:
        when = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        path.write_text(render_md(res, when=when), encoding="utf-8")
        print(f"\nbaseline recorded -> {path}")
        return 0

    if args.gate:
        ok, msgs = gate(res, path)
        print()
        for m in msgs:
            print(("  ✗ " if m.startswith(("REGRESSION", "no baseline")) else "  ✓ ") + m)
        print("\nGATE PASS" if ok else "\nGATE FAIL — do not ship this.")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
