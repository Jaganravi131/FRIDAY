"""Blind holdout scoring for retrieval (T-002).

Every retrieval number in this project so far was measured on queries the tuning agent
had seen — the committed bench, or probes written by the same agent that then fixed the
lexicon against them. Those numbers prove a mechanism works; they cannot prove
retrieval generalises. The fix is a holdout written by someone else.

This module is the harness half of that fix. The CONTENT half — 40+ paraphrase queries
and 10+ unanswerable ones, with expected predicates — must be written by a different
agent or by the user, and must never enter this repository. The harness enforces what
it can:

* **The file must live outside the repo.** A holdout inside the repo is one `cat` away
  from being tuned against, so loading one is refused.
* **The report contains aggregates only.** No query text, no per-query lines, in the
  text report, the JSON, or the error messages. An agent can learn "recall@5 is 0.61"
  and nothing about which questions were asked.
* **Small holdouts say so.** Below the board's 40/10 minimums the result carries a
  warning instead of pretending to be a measurement.

Queries run against the committed bench corpus, so `expected` is a bench predicate
(or `null` for unanswerable). Run it with `python -m friday holdout --file PATH`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

#: The board's minimums (T-002). Below these, warn rather than refuse: a small holdout
#: is still better than none, as long as nobody mistakes it for a measurement.
MIN_ANSWERABLE = 40
MIN_UNANSWERABLE = 10

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class HoldoutRow:
    query: str
    expected: str | None
    cls: str = "paraphrase"


@dataclass
class HoldoutResult:
    answerable: int = 0
    unanswerable: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    false_positive_rate: float = 0.0
    by_class: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _known_predicates() -> set[str]:
    from .bench import CORPUS

    return {p for facts in CORPUS.values() for (p, _v, _q) in facts}


def load_holdout(path: str) -> list[HoldoutRow]:
    """Parse and validate a holdout JSONL file. Errors name line numbers, never content."""
    p = Path(path).resolve()
    try:
        p.relative_to(_REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError(
            "holdout file must live outside the repository — a holdout the tuning "
            "agent can read is not blind"
        )
    known = _known_predicates()
    rows: list[HoldoutRow] = []
    seen: set[str] = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"holdout line {lineno}: not valid JSON") from None
        if not isinstance(obj, dict) or not str(obj.get("query") or "").strip():
            raise ValueError(f"holdout line {lineno}: missing a query")
        expected = obj.get("expected")
        if expected is not None and expected not in known:
            raise ValueError(
                f"holdout line {lineno}: expected predicate is not in the bench corpus"
            )
        key = str(obj["query"]).strip().lower()
        if key in seen:
            raise ValueError(f"holdout line {lineno}: duplicate query")
        seen.add(key)
        cls = str(obj.get("class") or ("unanswerable" if expected is None else "paraphrase"))
        rows.append(HoldoutRow(str(obj["query"]), expected, cls))
    if not rows:
        raise ValueError("holdout file has no rows")
    return rows


def run(path: str) -> HoldoutResult:
    """Score a holdout against the bench corpus in a throwaway root.

    Mirrors `bench.run`'s environment handling: `FRIDAY_ROOT` and `paths` are restored
    afterwards, and the DB connection closes in a `finally` (Windows cannot delete an
    open file — see bench.run).
    """
    from . import paths
    from .bench import _predicate_of, _write_corpus
    from .cli import main as cli
    from .retrieval.pipeline import search
    from .store import db

    rows = load_holdout(path)
    res = HoldoutResult()
    answerable = [r for r in rows if r.expected is not None]
    unanswerable = [r for r in rows if r.expected is None]
    res.answerable, res.unanswerable = len(answerable), len(unanswerable)
    if res.answerable < MIN_ANSWERABLE:
        res.warnings.append(
            f"only {res.answerable} answerable queries (board minimum {MIN_ANSWERABLE}) "
            "— treat this as a smoke test, not a measurement"
        )
    if res.unanswerable < MIN_UNANSWERABLE:
        res.warnings.append(
            f"only {res.unanswerable} unanswerable queries (board minimum "
            f"{MIN_UNANSWERABLE}) — the false-positive rate is barely sampled"
        )

    saved_env = os.environ.get("FRIDAY_ROOT")
    saved_root = paths.ROOT
    hits1 = hits5 = 0
    rr_sum = 0.0
    per_class: dict[str, list[float]] = {}
    fp = 0
    try:
        with tempfile.TemporaryDirectory(prefix="friday-holdout-") as td:
            root = Path(td)
            os.environ["FRIDAY_ROOT"] = str(root)
            paths.rebind(root)
            paths.ensure_layout()
            (root / "soul").mkdir(exist_ok=True)
            (root / "soul" / "SOUL.md").write_text("# SOUL\nholdout fixture\n", encoding="utf-8")
            _write_corpus(root)
            cli(["build"])
            conn = db.connect(paths.DB_PATH)
            try:
                t0 = time.perf_counter()
                for row in answerable:
                    preds = [_predicate_of(c) for c in search(conn, row.query).kept[:5]]
                    rank = preds.index(row.expected) + 1 if row.expected in preds else 0
                    hits1 += 1.0 if rank == 1 else 0.0
                    hits5 += 1.0 if rank else 0.0
                    rr_sum += (1.0 / rank) if rank else 0.0
                    per_class.setdefault(row.cls, []).append(1.0 if rank else 0.0)
                for row in unanswerable:
                    if search(conn, row.query).kept:
                        fp += 1
                res.seconds = round(time.perf_counter() - t0, 3)
            finally:
                conn.close()
    finally:
        if saved_env is None:
            os.environ.pop("FRIDAY_ROOT", None)
        else:
            os.environ["FRIDAY_ROOT"] = saved_env
        paths.rebind(saved_root)

    n = max(res.answerable, 1)
    res.recall_at_1 = round(hits1 / n, 4)
    res.recall_at_5 = round(hits5 / n, 4)
    res.mrr = round(rr_sum / n, 4)
    res.false_positive_rate = round(fp / max(res.unanswerable, 1), 4)
    res.by_class = {k: {"recall_at_5": round(sum(v) / len(v), 4), "n": len(v)}
                    for k, v in sorted(per_class.items())}
    return res


def format_report(res: HoldoutResult) -> str:
    """Aggregates only. No query text ever appears here — that is the point."""
    lines = [
        "FRIDAY blind holdout — retrieval",
        "==============================================================",
        f"  queries           {res.answerable} answerable, {res.unanswerable} unanswerable",
        f"  recall@1          {res.recall_at_1}",
        f"  recall@5          {res.recall_at_5}",
        f"  MRR               {res.mrr}",
        f"  false positives   {res.false_positive_rate}  (must be 0.0)",
    ]
    for cls, d in res.by_class.items():
        lines.append(f"    · {cls:16} recall@5 {d['recall_at_5']}  (n={d['n']})")
    lines.append(f"  time              {res.seconds}s")
    for w in res.warnings:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="friday holdout")
    ap.add_argument("--file", required=True, help="holdout JSONL, outside the repository")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        res = run(args.file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"holdout: {exc}")
        return 2
    if args.json:
        print(json.dumps(res.__dict__, indent=2))
    else:
        print(format_report(res))
    return 1 if res.false_positive_rate > 0 else 0
