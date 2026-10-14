"""The blind-holdout harness (T-002): scoring without seeing.

The point of a holdout is that the agent who tunes retrieval cannot read it. That is
only true if the *harness* enforces it: it must refuse a file inside the repo, and its
report must contain aggregates only — never a query. These tests pin both, plus the
schema validation that keeps a malformed holdout from scoring as a silent zero.

The holdout CONTENT is not in this repo and must never be: a different agent or the
user writes it. The dummy queries below exist only to exercise the harness.
"""

from __future__ import annotations

import json

import pytest

from friday.holdout import format_report, load_holdout, run

DUMMY = [
    {"query": "my rent", "expected": "lease_amount_monthly", "class": "noun"},
    {"query": "what is my sister's name", "expected": None, "class": "unanswerable"},
]


def _write(tmp_path, rows) -> str:
    p = tmp_path / "holdout.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def test_load_parses_answerable_and_unanswerable_rows(tmp_path):
    rows = load_holdout(_write(tmp_path, DUMMY))
    assert len(rows) == 2
    assert rows[0].expected == "lease_amount_monthly"
    assert rows[1].expected is None


def test_a_holdout_inside_the_repo_is_refused():
    """A holdout in the repo is one `cat` away from the agent tuning against it."""
    from pathlib import Path

    inside = Path(__file__).resolve().parent.parent / "holdout.jsonl"
    with pytest.raises(ValueError, match="outside the repository"):
        load_holdout(str(inside))


def test_an_unknown_expected_predicate_is_rejected(tmp_path):
    bad = [{"query": "my rent", "expected": "no_such_predicate"}]
    with pytest.raises(ValueError, match="line 1"):
        load_holdout(_write(tmp_path, bad))


def test_malformed_json_reports_the_line_without_echoing_content(tmp_path):
    p = tmp_path / "holdout.jsonl"
    p.write_text('{"query": "my rent", "expected": "lease_amount_monthly"}\n{not json\n',
                 encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_holdout(str(p))
    assert "line 2" in str(ei.value)
    assert "my rent" not in str(ei.value)


def test_duplicate_queries_are_rejected(tmp_path):
    dup = [DUMMY[0], dict(DUMMY[0])]
    with pytest.raises(ValueError, match="duplicate"):
        load_holdout(_write(tmp_path, dup))


def test_run_scores_and_the_report_never_contains_a_query(tmp_path):
    res = run(_write(tmp_path, DUMMY))
    assert res.answerable == 1 and res.unanswerable == 1
    assert res.recall_at_5 == 1.0            # "my rent" is the bench's easiest query
    assert res.false_positive_rate == 0.0    # the sister question must stay empty
    report = format_report(res)
    for row in DUMMY:
        assert row["query"] not in report


def test_main_refuses_an_in_repo_file_without_a_traceback(capsys):
    from pathlib import Path

    from friday.holdout import main

    inside = Path(__file__).resolve().parent.parent / "holdout.jsonl"
    assert main(["--file", str(inside)]) == 2
    assert "outside the repository" in capsys.readouterr().out


def test_a_small_holdout_warns_instead_of_pretending_to_measure(tmp_path):
    res = run(_write(tmp_path, DUMMY))
    assert any("40" in w for w in res.warnings)
