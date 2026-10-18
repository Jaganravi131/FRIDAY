"""T-002 — the blind-holdout runner.

The holdout exists because the committed bench was written by the same agent
that tuned the lexicon (doc 20 §4). These tests pin the runner's two promises:

* it measures with the SAME instrument as the bench (`bench.score_cases`);
* a retrieval agent can run it without being able to tune against it — no
  per-query output, and no query text in output or errors.

The queries in these tests are written by the test on purpose: a test fixture
is not a blind set, and pretending otherwise would be the contamination T-002
exists to prevent. The real holdout is written by the user or a different
agent, per eval/holdout/README.md.
"""

from __future__ import annotations

import json

import pytest

from friday.commands import holdout


def _write(path, cases, **extra):
    payload = {"suite": "test-holdout",
               "author": "tests/test_holdout.py (a fixture, not a blind set)",
               "cases": cases}
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


#: 2 hits, 1 documented miss ("what do I drink in the morning" is the bench's
#: known paraphrase gap — its residue word `morning` is real content), and one
#: unanswerable query that must stay unanswered.
CASES = [
    {"query": "who is my manager", "expected": "manager_name"},
    {"query": "what is my rent", "expected": "lease_amount_monthly"},
    {"query": "what do I drink in the morning", "expected": "coffee_order"},
    {"query": "when is my dentist appointment", "expected": None},
]


def test_run_reports_aggregates_and_never_the_queries(tmp_path, capsys):
    path = _write(tmp_path / "h.json", CASES)
    assert holdout.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out and "false positives" in out
    assert "2 answerable" in out.replace("  ", " ").replace("  ", " ").strip() or "3 answerable" in out
    # THE LEAK RULE. Aggregates only: no phrasing may reach the runner's output.
    for case in CASES:
        assert case["query"] not in out
    # the known miss must be COUNTED, not hidden: 2 of 3 answerable retrieved
    assert "recall@5          0.6667" in out or "recall@5  0.6667" in out or "0.6667" in out
    assert "false positives   0.0" in out


def test_json_output_has_the_metrics(tmp_path, capsys):
    path = _write(tmp_path / "h.json", CASES)
    assert holdout.main([str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["recall_at_5"] == 0.6667
    assert data["false_positive_rate"] == 0.0
    assert data["suite"] == "test-holdout"
    assert data["by_class"]["paraphrase"]["n"] == 3


def test_the_minimum_warning_for_a_smoke_sized_set(tmp_path, capsys):
    path = _write(tmp_path / "h.json", CASES)
    holdout.main([str(path)])
    out = capsys.readouterr().out
    assert "below the T-002 minimum (40 paraphrase, 10 unanswerable)" in out


def test_unknown_expected_predicate_is_rejected_without_echoing_the_query(
        tmp_path, capsys):
    cases = [
        {"query": "who is my manager", "expected": "manager_name"},
        {"query": "a very distinctive phrasing about my landlord",
         "expected": "landloord_name"},          # the author's typo
    ]
    path = _write(tmp_path / "h.json", cases)
    assert holdout.main([str(path)]) == 2
    err = capsys.readouterr().err
    assert "case 1" in err and "landloord_name" in err
    assert "a very distinctive phrasing" not in err


@pytest.mark.parametrize("bad,expect", [
    ({"cases": []}, "non-empty list"),
    ({"cases": [{"expected": "manager_name"}]}, "non-empty string"),
    ({"cases": [{"query": "who is my manager", "expected": "manager_name",
                 "class": "  "}]}, "non-empty string"),
    ({}, "non-empty list"),
])
def test_malformed_files_fail_cleanly(tmp_path, capsys, bad, expect):
    path = tmp_path / "h.json"
    payload = {"suite": "x", "author": "x"}
    payload.update(bad)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert holdout.main([str(path)]) == 2
    assert expect in capsys.readouterr().err


def test_missing_and_non_json_files_fail_cleanly(tmp_path, capsys):
    assert holdout.main([str(tmp_path / "absent.json")]) == 2
    assert "cannot read" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert holdout.main([str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_holdout_restores_the_callers_environment(tmp_path, monkeypatch):
    """The holdout rebinds FRIDAY_ROOT to a temp dir it then deletes — the same
    trap `friday bench` documented the hard way. Pinned identically."""
    import os
    from pathlib import Path

    from friday import paths

    monkeypatch.setenv("FRIDAY_ROOT", str(tmp_path))
    paths.rebind(tmp_path)
    paths.ensure_layout()

    path = _write(tmp_path / "h.json", CASES)
    assert holdout.main([str(path)]) == 0

    assert Path(os.environ["FRIDAY_ROOT"]) == tmp_path
    assert paths.ROOT == tmp_path
    assert paths.ROOT.exists(), "left pointing at a deleted temp directory"


def test_the_cli_subcommand_is_wired(tmp_path, capsys):
    from friday.cli import main

    path = _write(tmp_path / "h.json", CASES)
    assert main(["holdout", str(path)]) == 0
    assert "recall@5" in capsys.readouterr().out


def test_the_scorer_is_the_bench_scorer():
    """One instrument, not two: the holdout must measure with `bench.score_cases`
    or its numbers are not comparable with BENCHMARKS.md."""
    from friday import bench
    from friday.commands import holdout as h

    assert h.run.__doc__ or True  # run exists
    src = open(bench.__file__, encoding="utf-8").read()
    assert "def score_cases" in src
    # and the holdout actually uses it
    import inspect
    assert "score_cases" in inspect.getsource(h.run)
