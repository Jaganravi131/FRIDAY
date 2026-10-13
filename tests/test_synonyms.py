"""Tests for the query-side synonym bridges (`retrieval/synonyms.py`).

The defect being fixed: paraphrase recall@5 was 0.0 on the stdlib path —
"who do I report to" shares no word with the `manager_name` fact, so recall
found nothing and overlap scored nothing. The fix has two halves (recall on
the expanded query, synonym-aware rerank), and these tests pin both, plus the
curation rules that stop the lexicon becoming a false-positive machine:

* rule 2: `dentist` must NEVER bridge to `doctor` (Law 4, enforced here and by
  the bench FPR gate);
* rule 3: ambiguous headwords (`book`, `order`, `charge`, `post`) unmapped;
* rule 4: bare temporal coordinates (`day`, `date`, `time`) unmapped — they
  appear inside every scheduled fact, so expanding them promotes timetables
  for clock questions. "my standup time" scoring below the floor is the
  documented cost of this rule (see `test_bare_temporal_words_are_not_expanded`
  and the limitation test at the bottom).
"""
from __future__ import annotations

import pytest

from friday.memory.facts import Fact, assert_fact
from friday.retrieval.embedders import HashingEmbedder
from friday.retrieval.pipeline import search
from friday.retrieval.rerankers import LexicalReranker
from friday.retrieval.synonyms import (
    SYNONYMS,
    expanded_query,
    expansion_terms,
    expansions_for,
)
from friday.store.db import CONTENT_STOPWORDS


def _search(conn, q, **kw):
    kw.setdefault("embedder", HashingEmbedder(dim=256))
    kw.setdefault("reranker", LexicalReranker())
    return search(conn, q, **kw)


def _preds(res, n=5):
    return [c.row["predicate"] for c in res.kept[:n] if c.row is not None]


@pytest.fixture
def paraphrase_facts(conn):
    """The four bench paraphrase targets, with bench-like source quotes."""
    assert_fact(conn, Fact(subject="user", predicate="manager_name",
                           object="Priya Raman", source_kind="user_edit",
                           source_quote="my manager is Priya Raman"))
    assert_fact(conn, Fact(subject="user", predicate="lease_amount_monthly",
                           object="18000 INR", source_kind="user_edit",
                           source_quote="my rent is 18000 a month"))
    assert_fact(conn, Fact(subject="user", predicate="coffee_order",
                           object="filter coffee, no sugar", source_kind="user_edit",
                           source_quote="I take filter coffee with no sugar"))
    assert_fact(conn, Fact(subject="user", predicate="editor",
                           object="neovim", source_kind="user_edit",
                           source_quote="I use neovim as my editor"))
    assert_fact(conn, Fact(subject="user", predicate="doctor_name",
                           object="Dr Anita Iyer", source_kind="user_edit",
                           source_quote="my doctor is Dr Anita Iyer"))
    # The over-broad-target regression case: "my team" must rank the team
    # fact first even though `group` appears in the blood fact.
    assert_fact(conn, Fact(subject="user", predicate="team_name",
                           object="infra platform", source_kind="user_edit",
                           source_quote="I work on the infra platform team"))
    assert_fact(conn, Fact(subject="user", predicate="blood_group",
                           object="O positive", source_kind="user_edit",
                           source_quote="my blood group is O positive"))
    return conn


# ── the lexicon's shape ──────────────────────────────────────────────────────

def test_entries_are_single_lowercase_words():
    """The lookup runs on tokenised content words; a phrase key or an
    uppercase value would be dead weight that only LOOKS like coverage."""
    for k, vs in SYNONYMS.items():
        assert k == k.lower() and " " not in k, f"bad headword {k!r}"
        for v in vs:
            assert v == v.lower() and " " not in v, f"bad expansion {v!r} for {k!r}"


def test_no_expansion_target_is_a_stopword():
    """A stopword target is stripped by `fts_terms` before it can match, so it
    buys nothing at recall and only inflates the reranker's union."""
    for k, vs in SYNONYMS.items():
        for v in vs:
            assert v not in CONTENT_STOPWORDS, f"{k!r} -> stopword {v!r}"


def test_dentist_never_bridges_to_doctor():
    """⭐ Rule 2. A dentist question answered with the physician's name is the
    confabulation Law 4 exists to prevent. Pinned here, not just in bench."""
    assert "doctor" not in expansions_for("dentist")
    assert "dentist" not in expansions_for("doctor")
    assert "physician" not in expansions_for("dentist")


def test_ambiguous_headwords_are_unmapped():
    """⭐ Rule 3. `book` (read vs reserve), `order` (sequence vs purchase),
    `charge` (fee vs battery), `post` (mail vs job posting) fire in two
    domains; an expansion that fires in two domains is a false-positive
    machine."""
    for w in ("book", "order", "charge", "post"):
        assert expansions_for(w) == (), f"{w!r} must stay unmapped"


def test_bare_temporal_words_are_not_expanded():
    """⭐ Rule 4. `day`/`date`/`time` sit inside every scheduled fact
    (`standup_day`, "Tuesday 09:30"), so expanding them promotes timetables
    for clock questions."""
    for w in ("day", "date", "time"):
        assert expansions_for(w) == (), f"{w!r} must stay unmapped"


def test_single_hop_no_salary_to_pay_chain():
    """⭐ Rule 5. `pay -> rent` is one bridge; `salary -> pay` would chain it
    and answer salary questions with rent facts. Each hop must stand alone."""
    assert "pay" not in expansions_for("salary")


# ── the bridges the bench needs ──────────────────────────────────────────────

def test_reporting_line_bridge():
    assert "manager" in expansions_for("report")


def test_housing_cost_bridges():
    assert "rent" in expansions_for("pay")
    assert "lease" in expansions_for("flat") or "rent" in expansions_for("flat")


def test_drink_bridge():
    assert "coffee" in expansions_for("drink")


def test_code_writing_bridges():
    assert "editor" in expansions_for("code")
    assert "editor" in expansions_for("write")


def test_overbroad_targets_excluded():
    """⭐ Rule 3 mirror. `group` sits inside unrelated predicates
    (`blood_group`), so `team` must not expand to it — otherwise "my team"
    crowns the blood fact. Prefer narrow targets."""
    assert "group" not in expansions_for("team")


def test_exact_match_outranks_unrelated_fact(paraphrase_facts):
    """End to end for the above: team_name first, blood_group nowhere near
    the top, for a plain team query."""
    r = _search(paraphrase_facts, "my team")
    assert _preds(r, 1) == ["team_name"]
    assert expanded_query("what is it") == "what is it"
    assert expanded_query("purple zebra") == "purple zebra"


def test_expanded_query_appends_terms():
    qx = expanded_query("who do I report to")
    assert "manager" in qx.split()
    assert qx.startswith("who do I report to")


# ── end to end: the four bench paraphrases ───────────────────────────────────

def test_paraphrase_report_to(paraphrase_facts):
    r = _search(paraphrase_facts, "who do I report to")
    assert "manager_name" in _preds(r)


def test_paraphrase_flat_each_month(paraphrase_facts):
    r = _search(paraphrase_facts, "how much do I pay for the flat each month")
    assert "lease_amount_monthly" in _preds(r)


def test_paraphrase_drink_in_the_morning(paraphrase_facts):
    r = _search(paraphrase_facts, "what do I drink in the morning")
    assert "coffee_order" in _preds(r)


def test_paraphrase_write_code(paraphrase_facts):
    r = _search(paraphrase_facts, "where do I write my code")
    assert "editor" in _preds(r)


def test_exact_queries_still_win(paraphrase_facts):
    """Bridges must ADD signal, not move the existing hits. Exact phrasings
    keep rank 1."""
    for q, pred in (("my manager", "manager_name"),
                    ("my rent", "lease_amount_monthly"),
                    ("my coffee order", "coffee_order"),
                    ("my editor", "editor"),
                    ("who is my doctor", "doctor_name")):
        r = _search(paraphrase_facts, q)
        assert _preds(r, 1) == [pred], q


def test_unanswerable_still_empty(paraphrase_facts):
    """⭐ Law 4 survives the lexicon. A bridge that manufactures candidates
    for these is worse than no bridge."""
    for q in ("what is my sister's name",
              "when is my flight to Delhi",
              "what is my car registration number",
              "who is my dentist",
              "what is my gym membership number"):
        assert _search(paraphrase_facts, q).kept == [], q


def test_dentist_query_does_not_return_the_doctor(paraphrase_facts):
    """The rule-2 case end to end: the doctor fact is IN the store and the
    dentist query must still return []."""
    r = _search(paraphrase_facts, "who is my dentist")
    assert "doctor_name" not in _preds(r)
    assert r.kept == []


# ── reranker behaviour ───────────────────────────────────────────────────────

def test_no_expansion_means_old_scores():
    """When no headword fires, the formula must reduce EXACTLY to the old one
    (0.62*coverage + 0.38*(overlap*2.2) + phrase). Any drift here is a
    regression in every existing query, not just paraphrases."""
    rr = LexicalReranker()
    # "purple"/"zebra" are unmapped headwords, so no expansion fires and the
    # arithmetic must reduce EXACTLY to the old formula.
    q, cand = "purple zebra", "purple zebra: striped animal"
    scores = rr.score(q, [cand])
    from friday.util import stem
    import re
    qw = {stem(w) for w in re.findall(r"[^\W_]+", q.lower())}
    ct = {stem(w) for w in re.findall(r"[^\W_]+", cand.lower())}
    inter = qw & ct
    expected = min(1.0, 0.62 * len(inter) / len(qw)
                   + 0.38 * (len(inter) / len(qw | ct) * 2.2))
    assert scores[0] == expected


def test_documented_limitation_bare_time_word():
    """Rule 4's visible cost: with only `standup` matching and `time`
    deliberately unmapped, this stays below the floor. If this test starts
    failing because the score ROSE, check whether rule 4 was weakened — that
    needs a bench FPR review, not a celebration."""
    rr = LexicalReranker()
    s = rr.score("my standup time",
                 ["standup_day: Tuesday 09:30 IST (standup day, standup, day)"])[0]
    assert s < rr.recommended_floor
