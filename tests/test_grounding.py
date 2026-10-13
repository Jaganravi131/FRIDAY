"""The grounded-answer harness: a ratchet on the mock, and proof that the harness can FAIL.

A metric that cannot go red is decoration. So alongside the ratchet, two deliberately
bad clients must be caught: one that confabulates on every question, and one that
answers a neighbouring fact. If either passes, the harness is measuring nothing.
"""

from __future__ import annotations

import os

import pytest

from friday import grounding, paths
from friday.llm.client import Response


class _Always:
    """A client that gives the same reply to every question, whatever memory says."""

    name = "always"

    def __init__(self, text: str):
        self.text = text

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=768) -> Response:
        return Response(text=self.text, model=self.name)


def test_the_mock_is_grounded_on_the_bench_corpus():
    """Ratchet. The mock answers only from what retrieval hands it, so on this corpus it
    should be perfect. A drop means the retrieval-to-answer path broke, not that a model
    got worse. Do not lower these numbers to make a change pass; find the regression."""
    res = grounding.run()
    assert res.grounded_total == len(grounding.GROUNDED)
    assert res.ungrounded_total == len(grounding.UNGROUNDED)
    assert res.grounded_accuracy >= 1.0, [r for r in res.replies if r[0] == "grounded"]
    assert res.fabricated_rate == 0.0, [r for r in res.replies if r[0] == "ungrounded"]
    assert res.misdirected_rate == 0.0


def test_a_confabulating_client_is_caught():
    """If the agent answers a question it has no memory of with a stored value, the
    harness must say so. Here every reply names a real value, which is the worst case."""
    res = grounding.run(client=_Always("Your manager is Priya Raman."))
    assert res.fabricated_rate == 1.0
    assert res.grounded_accuracy < 1.0 or res.misdirected_rate > 0.0


def test_a_client_that_names_the_wrong_fact_is_caught():
    """A reply that names a stored value but NOT the expected one is misdirection:
    the agent answered a neighbouring fact. It must count, separately from a miss."""
    res = grounding.run(client=_Always("Suresh Kumar."))
    assert res.misdirected_rate > 0.0
    assert res.grounded_accuracy < 1.0


def test_the_harness_restores_the_callers_memory_root():
    """Same contract as bench.run: after the harness, FRIDAY_ROOT and the paths module
    must point where they pointed before. Otherwise every later operation in the process
    reads from a directory the harness already deleted."""
    before_env = os.environ.get("FRIDAY_ROOT")
    before_root = paths.ROOT
    grounding.run()
    assert os.environ.get("FRIDAY_ROOT") == before_env
    assert paths.ROOT == before_root


@pytest.mark.parametrize("question,expected", grounding.GROUNDED)
def test_every_grounded_expectation_exists_in_the_corpus(question, expected):
    """Sanity on the question set itself: a grounded question whose expected value is not
    stored would be a guaranteed miss, and the accuracy number would be measuring the
    test author rather than the agent."""
    from friday.bench import CORPUS

    stored = [v for facts in CORPUS.values() for _, v, _ in facts]
    assert any(expected.lower() in v.lower() for v in stored), (question, expected)
