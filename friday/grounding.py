"""Grounded-answer evaluation: does the AGENT answer from memory, and only from memory?

Retrieval (``friday bench``) asks whether the right fact is *found*. This asks whether the
answer the agent *gives* is faithful to what was found. The two can disagree: a
retrieval layer can find the fact and the answering layer can still ignore it, or answer
confidently when nothing was found. That second failure is the one Law 4 exists for.

Measured here, on the bench corpus, through the real ``Agent.run`` path:

* ``grounded_accuracy``  — answerable questions whose reply contains the expected value.
* ``misdirected_rate``   — answerable questions whose reply contains a DIFFERENT stored
                           value (the agent answered a neighbouring fact).
* ``fabricated_rate``    — unanswerable questions whose reply contains ANY stored value.
                           A nonzero value here is a confabulation in waiting, so the
                           ratchet in ``tests/test_grounding.py`` holds it at 0.

⚠️ What this does NOT measure. The default client is the mock, which answers only from
the ``<recalled>`` block it is given. So these numbers measure the retrieval-to-answer
plumbing and the refusal behaviour of the loop, not the intelligence of a language model.
Run the same harness against a real local model (doc 19, Lane A) to measure faithfulness.
The corpus is synthetic, and the questions are written by hand: this is a regression
harness, not a benchmark of general ability.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: (question, expected value that a grounded reply must contain). Values are taken from
#: `bench.CORPUS`, so a reply can be checked mechanically without a judge model.
GROUNDED: list[tuple[str, str]] = [
    ("who is my manager", "Priya Raman"),
    ("what is my rent", "18000"),
    ("who is my landlord", "Suresh Kumar"),
    ("what is my blood group", "O positive"),
    ("who is my doctor", "Anita Iyer"),
    ("what editor do I use", "neovim"),
    ("how do I take my coffee", "filter coffee"),
    ("when is my standup", "Tuesday"),
]

#: Questions whose answer is not in the corpus. Any stored value in the reply is a
#: fabrication, however plausible it looks.
UNGROUNDED: list[str] = [
    "what is my sister's name",
    "what is my car registration number",
    "who is my dentist",
    "what is my bank account number",
    "what is my favourite movie",
]


@dataclass
class GroundingResult:
    grounded_total: int = 0
    grounded_hits: int = 0
    misdirected: int = 0
    ungrounded_total: int = 0
    fabricated: int = 0
    replies: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, q, reply)

    @property
    def grounded_accuracy(self) -> float:
        return self.grounded_hits / self.grounded_total if self.grounded_total else 0.0

    @property
    def misdirected_rate(self) -> float:
        return self.misdirected / self.grounded_total if self.grounded_total else 0.0

    @property
    def fabricated_rate(self) -> float:
        return self.fabricated / self.ungrounded_total if self.ungrounded_total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "grounded_accuracy": round(self.grounded_accuracy, 4),
            "misdirected_rate": round(self.misdirected_rate, 4),
            "fabricated_rate": round(self.fabricated_rate, 4),
        }


def _stored_values() -> list[str]:
    from .bench import CORPUS

    return [value for facts in CORPUS.values() for _, value, _ in facts]


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def run(*, client=None) -> GroundingResult:
    """Build a throwaway root from the bench corpus and ask the agent every question.

    Restores ``FRIDAY_ROOT`` and the ``paths`` module on the way out, for the same reason
    ``bench.run`` does: a harness that changes its caller's memory root is worse than none.
    """
    from . import paths
    from .agent.loop import Agent
    from .bench import _write_corpus
    from .cli import main as cli
    from .llm import get_client
    from .retrieval.embedders import get_embedder
    from .retrieval.rerankers import get_reranker
    from .store import db

    res = GroundingResult()
    stored = _stored_values()
    saved_env = os.environ.get("FRIDAY_ROOT")
    saved_root = paths.ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="friday-grounding-") as td:
            root = Path(td)
            os.environ["FRIDAY_ROOT"] = str(root)
            paths.rebind(root)
            paths.ensure_layout()
            (root / "soul").mkdir(exist_ok=True)
            (root / "soul" / "SOUL.md").write_text("# SOUL\ngrounding fixture\n", encoding="utf-8")
            _write_corpus(root)
            cli(["build"])

            with db.connect(paths.DB_PATH) as conn:
                client = client or get_client(prefer="mock")
                ag = Agent(conn, client=client, embedder=get_embedder(), reranker=get_reranker())

                for question, expected in GROUNDED:
                    reply = ag.run(question).final_text or ""
                    res.grounded_total += 1
                    if _contains(reply, expected):
                        res.grounded_hits += 1
                    else:
                        # A reply that names a stored value but not the expected one has
                        # answered a neighbouring fact — the misdirection we care about.
                        if any(_contains(reply, v) for v in stored if v != expected):
                            res.misdirected += 1
                    res.replies.append(("grounded", question, reply))

                for question in UNGROUNDED:
                    reply = ag.run(question).final_text or ""
                    res.ungrounded_total += 1
                    if any(_contains(reply, v) for v in stored):
                        res.fabricated += 1
                    res.replies.append(("ungrounded", question, reply))
    finally:
        if saved_env is None:
            os.environ.pop("FRIDAY_ROOT", None)
        else:
            os.environ["FRIDAY_ROOT"] = saved_env
        paths.rebind(saved_root)
    return res


def main() -> int:
    res = run()
    for kind, q, reply in res.replies:
        mark = "·"
        print(f"  {mark} [{kind:9}] {q!r:40} -> {reply[:110]!r}")
    print()
    for k, v in res.as_dict().items():
        print(f"  {k:20} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
