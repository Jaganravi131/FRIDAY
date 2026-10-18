"""Rerankers. Law 4: retrieval without a reranker is noise.

Wide (k=20) -> rerank -> narrow (k=3) -> hard score floor -> `[]`.

The reranker is a CROSS-ENCODER: it sees query and candidate together, which is
why it beats both BM25 and cosine — those score the candidate independently and
can only approximate relevance. It is also the single highest-value 400 MB in the
whole stack (bge-reranker-base, ONNX, CPU).

Fallback is lexical overlap with a coverage term. It is genuinely weaker than a
cross-encoder, and the honest consequence is that you should raise the score floor
when it's active — so `Reranker.recommended_floor` exists and the pipeline reads it.
"""

from __future__ import annotations

from ..util import swallowed

import re
from typing import Protocol, Sequence

from ..store.db import _STOPWORDS
from ..util import stem
from .concepts import concept_of_stem, strip_possessives

#: ⚠️ `[^\W_]+`, NOT `\w+`. In Python `\w` includes the underscore, so the naive
#: pattern tokenises the stored predicate `lives_in` as ONE indivisible token. The
#: user asks "where do I live"; `live` can never equal `lives_in`, so coverage is 0
#: and every snake_case fact scores 0.000 regardless of how good the recall was.
#: Excluding `_` splits predicates into their words, which is what a lexical scorer
#: needs in order to compare a question against a field name at all.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

#: Weekday names and clock times carry the concept of a WHEN. "what time is the
#: scrum" and the fact `standup_day: Tuesday 09:30 IST` share NO word — the fact
#: answers WHEN without containing "time" — so a purely lexical scorer floors the
#: question at 0.380 no matter how good recall is. Renaming weekdays to one
#: temporal concept key (and adding the key for clock times) lets the fact
#: satisfy a temporal query term. The rename keeps a candidate's concept-count
#: stable, so no unrelated query's overlap denominator moves.
_WEEKDAY_STEMS = frozenset(
    stem(d) for d in (
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    )
)
_CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_TEMPORAL = "c:temporal"

#: ⚠️ CONTENT_STOPWORDS holds SURFACE forms; this scorer compares STEMS. Subtracting
#: the raw list therefore silently kept every stopword whose stem differs from its
#: surface form — `have` counted as the content term `hav`, and so did `were`/`wer`,
#: `some`/`som`, `really`/`real`, `please`/`pleas`, `actually`/`actual` and more.
#: Found while closing doc 20 §5's "usually / have" gap: "what do I usually have to
#: drink" was losing coverage to a word the stopword list believed it had dropped.
#: The stem-normalised view is derived here, once, rather than maintained by hand —
#: a hand-maintained second list is exactly the drift `test_there_is_one_canonical_
#: stopword_list` exists to prevent.
_STOPWORD_STEMS = frozenset(_STOPWORDS) | {stem(w) for w in _STOPWORDS}


class Reranker(Protocol):
    name: str
    recommended_floor: float

    def score(self, query: str, candidates: Sequence[str]) -> list[float]: ...


class OnnxReranker:
    """BAAI/bge-reranker-base via onnxruntime. Requires `friday[rerank]` + the model.

    Download once:
        hf download BAAI/bge-reranker-base onnx/model.onnx --local-dir models/reranker
    """

    def __init__(self, model_dir: str = "models/reranker", max_len: int = 512):
        import onnxruntime as ort  # lazy: optional dep
        from pathlib import Path

        onnx_path = Path(model_dir) / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = Path(model_dir) / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"no ONNX reranker under {model_dir}")
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = _load_tokenizer(model_dir)
        self.max_len = max_len
        self.name = "bge-reranker-base"
        self.recommended_floor = 0.35

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        enc = self.tokenizer(pairs, self.max_len)
        inputs = {k: v for k, v in enc.items() if k in {i.name for i in self.sess.get_inputs()}}
        out = self.sess.run(None, inputs)[0]
        return [_sigmoid(float(x)) for x in out.reshape(-1)]


def _load_tokenizer(model_dir: str):
    """Prefer HuggingFace tokenizers; fall back to a whitespace shim.

    The shim is NOT equivalent — a cross-encoder fed the wrong tokenisation gives
    confident nonsense. So if the real tokenizer is missing we raise, and
    `get_reranker` falls back to the lexical reranker instead.
    """
    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        tok.enable_truncation(512)

        class _Wrap:
            def __call__(self, pairs, max_len):
                encs = tok.encode_batch(pairs)
                import numpy as np

                return {
                    "input_ids": np.array([e.ids for e in encs], dtype="int64"),
                    "attention_mask": np.array(
                        [e.attention_mask for e in encs], dtype="int64"
                    ),
                    "token_type_ids": np.array(
                        [e.type_ids for e in encs], dtype="int64"
                    ),
                }

        return _Wrap()
    except Exception as e:  # pragma: no cover - depends on optional deps
        raise RuntimeError(f"reranker tokenizer unavailable: {e}") from e


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


class LexicalReranker:
    """Stdlib fallback: token overlap + query coverage + a small exact-phrase bonus.

    Coverage (fraction of query terms present) is weighted above raw overlap
    because the failure mode we care about is a candidate that matches one loud
    term and misses the rest of the question.

    ⚠️ COVERAGE IS COMPUTED OVER CONTENT TERMS ONLY, and this is load-bearing
    rather than cosmetic. FRIDAY's queries are conversational — "where do I live",
    "what is my monthly rent" — and a fact body is terse: `lives_in: Chennai`. If
    the stopwords count toward coverage then 3 of the 4 query terms can never
    possibly appear, coverage is capped near 0.25, and EVERY conversational query
    scores 0.000 and is dropped below the floor. That is not a conservative
    reranker, it is a dead one: the recall paths find the right fact and the
    reranker throws it away. `friday.store.db.fts_terms` already strips stopwords
    for exactly this reason; the reranker has to agree with the recall layer or the
    funnel discards what recall worked to find.
    """

    def __init__(self) -> None:
        self.name = "lexical"
        #: ⚠️ Higher than a cross-encoder's floor. A weaker reranker must be MORE
        #: willing to return [] — Law 4 says an empty list beats noise, and with a
        #: weak scorer the noise risk is higher, not lower.
        self.recommended_floor = 0.45

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        # The possessive strip is cluster-conditional ON PURPOSE (see
        # concepts.strip_possessives): a blind strip is a measured false positive.
        raw = {stem(w) for w in _WORD.findall(strip_possessives(query.lower()))}
        q = raw - _STOPWORD_STEMS
        # A query that is ALL stopwords ("what is it") carries no content signal, so
        # fall back to the raw terms rather than scoring against an empty set — the
        # same fallback `fts_terms` uses, for the same reason.
        if not q:
            q = raw
        if not q:
            return [0.0] * len(candidates)
        # ⭐ "what time" as an intent cue, THING-SCOPED (doc 20 §5). `time` maps to
        # the temporal concept only when the query also names a thing ("my standup
        # time"), because a bare `time` is a mass noun — "how much time do I have"
        # would otherwise be answered by any schedule fact, at coverage 1/1. The
        # cue lives here rather than in the recall layers because the miss it fixes
        # is a coverage miss: recall FINDS the fact, the floor drops it.
        if "tim" in q and len(q) > 1:
            q = {_TEMPORAL if s == "tim" else s for s in q}
        # Coverage is judged over CONCEPTS, not literal words: "report" in a question
        # and "manager" in a fact are the same concept. Stopwords were already removed
        # above, on stems, so the mapping only ever sees content terms.
        #
        # ⚠️ Coverage keeps DUPLICATE concept hits on the query side (a list, not a set).
        # "rent" and "monthly" both map to one concept; collapsing them to a set made
        # "what do I pay in rent each month" lose a term it had earned, and the
        # invariance test in test_retrieval.py caught it. Overlap still uses sets.
        q_terms = [concept_of_stem(s) for s in q]
        q_set = set(q_terms)
        ql = query.lower().strip()
        out = []
        for c in candidates:
            cl = strip_possessives(c.lower())
            ct = {_TEMPORAL if stem(w) in _WEEKDAY_STEMS else concept_of_stem(stem(w))
                  for w in _WORD.findall(cl)}
            if _CLOCK_TIME.search(cl):
                ct.add(_TEMPORAL)
            if not ct:
                out.append(0.0)
                continue
            hits = sum(1 for t in q_terms if t in ct)
            coverage = hits / len(q_terms)
            inter = q_set & ct
            overlap = len(inter) / len(q_set | ct)
            phrase = 0.15 if ql and ql in cl else 0.0
            out.append(min(1.0, 0.62 * coverage + 0.38 * (overlap * 2.2) + phrase))
        return out


def get_reranker(prefer: str = "auto", model_dir: str = "models/reranker") -> Reranker:
    """Best available. Never raises."""
    if prefer in ("auto", "onnx"):
        try:
            return OnnxReranker(model_dir)
        except Exception as exc:
            swallowed('friday/retrieval/rerankers.py:183', exc)
    return LexicalReranker()
