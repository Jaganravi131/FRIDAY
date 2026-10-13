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

import re
from typing import Protocol, Sequence

from ..store.db import _STOPWORDS
from ..util import stem
from .synonyms import expansions_for

#: ⚠️ `[^\W_]+`, NOT `\w+`. In Python `\w` includes the underscore, so the naive
#: pattern tokenises the stored predicate `lives_in` as ONE indivisible token. The
#: user asks "where do I live"; `live` can never equal `lives_in`, so coverage is 0
#: and every snake_case fact scores 0.000 regardless of how good the recall was.
#: Excluding `_` splits predicates into their words, which is what a lexical scorer
#: needs in order to compare a question against a field name at all.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


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

    #: ⭐ A pure synonym-bridge match carries a small need-scaled bonus. When a
    #: candidate shares NO incidental word with the query and the ONLY connection
    #: is a human-curated relation from `synonyms.py`, that is higher-precision
    #: evidence than an incidental exact hit, not lower — accidental matches are
    #: exact ("time" appears everywhere), while a curated bridge fires only where
    #: a person wrote the relation down. Without this, a query where half the
    #: content is circumstantial ("what do I drink IN THE MORNING") can never
    #: cross the floor no matter how exact the bridged half is, because coverage
    #: caps at 0.5 and 0.62 * 0.5 + overlap tops out around 0.41.
    #:
    #: The bonus scales with the UNCOVERED fraction (1 - coverage) and is ZERO
    #: at full coverage. That shape matters: a flat bonus would crown a bridged
    #: candidate over an exact one when both cover the query ("what is my rent"
    #: must rank the rent fact above the lease-end fact, not below it). Here the
    #: bonus only compensates for query words neither exact nor bridged — the
    #: circumstantial words a fact cannot be expected to contain — and it can
    #: never flip a full-coverage tie. It also never fires for an unmapped
    #: query, so the false-positive gate, not this number, keeps unanswerable
    #: queries at [].
    BRIDGE_BONUS = 0.18

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        raw_words = [w for w in _WORD.findall(query.lower())]
        raw = {stem(w) for w in raw_words}
        q = raw - _STOPWORDS
        # A query that is ALL stopwords ("what is it") carries no content signal, so
        # fall back to the raw terms rather than scoring against an empty set — the
        # same fallback `fts_terms` uses, for the same reason.
        if not q:
            q = raw
        if not q:
            return [0.0] * len(candidates)
        # Query words with content standing (raw form kept for the synonym lookup:
        # "reports" reaches the "report" entry through the stemmer below, but the
        # lookup itself is on whole words). Everything below is in STEM units, so
        # when no expansion fires the arithmetic is identical to the old formula.
        syn_map: dict[str, set[str]] = {}
        for w in raw_words:
            s = stem(w)
            if s in q:
                syn_map.setdefault(s, set()).update(
                    stem(e) for e in expansions_for(w))
        for v in syn_map.values():
            v -= _STOPWORDS
        expansions = set().union(*syn_map.values()) if syn_map else set()
        expansions -= q
        ql = query.lower().strip()
        out = []
        for c in candidates:
            cl = c.lower()
            ct = {stem(w) for w in _WORD.findall(cl)}
            if not ct:
                out.append(0.0)
                continue
            inter = q & ct
            bridged = {s for s in q
                       if s not in ct and syn_map.get(s, set()) & ct}
            coverage = (len(inter) + len(bridged)) / len(q)
            hit_expansions = (expansions & ct) - inter
            # ⭐ Bridged terms count slightly less as SHARED vocabulary than
            # exact hits do (0.9), while counting fully toward COVERAGE above.
            # Coverage asks "does the candidate address the query's words" —
            # a curated relation answers yes. Overlap asks "how much vocabulary
            # is literally shared" — a hypothesised term is weaker evidence of
            # sharing than a word both sides actually contain. The practical
            # effect is tie-breaking: at equal coverage the exact match
            # outranks the bridged one. With no expansions this reduces exactly
            # to len(inter) / len(q | ct).
            overlap = (len(inter) + 0.9 * len(hit_expansions)) / len(q | expansions | ct)
            phrase = 0.15 if ql and ql in cl else 0.0
            bridge = self.BRIDGE_BONUS * (1.0 - coverage) if not inter and bridged else 0.0
            out.append(min(1.0, 0.62 * coverage + 0.38 * (overlap * 2.2)
                           + phrase + bridge))
        return out


def get_reranker(prefer: str = "auto", model_dir: str = "models/reranker") -> Reranker:
    """Best available. Never raises."""
    if prefer in ("auto", "onnx"):
        try:
            return OnnxReranker(model_dir)
        except Exception:
            pass
    return LexicalReranker()
