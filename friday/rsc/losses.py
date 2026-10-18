"""The five-term RSC loss. docs/architecture/12 §6.3, extended by /13 §5.1.

    L_total = l1*L_reconstruction + l2*L_recall + l3*L_salience
            + l4*L_erase          + l5*L_decay

Two of these are the load-bearing walls and both exist because of a measured
failure elsewhere in this project:

  * `L_recall` — WITHOUT IT THE RSC LEARNS TO DROP VERBATIM DETAIL. That is the
    92%->33% summarisation collapse and RetNet's near-zero long-range recall, and
    a compressor trained only on reconstruction will reproduce both, because
    dropping a rare exact token barely moves the KL.
  * `L_erase` — the bi-temporal correction loss. Its labels come free from
    retraction pairs, and it is the term that decides GDN-2 vs EDA empirically: if
    L_erase plateaus under GDN-2 and drops under EDA, the address-level coupling
    was real for YOUR data. That's a publishable result either way.

`L_decay` is a CURRICULUM, not a constraint: strong early (alpha_t regressed toward
the hand-written decay policy) then annealed to zero so alpha_t is learned freely.
A working prior beats a cold start, and annealing means the prior never becomes a
ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


# ── primitives ─────────────────────────────────────────────────────────────────

def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-9) -> float:
    """KL(p || q) over two logit vectors, softmaxed internally.

    p = teacher (backbone with FULL history in context)
    q = student (backbone with RSC state + short recent window)
    """
    ps = _softmax(p)
    qs = _softmax(q)
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(ps, qs))


def _softmax(x: Sequence[float]) -> list[float]:
    if not x:
        return []
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = sum(e) or 1.0
    return [v / z for v in e]


def bce(pred: float, target: float, eps: float = 1e-7) -> float:
    p = min(1 - eps, max(eps, pred))
    return -(target * math.log(p) + (1 - target) * math.log(1 - p))


def mse(a: Sequence[float], b: Sequence[float]) -> float:
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


# ── the five terms ─────────────────────────────────────────────────────────────

def l_reconstruction(teacher_logits: Sequence[Sequence[float]],
                     student_logits: Sequence[Sequence[float]]) -> float:
    """Mean KL over continuation positions.

    Teacher: frozen backbone with the FULL history in context.
    Student: frozen backbone with the RSC's fixed-size state + a short recent window.
    If the state carries the history's *gist*, the two distributions agree.
    """
    n = min(len(teacher_logits), len(student_logits))
    if n == 0:
        return 0.0
    return sum(kl_divergence(teacher_logits[i], student_logits[i]) for i in range(n)) / n


def l_recall(state_read_fn: Callable[[str], float], needles: Sequence[dict]) -> float:
    """⭐ The guard rail. Negative log-probability of the EXACT needle tokens.

    `state_read_fn(query) -> p` returns the model's probability for the verbatim
    needle value given only the compiled state (plus the query). Each needle is
    {"query": …, "value": …, "depth": …} — the same shape
    `scripts/needle_test.py` generates, so training and eval share a definition of
    "recall" instead of having two.

    Depth is included in the dict so you can weight deep needles harder: a
    compressor that remembers position 500 and forgets position 7500 is exactly the
    failure the literature reports, and an unweighted mean hides it.
    """
    if not needles:
        return 0.0
    total = 0.0
    for nd in needles:
        p = max(1e-6, min(1.0, float(state_read_fn(nd["query"]))))
        weight = 1.0 + _depth_weight(nd.get("depth"))
        total += weight * -math.log(p)
    return total / len(needles)


def _depth_weight(depth: Any) -> float:
    try:
        d = float(depth)
    except (TypeError, ValueError):
        return 0.0
    # linearly up-weight needles planted deeper: 0 at depth 0, +1 at depth 8000
    return min(1.0, d / 8000.0)


def l_salience(preds: Sequence[float], labels: Sequence[int]) -> float:
    """BCE against the Memory Compiler's own output.

    Labels are free: any span that produced an S2 fact is positive
    (memory/supervision.SpanLabel). No human annotation, no cloud teacher.
    """
    n = min(len(preds), len(labels))
    if n == 0:
        return 0.0
    pos = sum(1 for y in labels[:n] if y)
    neg = n - pos
    # class-balance: positives are rare (most spans don't produce facts), and an
    # unbalanced BCE here just learns "always predict 0"
    w_pos = (n / (2.0 * pos)) if pos else 1.0
    w_neg = (n / (2.0 * neg)) if neg else 1.0
    total = 0.0
    for i in range(n):
        y = float(labels[i])
        total += (w_pos if y else w_neg) * bce(preds[i], y)
    return total / n


def l_erase(p_superseded_fn: Callable[[str, str], float],
            pairs: Sequence[dict]) -> float:
    """⭐ The bi-temporal correction loss. Free labels from retraction pairs.

    For every retraction event (fact_old.valid_to := now, fact_new asserted), the
    state must NOT return the superseded value when queried at "now":

        L_erase = -log( 1 - p_state(value_old | query(key_old), t=now) )

    This is the loss that makes EDA's independent erase address *necessary*. Under
    GDN-2 the erase direction is b_t ⊙ k_t — built from the CURRENT write key — so
    when key_old != key_new the model has no mechanism to reach the stale
    association. If this term plateaus on GDN-2 and drops on EDA, you have measured
    the address-level coupling on your own data.

    Note what is NOT in this loss: "the state must still support 'what did I used
    to think?'". It must not — that query is routed to the external store, which
    retains both values with their validity windows (Law 2b). Assert that in the
    eval, never in the loss.
    """
    if not pairs:
        return 0.0
    total = 0.0
    for pr in pairs:
        p = max(0.0, min(1.0 - 1e-6, float(p_superseded_fn(pr["key_old"], pr["value_old"]))))
        # address-distinct pairs are the EDA case; weight them so the ladder's rung 5
        # is judged on exactly the examples that discriminate it
        w = 2.0 if pr.get("address_distinct") else 1.0
        total += w * -math.log(1.0 - p)
    return total / len(pairs)


def l_decay(alpha_pred: Sequence[float], alpha_policy: Sequence[float],
            *, weight: float = 1.0) -> float:
    """Curriculum term: align channel-wise alpha_t with the hand-specified decay
    policy, then anneal `weight` to 0.

    `alpha_policy` comes from `memory.decay.policy_alpha(predicate_class, …)`,
    which converts a per-class half-life in days into a per-token retention.
    """
    if weight <= 0.0:
        return 0.0
    return weight * mse(alpha_pred, alpha_policy)


# ── the total ──────────────────────────────────────────────────────────────────

@dataclass
class LossWeights:
    reconstruction: float = 1.0
    recall: float = 1.0        # ⭐ never set this to 0
    salience: float = 0.5
    erase: float = 1.0         # ⭐ the GDN-2 vs EDA discriminator
    decay: float = 0.3         # annealed toward 0 over training

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction, "recall": self.recall,
            "salience": self.salience, "erase": self.erase, "decay": self.decay,
        }


@dataclass
class LossReport:
    terms: dict[str, float]
    weights: dict[str, float]
    total: float

    def line(self) -> str:
        parts = " ".join(f"{k}={v:.4f}" for k, v in self.terms.items())
        return f"L={self.total:.4f}  {parts}"


def total_loss(
    *,
    weights: LossWeights,
    teacher_logits: Sequence[Sequence[float]] = (),
    student_logits: Sequence[Sequence[float]] = (),
    recall_fn: Callable[[str], float] | None = None,
    needles: Sequence[dict] = (),
    salience_preds: Sequence[float] = (),
    salience_labels: Sequence[int] = (),
    erase_fn: Callable[[str, str], float] | None = None,
    retraction_pairs: Sequence[dict] = (),
    alpha_pred: Sequence[float] = (),
    alpha_policy: Sequence[float] = (),
    decay_weight_scale: float = 1.0,
) -> LossReport:
    """Compute all five terms and the weighted total.

    Missing inputs contribute 0.0 rather than raising — a partial batch (no needles
    this time, no retractions yet) must still train, because in Week 2 you will have
    salience labels and no retraction pairs, and the alternative is a trainer that
    refuses to start until your life has produced a contradiction.
    """
    terms: dict[str, float] = {}
    terms["reconstruction"] = l_reconstruction(teacher_logits, student_logits)
    terms["recall"] = l_recall(recall_fn, needles) if (recall_fn and needles) else 0.0
    terms["salience"] = l_salience(salience_preds, salience_labels)
    terms["erase"] = (
        l_erase(erase_fn, retraction_pairs) if (erase_fn and retraction_pairs) else 0.0
    )
    terms["decay"] = l_decay(alpha_pred, alpha_policy,
                             weight=weights.decay * decay_weight_scale)

    w = weights.as_dict()
    total = sum(w[k] * terms[k] for k in terms)
    return LossReport(terms=terms, weights=w, total=total)


def anneal_decay_weight(step: int, total_steps: int, start: float = 1.0,
                        floor: float = 0.0) -> float:
    """Cosine anneal the L_decay curriculum weight from `start` to `floor`.

    The prior should shape the model early and get out of the way later. If you
    leave it on, alpha_t converges to your hand-tuned half-lives and you have
    learned nothing — you've just implemented `policy_alpha` in 40M parameters.
    """
    if total_steps <= 0:
        return floor
    frac = min(1.0, max(0.0, step / total_steps))
    return floor + (start - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
