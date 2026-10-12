"""DDSG — the Decoupled Delta State Gate, FRIDAY's recurrent operator contract.

Reference implementation of the exact GDN-2 update ruled in
docs/architecture/13 §1.2, exposed under the name this project gave the idea
("Decoupled Delta State Gate") BEFORE the paper confirmed it. Honesty clause:
the recurrence is GDN-2's (NVlabs, arXiv 2605.22791); what is OURS is the
contract — an executable oracle any future numpy/Triton port must match
bit-for-approx, plus the test suite that pins the semantics. Claims about
novelty end there; the tests below are where correctness lives.

THE OPERATOR (state S is dk rows × dv cols; keys/queries live in R^dk,
                          values in R^dv; keys are L2-normalised at the block):

        S_t = ( I − k_t (b_t ⊙ k_t)ᵀ ) D_t S_{t−1}  +  k_t (w_t ⊙ v_t)ᵀ
        o_t = S_tᵀ q_t

                decay FIRST (D_t = Diag(α_t), channel-wise), then the targeted
                edit, then the write — order pinned by test_decay_runs_before_the_edit.

WHY DECOUPLED: with one scalar β (DeltaNet, KDA) erase and write are married —
you cannot add an association without scratching at the old one, nor scrub one
without writing something. Decoupled gates give FOUR modes, and the paper's
ablation says the ERASE gate carries most of the NIAH gain (63 → 90 on RULER
S-NIAH-3 @8K) — which is FRIDAY's entire memory problem:

    b=0, w>0        ADD     — pure new association (nothing old disturbed)
    b=1(k-dir),w=0  DELETE  — targeted scrub: read at that key returns ~0
    b>0, w>0        REPLACE — the classic delta-rule edit, per-channel now
    b=0, w=0        KEEP    — state passes through (times decay)

TIE-DOWNS (free control conditions, asserted as tests — not prose):
    b = β·1_dk, w = β·1_dv, α = 1      → classical DeltaNet exactly
    b = β·1_dk, w = β·1_dv, α = ᾱ·1    → Gated DeltaNet exactly
so the upgrade is falsifiable: ablate from tied scalars upward and every gate
shows what it bought.

Numerics: pure Python floats, tiny dims — this is an oracle, not a runtime.
The GPU port (Lane C of doc 19, chunkwise WY-form in Triton) reads THESE
golden vectors as its acceptance test before it earns NIAH measurements.
"""
from __future__ import annotations

import math


def zeros(dk: int, dv: int) -> list[list[float]]:
    """An empty state, dk rows × dv cols."""
    return [[0.0] * dv for _ in range(dk)]


def l2_normalize(x: list[float]) -> list[float]:
    """Keys enter the gate unit-length (the block design L2-normalises Q/K).
    Unit keys are what make 'erase gate = 1' mean 'reading at this key is
    exactly zero' rather than 'scaled by |k|²'."""
    n = math.sqrt(sum(xi * xi for xi in x))
    if n == 0.0:
        raise ValueError("zero vector has no direction")
    return [xi / n for xi in x]


def step(S: list[list[float]],
         k: list[float], q: list[float], v: list[float],
         alpha: list[float], b: list[float], w: list[float],
         ) -> tuple[list[list[float]], list[float]]:
    """One recurrent update + read, exactly the formula in the module docstring.

    No validation beyond shape asserts: this is an oracle — loud failure on
    nonsense input beats forgiving failure that ships wrong numbers downstream.
    """
    dk, dv = len(k), len(v)
    assert len(q) == dk, "query must match key dimension"
    assert len(alpha) == dk and len(b) == dk, "decay/erase are channel-wise on dk"
    assert len(w) == dv, "write gate is channel-wise on dv"
    assert len(S) == dk and all(len(row) == dv for row in S), "state must be dk×dv"

    # 1) DECAY FIRST: SD = Diag(alpha) S  — broad clearing before the edit.
    SD = [[alpha[i] * S[i][j] for j in range(dv)] for i in range(dk)]

    # 2) TARGETED EDIT: (I − k (b⊙k)ᵀ) SD.
    #    The left factor is the raw key (edit happens AT key k);
    #    the right factor b⊙k makes erasure channel-selective.
    bk = [b[i] * k[i] for i in range(dk)]
    r = [sum(bk[i] * SD[i][j] for i in range(dk)) for j in range(dv)]  # (b⊙k)ᵀ SD
    Se = [[SD[i][j] - k[i] * r[j] for j in range(dv)] for i in range(dk)]

    # 3) WRITE: + k (w⊙v)ᵀ — value channels gated independently of the erase.
    wv = [w[j] * v[j] for j in range(dv)]
    S1 = [[Se[i][j] + k[i] * wv[j] for j in range(dv)] for i in range(dk)]

    # 4) READ: o = Sᵀ q — read happens AFTER the update (recurrent convention).
    o = [sum(q[i] * S1[i][j] for i in range(dk)) for j in range(dv)]
    return S1, o


def run(seq: list[dict], dk: int, dv: int):
    """Convenience: fold `step` over a token sequence from a zero state.
    Each dict carries k,q,v,alpha,b,w. Returns (final_state, list_of_reads).
    Exists so golden-vector tests read as DATA, not as control flow."""
    S = zeros(dk, dv)
    reads = []
    for t in seq:
        S, o = step(S, t["k"], t["q"], t["v"], t["alpha"], t["b"], t["w"])
        reads.append(o)
    return S, reads
