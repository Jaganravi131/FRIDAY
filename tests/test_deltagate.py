"""Golden vectors for the Decoupled Delta State Gate (friday/research/deltagate.py).

Every expected number in this file was derived BY HAND from the formula
    S_t = ( I − k (b⊙k)ᵀ ) Diag(α) S_{t−1} + k (w⊙v)ᵀ , o = Sᵀ q
(docs/architecture/13 §1.2) — that is what makes this an oracle rather than a
test that circularly ratifies its own implementation. The future Triton port
must reproduce these numbers; the tie-down tests assert the module degrades
EXACTLY into DeltaNet and Gated DeltaNet when its new gates are tied, which is
what makes the decoupling a falsifiable upgrade instead of a rename.
"""
from __future__ import annotations

import pytest

from friday.research.deltagate import l2_normalize, run, step, zeros


def close(a, b):
    """approx-equality that also accepts nested rows (the state is a list of
    rows; approx() itself refuses nested structures)."""
    def flat(x):
        out = []
        for item in x:
            if isinstance(item, (list, tuple)):
                out.extend(flat(item))
            else:
                out.append(item)
        return out
    assert flat(a) == pytest.approx(flat(b), abs=1e-12)


e1, e2 = [1.0, 0.0], [0.0, 1.0]
ONES2, ZEROS2 = [1.0, 1.0], [0.0, 0.0]


def test_pure_add_reads_back_and_neither_key_disturbs_the_other():
    """ADD mode (b=0): two independent writes, each read back exactly."""
    S, o1 = step(zeros(2, 2), e1, e1, [3.0, 5.0], ONES2, ZEROS2, ONES2)
    close([S[0], S[1]], [[3.0, 5.0], [0.0, 0.0]])
    close(o1, [3.0, 5.0])
    S, o2 = step(S, e2, e2, [7.0, 11.0], ONES2, ZEROS2, ONES2)
    close([S[0], S[1]], [[3.0, 5.0], [7.0, 11.0]])
    close(o2, [7.0, 11.0])
    _, back = step(S, e1, e1, [0.0, 0.0], ONES2, ZEROS2, ZEROS2)  # KEEP-mode read
    close(back, [3.0, 5.0])


def test_targeted_erase_zeroes_only_its_own_direction():
    """DELETE mode: b=1 on the key channel removes THAT association; the
    orthogonal one survives untouched. S2 = [[3,5],[7,11]] from the add test."""
    S = [[3.0, 5.0], [7.0, 11.0]]
    S, o = step(S, e1, e1, [99.0, 99.0], ONES2, [1.0, 0.0], ZEROS2)
    close([S[0], S[1]], [[0.0, 0.0], [7.0, 11.0]])
    close(o, [0.0, 0.0])                      # the scrubbed key reads as nothing
    close([S[1][0], S[1][1]], [7.0, 11.0])    # the other memory is intact


def test_partial_erase_scales_the_reading_by_one_minus_b():
    """b=0.5 on the key channel halves that channel's reading (delta-rule
    semantics), WITH decay already applied: α=(0.5,1) turns [3,5] into
    [1.5,2.5] first, then the half-erase leaves [0.75,1.25]."""
    S = [[3.0, 5.0], [7.0, 11.0]]
    S, o = step(S, e1, e1, [0.0, 0.0], [0.5, 1.0], [0.5, 0.0], ZEROS2)
    close([S[0], S[1]], [[0.75, 1.25], [7.0, 11.0]])
    close(o, [0.75, 1.25])


def test_decay_runs_BEFORE_the_edit():
    """The ordering discriminator. k=(0.6,0.8) mixes rows, so decay-first and
    edit-first genuinely differ; hand-derived decay-first numbers:

        Diag(α)S = [[1.5,2.5],[7,11]]
        (b⊙k)ᵀ(DS) = (6.5, 10.3)
        DS − k·(b⊙k)ᵀ(DS) = [[-2.4,-3.68],[1.8,2.76]]

    (edit-then-decay would give [[-0.72,-1.04],[1.08,1.56]] — if this test
    ever starts matching THOSE numbers, the formula order was flipped)."""
    S = [[3.0, 5.0], [7.0, 11.0]]
    k = [0.6, 0.8]  # already unit length
    S, o = step(S, k, k, [0.0, 0.0], [0.5, 1.0], [1.0, 1.0], ZEROS2)
    close([S[0], S[1]], [[-2.4, -3.68], [1.8, 2.76]])
    close(o, [0.0, 0.0])  # full erase reads as EXACT zero at any unit direction


def test_state_is_dk_by_dv_not_its_transpose():
    """dk=2 rows × dv=3 cols: catches the classic S↔Sᵀ porting bug."""
    S, o1 = step(zeros(2, 3), e1, e1, [2.0, 4.0, 6.0], ONES2, ZEROS2, [1, 1, 1])
    close(S[0], [2.0, 4.0, 6.0]); close(S[1], [0.0, 0.0, 0.0])
    close(o1, [2.0, 4.0, 6.0])
    S, o2 = step(S, e2, e2, [1.0, 1.0, 1.0], ONES2, ZEROS2, [1, 1, 1])
    close(S[1], [1.0, 1.0, 1.0])
    S, _ = step(S, e1, e1, [0.0, 0.0, 0.0], ONES2, [1.0, 1.0], [0, 0, 0])
    close([S[0], S[1]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_write_gate_is_channel_selective_on_the_value():
    """w=0.5 halves what each VALUE channel commits — independent of erase."""
    S, o = step(zeros(2, 2), e1, e1, [3.0, 5.0], ONES2, ZEROS2, [0.5, 0.5])
    close([S[0], S[1]], [[1.5, 2.5], [0.0, 0.0]])
    close(o, [1.5, 2.5])


def test_tied_scalar_gates_recover_classical_DELTANET_exactly():
    """b = w = β·1, α = 1  →  S_t = (I − βkkᵀ)S + βkvᵀ (DeltaNet), computed here
    by an INDEPENDENT in-test implementation. The upgrade must contain its own
    control condition or it is not an upgrade."""
    beta = 0.25
    S = [[1.0, -1.0], [0.5, 2.0]]
    k1 = l2_normalize([3.0, 4.0])       # (0.6, 0.8)
    k2 = l2_normalize([-1.0, 2.0])      # (-1/√5, 2/√5)
    v1, v2 = [2.0, -3.0], [4.0, 1.0]

    def classic(S_, k_, v_):
        r = [sum(k_[i] * S_[i][j] for i in range(2)) for j in range(2)]
        return [[S_[i][j] - beta * k_[i] * r[j] + beta * k_[i] * v_[j]
                 for j in range(2)] for i in range(2)]

    for kk, vv in ((k1, v1), (k2, v2)):
        S_new, _ = step(S, kk, kk, vv, ONES2, [beta, beta], [beta, beta])
        exp = classic(S, kk, vv)
        close([S_new[0], S_new[1]], exp)
        S = exp


def test_tied_gates_with_scalar_decay_recover_GATED_DELTANET_exactly():
    """b = w = β·1, α = ᾱ·1  →  S_t = (I − βkkᵀ)(ᾱI)S + βkvᵀ — the ICLR-2025
    Gated DeltaNet, again via independent in-test computation."""
    beta, a_bar = 0.3, 0.5
    S = [[2.0, 0.0], [1.0, -1.0]]
    k = l2_normalize([1.0, 1.0])
    v = [1.0, 2.0]

    def gdn(S_, k_, v_):
        D = [[a_bar * S_[i][j] for j in range(2)] for i in range(2)]
        r = [sum(k_[i] * D[i][j] for i in range(2)) for j in range(2)]
        return [[D[i][j] - beta * k_[i] * r[j] + beta * k_[i] * v_[j]
                 for j in range(2)] for i in range(2)]

    S_new, _ = step(S, k, k, v, [a_bar, a_bar], [beta, beta], [beta, beta])
    close([S_new[0], S_new[1]], gdn(S, k, v))


def test_run_folds_a_sequence_and_l2_normalize_rejects_zero():
    seq = [
        {"k": e1, "q": e1, "v": [3.0, 5.0], "alpha": ONES2, "b": ZEROS2, "w": ONES2},
        {"k": e2, "q": e2, "v": [7.0, 11.0], "alpha": ONES2, "b": ZEROS2, "w": ONES2},
        {"k": e1, "q": e1, "v": [0.0, 0.0], "alpha": ONES2, "b": [1.0, 0.0], "w": ZEROS2},
    ]
    S, reads = run(seq, dk=2, dv=2)
    close([reads[0], reads[1], reads[2]], [[3.0, 5.0], [7.0, 11.0], [0.0, 0.0]])
    close([S[0], S[1]], [[0.0, 0.0], [7.0, 11.0]])
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])
    with pytest.raises(AssertionError):
        step(zeros(2, 2), e1, [1.0, 0.0, 0.0], [0.0, 0.0], ONES2, ZEROS2, ZEROS2)
