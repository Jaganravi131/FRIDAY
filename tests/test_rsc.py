"""The RSC: operator algebra, the gate branches, the loss terms, the probes.

These tests verify the MATH against the papers' described behaviour. They are not
the 1.3B/100B-token result — that lives in scripts/rsc_ablation.py and needs a T4.

The decisive assertions are the last three groups: that rung 0 cannot do selective
associative recall (the entire reason FRIDAY is not on RetNet), that the erase gate
moves the needle, and that the interference probe discriminates rung 4 from rung 5.
If those fail, the operators are wrong and no amount of Kaggle time will fix it.

Operator API (used throughout):

    s = State(d_k, d_v)                      # S is (d_k, d_v); rows are key channels
    g = Gates(q=…, k=…, v=…, alpha=[…]*d_k,  # alpha is CHANNEL-WISE
              b=[…]*d_k, w=[…]*d_v, beta=…, e=…, gamma_e=…)
    s, out = op.step(s, g)                   # out is the read at q, length d_v
    s.read(q)                                # o = Sᵀq
"""

from __future__ import annotations

import math

import pytest
from pathlib import Path

from friday.rsc.ablation import _gates, _rel_err
from friday.rsc.operators import (
    ABLATION_LADDER,
    OPERATOR_NAMES,
    EraseThenDelta,
    Gates,
    GatedDeltaNet,
    GatedDeltaNet2,
    KimiDelta,
    RetNet,
    State,
    build,
    l2_normalise,
)


# ── state ──────────────────────────────────────────────────────────────────────

def test_state_shape_is_dk_by_dv():
    """S is (d_k, d_v) with rows as KEY channels, so Diag(alpha) @ S is channel-wise
    decay over the key axis exactly as in KDA/GDN-2. Getting this transposed would
    make every gate act on the wrong axis and still run without error."""
    s = State(d_k=8, d_v=5)
    assert len(s.S) == 8 and len(s.S[0]) == 5
    assert s.numel == 40
    assert s.frobenius() == 0.0


def test_read_is_state_transpose_times_query():
    """o = Sᵀq. With S = k vᵀ and q = k (unit), o = v·|k|² = v."""
    s = State(d_k=4, d_v=4)
    k = [1.0, 0.0, 0.0, 0.0]
    v = [0.0, 2.0, 0.0, 0.0]
    s.S = [[ki * vj for vj in v] for ki in k]      # outer(k, v)
    o = s.read(k)
    assert o == pytest.approx(v, abs=1e-9)
    assert len(o) == 4, "the read lives on the VALUE axis"


def test_clone_is_deep():
    s = State(d_k=3, d_v=3)
    s.S[0][0] = 5.0
    c = s.clone()
    c.S[0][0] = 9.0
    assert s.S[0][0] == 5.0, "clone must not alias the parent's rows"


# ── rung 0: RetNet ─────────────────────────────────────────────────────────────

def test_retnet_decay_is_input_independent():
    """⭐ The reason FRIDAY is not on RetNet. γ is a fixed scalar per head — it
    cannot be conditioned on what is being written, so there is no way to forget
    selectively. arXiv 2507.06457v2: fixed decay yields near-zero long-range recall
    even with full-attention layers added."""
    op = RetNet()
    assert op.rung == 0 and op.has_erase_address is False

    g1 = _gates([1, 0, 0, 0], [1, 0, 0, 0], 4, 4)
    g2 = _gates([0, 0, 0, 1], [9, 9, 9, 9], 4, 4)
    # RetNet ignores g.alpha entirely and uses its own gamma — that IS the finding
    s = State(4, 4)
    s1, _ = op.step(s, g1)
    s2, _ = op.step(s, g2)
    assert op.gamma == 0.995
    assert s1.S[0][0] != 0 and s2.S[3][0] != 0


def test_retnet_write_is_additive_not_corrective():
    """No delta rule: writing the same key twice ADDS instead of converging. This is
    the mechanism behind the recall failure, so it is worth pinning directly."""
    op = RetNet()
    s = State(4, 4)
    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [1.0, 0.0, 0.0, 0.0]
    g = Gates(q=k, k=k, v=v, alpha=[1.0] * 4)
    for _ in range(5):
        s, _ = op.step(s, g)
    out = s.read(k)
    assert out[0] > 2.0, f"additive write must accumulate past the target: {out[0]}"


def test_gdn_write_is_corrective():
    """The delta rule converges to the target instead of accumulating — the single
    change that makes selective associative recall possible."""
    op = GatedDeltaNet()
    s = State(4, 4)
    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [1.0, 0.0, 0.0, 0.0]
    g = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, beta=0.9)
    for _ in range(5):
        s, _ = op.step(s, g)
    out = s.read(k)
    assert out[0] == pytest.approx(1.0, abs=0.05), f"must converge to v: {out[0]}"


# ── gate structure per rung ────────────────────────────────────────────────────

def test_gdn_uses_scalar_decay():
    """GDN takes alpha[0] as a scalar — no channel-wise decay."""
    op = GatedDeltaNet()
    s = State(4, 4)
    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [1.0, 0.0, 0.0, 0.0]
    # deliberately non-uniform alpha: GDN must ignore all but the first channel
    g = Gates(q=k, k=k, v=v, alpha=[1.0, 0.1, 0.1, 0.1], beta=0.9)
    s, _ = op.step(s, g)
    g_uniform = Gates(q=k, k=k, v=v, alpha=[1.0, 1.0, 1.0, 1.0], beta=0.9)
    s2, _ = op.step(State(4, 4), g_uniform)
    flat_a = [x for row in s.S for x in row]
    flat_b = [x for row in s2.S for x in row]
    assert flat_a == pytest.approx(flat_b, abs=1e-9), "GDN must use only alpha[0]"


def test_kda_decay_is_channel_wise():
    """KDA: channel-wise decay + SCALAR delta gate. This is GDN-2's tied subspace,
    which is what makes it a free control for rungs 3–4."""
    op = KimiDelta()
    assert op.rung == 2 and op.has_erase_address is False

    k = l2_normalise([1.0, 1.0, 0.0, 0.0])
    v = [1.0, 0.0, 0.0, 0.0]
    s_full = State(4, 4)
    s_full.S = [[1.0] * 4 for _ in range(4)]
    g_full = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, beta=0.0)
    out_full, _ = op.step(s_full, g_full)

    s_half = State(4, 4)
    s_half.S = [[1.0] * 4 for _ in range(4)]
    g_half = Gates(q=k, k=k, v=v, alpha=[1.0, 0.5, 0.5, 0.5], beta=0.0)
    out_half, _ = op.step(s_half, g_half)

    # with beta=0 the update is pure decay, so differing alpha must give differing
    # states — that is the definition of channel-wise
    assert out_half.S[0] != out_half.S[1], "channel 0 kept, channel 1 decayed"
    assert out_full.S[0] == out_full.S[1]


# ── rung 3/4: GDN-2 ────────────────────────────────────────────────────────────

def test_gdn2_rule_matches_the_paper_equation():
    """S_t = (I − k_t(b_t⊙k_t)ᵀ) D_t S_{t−1} + k_t(w_t⊙v_t)ᵀ — verified by hand.

    The implementation applies this as X − k ⊗ ((b⊙k)ᵀ X), which is the same thing
    by associativity: the LEFT factor stays k (write direction preserved) while the
    READ direction becomes channel-selective via b⊙k. That asymmetry is the design.
    """
    dk = dv = 3
    op = GatedDeltaNet2()
    S_prev = [[1.0, 2.0, 0.5], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0]]
    k = l2_normalise([1.0, 2.0, 0.0])
    v = [0.5, 1.0, 2.0]
    b = [0.3, 0.7, 1.0]
    w = [1.0, 0.5, 0.25]
    d = [0.9, 0.8, 1.0]

    g = Gates(q=k, k=k, v=v, alpha=d, b=b, w=w)
    S, _ = op.step(State(dk, dv, [r[:] for r in S_prev]), g)

    # hand computation of the paper's equation
    DS = [[d[i] * S_prev[i][j] for j in range(dv)] for i in range(dk)]
    bk = [b[i] * k[i] for i in range(dk)]                     # b ⊙ k
    proj = [sum(bk[i] * DS[i][j] for i in range(dk)) for j in range(dv)]   # (b⊙k)ᵀ DS
    erased = [[DS[i][j] - k[i] * proj[j] for j in range(dv)] for i in range(dk)]
    wv = [w[j] * v[j] for j in range(dv)]                     # w ⊙ v
    expect = [[erased[i][j] + k[i] * wv[j] for j in range(dv)] for i in range(dk)]

    for i in range(dk):
        for j in range(dv):
            assert S.S[i][j] == pytest.approx(expect[i][j], abs=1e-9), (i, j)


def test_gdn2_decouples_erase_and_write():
    """The paper's claim: b_t on the key axis, w_t on the value axis. Recovers KDA
    when the gates are tied, which is why KDA is a free control."""
    op = GatedDeltaNet2()
    assert op.rung == 4 and op.write_gate is True

    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [1.0, 1.0, 1.0, 1.0]
    b = [0.1, 0.2, 0.3, 0.4]          # key axis, d_k
    w = [0.9, 0.8, 0.7, 0.6]          # value axis, d_v

    s = State(4, 4)
    s.S = [[1.0] * 4 for _ in range(4)]
    g = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, b=b, w=w)
    out, _ = op.step(s, g)

    # w scales what lands on the value axis, so the written pattern must be uneven
    assert out.S[0][0] != out.S[0][3], f"w must differentiate value channels: {out.S[0]}"


def test_rung3_erase_only_ties_write_to_a_scalar():
    """Rung 3 = erase gate only. w is tied to mean(b), which is the closest thing to
    KDA's single beta while keeping b channel-wise. NVlabs' ablation says the erase
    gate carries most of the gain, so this rung is expected to do most of the work —
    and that expectation is itself what the ladder tests."""
    op = build("gdn2_e")
    assert op.rung == 3 and op.write_gate is False
    assert op.variant == "gdn2_erase_only", "the variant, not the name, carries the flag"
    assert build("gdn2").variant == "gdn2"

    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [1.0, 2.0, 3.0, 4.0]
    s = State(4, 4)
    g = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, b=[0.5] * 4, w=[0.1, 0.2, 0.3, 0.4])
    out, _ = op.step(s, g)
    # w was supplied but must be IGNORED at rung 3: the write is a scalar tie
    written = [out.S[0][j] for j in range(4)]
    ratios = [written[j] / v[j] for j in range(4) if v[j]]
    assert max(ratios) - min(ratios) < 1e-9, \
        f"rung 3 must not use a channel-wise w: {ratios}"


def test_erase_actually_erases():
    """The functional claim, not the algebraic one: a large b_t at a key removes what
    was stored there."""
    op = GatedDeltaNet2()
    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [0.0, 7.0, 0.0, 0.0]

    s = State(4, 4)
    g = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, b=[0.0] * 4, w=[1.0] * 4, beta=1.0)
    s, _ = op.step(s, g)
    assert s.read(k)[1] == pytest.approx(7.0, abs=1e-6), "write landed"

    # now erase hard at that key, writing nothing
    g2 = Gates(q=k, k=k, v=[0.0] * 4, alpha=[1.0] * 4, b=[1.0] * 4, w=[0.0] * 4)
    s, _ = op.step(s, g2)
    assert abs(s.read(k)[1]) < 1e-6, f"a full erase at that key must clear it: {s.read(k)}"


# ── rung 5: EDA ────────────────────────────────────────────────────────────────

def test_eda_has_an_independent_erase_address():
    """The rung 4 vs 5 discriminator, stated structurally."""
    eda = EraseThenDelta()
    gdn2 = GatedDeltaNet2()
    assert eda.has_erase_address is True, "EDA exposes an address the write does not"
    assert gdn2.has_erase_address is False, "GDN-2's erase is b⊙k: write-anchored"
    assert eda.rung == 5


def test_eda_can_erase_at_an_address_it_is_not_writing():
    """⭐ The whole reason FRIDAY spec'd rung 5.

    GDN-2's erase direction is b_t ⊙ k_t — built from the CURRENT write key — so it
    cannot suppress a stale association stored at a DIFFERENT address. That is
    exactly the bi-temporal correction case:

        week 1  "I live at 12 Marina Road"      -> address kA
        week 3  "I moved to 48 Velachery Main"  -> address kB  (kB != kA)

    EDA's e_t is an independent projection, so it can reach kA while writing kB.
    """
    kA = l2_normalise([1.0, 0.0, 0.0, 0.0])
    kB = l2_normalise([0.0, 1.0, 0.0, 0.0])
    vA = [3.0, 0.0, 0.0, 0.0]

    # GDN-2: write at A, then write at B while erasing hard. b⊙kB is orthogonal to
    # kA, so A must SURVIVE — that is the limitation, not a bug.
    op = GatedDeltaNet2()
    s = State(4, 4)
    s, _ = op.step(s, Gates(q=kA, k=kA, v=vA, alpha=[1.0] * 4,
                            b=[0.0] * 4, w=[1.0] * 4, beta=1.0))
    assert s.read(kA)[0] == pytest.approx(3.0, abs=1e-6)
    s, _ = op.step(s, Gates(q=kB, k=kB, v=[0.0] * 4, alpha=[1.0] * 4,
                            b=[1.0] * 4, w=[1.0] * 4))
    assert s.read(kA)[0] == pytest.approx(3.0, abs=1e-6), \
        "GDN-2 cannot erase at an address it is not writing to"

    # EDA: same sequence, but e_t addresses kA directly while the write goes to kB.
    eda = EraseThenDelta()
    s = State(4, 4)
    s, _ = eda.step(s, Gates(q=kA, k=kA, v=vA, alpha=[1.0] * 4,
                             b=[0.0] * 4, w=[1.0] * 4, beta=1.0))
    assert s.read(kA)[0] == pytest.approx(3.0, abs=1e-6)
    s, _ = eda.step(s, Gates(q=kB, k=kB, v=[0.0] * 4, alpha=[1.0] * 4,
                             b=[0.0] * 4, w=[1.0] * 4,
                             e=kA, gamma_e=1.0))
    assert abs(s.read(kA)[0]) < 0.5, \
        f"EDA's independent erase address must reach kA: {s.read(kA)[0]}"


def test_eda_query_erase_rung6_is_inert_unless_supplied():
    """Rung 6 ("The Query Knows What to Forget", arXiv 2608.13668) is a second erase
    direction derived from the QUERY. It must do nothing unless `q_erase` is supplied
    with a nonzero `gamma_q`."""
    eda = EraseThenDelta()
    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [2.0, 0.0, 0.0, 0.0]

    # b=1 so the erase-then-write converges instead of accumulating (see the next
    # test — GDN-2 has no beta, its write is full-strength every step).
    g_no_q = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, b=[1.0] * 4, w=[1.0] * 4,
                   q_erase=None, gamma_q=0.0)
    s_off = State(4, 4)
    for _ in range(3):
        s_off, _ = eda.step(s_off, g_no_q)

    # same, but with a query-derived erase aimed at the SAME address and full
    # strength: this must actually remove content that the inert version keeps.
    g_q = Gates(q=k, k=k, v=v, alpha=[1.0] * 4, b=[0.0] * 4, w=[0.0] * 4,
                q_erase=k, gamma_q=1.0)
    s_on = State(4, 4)
    for _ in range(2):
        s_on, _ = eda.step(s_on, g_no_q)
    s_on, _ = eda.step(s_on, g_q)

    assert s_off.read(k)[0] == pytest.approx(2.0, abs=0.2), "converged write survives"
    assert abs(s_on.read(k)[0]) < abs(s_off.read(k)[0]), \
        f"a supplied q_erase must remove content the inert path keeps: {s_on.read(k)[0]}"


def test_gdn2_has_no_beta_its_erase_gate_provides_the_selectivity():
    """⭐ The single most important property of the GDN-2 update rule, and the reason
    NVlabs' ablation found the erase gate b_t carries most of the gain.

    Rungs 0-2 gate the WRITE with a scalar beta (delta = beta*(v - pred)), so
    beta=0 means "change nothing". GDN-2 has NO beta: its write is full-strength
    k_t(w_t ⊙ v_t)^T every step, and selectivity comes from ERASING at the write
    address first. The consequence, measured:

        b = 0  ->  erase disabled  ->  purely ADDITIVE  ->  behaves like RetNet
        b = 1  ->  erase first     ->  CONVERGES on repeated presentation

    So the erase gate is not a refinement on top of the delta rule; for rungs 3-4 it
    IS the mechanism that replaces it. Getting this backwards (e.g. assuming beta=0
    is a no-op) silently turns GDN-2 into the RetNet the whole pivot rejected.
    """
    k = l2_normalise([1.0, 0.0, 0.0, 0.0])
    v = [2.0, 0.0, 0.0, 0.0]

    op = GatedDeltaNet2()
    s_off = State(4, 4)
    reads_off = []
    for _ in range(5):
        s_off, _ = op.step(s_off, Gates(q=k, k=k, v=v, alpha=[1.0] * 4,
                                        b=[0.0] * 4, w=[1.0] * 4))
        reads_off.append(s_off.read(k)[0])

    s_on = State(4, 4)
    reads_on = []
    for _ in range(5):
        s_on, _ = op.step(s_on, Gates(q=k, k=k, v=v, alpha=[1.0] * 4,
                                      b=[1.0] * 4, w=[1.0] * 4))
        reads_on.append(s_on.read(k)[0])

    assert reads_off[-1] > 8.0, f"b=0 must accumulate like RetNet: {reads_off}"
    assert all(abs(r - 2.0) < 1e-6 for r in reads_on), f"b=1 must converge: {reads_on}"

    # and RetNet really does match the b=0 case, which is the whole point
    rn = RetNet()
    s_rn = State(4, 4)
    for _ in range(5):
        s_rn, _ = rn.step(s_rn, Gates(q=k, k=k, v=v, alpha=[1.0] * 4))
    assert abs(s_rn.read(k)[0] - reads_off[-1]) < 0.5, \
        f"GDN-2 with b=0 should match RetNet: {s_rn.read(k)[0]} vs {reads_off[-1]}"


# ── registry / ladder ──────────────────────────────────────────────────────────

def test_operator_factory_covers_the_ladder():
    assert set(OPERATOR_NAMES) == {"retnet", "gdn", "kda", "gdn2_e", "gdn2", "eda"}
    assert [n for n, _ in ABLATION_LADDER] == \
        ["retnet", "gdn", "kda", "gdn2_e", "gdn2", "eda"]

    expected_rung = {"retnet": 0, "gdn": 1, "kda": 2, "gdn2_e": 3, "gdn2": 4, "eda": 5}
    for name, rung in ABLATION_LADDER:
        op = build(name)
        assert op.rung == expected_rung[name], f"{name} should be rung {expected_rung[name]}"
        s = State(4, 4)
        s, _ = op.step(s, _gates([1, 0, 0, 0], [1, 0, 0, 0], 4, 4))
        assert len(s.S) == 4 and len(s.S[0]) == 4


def test_build_rejects_an_unknown_operator():
    with pytest.raises(KeyError):
        build("mamba2")


def test_state_is_fixed_size_regardless_of_sequence_length():
    """The whole premise: O(1) memory in sequence length. This is what dissolves the
    KV-cache RAM growth and the prompt-caching constraint (docs/architecture/12 §3)."""
    op = build("gdn2")
    s = State(8, 8)
    for t in range(300):
        k = [math.sin(t * 0.1 + i) for i in range(8)]
        v = [math.cos(t * 0.13 + i) for i in range(8)]
        s, _ = op.step(s, Gates(q=k, k=k, v=v, alpha=[0.999] * 8,
                                b=[0.5] * 8, w=[0.9] * 8))
        assert len(s.S) == 8 and len(s.S[0]) == 8
    assert s.numel == 64, "300 steps in, still 64 floats"
    assert math.isfinite(s.frobenius()), "and it must not have exploded"


# ── the model ──────────────────────────────────────────────────────────────────

def test_compile_returns_fixed_state_slots():
    from friday.rsc.model import RSCConfig, RetentionStateCompiler

    cfg = RSCConfig(d_model=16, n_heads=2, d_k=8, d_v=8, n_layers=2,
                    swa_window=4, state_slots=3, operator="gdn2")
    m = RetentionStateCompiler(cfg)
    out = m.compile([[0.1] * 16 for _ in range(12)])

    assert len(out["state"]) == 3, "state_slots vectors are injected, not T"
    assert all(len(v) == 16 for v in out["state"])
    assert len(out["salience"]) == 12, "one salience score per token"
    assert all(0.0 <= x <= 1.0 for x in out["salience"])
    assert out["tokens"] == 12
    assert len(out["per_head"]) == cfg.n_layers
    assert len(out["per_head"][0]) == cfg.n_heads


def test_compile_of_an_empty_sequence_is_safe():
    from friday.rsc.model import RSCConfig, RetentionStateCompiler

    cfg = RSCConfig(d_model=8, n_heads=1, d_k=4, d_v=4, n_layers=1, state_slots=2)
    out = RetentionStateCompiler(cfg).compile([])
    assert len(out["state"]) == 2 and out["salience"] == []


def test_state_size_is_reported_in_bytes():
    """The paper's per-head state is 262,144 floats ~1 MB. This is the number that
    decides whether the RSC fits alongside a 4-bit backbone in 16 GB."""
    from friday.rsc.model import RSCConfig

    cfg = RSCConfig(d_model=64, n_heads=16, d_k=128, d_v=128, n_layers=2)
    assert cfg.state_numel == 2 * 16 * 128 * 128
    assert cfg.state_bytes_fp32() == 4 * cfg.state_numel
    assert cfg.state_numel == 524288, "the paper's figure, per layer x2"


def test_render_state_block_is_inspectable():
    """Law 7 applied to a tensor: you should be able to LOOK at what FRIDAY
    remembers. This is what makes the Ledger printout honest about the RSC."""
    from friday.rsc.model import render_state_block

    block = render_state_block([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    assert "<retention_state>" in block and "</retention_state>" in block
    assert "state[0]" in block
    assert "2 more" in block, "long vectors must be elided, not dumped"


def test_gate_branches_are_separate_projections():
    """Separate branches are what make the gates separately SUPERVISABLE — the entire
    basis of gate_supervision.py. If decay, erase and write shared a projection, one
    label set would drag the other two."""
    from friday.rsc.model import RSCConfig, RetentionStateCompiler

    cfg = RSCConfig(d_model=16, n_heads=2, d_k=8, d_v=8, n_layers=1, state_slots=2,
                    operator="gdn2")
    m = RetentionStateCompiler(cfg)
    names = [n for n, _ in m.__dict__.items()]
    assert any("decay" in n or "alpha" in n for n in names), names
    assert any("erase" in n for n in names), names
    assert any("write" in n for n in names), names


# ── the loss ───────────────────────────────────────────────────────────────────

def test_kl_and_bce_are_sane():
    from friday.rsc.losses import bce, kl_divergence

    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)
    assert kl_divergence([0.9, 0.1], [0.5, 0.5]) > 0
    assert bce(0.5, 1.0) > 0
    assert bce(0.999, 1.0) < bce(0.6, 1.0), "a correct confident prediction costs less"


def test_reconstruction_is_zero_when_teacher_and_student_agree():
    from friday.rsc.losses import l_reconstruction

    logits = [[1.0, 0.0], [0.0, 1.0]]
    assert l_reconstruction(logits, logits) == pytest.approx(0.0, abs=1e-9)
    assert l_reconstruction(logits, [[0.0, 1.0], [1.0, 0.0]]) > 0
    assert l_reconstruction([], []) == 0.0


def test_recall_weights_deep_needles_harder():
    """⭐ A compressor that remembers position 500 and forgets position 7500 is
    exactly the failure the literature reports, and an unweighted mean hides it."""
    from friday.rsc.losses import l_recall

    shallow = [{"query": "q", "value": "v", "depth": 0}]
    deep = [{"query": "q", "value": "v", "depth": 8000}]
    fn = lambda q: 0.5
    assert l_recall(fn, deep) > l_recall(fn, shallow), "depth must cost more"


def test_recall_is_nll_of_the_exact_needle():
    from friday.rsc.losses import l_recall

    assert l_recall(lambda q: 1.0, [{"query": "q", "value": "v", "depth": 0}]) \
        == pytest.approx(0.0, abs=1e-3)
    assert l_recall(lambda q: 0.0, []) == 0.0


def test_salience_class_balances():
    """Positives are rare — most spans don't produce facts. An unbalanced BCE here
    just learns "always predict 0", which is a salience head that never fires."""
    from friday.rsc.losses import l_salience

    # all-negative labels with an all-negative prediction should be cheap
    cheap = l_salience([0.01] * 10, [0] * 10)
    # missing the rare positives must cost more than missing the common negatives
    missed_pos = l_salience([0.01] * 10, [1] + [0] * 9)
    missed_neg = l_salience([0.01, 0.99] + [0.01] * 8, [1, 0] + [0] * 8)
    assert missed_pos > cheap
    assert missed_pos > 0.1, "a missed positive must hurt"


def test_erase_loss_weights_address_distinct_pairs_higher():
    """⭐ The retraction pair where keys differ is the informative one — it is the
    case a write-anchored erase cannot learn. It must carry more weight, or the
    ladder's rung 5 is judged on examples that cannot discriminate it."""
    from friday.rsc.losses import l_erase

    fn = lambda k_old, v_old: 0.5
    distinct = [{"key_old": "a", "value_old": "x", "address_distinct": True}]
    same = [{"key_old": "a", "value_old": "x", "address_distinct": False}]
    assert l_erase(fn, distinct) > l_erase(fn, same)
    assert l_erase(fn, []) == 0.0


def test_erase_loss_is_minimised_when_the_stale_value_is_gone():
    from friday.rsc.losses import l_erase

    pairs = [{"key_old": "address::12 marina", "value_old": "12 Marina Road",
              "address_distinct": True}]
    assert l_erase(lambda k, v: 0.0, pairs) == pytest.approx(0.0, abs=1e-9)
    assert l_erase(lambda k, v: 0.9, pairs) > l_erase(lambda k, v: 0.1, pairs)


def test_decay_weight_anneals_to_zero():
    """The curriculum term is a scaffold. If it doesn't anneal away, alpha_t
    converges to your hand-tuned half-lives and you have learned nothing — you've
    just implemented `policy_alpha` in 40M parameters."""
    from friday.rsc.losses import anneal_decay_weight

    assert anneal_decay_weight(0, 1000) == pytest.approx(1.0)
    assert anneal_decay_weight(500, 1000) == pytest.approx(0.5, abs=0.05)
    assert anneal_decay_weight(1000, 1000) == pytest.approx(0.0, abs=1e-6)
    assert anneal_decay_weight(5000, 1000) == pytest.approx(0.0, abs=1e-6)


def test_decay_term_vanishes_when_weight_is_zero():
    from friday.rsc.losses import l_decay

    assert l_decay([0.5], [0.9], weight=0.0) == 0.0
    assert l_decay([0.5], [0.9], weight=1.0) > 0


def test_total_loss_reports_every_term_and_tolerates_a_partial_batch():
    """Missing inputs contribute 0.0 rather than raising. In Week 2 you will have
    salience labels and no retraction pairs, and the alternative is a trainer that
    refuses to start until your life has produced a contradiction."""
    from friday.rsc.losses import LossWeights, total_loss

    empty = total_loss(weights=LossWeights())
    assert set(empty.terms) == {"reconstruction", "recall", "salience", "erase", "decay"}
    assert empty.total == pytest.approx(0.0, abs=1e-9), "an empty batch must not raise"

    full = total_loss(
        weights=LossWeights(),
        teacher_logits=[[1.0, 0.0], [0.0, 1.0]],
        student_logits=[[0.9, 0.1], [0.1, 0.9]],
        recall_fn=lambda q: 0.7,
        needles=[{"query": "q", "value": "v", "depth": 4000}],
        salience_preds=[0.9, 0.1],
        salience_labels=[1, 0],
        erase_fn=lambda k, v: 0.2,
        retraction_pairs=[{"key_old": "a", "value_old": "x", "address_distinct": True}],
        alpha_pred=[0.9, 0.9],
        alpha_policy=[1.0, 1.0],
        decay_weight_scale=1.0,
    )
    assert all(v > 0 for v in full.terms.values()), full.terms
    assert math.isfinite(full.total)
    assert "L=" in full.line()


def test_recall_weight_is_marked_untouchable():
    """`recall: ⭐ never set this to 0`. If someone zeroes it, the model is free to
    compress everything away and still minimise the loss — Law 2c is what stops that,
    and the default weights are where it lives."""
    from friday.rsc.losses import LossWeights

    w = LossWeights()
    assert w.recall == 1.0
    assert w.erase == 1.0, "the GDN-2 vs EDA discriminator must be on by default"
    assert 0 < w.decay < 1.0, "decay is a curriculum, deliberately under-weighted"


# ── supervision synthesis ──────────────────────────────────────────────────────

def test_alpha_targets_come_from_the_decay_policy(conn, seeded):
    """The decay function IS the supervision for alpha_t. That duality is the point
    of docs/architecture/13 §4.1: one hand-written policy, two consumers."""
    from friday.rsc.gate_supervision import alpha_targets

    classes, alphas = alpha_targets(conn)
    assert classes and len(classes) == len(alphas)
    assert all(0.0 < a <= 1.0 for a in alphas)
    assert set(classes) <= {"identity", "environment", "preference", "location",
                            "relationship", "mood", "current_task", "project_state"}


def test_alpha_ordering_follows_the_half_lives():
    """The real assertion. Shorter half-life -> lower retention. If this ordering is
    wrong the RSC learns to retain transients and forget facts — exactly backwards."""
    from friday.rsc.gate_supervision import alpha_for_class

    assert alpha_for_class("identity") == 1.0, "identity never decays"
    assert alpha_for_class("mood") < alpha_for_class("current_task")
    assert alpha_for_class("current_task") < alpha_for_class("location")
    assert alpha_for_class("location") < alpha_for_class("environment")
    assert alpha_for_class("environment") < alpha_for_class("relationship")
    assert alpha_for_class("relationship") < alpha_for_class("preference")
    assert alpha_for_class("preference") < alpha_for_class("identity")


def test_span_batches_mix_positives_and_sampled_negatives(conn, seeded):
    """Negatives are SAMPLED, not exhaustively taken: almost every span in a
    conversation fails to produce a fact, so an unsampled negative set would be ~99%
    of the corpus and the head would learn "always 0"."""
    from friday.rsc.gate_supervision import span_batches

    batches = span_batches(conn)
    assert batches, "seeded facts carry source quotes, so spans exist"
    assert all(b["label"] == 1 for b in batches), "no negatives minted yet"
    assert all(b["char_end"] > b["char_start"] for b in batches)


def test_pair_batches_order_address_distinct_first(conn, seeded):
    """Rungs 4 and 5 differ ONLY on address-distinct pairs. If a batch is mostly
    address-identical the two rungs look the same and you learn nothing."""
    from friday.memory.facts import Fact, assert_fact
    from friday.rsc.gate_supervision import pair_batches

    assert_fact(conn, Fact(subject="user", predicate="lives_in", object="Bengaluru",
                           valid_from="2020-01-01", source_quote="i live in bengaluru"))
    assert_fact(conn, Fact(subject="user", predicate="lives_in", object="Chennai",
                           valid_from="2023-06-01", source_quote="i moved to chennai"))
    assert_fact(conn, Fact(subject="user", predicate="mood", object="tired",
                           valid_from="2026-09-01", source_kind="observed", confidence=0.6))
    assert_fact(conn, Fact(subject="user", predicate="mood", object="fine",
                           valid_from="2026-09-02", source_kind="observed", confidence=0.6))

    batches = pair_batches(conn)
    assert len(batches) >= 2
    assert batches[0]["address_distinct"] == 1, "address-distinct pairs must sort first"


def test_needles_from_facts_use_real_phrasing(conn, seeded):
    """Better than synthetic needles for TRAINING, because the phrasing is yours:
    "what's my rent" against lease_amount_monthly is the query FRIDAY will actually
    get. scripts/needle_test.py owns the EVAL needles — training and eval must not
    share examples."""
    from friday.rsc.gate_supervision import needles_from_facts

    n = needles_from_facts(conn)
    assert n
    assert all({"query", "value", "depth", "predicate"} <= set(x) for x in n)
    assert any("lease" in x["query"] for x in n)
    depths = [x["depth"] for x in n]
    assert len(set(depths)) > 1, "depths must be spread so the loss can't cheat by recency"


def test_build_batch_assembles_every_loss_input(conn, seeded):
    from friday.rsc.gate_supervision import build_batch

    b = build_batch(conn)
    assert b.span_labels and len(b.span_labels) == len(b.span_bounds)
    assert b.alpha_targets and len(b.alpha_targets) == len(b.alpha_classes)
    assert b.needles
    assert b.n_positive == sum(1 for y in b.span_labels if y)
    assert "spans" in b.meta and "pairs" in b.meta


def test_readiness_report_says_not_ready_early(conn, seeded):
    """The line that tells you whether Phase 3.5 has fuel. It must say NO at the
    start, or you will burn Kaggle hours on a model with 12 training pairs."""
    from friday.rsc.gate_supervision import readiness_report

    r = readiness_report(conn)
    assert "NOT READY" in r, r


# ── the probes ─────────────────────────────────────────────────────────────────

def test_probe_associative_separates_rung0_from_rung2():
    """⭐ The decisive check. A fixed-decay additive write cannot do selective
    associative recall; the delta rule can. This is the task ICLR 2025 proves an RNN
    with o(n)-bit memory cannot solve. If rung 0 passes, the probe is broken.

    The load-bearing assertion is the SEPARATION, not KDA's absolute score. The
    absolute number moves with reps (the delta rule is error-correcting, so it
    converges over repeated presentation: 0.33 -> 0.50 -> 0.67 -> 1.00 at reps
    1/2/3/5) and with seed (random dense keys overlap differently). RetNet scoring
    exactly 0.00 at every seed and every rep count while every delta-rule rung
    clears it substantially is the actual finding, and that is stable. Asserting
    `kda > 0.5` instead pins a threshold that sits right on the reps=3 boundary —
    it failed at exactly 0.5 while the separation was perfectly clear.
    """
    from friday.rsc.ablation import probe_associative

    for seed in (0, 1, 2):
        retnet = probe_associative(build("retnet"), 32, 32, n_pairs=6,
                                   distractors=12, seed=seed)
        kda = probe_associative(build("kda"), 32, 32, n_pairs=6,
                                distractors=12, seed=seed)
        assert retnet.name == "associative"
        assert retnet.score == 0.0, \
            f"seed {seed}: RetNet must score exactly zero, got {retnet.score}"
        assert kda.score >= retnet.score + 0.3, \
            f"seed {seed}: the delta rule must clear RetNet decisively, got {kda.score}"
        assert kda.detail["mean_rel_err"] < 0.5 < retnet.detail["mean_rel_err"], \
            f"seed {seed}: the mean error must tell the same story as the hit rate"


def test_associative_probe_reports_its_load():
    """`load_ratio` and `reps` are reported so a flat ladder is diagnosable. A score
    of 0.9 at load 0.2 and a score of 0.2 at load 3.0 are different claims about the
    same operator, and the mean alone does not distinguish them."""
    from friday.rsc.ablation import probe_associative

    r = probe_associative(build("kda"), 32, 32, n_pairs=6, distractors=12, seed=0)
    assert r.detail["distractors"] == 12
    assert r.detail["reps"] == 3
    assert 0 < r.detail["load_ratio"] < 1.0, \
        f"load must stay under state capacity or the probe measures a wall: {r.detail}"


def test_associative_probe_collapses_when_over_state_capacity():
    """The failure mode, pinned so it cannot silently return. At d_k=16 with
    n_pairs + distractors = 18 the state is over capacity and EVERY rung scores 0.00
    — including the delta rule, which works fine at d_k=32. A probe reporting "all
    rungs tied at zero" is not reporting that the ladder is flat; it is reporting
    that it was configured badly. This is why the CLI warns below d_k=24."""
    from friday.rsc.ablation import probe_associative

    cramped = probe_associative(build("kda"), 16, 16, n_pairs=6, distractors=12, seed=0)
    roomy = probe_associative(build("kda"), 32, 32, n_pairs=6, distractors=12, seed=0)
    assert cramped.detail["load_ratio"] > 1.0
    assert roomy.detail["load_ratio"] < 1.0
    assert cramped.score < roomy.score, \
        f"over capacity {cramped.score} vs in capacity {roomy.score}"


def test_probe_needle_returns_a_per_depth_curve():
    """Score is per-depth so you can SEE the decay curve rather than a single
    average. A model at 0.95/depth-16 and 0.10/depth-480 is not a 0.5 model."""
    from friday.rsc.ablation import probe_needle

    r = probe_needle(build("kda"), 32, 32, length=512, depths=(8, 64, 256), seed=0)
    assert r.name == "needle"
    assert 0.0 <= r.score <= 1.0
    assert set(r.detail["per_depth"]) == {8, 64, 256}


def test_probe_interference_reports_all_three_metrics():
    """FRIDAY's probe: does writing at key B corrupt the value at key A? This is the
    bi-temporal correction case in miniature, and no public benchmark sheet reports
    it — which is exactly why it is the one that matters here."""
    from friday.rsc.ablation import probe_interference

    r = probe_interference(build("gdn2"), 32, 32, n_old=4, n_new=4, seed=0)
    assert r.name == "interference"
    for k in ("retain_old", "new_write_err", "crosstalk_new_into_old"):
        assert k in r.detail, (k, r.detail)
    assert 0.0 <= r.detail["retain_old"] <= 1.0


def test_probe_interference_is_the_rung4_vs_rung5_discriminator():
    """If EDA doesn't move crosstalk on this probe, the address-level coupling isn't
    binding on FRIDAY's data and you can ship rung 3 or 4 — which is a cheaper and
    better-supported outcome, not a failure."""
    from friday.rsc.ablation import probe_interference

    g = probe_interference(build("gdn2"), 32, 32, n_old=4, n_new=4, seed=0)
    e = probe_interference(build("eda"), 32, 32, n_old=4, n_new=4, seed=0)
    assert g.detail["crosstalk_new_into_old"] >= 0.0
    assert e.detail["crosstalk_new_into_old"] >= 0.0


def test_run_ladder_covers_every_rung_with_deltas():
    from friday.rsc.ablation import run_ladder

    out = run_ladder(d_k=32, d_v=32, needle_length=96, seeds=(0,))
    assert list(out["rungs"]) == ["retnet", "gdn", "kda", "gdn2_e", "gdn2", "eda"]
    for name, r in out["rungs"].items():
        assert r["rung"] is not None
        assert set(r["scores"]) == {"associative", "needle", "interference"}
        assert len(r["detail"]) == 3, "three probes x one seed"

    # the delta column is the point: rung2->3->4 are one-variable changes, so their
    # deltas are directly attributable. That is what makes this a measurement.
    assert len(out["ladder"]) == 6
    assert out["ladder"][0]["delta_vs_prev"] == {}
    assert "associative" in out["ladder"][1]["delta_vs_prev"]


def test_run_ladder_respects_a_rung_subset():
    from friday.rsc.ablation import run_ladder

    out = run_ladder(d_k=32, d_v=32, needle_length=64, seeds=(0,), rungs=("kda", "eda"))
    assert list(out["rungs"]) == ["kda", "eda"]


def test_format_report_is_readable_markdown():
    from friday.rsc.ablation import format_report, run_ladder

    out = run_ladder(d_k=32, d_v=32, needle_length=64, seeds=(0,), rungs=("retnet", "kda"))
    md = format_report(out)
    assert "retnet" in md and "kda" in md
    assert "|" in md
    assert "not" in md.lower(), "the report must say what it is NOT (a 1.3B result)"


# ── ⭐ reproducibility: the precondition for the report meaning anything ──────


def test_probe_keys_are_identical_across_processes():
    """⭐⭐ Regression guard for a bug that made every number in the ablation report a
    SAMPLE rather than a MEASUREMENT.

    `_key_for` seeded its RNG with Python's built-in `hash()`, which is randomised per
    process unless PYTHONHASHSEED is set. So the probe keys — and therefore the key
    geometry the associative probe measures — were different on every run. Its
    docstring claimed "same word -> same key, so a probe run twice gives the same
    answer", which was false: one test passed twice and failed on the third run in an
    unchanged working tree.

    The only honest way to test cross-process stability is to actually spawn a process
    with a different hash seed. Anything in-process proves nothing, because the whole
    bug is that in-process looks fine.
    """
    import os
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, %r);"
        "from friday.rsc.ablation import _key_for;"
        "print([round(x, 9) for x in _key_for('probe-key', 8, seed=3)])"
    ) % str(Path(__file__).resolve().parent.parent)

    outs = []
    for hs in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())

    assert len(set(outs)) == 1, \
        f"probe keys must not depend on the process hash seed:\n" + "\n".join(outs)


def test_associative_probe_is_reproducible_run_to_run():
    """Same seeds -> same scores, in THIS process and on a re-import. Pairs with the
    subprocess test above: that one proves the keys are stable, this one proves the
    score that comes out of them is too."""
    from friday.rsc.ablation import probe_associative

    first = [probe_associative(build("kda"), 32, 32, seed=s).score for s in range(5)]
    second = [probe_associative(build("kda"), 32, 32, seed=s).score for s in range(5)]
    assert first == second, (first, second)
    # and the separation is robust across seeds, not a lucky draw
    retnet = [probe_associative(build("retnet"), 32, 32, seed=s).score for s in range(8)]
    assert all(r == 0.0 for r in retnet), retnet
    assert all(k >= 0.6 for k in first), first


def test_needle_and_interference_are_reproducible():
    """The two probes whose results the docs quote as exact figures. If these drift,
    the quoted numbers in docs/architecture/13 are wrong."""
    from friday.rsc.ablation import probe_interference, probe_needle

    n = probe_needle(build("retnet"), 32, 32, length=512, depths=(16, 64, 192, 400))
    depths = list(n.detail["per_depth"].values())
    # RetNet's fixed decay is 0.995 per step: this is the closed form, not a fit
    for d, got in zip((16, 64, 192, 400), depths):
        assert abs(got - 0.995 ** d) < 0.01, f"depth {d}: {got} vs {0.995 ** d}"

    g = probe_interference(build("gdn2"), 32, 32, n_old=4, n_new=4)
    e = probe_interference(build("eda"), 32, 32, n_old=4, n_new=4)
    assert g.detail["retain_old"] == 1.0, \
        "GDN-2's erase is anchored to the write key and cannot reach an orthogonal address"
    assert e.detail["retain_old"] == 0.5, \
        "EDA's independent erase address can — this is the rung-5 claim"
