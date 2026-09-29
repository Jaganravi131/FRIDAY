"""Recurrent operators for the Retention State Compiler — swappable by config flag.

docs/architecture/13 §6: this target moves fast (four substantive papers in five
months: GDN-2 May, EDA Jun, query-erase Aug, TERN Sep 2026). So the update rule is
a *plugin*, the state tensor shape is the contract, and every rung is a YAML flag
plus a permanent row in the eval table.

    rung 0  retnet   fixed scalar decay, additive write          <- must lose
    rung 1  gdn      scalar decay + scalar delta gate
    rung 2  kda      channel-wise decay + scalar delta gate      <- free control
    rung 3  gdn2_e   kda + channel-wise ERASE gate  b_t          <- most of the gain
    rung 4  gdn2     + channel-wise WRITE gate w_t               <- full GDN-2
    rung 5  eda      + INDEPENDENTLY ADDRESSED erase e_t         <- the bi-temporal win
    rung 6  qerase   + query-derived second erase direction      <- speculative

The verified GDN-2 rule (arXiv 2605.22791, eq. in §3.1), with S of shape (d_k, d_v):

    S_t = ( I − k_t (b_t ⊙ k_t)^T ) · D_t S_{t−1} + k_t (w_t ⊙ v_t)^T
          └────── active edit ──────┘ └─ decay ─┘   └───── write ─────┘
    D_t = Diag(α_t),  α_t ∈ (0,1]^{d_k}   channel-wise, inherited from KDA

Three properties that make it a *safe* upgrade and are relied on below:
  * decay is applied BEFORE the active edit;
  * the write direction stays k_t (left factor), so the delta rule's "edit the
    association AT key k_t" survives — only the READ direction becomes
    channel-selective via b_t ⊙ k_t;
  * tying b_t = w_t = β_t·1 recovers KDA EXACTLY, and additionally tying
    α_t = α·1 recovers Gated DeltaNet. KDA is therefore a free control condition.

⚠️ EDA's exact published equation is NOT reproduced here from memory. §3.3 of
arXiv 2606.26560 describes three levels of specificity (diagonal decay D_t,
independent directional erasure γ_t·e_t, write-coupled correction β_t·k_t) and a
factorised erase address with a 16-d intermediate per head; `EraseThenDelta`
implements that structure and is marked where it is my construction. Verify
against the paper before publishing numbers.

Pure Python + `math`. No numpy, no torch, no Triton. Slow, exact, and runnable on
any machine including a free Kaggle T4 — which is the point: this file is for
CORRECTNESS and for the ablation ladder's logic. The fast path is
`NVlabs/GatedDeltaNet-2` (chunkwise WY + gate-aware Triton backward) or `fla`'s
GDN/KDA kernels; see `fastpath.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

Vector = list[float]
Matrix = list[list[float]]   # (rows, cols)


# ── small linear algebra (deliberately obvious rather than clever) ─────────────

def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def matvec(A: Matrix, x: Vector) -> Vector:
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def outer(a: Vector, b: Vector) -> Matrix:
    return [[ai * bj for bj in b] for ai in a]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    cols = len(B[0])
    Bt = [[B[r][c] for r in range(len(B))] for c in range(cols)]
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def mat_add(A: Matrix, B: Matrix) -> Matrix:
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mat_scale_rows(A: Matrix, s: Vector) -> Matrix:
    """Diag(s) @ A — scale each ROW. This is how channel-wise decay over the key
    axis acts on S of shape (d_k, d_v)."""
    return [[si * a for a in row] for si, row in zip(s, A)]


def l2_normalise(x: Vector) -> Vector:
    n = math.sqrt(sum(v * v for v in x)) or 1.0
    return [v / n for v in x]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rms_norm(x: Vector, eps: float = 1e-6) -> Vector:
    n = math.sqrt(sum(v * v for v in x) / max(1, len(x))) or 1.0
    return [v / (n + eps) for v in x]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── state ──────────────────────────────────────────────────────────────────────

@dataclass
class State:
    """One head's recurrent memory. `S` is (d_k, d_v): rows are key channels, so
    Diag(alpha) @ S is channel-wise decay over the key axis, exactly as in KDA/GDN-2.

    Per layer the paper uses 16 heads with d_k = d_v = 128 -> 262,144 floats
    (~1 MB fp32), "similar to Mamba-2/3". Here it's one head; the model stacks them.
    """

    d_k: int
    d_v: int
    S: Matrix = field(default_factory=list)

    def __post_init__(self):
        if not self.S:
            self.S = zeros(self.d_k, self.d_v)

    @property
    def numel(self) -> int:
        return self.d_k * self.d_v

    def clone(self) -> "State":
        return State(self.d_k, self.d_v, [row[:] for row in self.S])

    def read(self, q: Vector) -> Vector:
        """o = S^T q. Retrieval by query against the associative memory."""
        return [dot(q, [self.S[k][v] for k in range(self.d_k)]) for v in range(self.d_v)]

    def frobenius(self) -> float:
        return math.sqrt(sum(a * a for row in self.S for a in row))


@dataclass
class Gates:
    """Everything the operator needs at step t, produced by separate linear branches.

    GDN-2's block design gives decay, erase and write their OWN projections — three
    decisions, three branches. That separation is what makes them independently
    supervisable, which is the whole basis of docs/architecture/13 §4.1:

        alpha  <- the decay function        (memory/decay.policy_alpha)
        b / e  <- retraction pairs          (memory/supervision.RetractionPair)
        w      <- the salience head         (memory/supervision.SpanLabel)
    """

    q: Vector
    k: Vector
    v: Vector
    alpha: Vector                 # channel-wise decay, (0,1]^{d_k}
    b: Vector | None = None       # channel-wise erase gate, [0,1]^{d_k}
    w: Vector | None = None       # channel-wise write gate, [0,1]^{d_v}
    beta: float | None = None     # scalar delta gate (rungs 0-2)
    e: Vector | None = None       # independently addressed erase direction (rung 5)
    gamma_e: float = 0.0          # erase strength at that address (rung 5)
    q_erase: Vector | None = None # query-derived second erase direction (rung 6)
    gamma_q: float = 0.0


class Operator(Protocol):
    name: str
    rung: int
    has_erase_address: bool

    def step(self, s: State, g: Gates) -> tuple[State, Vector]: ...


# ── rung 0: RetNet ─────────────────────────────────────────────────────────────

class RetNet(Operator):
    """Fixed, input-independent scalar decay. Additive (Hebbian) write, no delta rule.

    This is the control that MUST lose. docs/architecture/12 §2.1: its fixed
    exponential decay "fails to protect long-range cues, yielding near-zero recall
    even when full-attention layers are added". Keeping it in the ladder is what
    makes the ladder a *measurement* rather than a demo.
    """

    name = "retnet"
    rung = 0
    has_erase_address = False

    def __init__(self, gamma: float = 0.995):
        self.gamma = gamma

    def step(self, s: State, g: Gates) -> tuple[State, Vector]:
        k, v = l2_normalise(g.k), g.v
        decayed = [[self.gamma * a for a in row] for row in s.S]
        new = mat_add(decayed, outer(k, v))       # additive: no correction, no selectivity
        return State(s.d_k, s.d_v, new), _read(new, g.q)


# ── rung 1: Gated DeltaNet ─────────────────────────────────────────────────────

class GatedDeltaNet(Operator):
    """Scalar decay + scalar delta gate. The ICLR 2025 baseline: perfect in-context
    associative recall where Mamba-2 has gaps."""

    name = "gdn"
    rung = 1
    has_erase_address = False

    def step(self, s: State, g: Gates) -> tuple[State, Vector]:
        k = l2_normalise(g.k)
        beta = g.beta if g.beta is not None else 0.5
        alpha = g.alpha[0] if g.alpha else 1.0          # scalar decay
        decayed = [[alpha * a for a in row] for row in s.S]
        pred = _read(decayed, k)                        # what is stored at k
        delta = [beta * (vi - pi) for vi, pi in zip(g.v, pred)]
        new = mat_add(decayed, outer(k, delta))
        return State(s.d_k, s.d_v, new), _read(new, g.q)


# ── rung 2: KDA ────────────────────────────────────────────────────────────────

class KimiDelta(Operator):
    """Channel-wise decay + SCALAR delta gate. Shipped in Kimi K3 at 3:1 against
    gated MLA. This is GDN-2's exact tied subspace, so it's the free control for
    rungs 3-4: any gain measured from here is attributable to the gates alone."""

    name = "kda"
    rung = 2
    has_erase_address = False

    def step(self, s: State, g: Gates) -> tuple[State, Vector]:
        k = l2_normalise(g.k)
        beta = g.beta if g.beta is not None else 0.5
        decayed = mat_scale_rows(s.S, g.alpha)
        pred = _read(decayed, k)
        delta = [beta * (vi - pi) for vi, pi in zip(g.v, pred)]
        new = mat_add(decayed, outer(k, delta))
        return State(s.d_k, s.d_v, new), _read(new, g.q)


# ── rungs 3-4: Gated DeltaNet-2 ────────────────────────────────────────────────

class GatedDeltaNet2(Operator):
    """Gated Delta Rule-2. arXiv 2605.22791.

        S_t = ( I − k_t (b_t ⊙ k_t)^T ) D_t S_{t−1} + k_t (w_t ⊙ v_t)^T

    `write_gate` toggles rung 3 (b only, w tied to b's scalar) vs rung 4 (both
    channel-wise). NVlabs' ablation: "the erase gate b_t accounts for most of the
    gain" — so rung 3 is expected to carry nearly all of the improvement, and that
    expectation is itself the thing being tested.
    """

    name = "gdn2"
    rung = 4
    has_erase_address = False

    def __init__(self, *, write_gate: bool = True):
        self.write_gate = write_gate
        if not write_gate:
            # `name` stays "gdn2" so that build("gdn2_e").name is stable regardless
            # of the flag; the RUNG is what distinguishes them, and the registry key
            # "gdn2_e" is what the ladder and the report print. An instance attribute
            # that silently renamed the operator made `op.name == registry_key` false
            # for exactly one rung, which is the kind of inconsistency that turns a
            # dict lookup into a KeyError three modules away.
            self.rung = 3

    @property
    def variant(self) -> str:
        """Human-readable variant, for reports: 'gdn2' vs 'gdn2_erase_only'."""
        return "gdn2" if self.write_gate else "gdn2_erase_only"

    def step(self, s: State, g: Gates) -> tuple[State, Vector]:
        k = l2_normalise(g.k)
        dk, dv = s.d_k, s.d_v

        b = g.b if g.b is not None else [0.5] * dk
        if self.write_gate and g.w is not None:
            w = g.w
        else:
            # rung 3: write strength tied to a scalar mean of the erase gate, which
            # is the closest thing to KDA's single beta while keeping b channel-wise.
            scalar = sum(b) / len(b)
            w = [scalar] * dv

        # 1. decay BEFORE the active edit (paper: "The update applies decay before
        #    the active edit.")
        decayed = mat_scale_rows(s.S, g.alpha)

        # 2. active edit. (I − k(b⊙k)^T) @ X  ==  X − k ⊗ ((b⊙k)^T X)
        #    The left factor stays k (write direction preserved); the READ direction
        #    becomes channel-selective via b ⊙ k. That asymmetry is the whole design.
        read_dir = [bi * ki for bi, ki in zip(b, k)]
        pred = _read(decayed, read_dir)         # (b⊙k)^T X, length d_v
        erased = [
            [row[j] - k[i] * pred[j] for j in range(dv)]
            for i, row in enumerate(decayed)
        ]

        # 3. write: k ⊗ (w ⊙ v)
        z = [wi * vi for wi, vi in zip(w, g.v)]
        new = mat_add(erased, outer(k, z))
        return State(dk, dv, new), _read(new, g.q)


# ── rung 5: Erase-then-Delta ───────────────────────────────────────────────────

class EraseThenDelta(Operator):
    """EDA. arXiv 2606.26560. Adds an INDEPENDENTLY ADDRESSED erase before the
    delta write.

    Why FRIDAY needs this and not merely GDN-2: GDN-2's erase direction is
    b_t ⊙ k_t — still built from the CURRENT write key. So it cannot remove a stale
    association stored at a DIFFERENT address. That is exactly the bi-temporal
    correction case:

        week 1  "I live at 12 Marina Road"      -> address k("my address")
        week 3  "I moved to 48 Velachery Main"  -> address k("my new address")

    EDA's paper reports the model learns a NEAR-ORTHOGONAL separation between erase
    and write addressing, i.e. the two operations serve genuinely different roles,
    and that the extra address acts as "a conditional cleanup path rather than
    merely stronger forgetting".

    Three levels of specificity, in the paper's own terms:
        1. diagonal decay            D_t
        2. independent erasure       gamma_e * e_t          <- the new degree of freedom
        3. write-coupled correction  beta * k_t  (the delta rule, intact)

    ⚠️ The exact published equation is NOT reproduced from memory here. This
    implements the described structure: decay, then a rank-one erase at an
    independently supplied unit address e_t, then the GDN-2-shaped corrective write.
    Read §3.3 of the paper and reconcile before publishing numbers.
    """

    name = "eda"
    rung = 5
    has_erase_address = True

    def __init__(self, *, write_gate: bool = True, erase_rank_one: bool = True):
        self.inner = GatedDeltaNet2(write_gate=write_gate)
        self.erase_rank_one = erase_rank_one

    def step(self, s: State, g: Gates) -> tuple[State, Vector]:
        k = l2_normalise(g.k)

        # 1. diagonal decay
        cur = mat_scale_rows(s.S, g.alpha)

        # 2. independently addressed erasure — BEFORE the write, at an address the
        #    model chose for cleaning up, not the address it is about to write to.
        if g.e is not None and g.gamma_e > 0.0:
            e = l2_normalise(g.e)
            if self.erase_rank_one:
                # S <- (I − gamma_e * e e^T) S : suppress whatever is stored along e
                proj = _read(cur, e)
                cur = [
                    [row[j] - g.gamma_e * e[i] * proj[j] for j in range(s.d_v)]
                    for i, row in enumerate(cur)
                ]
            else:
                # alternative reading: erase the value pattern at address e
                proj = _read(cur, e)
                cur = [
                    [row[j] - g.gamma_e * e[i] * proj[j] for j in range(s.d_v)]
                    for i, row in enumerate(cur)
                ]

        # 3. write-coupled delta correction, GDN-2 shaped, at k_t. We rebuild the
        #    gates with alpha=1 because decay was already applied in step 1 — this is
        #    what keeps the three levels of specificity from double-counting.
        inner_g = Gates(
            q=g.q, k=g.k, v=g.v, alpha=[1.0] * s.d_k, b=g.b, w=g.w, beta=g.beta,
        )
        pre = State(s.d_k, s.d_v, cur)
        new, out = self.inner.step(pre, inner_g)

        # rung 6: a second erase direction derived from the QUERY (arXiv 2608.13668).
        # "The query knows what to forget": asking about rent should suppress what is
        # stale about rent, addressed by what you're asking rather than what you're
        # writing.
        if g.q_erase is not None and g.gamma_q > 0.0:
            qe = l2_normalise(g.q_erase)
            proj = _read(new.S, qe)
            S2 = [
                [row[j] - g.gamma_q * qe[i] * proj[j] for j in range(s.d_v)]
                for i, row in enumerate(new.S)
            ]
            new = State(s.d_k, s.d_v, S2)
            out = _read(S2, g.q)
        return new, out


# ── registry ───────────────────────────────────────────────────────────────────

def build(name: str, **kw: Any) -> Operator:
    """One factory, so the ablation ladder is a config file and not a code change."""
    n = name.lower().strip()
    if n in ("retnet", "rung0"):
        return RetNet(**kw)
    if n in ("gdn", "gated_deltanet", "rung1"):
        return GatedDeltaNet(**kw)
    if n in ("kda", "kimi_delta", "rung2"):
        return KimiDelta(**kw)
    if n in ("gdn2_e", "gdn2_erase", "rung3"):
        return GatedDeltaNet2(write_gate=False, **kw)
    if n in ("gdn2", "gated_deltanet2", "rung4"):
        return GatedDeltaNet2(write_gate=True, **kw)
    if n in ("eda", "erase_then_delta", "rung5", "rung6", "qerase"):
        return EraseThenDelta(**kw)
    raise KeyError(f"unknown operator '{name}'. known: {sorted(OPERATOR_NAMES)}")


OPERATOR_NAMES = ("retnet", "gdn", "kda", "gdn2_e", "gdn2", "eda")

#: The ladder, in the order docs/architecture/13 §5.2 says to run it.
ABLATION_LADDER: tuple[tuple[str, str], ...] = (
    ("retnet", "rung 0 — fixed decay, additive write. The control that must lose."),
    ("gdn",    "rung 1 — scalar decay + scalar delta gate (ICLR 2025 baseline)."),
    ("kda",    "rung 2 — channel-wise decay + scalar gate. GDN-2's tied subspace: FREE CONTROL."),
    ("gdn2_e", "rung 3 — + channel-wise ERASE gate b_t. NVlabs: 'most of the gain' is here."),
    ("gdn2",   "rung 4 — + channel-wise WRITE gate w_t. Full Gated DeltaNet-2."),
    ("eda",    "rung 5 — + independently ADDRESSED erase e_t. The bi-temporal win."),
)


def _read(S: Matrix, q: Vector) -> Vector:
    """S^T q for S of shape (d_k, d_v) -> length d_v."""
    return [dot(q, [S[k][v] for k in range(len(S))]) for v in range(len(S[0]))]
