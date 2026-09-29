"""The Retention State Compiler — the module, not the backbone.

docs/architecture/12 §6, /13 §5. A small (~10-100M param target; ~2M in this
reference implementation) recurrent network that compiles unbounded conversation
history into a FIXED-SIZE state, which is then injected into a frozen backbone as a
prefix-state / soft prompt.

    history (unbounded, lossless, on disk)
        -> RSC: gated recurrence over heads, SWA over the recent window
        -> S in R^{L x d}            FIXED SIZE
        -> frozen LFM2 backbone sees [RSC state] + [identity prefix]
                                     + [retrieved facts] + [current turn]

Two outputs, not one — and the second is why this is FRIDAY's and not a generic
context compressor:

    1. the STATE          -> conversational gist, current thread, register
    2. the SALIENCE HEAD  -> "this span contains a durable fact" -> S2 extraction

Compression and fact extraction in one forward pass. That unifies S1 (episode) with
the Ledger's transcript slot, which is what the original proposal intuited.

⚠️ LAW 2b IS NOT NEGOTIABLE HERE. The state is lossy BY CONSTRUCTION and loses
associative recall first. It carries *context*. Facts live in SQLite/Markdown and
arrive by retrieval. Nothing in this file may be used as the authoritative record —
`L_recall` exists to enforce that at training time and `scripts/needle_test.py`
enforces it at eval time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .operators import (
    Gates,
    Operator,
    State,
    build,
    dot,
    l2_normalise,
    mat_scale_rows,
    rms_norm,
    sigmoid,
)


# ── a tiny dense layer, because there is no torch here ─────────────────────────

@dataclass
class Linear:
    """W (out, in) + bias. Initialised small; trained by `losses.sgd_step`."""

    out_dim: int
    in_dim: int
    seed: int = 0
    W: list[list[float]] = field(default_factory=list)
    b: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.W:
            rng = random.Random(self.seed)
            scale = 1.0 / math.sqrt(max(1, self.in_dim))
            self.W = [[rng.gauss(0.0, scale) for _ in range(self.in_dim)]
                      for _ in range(self.out_dim)]
            self.b = [0.0] * self.out_dim

    def __call__(self, x: Sequence[float]) -> list[float]:
        return [dot(row, x) + bi for row, bi in zip(self.W, self.b)]

    def params(self) -> list[float]:
        return [v for row in self.W for v in row] + list(self.b)

    def numel(self) -> int:
        return self.out_dim * self.in_dim + self.out_dim


def swish(x: float) -> float:
    return x * sigmoid(x)


# ── config ─────────────────────────────────────────────────────────────────────

@dataclass
class RSCConfig:
    """Mirrors the paper's block design where it matters, at a scale a T4 can chew.

    Paper (1.3B): 16 heads, d_k = d_v = 128, state 262,144 floats/layer ~1 MB.
    Here: small enough to unit-test in pure Python, structured so the shapes match
    and a real kernel can be dropped in without changing the interface.
    """

    d_model: int = 64
    n_heads: int = 4
    d_k: int = 16
    d_v: int = 16
    n_layers: int = 2
    swa_window: int = 64          # paper: w=2048 for local evidence
    operator: str = "gdn2"        # one of operators.OPERATOR_NAMES
    state_slots: int = 8          # L: how many state vectors are injected
    dropout_gate: float = 0.0
    seed: int = 1234

    @property
    def head_dim_in(self) -> int:
        return self.d_k

    @property
    def state_numel(self) -> int:
        return self.n_layers * self.n_heads * self.d_k * self.d_v

    def state_bytes_fp32(self) -> int:
        return 4 * self.state_numel


# ── the module ─────────────────────────────────────────────────────────────────

class RetentionStateCompiler:
    """Per layer: recurrent mixer (n_heads x operator) -> MLP -> SWA -> MLP.

    Gate branches are SEPARATE projections — decay, erase, write each get their own.
    That is not incidental: separate branches are what make them separately
    *supervisable*, which is the entire basis of gate_supervision.py.
    """

    def __init__(self, cfg: RSCConfig | None = None):
        self.cfg = cfg or RSCConfig()
        c = self.cfg
        rng = random.Random(c.seed)

        self.in_proj = Linear(c.d_model, c.d_model, seed=rng.randrange(10**6))
        self.qkv = Linear(3 * c.n_heads * c.d_k, c.d_model, seed=rng.randrange(10**6))

        # three gate branches per head, per layer — alpha / b / w are independent
        self.decay_proj = [
            Linear(c.n_heads * c.d_k, c.d_model, seed=rng.randrange(10**6))
            for _ in range(c.n_layers)
        ]
        self.erase_proj = [
            Linear(c.n_heads * c.d_k, c.d_model, seed=rng.randrange(10**6))
            for _ in range(c.n_layers)
        ]
        self.write_proj = [
            Linear(c.n_heads * c.d_v, c.d_model, seed=rng.randrange(10**6))
            for _ in range(c.n_layers)
        ]
        # EDA: the independently addressed erase direction. Factorised with a 16-d
        # intermediate per head, as in TERN's description of EDA:
        #   e_t = W2 W1 z_t / ||W2 W1 z_t||
        self.erase_addr_1 = [
            Linear(c.n_heads * 16, c.d_model, seed=rng.randrange(10**6))
            for _ in range(c.n_layers)
        ]
        self.erase_addr_2 = [
            Linear(c.n_heads * c.d_k, c.n_heads * 16, seed=rng.randrange(10**6))
            for _ in range(c.n_layers)
        ]
        self.erase_strength = [
            Linear(c.n_heads, c.d_model, seed=rng.randrange(10**6))
            for _ in range(c.n_layers)
        ]

        self.out_gate = Linear(c.d_model, c.d_model, seed=rng.randrange(10**6))
        self.out_proj = Linear(c.d_model, c.d_model, seed=rng.randrange(10**6))
        self.mlp1 = Linear(c.d_model, c.d_model, seed=rng.randrange(10**6))
        self.mlp2 = Linear(c.d_model, c.d_model, seed=rng.randrange(10**6))

        # ⭐ the salience side-channel: binary "did this span produce a fact?"
        self.salience_head = Linear(1, c.d_model, seed=rng.randrange(10**6))

        self.ops: list[list[Operator]] = [
            [build(c.operator) for _ in range(c.n_heads)] for _ in range(c.n_layers)
        ]

    # ── forward ────────────────────────────────────────────────────────────────

    def compile(self, hidden: Sequence[Sequence[float]], *,
                supervision: dict | None = None) -> dict[str, Any]:
        """Run the recurrence over a sequence of hidden vectors.

        `hidden` is (T, d_model) — the frozen backbone's embeddings for the history,
        or a cheap stand-in during unit tests. Returns:

            state       (state_slots, d_model)  the fixed-size thing we inject
            salience    (T,)                    per-token durable-fact probability
            per_head    (n_layers, n_heads, State)   final states, for probes
            gate_stats  summary of alpha/b/w/e — for telemetry and debugging

        `supervision` optionally supplies per-token targets used to *bias* the gates
        at inference (curriculum/annealing); training uses the losses instead.
        """
        c = self.cfg
        T = len(hidden)
        if T == 0:
            return {"state": [[0.0] * c.d_model for _ in range(c.state_slots)],
                    "salience": [], "per_head": [], "gate_stats": {}}

        states = [[State(c.d_k, c.d_v) for _ in range(c.n_heads)] for _ in range(c.n_layers)]
        window: list[list[float]] = []
        salience: list[float] = []
        gate_acc: dict[str, list[float]] = {"alpha": [], "b": [], "w": [], "gamma_e": []}
        outs: list[list[float]] = []

        for t in range(T):
            x = list(hidden[t])
            xin = [swish(v) for v in self.in_proj(x)]

            qkv = self.qkv(xin)
            per_head = c.n_heads * c.d_k
            qs = [qkv[i * c.d_k:(i + 1) * c.d_k] for i in range(c.n_heads)]
            ks = [qkv[per_head + i * c.d_k: per_head + (i + 1) * c.d_k] for i in range(c.n_heads)]
            vs = [qkv[2 * per_head + i * c.d_v: 2 * per_head + (i + 1) * c.d_v]
                  for i in range(c.n_heads)]

            # paper's block design: q,k get short conv + SiLU + L2 normalisation for
            # stability; v gets conv + SiLU only. The short conv is approximated here
            # by mixing in the previous token, which is what a width-2 causal conv does.
            prev = window[-1] if window else None
            h_cur = xin

            concat_heads: list[float] = []
            for L in range(c.n_layers):
                a_raw = self.decay_proj[L](h_cur)
                b_raw = self.erase_proj[L](h_cur)
                w_raw = self.write_proj[L](h_cur)

                for h in range(c.n_heads):
                    sl = slice(h * c.d_k, (h + 1) * c.d_k)
                    q = l2_normalise([swish(v) for v in qs[h]])
                    k = l2_normalise([swish(v) for v in ks[h]])
                    if prev is not None:
                        k = l2_normalise([0.75 * a + 0.25 * b
                                          for a, b in zip(k, _slice(prev, h, c.d_k))])
                    v = [swish(x) for x in vs[h]]

                    # decay in (0,1]: softplus-then-sigmoid keeps it off the 0/1 rails
                    alpha = [sigmoid(a_raw[h * c.d_k + i]) for i in range(c.d_k)]
                    b = [sigmoid(b_raw[h * c.d_k + i]) for i in range(c.d_k)]
                    w = [sigmoid(w_raw[h * c.d_v + i]) for i in range(c.d_v)]

                    g = Gates(q=q, k=k, v=v, alpha=alpha, b=b, w=w,
                              beta=sum(b) / len(b))

                    op = self.ops[L][h]
                    if getattr(op, "has_erase_address", False):
                        e = self._erase_address(L, h, h_cur, c)
                        strength = sigmoid(self.erase_strength[L](h_cur)[h])
                        g.e = e
                        g.gamma_e = strength
                        gate_acc["gamma_e"].append(strength)

                    states[L][h], _ = op.step(states[L][h], g)
                    gate_acc["alpha"].extend(alpha)
                    gate_acc["b"].extend(b)
                    gate_acc["w"].extend(w)

                # read out this layer: concatenate head reads, project back to d_model
                reads: list[float] = []
                for h in range(c.n_heads):
                    reads.extend(states[L][h].read(qs[h]))
                mixed = [swish(v) for v in Linear(c.d_model, len(reads), seed=7)(reads)] \
                    if len(reads) != c.d_model else [swish(v) for v in reads]
                h_cur = _residual(h_cur, mixed[: c.d_model])

                # SWA over the recent window: "SWA handling local evidence" while the
                # recurrent operator "preserves longer-range associations".
                h_cur = self._swa(h_cur, window, L)
                h_cur = _residual(h_cur, [swish(v) for v in self.mlp1(h_cur)])
                h_cur = _residual(h_cur, [swish(v) for v in self.mlp2(h_cur)])

            window.append(h_cur)
            if len(window) > c.swa_window:
                window.pop(0)

            salience.append(sigmoid(self.salience_head(h_cur)[0]))
            gated = [swish(v) for v in self.out_gate(h_cur)]
            outs.append([a * b for a, b in zip(rms_norm(h_cur), gated)])

        out_proj = [self.out_proj(o) for o in outs]
        state = self._pool_state(out_proj, states)
        return {
            "state": state,
            "salience": salience,
            "outputs": out_proj,
            "per_head": states,
            "gate_stats": {k: _mean(v) for k, v in gate_acc.items() if v},
            "tokens": T,
        }

    # ── pieces ─────────────────────────────────────────────────────────────────

    def _erase_address(self, L: int, h: int, h_cur: list[float], c: RSCConfig) -> list[float]:
        """e_t = W2 W1 z_t / ||W2 W1 z_t||  — factorised, 16-d intermediate per head.

        The factorisation is not a size optimisation here: it is what lets the erase
        direction live in a DIFFERENT subspace from the write key, which is the whole
        point of decoupling addresses rather than strengths.
        """
        z = self.erase_addr_1[L](h_cur)
        mid = z[h * 16:(h + 1) * 16]
        full = self.erase_addr_2[L](_pad(mid, c.d_model))
        e = full[h * c.d_k:(h + 1) * c.d_k]
        return l2_normalise(e)

    def _swa(self, h: list[float], window: list[list[float]], layer: int) -> list[float]:
        """Causal sliding-window attention over the recent window, single head.

        This is the "local precision" half of the hybrid. It is NOT the retrieval
        store: sqlite-vec/FTS5 is reached by a tool call, is unbounded, and is not
        differentiable. Conflating the two is the category error
        docs/architecture/13 §3.3 corrects.
        """
        if not window:
            return h
        c = self.cfg
        recent = window[-c.swa_window:]
        qn = l2_normalise(h)
        scores = [dot(qn, l2_normalise(p)) / math.sqrt(len(qn)) for p in recent]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps) or 1.0
        acc = [0.0] * len(h)
        for w, p in zip(exps, recent):
            for i in range(len(acc)):
                acc[i] += (w / z) * (p[i] if i < len(p) else 0.0)
        return acc

    def _pool_state(self, outputs: list[list[float]], states: list[list[State]]) -> list[list[float]]:
        """Reduce everything to a FIXED number of d_model vectors.

        This is the step that makes the memory O(1): however long the history, the
        injected block is `state_slots x d_model`. The last `state_slots` outputs are
        the most recent summary; a mean over head states contributes the associative
        gist. Both are cheap and neither grows with T.
        """
        c = self.cfg
        tail = outputs[-c.state_slots:]
        while len(tail) < c.state_slots:
            tail.insert(0, [0.0] * c.d_model)

        # fold head states in as an extra slot-group so the injected block carries
        # something of the associative structure, not just the recent summary
        flat: list[float] = []
        for L in states:
            for h in L:
                flat.extend([v for row in h.S for v in row][: c.d_model])
        if flat:
            n = c.d_model
            pooled = [sum(flat[i::n][:16]) / max(1, len(flat[i::n][:16])) for i in range(n)]
            tail[-1] = [0.5 * a + 0.5 * b for a, b in zip(tail[-1], pooled)]
        return tail

    def numel(self) -> int:
        total = 0
        for attr in vars(self).values():
            if isinstance(attr, Linear):
                total += attr.numel()
            elif isinstance(attr, list):
                for x in attr:
                    if isinstance(x, Linear):
                        total += x.numel()
        return total


# ── helpers ────────────────────────────────────────────────────────────────────

def _slice(v: Sequence[float], h: int, d: int) -> list[float]:
    return list(v[h * d:(h + 1) * d])


def _pad(v: Sequence[float], n: int) -> list[float]:
    out = list(v)
    if len(out) < n:
        out += [0.0] * (n - len(out))
    return out[:n]


def _residual(a: Sequence[float], b: Sequence[float]) -> list[float]:
    n = min(len(a), len(b))
    return [a[i] + b[i] for i in range(n)]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def render_state_block(state: Sequence[Sequence[float]], *, decimals: int = 4) -> str:
    """Serialise the compiled state into the Ledger's `recalled` prefix.

    A real deployment injects this as *vectors* into the backbone's embedding space.
    The text form exists so the state is inspectable in the Ledger printout — you
    should be able to look at what FRIDAY remembers, which is Law 7 applied to a
    tensor.
    """
    rows = []
    for i, v in enumerate(state):
        head = ", ".join(f"{x:+.{decimals}f}" for x in v[:8])
        rows.append(f"state[{i}] = [{head}, …{len(v)-8} more]")
    return "<retention_state>\n" + "\n".join(rows) + "\n</retention_state>"
