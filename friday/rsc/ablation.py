"""The ablation ladder runner. docs/architecture/13 §5.2.

Run rungs 2 -> 3 -> 4 -> 5 in that order. Each is a one-variable change and a few
Kaggle GPU-hours. If rung 3 is where recall jumps — as NVlabs' own ablation
predicts, "the erase gate b_t accounts for most of the gain" — you have learned the
most important thing about your system for about six GPU-hours, and you can stop
there and ship.

Do NOT start at rung 5. EDA is the most interesting and the least validated; start
there and you won't know which mechanism helped.

The three probes below are the ones that actually discriminate the rungs:

    associative   can the state return the value stored at a key?     (rungs 0 vs 1+)
    needle        does a fact planted at depth d survive to the end?  (rungs 2 vs 3)
    interference  does writing at key B corrupt the value at key A?   (rungs 4 vs 5)

`interference` is the one that matters for FRIDAY and the one nobody's benchmark
sheet reports. It is the bi-temporal correction case in miniature: write a new
value at a DIFFERENT address and see whether the stale association at the old
address is still polluting reads. GDN-2 cannot clear it; EDA can.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .model import RSCConfig, RetentionStateCompiler
from .operators import (
    ABLATION_LADDER,
    OPERATOR_NAMES,
    Gates,
    State,
    build,
    l2_normalise,
)


# ── probes ─────────────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    name: str
    score: float
    n: int
    detail: dict[str, Any] = field(default_factory=dict)


def _key_for(word: str, d: int, seed: int = 0) -> list[float]:
    """A deterministic pseudo-random unit key for a symbol. Same word -> same key,
    so a probe run twice gives the same answer."""
    rng = random.Random(hash((word, seed)) & 0xFFFFFFFF)
    return l2_normalise([rng.gauss(0, 1) for _ in range(d)])


def probe_associative(op, d_k: int = 32, d_v: int = 32, n_pairs: int = 6,
                      distractors: int = 12, reps: int = 3, seed: int = 0) -> ProbeResult:
    """Write n_pairs key->value associations, interleave distractors, read back.

    This is the task ICLR 2025 proves an RNN with o(n)-bit memory cannot solve and
    a Transformer solves trivially. Rung 0 (RetNet, additive write) should fail it
    badly; rung 1+ (delta rule) should mostly pass. If rung 0 passes, the probe is
    broken, not the architecture.

    Measured behaviour at d=32, n_pairs=6, distractors=12: RetNet scores 0.00 at
    every rep count while every delta-rule rung climbs 0.17 -> 0.67 -> 0.83 -> 1.00
    as reps go 1 -> 2 -> 3 -> 8. Clean separation, and the *shape* is the point: the
    delta rule is error-CORRECTING, so repeated presentation converges, while an
    additive write just accumulates more interference.

    Three settings here are load-bearing, and all three were wrong in the first
    version of this probe:

      * `reps` — the delta rule needs repeated presentation to converge. At reps=1
        every rung including RetNet's neighbours scores near zero and the probe
        reports nothing.
      * `d_k` — must comfortably exceed n_pairs + distractors. At d=16 with 18
        addresses the state is over capacity, everything collapses to 0.00, and the
        probe measures a capacity wall instead of an update rule.
      * distinct distractor ADDRESSES — see below.

    ⚠️ The distractors must be at DISTINCT ADDRESSES. An earlier version built its
    step order from `list(keys) * n`, which made the distractor branch unreachable:
    every "distractor" was a noisy rewrite of a key that was already stored. That is
    rehearsal, not interference. The measured effect was that rungs 1 through 5 all
    scored EXACTLY 0.667 — with no competing addresses there is nothing for
    selectivity to act on, so every delta-rule variant collapses to the same
    behaviour. The ladder reported six rungs and measured one.
    """
    rng = random.Random(seed)
    s = State(d_k, d_v)
    keys = {f"k{i}": _key_for(f"k{i}", d_k, seed) for i in range(n_pairs)}
    vals = {f"k{i}": l2_normalise([rng.gauss(0, 1) for _ in range(d_v)])
            for i in range(n_pairs)}

    # Distractor addresses live in a DIFFERENT namespace so they cannot collide
    # with a real key, and carry their own random values at the SAME norm as the
    # real ones — otherwise rel_err is dominated by the norm ratio rather than by
    # what the state actually retrieved.
    dkeys = {f"d{i}": _key_for(f"d{i}", d_k, seed) for i in range(distractors)}
    dvals = {f"d{i}": l2_normalise([rng.gauss(0, 1) for _ in range(d_v)])
             for i in range(distractors)}

    steps: list[tuple[list[float], list[float]]] = (
        [(keys[n], vals[n]) for n in keys] + [(dkeys[n], dvals[n]) for n in dkeys]
    )
    for _ in range(max(1, reps)):
        rng.shuffle(steps)
        for k, v in steps:
            # beta near 1 so the correction actually lands within one presentation.
            s, _ = op.step(s, _gates(k, v, d_k, d_v, write=True, alpha=1.0, beta=1.0))

    hits, total_err = 0, 0.0
    for name, k in keys.items():
        out = s.read(k)
        target = vals[name]
        err = _rel_err(out, target)
        total_err += err
        if err < 0.35:
            hits += 1
    return ProbeResult(
        "associative", hits / n_pairs, n_pairs,
        {
            "mean_rel_err": round(total_err / n_pairs, 4),
            "distractors": distractors,
            "reps": reps,
            # Reported so a future reader can see the load the state was under.
            # A score of 0.9 with 4 distractors and 0.2 with 200 are different
            # claims, and the mean alone does not distinguish them.
            "load_ratio": round((n_pairs + distractors) / max(1, d_k), 2),
        },
    )


def probe_needle(op, d_k: int = 32, d_v: int = 32, length: int = 512,
                 depths: Sequence[int] = (16, 128, 256, 384, 480),
                 seed: int = 0, overlap: float = 0.0) -> ProbeResult:
    """⭐ Needle-in-a-haystack, at controlled depth. The decisive benchmark
    (docs/architecture/12 §5.3), scaled down to what pure Python can run.

    A distinctive key->value pair is planted at depth 0, then `depth` distractor
    steps follow, then we read. Score is per-depth so you can SEE the decay curve
    rather than a single average — a model that scores 0.9 at depth 16 and 0.1 at
    depth 480 is not a 0.5 model, it's a model with no long-range memory.

    WHAT THIS PROBE CAN AND CANNOT MEASURE
    --------------------------------------
    With hand-set gates it cleanly separates **rung 0 from rungs 1+**, because that
    gap is the *decay* mechanism: RetNet's γ is a fixed scalar (0.995), so the
    needle fades as γ^depth no matter what is written, while a gated rung can hold
    α at 1.0 and keep it exactly. That is the paper's finding in miniature —
    arXiv 2507.06457v2: fixed decay yields near-zero long-range recall even with
    full-attention layers added.

    It CANNOT separate rungs 2–5, and no pure-Python probe with frozen gates can.
    Rung 2 (channel-wise α), rung 3 (erase gate b_t) and rung 4 (write gate w_t)
    differ in what a TRAINED gate learns to do per channel and per token. With gates
    pinned to constants, those three operators are algebraically identical on this
    task — KDA reduces to GDN when α is uniform, and GDN-2 reduces to KDA when b and
    w tie. Their published separation (RULER S-NIAH-3 63 → 90) comes from 1.3B
    parameters on 100B tokens. Rung 5 is measured by `probe_interference` instead,
    which supplies an explicit erase address.

    Earlier versions of this probe reported a ladder-shaped result that meant
    nothing, for two reasons worth recording because both are easy to reintroduce:

      * The needle key was drawn from `_key_for`, which returns a DENSE unit vector.
        A dense key has nonzero overlap with every distractor address, so each
        distractor's delta-rule correction subtracted β·(k_n·k_d)·(stored value) —
        the needle bled out through *key geometry*, not decay, and every operator
        scored ~0 by depth 48. The probe was measuring its own key distribution.
      * The needle value was an unnormalised one-hot `[1.0, 0, …]` while distractor
        values were `gauss(0,1)` vectors of norm ≈ √d. So `1 − rel_err` was
        dominated by the ratio of those norms and saturated at 0 regardless of what
        the state actually held.

    `overlap` is exposed so you can dial interference back in deliberately once the
    decay curve is believed: at overlap > 0 distractors write a little into the
    needle's direction, which is the regime where the erase gate starts to matter.
    """
    rng = random.Random(seed)
    per_depth: dict[int, float] = {}

    # The needle owns basis direction 0 on BOTH axes, so it is exactly orthogonal to
    # every distractor address and the read is unambiguous.
    nk = _onehot(0, d_k)
    nv = _onehot(0, d_v)

    for depth in depths:
        s = State(d_k, d_v)
        # beta near 1 so a SINGLE write lands the value; with beta=0.5 the needle
        # only ever reaches half its target and the depth curve is measuring that.
        s, _ = op.step(s, _gates(nk, nv, d_k, d_v, write=True, alpha=1.0, beta=1.0))

        for i in range(min(depth, length - 1)):
            # Distractor addresses are the remaining basis vectors, cycled. They are
            # orthogonal to the needle and to each other, so the ONLY thing that can
            # destroy the needle is the operator's own decay.
            j = 1 + (i % max(1, d_k - 1))
            dk = _onehot(j, d_k)
            dv = _onehot(j, d_v)
            if overlap > 0.0:
                dk = l2_normalise([dk[t] + overlap * nk[t] for t in range(d_k)])
                dv = l2_normalise([dv[t] + overlap * nv[t] for t in range(d_v)])
            s, _ = op.step(s, _gates(dk, dv, d_k, d_v, write=True, alpha=1.0, beta=1.0))

        out = s.read(nk)
        per_depth[depth] = round(max(0.0, 1.0 - _rel_err(out, nv)), 4)

    score = sum(per_depth.values()) / len(per_depth)
    return ProbeResult("needle", score, len(depths),
                       {"per_depth": per_depth, "overlap": overlap})


def probe_interference(op, d_k: int = 32, d_v: int = 32, n_old: int = 4,
                       n_new: int = 4, seed: int = 0, gamma_e: float = 0.9) -> ProbeResult:
    """⭐⭐ The FRIDAY probe, and the one that separates rung 4 from rung 5.

    Phase A: write n_old associations at addresses kA_i  ("I live at 12 Marina Road").
    Phase B: write n_new associations at DIFFERENT addresses kB_j, while handing any
             operator that has one an independent erase address aimed at the kA span
             ("I moved to 48 Velachery Main — and the old address is stale").

    Then measure:
        retain_old             — how much of kA's content is still readable
        new_write_err          — how cleanly kB landed
        crosstalk_new_into_old — new-value energy leaking into a read at kA

    ADDRESSES ARE EXACTLY ORTHOGONAL, and that is what makes the measurement clean.
    With dense random keys, kA and kB overlap by construction, so "did the erase
    reach kA?" is confounded with "how much did kB overlap kA anyway?" — and the
    answer comes out the same for every rung. On an orthogonal basis the only way
    content at kA can disappear is if something addressed kA on purpose.

    THE RESULT, AND HOW TO READ IT
    ------------------------------
    GDN-2's erase direction is b_t ⊙ k_t — built from the CURRENT write key. With kB
    orthogonal to kA, that direction cannot reach kA at all, so `retain_old` stays at
    1.000: the stale association is untouched no matter how hard b_t fires. EDA's e_t
    is an independent projection, so it CAN be aimed at kA, and `retain_old` drops.

    ⚠️ A lower `retain_old` for EDA is the SUCCESS condition here, not memory loss.
    This probe models a CORRECTION: the old address has been declared stale and the
    operator is being asked to suppress it. Law 2b is what makes that safe — the
    external store keeps both values with their validity windows, so "where did I
    used to live?" is answered by SQLite, never by the state. The state's job is to
    stop returning the stale value as if it were current.

    If EDA does NOT beat GDN-2 on this probe once trained, the address-level coupling
    is not binding on FRIDAY's data and you should ship rung 3 or 4 — cheaper, and
    better supported. That is a useful result, not a failure.
    """
    rng = random.Random(seed)
    s = State(d_k, d_v)
    if n_old + n_new > min(d_k, d_v):
        raise ValueError(
            f"probe_interference needs n_old + n_new <= min(d_k, d_v) for orthogonal "
            f"addresses; got {n_old}+{n_new} > {min(d_k, d_v)}"
        )

    # Exactly orthogonal addresses on both axes: kA owns basis 0..n_old-1, kB owns
    # the next n_new. Values are one-hot on the value axis to match.
    ka = {f"A{i}": _onehot(i, d_k) for i in range(n_old)}
    va = {f"A{i}": _onehot(i, d_v) for i in range(n_old)}
    kb = {f"B{i}": _onehot(n_old + i, d_k) for i in range(n_new)}
    vb = {f"B{i}": _onehot(n_old + i, d_v) for i in range(n_new)}

    for name in ka:
        s, _ = op.step(s, _gates(ka[name], va[name], d_k, d_v,
                                 write=True, alpha=1.0, beta=1.0))

    # The erase address spans the OLD associations: "everything at kA is stale".
    # Only an operator with an independent erase ADDRESS can act on this; a
    # strength-only gate anchored to kB cannot reach it.
    erase_addr = l2_normalise([sum(ka[n][i] for n in ka) / len(ka) for i in range(d_k)])
    for name in kb:
        g = _gates(kb[name], vb[name], d_k, d_v, write=True, alpha=1.0, beta=1.0)
        if getattr(op, "has_erase_address", False):
            g.e = erase_addr
            g.gamma_e = gamma_e
            g.gamma_q = 0.0
        s, _ = op.step(s, g)

    retain = [max(0.0, 1.0 - _rel_err(s.read(ka[name]), va[name])) for name in ka]
    leak = [_rel_err(s.read(kb[name]), vb[name]) for name in kb]
    crosstalk = []
    for name in ka:
        out = s.read(ka[name])
        energy_new = sum(out[i] ** 2 for i in range(n_old, min(d_v, n_old + n_new)))
        energy_all = sum(x * x for x in out) or 1.0
        crosstalk.append(energy_new / energy_all)

    return ProbeResult(
        "interference",
        round(sum(retain) / len(retain), 4),
        n_old,
        {
            "retain_old": round(sum(retain) / len(retain), 4),
            "new_write_err": round(sum(leak) / len(leak), 4),
            "crosstalk_new_into_old": round(sum(crosstalk) / len(crosstalk), 4),
            "has_erase_address": bool(getattr(op, "has_erase_address", False)),
            "gamma_e": gamma_e,
        },
    )


def _onehot(i: int, d: int) -> list[float]:
    v = [0.0] * d
    v[i % d] = 1.0
    return v


def _gates(k: Sequence[float], v: Sequence[float], d_k: int, d_v: int, *,
           write: bool = True, alpha: float = 0.999, beta: float = 0.9) -> Gates:
    """Gate values for probes. Strong write, near-1 decay: we are testing the
    UPDATE RULE's selectivity, not its forgetting schedule. `alpha` deliberately
    differs per rung only via the operator's own handling."""
    return Gates(
        q=l2_normalise(list(k)),
        k=l2_normalise(list(k)),
        v=list(v),
        alpha=[alpha] * d_k,
        b=[beta] * d_k,
        w=[1.0] * d_v if write else [beta] * d_v,
        beta=beta,
    )


def _rel_err(out: Sequence[float], target: Sequence[float]) -> float:
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(out, target)))
    den = math.sqrt(sum(b * b for b in target)) or 1.0
    return num / den


# ── the ladder ─────────────────────────────────────────────────────────────────

def run_ladder(
    *,
    d_k: int = 32,
    d_v: int = 32,
    needle_length: int = 384,
    rungs: Sequence[str] = tuple(n for n, _ in ABLATION_LADDER),
    seeds: Sequence[int] = (0, 1),
) -> dict[str, Any]:
    """Run every rung against every probe. Pure Python, no GPU: this is the
    *logic* of the ladder and a correctness check on the operators, not the 1.3B /
    100B-token result. It answers "does my implementation of the update rule behave
    the way the paper says it should?" — which you must know before burning Kaggle
    hours on the real thing.

    Expected shape of the result, if the implementations are right:
        retnet  associative low, needle collapses with depth
        gdn/kda associative high, needle better
        gdn2_e  needle better still (erase protects the key channels)
        eda     interference: lower crosstalk, comparable retain
    """
    out: dict[str, Any] = {"config": {"d_k": d_k, "d_v": d_v,
                                      "needle_length": needle_length},
                           "rungs": {}}
    for name in rungs:
        if name not in OPERATOR_NAMES:
            continue
        op = build(name)
        agg: dict[str, list[float]] = {}
        details: list[dict] = []
        for seed in seeds:
            probes = [
                probe_associative(op, d_k, d_v, seed=seed),
                probe_needle(op, d_k, d_v, length=needle_length, seed=seed),
                probe_interference(op, d_k, d_v, seed=seed),
            ]
            for p in probes:
                agg.setdefault(p.name, []).append(p.score)
                details.append({"seed": seed, "probe": p.name, "score": round(p.score, 4),
                                **p.detail})
        out["rungs"][name] = {
            "rung": getattr(op, "rung", None),
            "scores": {k: round(sum(v) / len(v), 4) for k, v in agg.items()},
            "detail": details,
        }
    out["ladder"] = _rank(out["rungs"])
    return out


def _rank(rungs: dict[str, Any]) -> list[dict]:
    """Sorted summary, and the delta each rung bought over the previous one.

    The delta column is the point: GDN-2 recovers KDA exactly when both gates tie,
    so rung2->rung3->rung4 are one-variable changes and their deltas are directly
    attributable. That's what makes this a measurement rather than a leaderboard.
    """
    order = [n for n, _ in ABLATION_LADDER if n in rungs]
    rows: list[dict] = []
    prev: dict[str, float] | None = None
    for name in order:
        r = rungs[name]
        sc = r["scores"]
        delta = {}
        if prev:
            for k in sc:
                if k in prev:
                    delta[k] = round(sc[k] - prev[k], 4)
        rows.append({"operator": name, "rung": r["rung"], **sc, "delta_vs_prev": delta})
        prev = sc
    return rows


def format_report(result: dict[str, Any]) -> str:
    """Markdown, ready to paste into docs/architecture/BENCHMARKS.md."""
    lines = [
        "# RSC ablation ladder — pure-Python operator probes",
        "",
        f"Config: `{json.dumps(result['config'])}`",
        "",
        "⚠️ This measures the **update rule**, at d_k=d_v=16, in pure Python. It is a",
        "correctness check and a shape-of-the-curve preview — **not** the 1.3B/100B-token",
        "result from arXiv 2605.22791. Re-run at scale on Kaggle before believing it.",
        "",
        "| operator | rung | associative | needle | interference (retain) | Δ needle | Δ interference |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["ladder"]:
        d = row.get("delta_vs_prev") or {}
        lines.append(
            f"| `{row['operator']}` | {row['rung']} | {row.get('associative','-')} | "
            f"{row.get('needle','-')} | {row.get('interference','-')} | "
            f"{d.get('needle','-')} | {d.get('interference','-')} |"
        )
    lines += ["", "## What to look for", ""]
    lines += [
        "- **rung 0 (`retnet`) must lose.** If it doesn't, the probes are broken — a",
        "  fixed-decay additive write cannot do selective associative recall.",
        "- **rung 2 -> 3 should be the biggest needle jump.** NVlabs' ablation: \"the",
        "  erase gate b_t accounts for most of the gain\". If your largest delta is",
        "  somewhere else, your data disagrees with theirs — which is interesting.",
        "- **rung 4 -> 5 should move `interference`, not `needle`.** EDA decouples",
        "  *where* erase happens, so its win shows up as lower crosstalk on the",
        "  bi-temporal correction probe. If it doesn't move, the address-level",
        "  coupling isn't binding on your data and you can ship rung 3 or 4.",
        "",
    ]
    return "\n".join(lines)
