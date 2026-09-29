# The fast path: going from this reference implementation to a trained RSC

`operators.py` is **pure Python on purpose**. It is exact, dependency-free, and
runnable on any machine — including this sandbox and a free Kaggle T4 — which makes
it the right place to get the *math* right. It is also roughly four orders of
magnitude too slow to train anything real.

This file is the bridge. It exists so that "the reference implementation is slow"
never silently becomes "so we never trained it".

---

## 1. Which rung you are on determines which code you use

| Rung | Operator | Fastest available path |
|---|---|---|
| 0 | `retnet` | `fla` — `fla.ops.retention` (battle-tested Triton) |
| 1 | `gdn` | `fla` — `fla.ops.gated_delta_rule` |
| 2 | `kda` | `fla` — `fla.ops.kda` |
| 3 | `gdn2_e` | ⚠️ **`fla` has no GDN-2.** Start from `fla`'s GDN and add the channel-wise erase gate yourself |
| 4 | `gdn2` | **`NVlabs/GatedDeltaNet-2`** — official PyTorch + Triton, chunkwise WY form, gate-aware backward |
| 5 | `eda` | ⚠️ **No public kernel found.** Port from the paper, or extend rung 4 |

**Practical consequence:** rungs 0–2 are free. Rung 3 is a small diff on `fla`.
Rung 4 is an import. **Rung 5 is real engineering** — which is a further reason
`docs/architecture/13 §5.2` says to run 2 → 3 → 4 first and treat EDA as a stretch
goal gated on `L_erase` plateauing.

## 2. NVlabs repo: what to take and what to leave

`github.com/NVlabs/GatedDeltaNet-2` (323★, active) is a **pretraining harness**, not
a library:

```
take    lit_gpt/          the GDN-2 mixer definition + Triton kernels
take    cache.py          the recurrent state cache (decode path)
leave   pretrain.py       lit-gpt's loop; you want your own trainer
leave   data.py           their FineWeb-Edu pipeline
leave   Dockerfile        their environment, not your T4's
```

**Extract the mixer. Do not adopt the harness.** Your trainer needs to compute the
five-term loss from `losses.py` against a *frozen backbone*, which is a different
program from next-token pretraining on FineWeb-Edu.

⚠️ Check `LICENSE` (NVIDIA source release) before redistributing derivatives.

## 3. T4 reality

| Concern | Assessment |
|---|---|
| **Triton on T4** | T4 is **sm_75 (Turing)**. Triton supports sm_70+, so kernels should compile — but they are tuned for Hopper, so expect poor occupancy. Budget time for this, not for the model |
| **Fallback** | Keep a pure-PyTorch recurrent path for correctness checks. The RSC runs **offline / between turns**, so decode throughput genuinely does not matter — only training throughput does |
| **Memory** | ~40M trainable params + frozen 4-bit backbone ≈ **2–6 GB** of the T4's 15 GB. Comfortable |
| **Sequence length** | Needle probes need 8K contexts. Feasible at 40M params with chunking, but watch it. Curriculum: train at 2K, evaluate at 8K, extend only if the loss curve demands it |
| **Throughput** | ~5–20K tokens/s realistically on 2×T4 for a small model ⇒ **~1.1B tokens/week** at 30 h/week. Plenty for a 40M module; nowhere near enough for from-scratch pretraining (docs/architecture/12 §7.1) |
| **Preemption** | Free Kaggle can reclaim the VM. Checkpoint on a **wall-clock** interval and copy off immediately |

## 4. What the frozen backbone contributes

The RSC is distilled **against** a backbone, not trained in isolation:

```
teacher = backbone( full_history )            -> logits
student = backbone( RSC(history), recent )    -> logits
L_reconstruction = KL(teacher || student)
```

So you need the backbone's logits, which means the backbone must be loadable in
PyTorch on the T4 in 4-bit. **LFM2 is a `transformers` architecture**, so
`AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)` is the path — *not*
llama.cpp, which cannot give you logits for a KL term.

This is a different dependency from serving: you serve with `llama-server` (GGUF)
and distil with `transformers` (4-bit safetensors). Same weights, two runtimes,
two reasons.

## 5. The order to do it in

1. **Get `scripts/rsc_ablation.py` passing here.** If rung 0 doesn't lose the
   associative probe, your operators are wrong and nothing downstream means anything.
2. **Reproduce rungs 0–2 with `fla`** on a T4. Confirm the pure-Python and kernel
   versions agree on the same synthetic probes to ~1e-3. This validates *both*.
3. **Add the channel-wise erase gate (rung 3)** as a diff on `fla`'s GDN. Re-run.
4. **Import NVlabs' GDN-2 (rung 4)**, re-run the same probes, confirm agreement.
5. **Only now** wire up `losses.py` against a frozen 4-bit LFM2 and start distilling.
6. **EDA (rung 5)** last, and only if `L_erase` has plateaued at rung 4.

Steps 1–4 are correctness work and take hours. Step 5 is the research and takes
weeks. Doing step 5 first is how you end up debugging a kernel and a loss function
and a tokenizer at the same time, with no idea which one is lying to you.

## 6. Do not skip the needle test

`scripts/needle_test.py` runs against a *served model*. After every training run,
re-run it (Law 2c: **recall is a property of a checkpoint, not an architecture** —
CoT post-training has been shown to *degrade* long-range recall in hybrids, and the
Constitutional Eval Gate will not catch it, because the gate measures task success,
not retrieval).

A checkpoint that scores higher on the eval suite and lower on needle recall is a
**regression**, not a promotion.
