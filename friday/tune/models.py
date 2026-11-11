"""Candidate base models for FRIDAY fine-tuning, with the honest maths (doc 18 §4.2).

The laptop is 16 GB RAM, Ryzen 5, Radeon iGPU, 512 GB SSD — not a training rig.
QLoRA runs 4-bit inference + paged optimizer states, so the VRAM/RAM floor for
training is roughly:  (params × 0.5 GB) [4-bit weights]  + (~0.06–0.12 × params)
[LoRA adapters + activations for a sane sequence length]. Inference afterwards is
~params × 0.6–0.7 GB at Q4_K_M context-included.

These figures are planning numbers, not benchmarks — they exist so the choice of
base model is a table lookup instead of three blog posts. Re-measure on real
hardware and update the table when you do; a table nobody maintains is a lie with
column headers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    repo: str                 # huggingface repo id
    params_b: float
    license: str
    why: str
    qlora_train_gb: float     # approx peak (system RAM if CPU-offload, else VRAM)
    gguf_q4km_gb: float       # approx on-disk/runtime weight size
    colab_free_ok: bool       # fits the free T4 (15 GB VRAM) comfortably?
    local_16gb_ok: bool       # plausible on the VivoBook CPU at all?


CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice(
        repo="unsloth/Qwen3-1.7B",
        params_b=1.7, license="Apache-2.0",
        why=("The local-only ceiling: QLoRA in ~3 GB with CPU offload (~8-15 min/epoch "
             "on Colab; hours on the VivoBook CPU). It will learn your register and "
             "facts, it will not reason deeply. Use it to prove the loop end-to-end "
             "before spending time on a bigger run."),
        qlora_train_gb=3.0, gguf_q4km_gb=1.1, colab_free_ok=True, local_16gb_ok=True),
    ModelChoice(
        repo="unsloth/Qwen3-4B-Instruct-2507",
        params_b=4.0, license="Apache-2.0",
        why=("The sweet spot for Colab T4: trains in ~5-6 GB with room for context, "
             "Apache-2.0, markedly better instruction-following than 1.7B. DEFAULT "
             "for the first real personal run."),
        qlora_train_gb=5.5, gguf_q4km_gb=2.5, colab_free_ok=True, local_16gb_ok=False),
    ModelChoice(
        repo="unsloth/gemma-3-4b-it",
        params_b=4.3, license="Gemma Terms",
        why=("Comparable to Qwen3-4B; pick by taste after one run each. Gemma's "
             "license is not Apache — fine for personal use, read it before sharing "
             "weights."),
        qlora_train_gb=5.8, gguf_q4km_gb=2.7, colab_free_ok=True, local_16gb_ok=False),
    ModelChoice(
        repo="unsloth/Qwen3-8B",
        params_b=8.2, license="Apache-2.0",
        why=("The Colab ceiling: ~9-10 GB QLoRA peak fits the free T4 with the unsloth "
             "offload defaults, and this is where responses start feeling like a real "
             "assistant. Train here; serve the 4.7 GB GGUF on the laptop."),
        qlora_train_gb=9.5, gguf_q4km_gb=4.7, colab_free_ok=True, local_16gb_ok=False),
)


def default_choice() -> ModelChoice:
    return CHOICES[1]          # Qwen3-4B — best quality that always fits free T4


def render_table() -> str:
    lines = ["base model | params | QLoRA train peak | GGUF Q4_K_M | free Colab T4 | VivoBook 16 GB",
             "---|---|---|---|---|---"]
    for c in CHOICES:
        lines.append(f"`{c.repo}` | {c.params_b} B | ~{c.qlora_train_gb} GB | "
                     f"~{c.gguf_q4km_gb} GB | {'✓' if c.colab_free_ok else '—'} | "
                     f"{'✓ (slow)' if c.local_16gb_ok else '✗'}")
    return "\n".join(lines)
