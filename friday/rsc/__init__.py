"""⭐ The Retention State Compiler — Innovation #9 / #9b.

A trained gated-delta module that compiles unbounded conversation history into a
FIXED-SIZE state, injected into a frozen backbone, plus a salience side-channel that
feeds S2 fact extraction. Compression and memory in one forward pass.

Docs: docs/architecture/12 §6 (why), docs/architecture/13 (the operator, the
supervision synthesis, the ablation ladder).
"""

from .ablation import (
    ProbeResult,
    format_report,
    probe_associative,
    probe_interference,
    probe_needle,
    run_ladder,
)
from .gate_supervision import (
    Batch,
    alpha_for_class,
    alpha_targets,
    build_batch,
    needles_from_facts,
    pair_batches,
    readiness_report,
    sample_negatives,
    span_batches,
)
from .losses import (
    LossReport,
    LossWeights,
    anneal_decay_weight,
    bce,
    kl_divergence,
    l_decay,
    l_erase,
    l_recall,
    l_reconstruction,
    l_salience,
    total_loss,
)
from .model import RSCConfig, RetentionStateCompiler, render_state_block
from .operators import (
    ABLATION_LADDER,
    OPERATOR_NAMES,
    EraseThenDelta,
    Gates,
    GatedDeltaNet,
    GatedDeltaNet2,
    KimiDelta,
    Operator,
    RetNet,
    State,
    build,
)

__all__ = [
    "ProbeResult", "format_report", "probe_associative", "probe_interference",
    "probe_needle", "run_ladder",
    "Batch", "alpha_for_class", "alpha_targets", "build_batch", "needles_from_facts",
    "pair_batches", "readiness_report", "sample_negatives", "span_batches",
    "LossReport", "LossWeights", "anneal_decay_weight", "bce", "kl_divergence",
    "l_decay", "l_erase", "l_recall", "l_reconstruction", "l_salience", "total_loss",
    "RSCConfig", "RetentionStateCompiler", "render_state_block",
    "ABLATION_LADDER", "OPERATOR_NAMES", "EraseThenDelta", "Gates", "GatedDeltaNet",
    "GatedDeltaNet2", "KimiDelta", "Operator", "RetNet", "State", "build",
]
