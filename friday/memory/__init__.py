"""The Memory Compiler: S0 trace -> S1 episode -> S2 fact -> S3 skill -> S4 weight.

Innovation #1. Innovation #2 (Markdown as truth) and #3 (bi-temporal) live here too,
along with the RSC supervision taps that docs/architecture/13 §4.1 requires.
"""

from .compiler import CompileStats, compile_all, compile_file, rebuild
from .decay import DecayResult, decay, decay_row, policy_alpha
from .facts import (
    Fact,
    assert_fact,
    beliefs_at,
    current_facts,
    facts_at,
    history_of,
)
from .mdfacts import FactLine, FactsFile, parse_file, render_file, upsert_fact_line
from .supervision import (
    RetractionPair,
    SpanLabel,
    export_training_set,
    ready_for_phase_3_5,
    record_pair,
    record_span,
    span_counts,
    pair_counts,
    supervision_report,
)
from .traces import Trace, TraceWriter, behavioural_signals

__all__ = [
    "CompileStats", "compile_all", "compile_file", "rebuild",
    "DecayResult", "decay", "decay_row", "policy_alpha",
    "Fact", "assert_fact", "beliefs_at", "current_facts", "facts_at", "history_of",
    "FactLine", "FactsFile", "parse_file", "render_file", "upsert_fact_line",
    "RetractionPair", "SpanLabel", "export_training_set", "ready_for_phase_3_5",
    "record_pair", "record_span", "span_counts", "pair_counts", "supervision_report",
    "Trace", "TraceWriter", "behavioural_signals",
]
