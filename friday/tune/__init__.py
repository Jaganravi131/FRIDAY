"""Fine-tuning pipeline — the "learns you" loop, made executable (docs/architecture/18 §4).

Nothing here trains a model. Training a model takes GPU hours, and this core stays
stdlib-only — that is a hard architectural rule (CI checks the imports). What this
package does is everything around the training run, which is the part that decides
whether the run is worth anything:

  dataset.py   — mine the bi-temporal memory into an instruction dataset. Honour
                 the provenance hierarchy IN the export: only facts the user
                 actually stated (or edited) may become "the user said" examples;
                 inferred facts become context, never ground truth. Redaction runs
                 on every row before it leaves the machine.
  notebook.py  — emit a ready-to-run Google Colab notebook (free T4) that installs
                 unsloth, QLoRA-fine-tunes a small open base model on the exported
                 dataset, exports a Q4_K_M GGUF for llama.cpp, and downloads it.
                 The ₹0 constraint is a design input, not a bug: free GPU tiers
                 exist precisely so a 16 GB laptop doesn't have to pretend it has 24.
  models.py    — the candidate base models with the honest RAM/VRAM maths for each,
                 so choosing is a lookup, not a blog-reading session.

The sandbox this is developed in cannot reach huggingface.co (measured, egress
blocked), so the download + train steps EXECUTE where egress is open: your laptop
or Colab. The pipeline is tested against stubs here, never faked.
"""
from __future__ import annotations

__all__ = ["dataset", "models", "notebook"]
