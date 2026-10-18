"""FRIDAY — cognition core.

Design laws this package enforces in code (see docs/architecture/00-BLUEPRINT.md):

  Law 2   Shrink the prefix, don't rewrite it      -> friday/ledger/ladder.py
  Law 2b  Never ask a lossy state to be a database -> friday/rsc/ + friday/retrieval/
  Law 2c  Recall is a checkpoint property          -> scripts/needle_test.py in every gate
  Law 3   Just-in-time retrieval beats pre-packing -> friday/ledger/compiler.py
  Law 4   Retrieval without a reranker is noise    -> friday/retrieval/pipeline.py
  Law 5   Enforce permissions in the data layer    -> friday/agent/policy.py
  Law 7   Memory files are human-editable          -> friday/memory/compiler.py

Stdlib-only. Optional extras degrade to documented fallbacks, never to ImportError.
"""

__version__ = "0.1.0"
