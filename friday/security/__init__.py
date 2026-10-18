"""FRIDAY's trust boundary.

Three modules, three layers, in descending order of how much they can be trusted:

  `trust`      structural — makes below-USER text incapable of closing a slot fence
               or spoofing an authority tag. Cannot be evaded by wording.
  `policy`     authority — below-USER text cannot authorise a destructive or
               externally-visible action. (Lives in friday/agent/policy.py.)
  `injection`  heuristic — scores intent, so an attempt is VISIBLE to the user
               rather than silently neutralized.
  `redact`     at write time — secrets never reach disk, the DB, the FTS index, the
               vector store or the traces, so there is one place to get it right
               instead of five places to remember to clean up.

The single rule underneath all of it: **only the user may issue instructions.**
Everything else in the context is data, however imperative it reads.
"""

from __future__ import annotations

from .injection import InjectionReport, Verdict, detect, is_hostile, withheld_notice
from .redact import RedactionReport, has_secrets, redact, scan
from .trust import (
    PREFIX_RULE,
    PROTECTED_TAGS,
    NeutraliseReport,
    Trust,
    fence,
    neutralize,
    neutralize_row,
    trust_of,
)

__all__ = [
    "PREFIX_RULE",
    "PROTECTED_TAGS",
    "InjectionReport",
    "NeutraliseReport",
    "RedactionReport",
    "Trust",
    "Verdict",
    "detect",
    "fence",
    "has_secrets",
    "is_hostile",
    "neutralize",
    "neutralize_row",
    "redact",
    "scan",
    "trust_of",
    "withheld_notice",
]
