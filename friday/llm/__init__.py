"""The LLM client surface: one real client, one mock, and a selector that never raises.

Re-exported here so callers write `from friday.llm import get_client` instead of
reaching into `friday.llm.client`, and so the two-tier story is discoverable from the
package root rather than buried in a module docstring.

WHY THERE IS A MOCK, AND WHY IT IS NOT A TEST TOY
-------------------------------------------------
`MockClient` is what lets the whole system run end to end — retrieval, ledger, tools,
traces, telemetry, supervision — before a model has been downloaded. Phase 0's exit
test is "ask on Monday, follow up on Friday"; you should be able to prove the
*memory* half of that on day one, on a laptop with no GPU and no 2.6 GB download.
Everything except generation is real when the mock is in place, which is what makes
it a useful harness rather than a stub that hides bugs.

₹0 BUDGET, AND WHAT THAT FORCES
-------------------------------
`CLOUD_TIERS_ENABLED` is False by default (docs/architecture/12 §8): local
llama-server / Ollama only, no paid escalation. When the top of the Escalation
Ladder is missing, `escalate` must degrade to *ask the user*, never to *guess
confidently*. `NoRouteError` exists to make that explicit — a missing route raises
rather than falling through to a silent hallucination, which is the one failure mode
a personal agent cannot be allowed to have.
"""

from __future__ import annotations

from .client import (
    Client,
    MockClient,
    NoRouteError,
    OpenAICompatibleClient,
    Response,
    ToolCall,
    get_client,
)

__all__ = [
    "Client",
    "MockClient",
    "NoRouteError",
    "OpenAICompatibleClient",
    "Response",
    "ToolCall",
    "get_client",
]
