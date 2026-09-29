"""Embedders, best-available-first.

The interface is one method: `embed(text) -> list[float]`. Three implementations,
chosen at runtime by what is installed:

  1. `SentenceTransformerEmbedder` — bge-m3 or Qwen3-Embedding-0.6B. Best, and
     the one to use for real. bge-m3 is worth it here specifically because you
     code-switch Tamil and English (docs/architecture/03 §5).
  2. `ServerEmbedder` — asks a running `llama-server` for embeddings, so you get a
     real model with no Python ML dependencies at all.
  3. `HashingEmbedder` — deterministic hashed bag-of-ngrams, stdlib only.

⚠️ Be honest about what #3 is: **it is a lexical index wearing a vector's clothes.**
It captures term overlap and nothing semantic — "rent" and "lease" will not be
close. It exists so that the *system* runs, is testable, and degrades gracefully
on a machine with no ML stack. It is not a substitute for a real embedder, and the
retrieval floor (`RETRIEVAL_SCORE_FLOOR`) matters more than usual when it's active,
because FTS5 is doing nearly all the work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import Protocol


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


# ── 1. real model, in-process ──────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """bge-m3 (Tamil+English) or Qwen3-Embedding-0.6B. Requires `friday[embed]`."""

    def __init__(self, model_name: str = "BAAI/bge-m3", dim: int = 1024, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # lazy: optional dep

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(dim)
        self.name = f"st:{model_name}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return [[float(x) for x in v] for v in vecs]


# ── 2. real model, over HTTP (no Python ML deps) ───────────────────────────────

class ServerEmbedder:
    """Use an embedding GGUF served by llama-server:

        llama-server -m bge-m3-Q8_0.gguf --embeddings -c 2048 --port 8081

    Keeps the Python side stdlib-only, which matters on a 16 GB laptop where you
    do not want two runtimes resident.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8081", dim: int = 1024, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.timeout = timeout
        self.name = f"server:{self.base_url}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": texts}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        out = [item["embedding"] for item in sorted(data["data"], key=lambda d: d.get("index", 0))]
        return [[float(x) for x in v] for v in out]

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False


# ── 3. stdlib fallback ─────────────────────────────────────────────────────────

_NGRAM = re.compile(r"[a-z0-9\u0980-\u09ff\u0b80-\u0bff]+", re.UNICODE)  # latin, bengali, tamil


class HashingEmbedder:
    """Deterministic hashed bag of character+word n-grams, L2-normalised.

    Properties that make it useful as a fallback:
      * same text -> same vector, forever, on any machine (so indices are stable
        and rebuildable, which the artifacts-are-derived law requires)
      * no dependencies, no model download, ~microseconds per call
      * handles Tamil script, because it hashes characters rather than relying on
        an English-only vocabulary

    Properties that make it NOT a real embedder: zero semantic generalisation.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = "hashing"

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = _NGRAM.findall(text.lower())
        for t in toks:
            self._bump(v, f"w:{t}", 1.0)
        for a, b in zip(toks, toks[1:]):
            self._bump(v, f"b:{a}_{b}", 0.7)          # word bigrams
        joined = "".join(toks)
        for i in range(len(joined) - 2):
            self._bump(v, f"c:{joined[i:i+3]}", 0.35)  # char trigrams: script-agnostic
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _bump(self, v: list[float], key: str, weight: float) -> None:
        h = hashlib.blake2b(key.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0     # signed hashing cuts collision bias
        v[idx] += sign * weight


# ── selection ──────────────────────────────────────────────────────────────────

def get_embedder(dim: int = 1024, prefer: str = "auto") -> Embedder:
    """Best available. Never raises: falls through to HashingEmbedder."""
    if prefer in ("auto", "server"):
        try:
            se = ServerEmbedder(dim=dim)
            if se.alive():
                return se
        except Exception:
            pass
        if prefer == "server":
            return HashingEmbedder(dim)
    if prefer in ("auto", "st"):
        try:
            return SentenceTransformerEmbedder(dim=dim)
        except Exception:
            pass
    return HashingEmbedder(dim)
