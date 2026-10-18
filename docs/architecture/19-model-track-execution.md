# 19 — The model track: current state and the execution plan

> Written 2026-10-12, when the architecture rulings in docs 12/13 were turned
> into an executable contract. The question this doc answers: *we discussed
> RetNet vs Transformer vs fine-tuning a lightweight model — what is the plan,
> and who runs what, where?*

## 0 · State of the model side today — facts, not vibes

| Piece | Where it runs | Status |
|---|---|---|
| Chat client (OpenAI-compatible) | repo, `friday/llm/client.py` | works against a llama.cpp server **today**, CI-pinned |
| Memory: ledger + RSC + retrieval | repo core | green, bench-gated (`friday bench --gate`) |
| Fine-tune pipeline | `friday/tune/` (`--export`, `--notebook`, `--models`) | shipped; `scripts/tune_verify.py` is the "make-sure" artifact (doc 18 §4.3) |
| Architecture rulings | docs 12, 13, 15, 16 | **done**: linear-attention pivot (12), GDN-2/DDSG verdict (13), ₹0 levers (15), external review (16) |
| DDSG operator contract | `friday/research/deltagate.py` + `tests/test_deltagate.py` | **shipped**: golden vectors, four modes, tie-downs |
| Actual weights / training runs | the VivoBook + free Colab T4 | **not in the sandbox** — never will be: weights can't be downloaded here, and ₹0 rules the choices |

The RetNet–Transformer discussion resolved like this: pure RetNet's *fixed*
decay fails recall (arXiv 2507.06457); production hybrids go ~3:1
local-attention:full-attention; GDN-2's *decoupled, data-dependent* erase gate
is where the retrieval gain lives (RULER S-NIAH-3 63→90). So: a Transformer
from the shelf carries intelligence *now* (Lane A/B); the hybrid linear-
attention work is our contribution *later* (Lane C), gated by numbers.

## 1 · Lane A — real intelligence on the VivoBook NOW (the user executes)

1. Download a **llama.cpp Vulkan** build (ships with the AMD driver; no ROCm,
   no CUDA needed — doc 15 verified this runs on Radeon iGPUs on Windows).
2. Pick weights, 16 GB RAM arithmetic: **Qwen3-8B Q4_K_M (~5.2 GB)** by
   default; **Qwen3-14B Q4_K_M (~8.5 GB)** if OS+browser leave headroom —
   measure, don't assume.
3. Serve: `llama-server -m qwen3-8b-q4_k_m.gguf -ngl 99 --port 8080`
4. Point FRIDAY at it: `friday ask "…" --url http://127.0.0.1:8080 --model qwen3`
   — **no repo change needed**; the client contract is CI-tested against any
   OpenAI-compatible `/v1/chat/completions`.

**Buys:** real reasoning behind the ledger, the tools, and the connectors.
**Honesty note:** doc 15's tok/s numbers were web-verified when written;
re-measure on the actual box (`friday ask --verbose` shows the ledger either
way).

## 2 · Lane B — the personal fine-tune (free Colab T4)

1. `friday tune --models` → pick the base (the table is the honest QLoRA×VRAM
   arithmetic; ≤8B fits a 15 GB T4).
2. `friday tune --export friday_dataset.jsonl` — provenance-enforced
   (`stated`/`user_edit` only; `inferred` facts are *skipped*, counted,
   reported) and redacted at the export boundary.
3. `friday tune --notebook friday_colab.ipynb --base <id>` → open in Colab
   (T4 runtime) → run cells.
4. Verify with `scripts/tune_verify.py`, merge the adapter → GGUF → it
   *replaces* the Lane-A weights.

**Buys:** the *person* — register, rhythms, unstated assumptions. Facts stay
with retrieval (doc 18 §4.1 is the full argument). **Cost: ₹0. Time: an
evening.**

## 3 · Lane C — architecture innovation: DDSG → Phase 3.5 (research)

**This week's landing:** the doc-13 ruling is now an *executable contract*:

* `friday/research/deltagate.py` — the exact operator,
  `S_t = (I − k(b⊙k)ᵀ)Diag(α)S + k(w⊙v)ᵀ`, decay-before-edit, decoupled
  erase/write: four modes (ADD / DELETE / REPLACE / KEEP).
* `tests/test_deltagate.py` — **hand-derived** golden vectors; an ordering
  discriminator (decay-before-edit is pinned, not assumed); a dk×dv
  transpose guard; and **tie-down identities to DeltaNet and Gated DeltaNet
  computed by independent in-test implementations** — so the decoupling is a
  falsifiable upgrade with a free control condition, not a rename.

The ladder from here (each rung's acceptance test is *the previous rung's
vectors and controls*):

1. **numpy port** — same golden vectors must match (Colab or local).
2. **Triton chunkwise WY-form kernel** (Colab GPU) — acceptance: golden
   vectors + tied-down controls reproduce *before any training run starts*.
3. **Hybrid block** `GDN-2 → MLP → SWA → MLP` at 1.3B–3B, a scale-taste run on
   free quota — **kill criterion pre-registered:** if the tied-scalar control
   ties the decoupled variant on S-NIAH at our scale, we fold back to KDA and
   say so in doc 13's table. (Doc 12 §5.3 named this benchmark decisive.)
4. **Phase-3.5 RSC integration** — and only then: the recurrent operator
   becomes the memory compiler's *working state*. FRIDAY's SQLite stays the
   system of record; the state is a cache over truth, never the truth. Law 1
   does not move.

## 4 · Deliberately NOT on the plan

* **Pretraining from scratch.** Chinchilla arithmetic (1B params ≈ 600 H100·h,
  ~50B tokens) against a ₹0 budget is a hard no; we fine-tune and wire
  smarter.
* **Model work inside this sandbox.** Weight egress is blocked here; every
  GPU step runs on the VivoBook or Colab by design (doc 18 §5).
* **A Triton kernel before the golden-vector gate.** Order is the contract.

## 5 · The scoreboard (how we know it's working, per lane)

* **Lane A:** answers cite retrieval sources; tok/s measured on the box;
  ledger printout stays on by default.
* **Lane B:** `tune_verify` at-or-above baseline; the known paraphrase-recall
  hole is re-measured — improves, or we write that it didn't.
* **Lane C:** an ablation table in doc 13's style — erase-on vs erase-off on
  S-NIAH, with numbers, not adjectives.
