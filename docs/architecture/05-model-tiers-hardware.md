# 05 — Model Tiers, Hardware Reality & the Escalation Ladder

> **Your machine:** ASUS VivoBook · AMD Ryzen 5 · Radeon (integrated) · **16 GB RAM** ·
> 512 GB SSD · Windows 11
>
> This document contains **measured numbers from comparable hardware**, not hopes. Read §1 first
> — it will save you a week of disappointment.

---

## 1. Hardware truth table

### What your machine is good at
| Workload | Why | Verdict |
|---|---|---|
| **Small dense model decode (≤4B, Q4)** | Memory-bandwidth-bound; ~20 t/s measured for Qwen3-4B Q4_K_XL on comparable CPU-only rigs | ✅ **Excellent** — well above reading speed (~4 t/s) |
| **Prompt prefill via iGPU** | Radeon 780M-class iGPU measured **54 t/s prefill** (Ollama + ROCm 7.2, gfx1102 spoof) vs 14–16 t/s CPU-only | ✅ **Big win** — offload prefill, keep decode on CPU |
| **Tiny models (≤1.7B)** | Qwen3-0.6B Q8: **~60–86 t/s**; Gemma-3-1B Q8: **~42 t/s** | ✅ Runs constantly, effectively free |
| **Vision (small VLM)** | Qwen3-VL-4B: MMMU **67.4**, DocVQA **95.3**, ~3.5 GB @ Q4 | ✅ Document/screen reading is genuinely viable |
| **Speech** | Silero VAD (tiny), faster-whisper base/small (CTranslate2 CPU), Piper TTS | ✅ Full local voice pipeline fits |
| **Embeddings + reranking** | Qwen3-Embedding-0.6B GGUF + bge-reranker-base ONNX, <1 GB together | ✅ The whole retrieval stack is free |
| **Storage** | 512 GB | ✅ **Your real superpower** — keep every trace forever |

### What your machine is bad at (accept this now)
| Workload | Reality |
|---|---|
| **iGPU *decode*** | On a Radeon 780M-class iGPU: dense 7B/12B decode measured **4.5 t/s** (ROCm+FA), **4.7 t/s** (Vulkan RADV), **4.5 t/s** (native HIP gfx1103) vs **5.3 t/s CPU-only**. **The GPU does not beat the CPU for generation.** One researcher's summary: *"The Surprising Finding: CPU Beats GPU on Generation."* The iGPU wins **prefill** only. |
| **Dense 7B–8B decode** | Llama-3.1-8B Q4: **~11 t/s**; Qwen3-8B-128K Q4: **~9 t/s**; Qwen3-4B Q8: **~13 t/s**. Usable but not pleasant. |
| **Dense 12B+** | gemma-3-12b Q4: **~7 t/s**; Q6: **~6 t/s**. Too slow for conversation. |
| **Qwen3-30B-A3B MoE** | **Won't load.** 19 GB download; users with exactly 16 GB report *"I have 16GB ram and Q4 can't be loaded."* Windows alone consumes 4–5 GB. On DDR4 laptop bandwidth, expect **~3 t/s** even if it fit (one user with a 30B-A3B MoE on constrained bandwidth measured **3.1 t/s**). **Do not plan around this model.** |
| **Local fine-tuning** | No CUDA. ROCm on Windows is still experimental. CPU training is days-scale. **Not viable.** |
| **24/7 always-on capture + inference** | A VivoBook is a thin-and-light. Sustained load = thermal throttle + fan noise + battery wear. **Duty-cycle it.** |

### The bottleneck, precisely
Token generation is **memory-bandwidth-bound**. A DDR4 laptop in dual channel gives ~38 GB/s;
single-channel is half that. Qwen3-4B Q4 is ~2.5 GB, so at 20 t/s you're using ~50 GB/s of
effective bandwidth — you are already at or near the ceiling. **More compute won't help; more
bandwidth will.**

Verify yours:
```bash
# Windows: Task Manager → Performance → Memory → look at "Speed" and "Slots used"
# If it says 1 of 2 slots used → you have a free SODIMM slot. This is the best news you'll get today.
sudo dmidecode -t memory | grep -E 'Size|Speed|Locator'   # in WSL2 with admin
```

---

## 2. The two upgrades worth making

### 🥇 Upgrade #1: 32 GB RAM (~₹4,000–6,500) — *do this before anything else*

If your VivoBook has a free SODIMM slot, this single change **removes every memory constraint in
this document** and unlocks the MoE loophole:

| | 16 GB (now) | 32 GB (after) |
|---|---|---|
| Qwen3-4B Q4 | ✅ 20 t/s | ✅ 20 t/s |
| Qwen3-8B Q4 | ⚠️ 9–11 t/s, tight | ✅ comfortable |
| **Qwen3-30B-A3B MoE (Q4, ~17–19 GB)** | ❌ **won't load** | ✅ **~27 t/s CPU-only** (measured `Qwen3-30B-A3B-IQ4_XS` at 27 t/s; multiple reports of 12–15 t/s on dual-channel DDR4, 30+ t/s on DDR5) |
| gpt-oss-20b mxfp4 | ❌ | ✅ ~23 t/s |
| Concurrent model + capture + browser | ❌ swapping | ✅ |
| Local QLoRA on a 1.5B model | ⚠️ marginal | ✅ actually feasible |

**Why MoE is the loophole:** Qwen3-30B-A3B has 30B parameters but activates only **~3B per
token**. Generation speed scales with *active* parameters (bandwidth), while quality scales with
*total* parameters. You need the RAM to hold all 30B, but you only pay 3B worth of bandwidth per
token. It is the single best model-per-rupee on constrained hardware — *if* you have the RAM.

The llama.cpp flag that makes it work:
```bash
llama-server -m Qwen3-30B-A3B-IQ4_XS.gguf \
  --cpu-moe \                    # keep MoE expert weights on CPU
  -ot ".ffn_.*_exps.=CPU" \      # or: offload experts by tensor pattern
  -ngl 99 --flash-attn 1         # everything else to the iGPU (prefill!)
```
Tuning `--n-cpu-moe N` trades VRAM against prefill speed. On an 8 GB-VRAM setup, going from
`--cpu-moe` (all experts on CPU) to `--n-cpu-moe 38` moved prefill from **2.78 → 53.14 t/s** and
generation from **13.38 → 33.64 t/s**. Sweep it on your machine and keep the best.

### 🥈 Upgrade #2: free Kaggle GPU quota — *₹0, and it's the whole self-improvement loop*

| Platform | Free GPU | VRAM | Quota | Session cap | Verdict |
|---|---|---|---|---|---|
| **Kaggle** | **T4 × 2** (or P100) | 16 GB each / 32 GB combined | **30 GPU-hours/week, visible, resets Sunday UTC** | 12 h | ✅ **Use this.** Predictable quota, no credit card |
| Google Colab free | T4 | 15 GB | *No fixed quota — demand-throttled*, availability "varies over time" | ~12 h hard cap, **90-min idle timeout** | ⚠️ Backup between Kaggle windows. Can reclaim the VM without warning → checkpoint on a wall-clock interval |
| Hugging Face | T4 | 16 GB | Limited | session resets | ⚠️ |

**What fits on a free T4 (15–16 GB) with QLoRA + Unsloth:**
| Model | QLoRA + Unsloth VRAM | Fits free T4? |
|---|---|---|
| SmolLM2 135M | <2 GB | ✅ trivial |
| Qwen3 0.6B | ~2 GB | ✅ |
| Qwen2.5 / Qwen3.5 1.5–2B | ~3–4 GB | ✅ |
| **Qwen3.5 / Qwen3 4B** | **~5 GB** | ✅ **← the sweet spot for FRIDAY** |
| Phi-4-mini 3.8B | ~5 GB | ✅ |
| Gemma 4 E4B (vision+text) | ~5–6 GB | ✅ |
| Qwen2.5 7B / Llama-3.1-8B | ~6–10 GB | ✅ at short seq len, batch 1, gradient checkpointing |
| 12B | ~10–16 GB | ⚠️ tight |

**Reference result:** a Qwen2.5-1.5B + LoRA (r=16) fine-tune on 106 examples took **70 seconds**
on a free Colab T4, loss **3.37 → 0.14**, exported to a **940 MB Q4_K_M GGUF**, and ran straight
into Ollama. **Total cost: ₹0.**

Unsloth is **1.6–2× faster** and uses **~60% less VRAM** than stock HF training, and it
auto-patches the **bfloat16 → float16 tensor-core workaround** that T4s (and other pre-Ampere
cards) need to avoid gradient overflow. That fix is why T4 fine-tuning works at all.

**Workflow:**
```bash
uv tool install google-colab-cli            # or use Kaggle notebooks / kaggle CLI
colab --auth oauth2 new -s trainer
colab run --gpu T4 train.py                 # or: kaggle kernels push
colab download -s trainer /content/gguf/friday-4b-v3.Q4_K_XL.gguf ./artifacts/adapters/
```
Then locally:
```
FROM ./artifacts/adapters/friday-4b-v3.Q4_K_XL.gguf
SYSTEM """<contents of soul/SOUL.md>"""
PARAMETER num_ctx 8192
PARAMETER temperature 0.3
```
```bash
ollama create friday:v3 -f Modelfile && ollama run friday:v3
```

---

## 3. The model menu (concrete, for 16 GB)

All numbers are CPU-only Q4/Q8 measured on comparable Ryzen-class hardware unless noted.

### Resident set (loaded at boot, ~7 GB total)
| Role | Model | Quant | RAM | Speed | Notes |
|---|---|---|---|---|---|
| **L0 Intent/router/redaction** | `qwen3:0.6b` | Q8_0 | ~0.7 GB | **~60–86 t/s** | Runs on *every* turn. Effectively free. Also: tagging, PII detection, query rewrite |
| **L1 Main brain** | `qwen3:4b-instruct-2507` | **UD-Q4_K_XL** | ~2.6 GB | **~20 t/s** | ⭐ Your daily driver. Q8 is ~13 t/s — take the Q4, the speed matters more |
| **Vision** | `qwen3-vl:4b` | Q4 | ~3.5 GB | ~10–15 t/s | DocVQA **95.3**, MMMU **67.4**, 256K ctx. Load on demand, unload after 2 min idle |
| **Embeddings** | `qwen3-embedding:0.6b` (GGUF) or `bge-m3` | — | ~0.6 GB | fast | 1024-d. Prefer `bge-m3` if Tamil quality matters |
| **Reranker** | `bge-reranker-base` (ONNX) | — | ~0.4 GB | ~50–200 ms/query, CPU | The single highest-value 400 MB you'll spend |
| **VAD** | `silero-vad v5` | — | ~50 MB | 30 ms frames | **Pre-warm at process start** — cold init costs ~1 s on the first turn |
| **STT** | `faster-whisper small` (CTranslate2, int8) | — | ~1 GB | near-real-time chunks | `base` if too slow; Deepgram Nova-3 (cloud) is materially better (110–160 ms) |
| **TTS (local)** | `piper` | — | ~200 MB | instant | Functional, not natural |
| **Wake word** | `microWakeWord` / `openWakeWord` | — | ~50 MB | always-on | Train a custom "Friday" wake word; ESP32-portable later |

**Total resident: ~9–10 GB.** Leaves ~6 GB for Windows + WSL2 + browser. **Tight but workable.**
Mitigations below.

### On-demand (loaded only when the ladder escalates)
| Tier | Model | Quant | RAM | Speed | When |
|---|---|---|---|---|---|
| **L1+** | `qwen3:8b` | UD-Q4_K_XL | ~5 GB | **~9–11 t/s** | Hard local reasoning; unload L1 first |
| **L1-alt** | `phi4-mini` (3.8B) | Q6_K | ~3 GB | ~18 t/s | Reasoning/math/code per GB — better than 4B on hard tasks |
| **L1-alt** | `gemma-3n:E4B` | UD-Q4_K_XL | ~3 GB | ~13 t/s | Google's edge model, native function calling, memory-efficient |
| **L1-fast** | `Huihui-Ling-mini-2.0-abliterated MXFP4_MOE` | — | ~2 GB | **~58 t/s** | Tiny MoE — shockingly fast. Good for a snappy conversational tier |
| **L1-fast** | `LFM2-8B-A1B` | UD-Q4_K_XL | ~5 GB | **~38 t/s** | 8B with 1B active — the best speed/quality on your hardware |
| **Draft model** | `qwen3:0.6b` Q8 | — | 0.7 GB | — | **Speculative decoding** for L1 |

> ⚠️ **Note `LFM2-8B-A1B` at ~38 t/s and `Huihui-Ling-mini-2.0` at ~58 t/s.** These MoE-style
> small-active-parameter models are the most under-used trick for 16 GB machines. Benchmark them
> against Qwen3-4B on *your* tasks — if quality is within a few points, take the 2× speed.

### Cloud (only via the ladder, never by default)
| Tier | Model | Why |
|---|---|---|
| **L2** | Claude / GPT / Gemini mid-tier via **OpenRouter** | One key, 300+ models, per-request routing, hard spend caps |
| **L2-voice** | `gpt-realtime-2.1` **or** Gemini Live (`gemini-3.1-flash-live-preview`) | Full-duplex speech-to-speech, **300–550 ms end-to-end**. Gemini Live has **affective dialog** (adapts tone to your expression) and **native bidirectional transcription** — no sidecar ASR |
| **L3** | Frontier reasoning model | Deep research, multi-step planning |
| **L4-teacher** | Frontier model, **search-time only** | GEPA reflection + SEAL-style training-data synthesis. OpenJarvis: amortises to **<$0.001/query within 6 months** at 100 queries/day |

**Speculative decoding — the free latency win:**
```bash
llama-cli -hf Qwen/Qwen3-4B-Instruct-2507-GGUF:Q4_K_XL \
          -hfrd Qwen/Qwen3-0.6B-GGUF:Q8_0 \
          --draft-max 16 -ngl 99 -fa 1
```
The 0.6B drafts, the 4B verifies. On bandwidth-constrained hardware this is a real speedup
because verification is parallel while generation is serial. **Measure it** — acceptance rate
varies by task, and a bad draft model makes it slower.

---

## 4. Innovation #5 — The Escalation Ladder

**Principle: the cheapest tier that can do the job wins. Confidence-gated, measured, reversible.**

Justification from the research: **local models already handle 88.7% of single-turn queries**
(Stanford, *Intelligence Per Watt*). OpenJarvis lands within **3.2 pp** of the best cloud model at
**~800× lower marginal cost** and **~4× lower latency** — with the residual gap concentrated in
*reasoning- and research-heavy* tasks. That is exactly the ~11% you should escalate.

```
                        user turn
                            │
              ┌─────────────▼──────────────┐
              │  L0  GATE  (qwen3:0.6b)    │  ~15–40 ms, ~0 cost
              │  intent · complexity ·     │  runs on EVERY turn
              │  PII · language · routing  │
              └─────────────┬──────────────┘
                            │
        ┌───────────────────┼───────────────────────┐
        │                   │                       │
   trivial/            needs memory/            needs the
   chit-chat           tools/context            world
        │                   │                       │
┌───────▼───────┐  ┌────────▼────────┐   ┌──────────▼──────────┐
│ L0.5 CANNED   │  │ L1  LOCAL       │   │ L1.5 LOCAL+SEARCH   │
│ templates,    │  │ qwen3:4b        │   │ L1 + web/memory     │
│ time, unit    │  │ ~20 t/s, ₹0     │   │ tools, still local  │
│ conversion    │  │ ← 70–85% of     │   │                     │
│ ~0 ms, ₹0     │  │    turns        │   │                     │
└───────────────┘  └────────┬────────┘   └──────────┬──────────┘
                            │                       │
                            └───────────┬───────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  CONFIDENCE GATE           │
                          │  logprob entropy · "I don't│
                          │  know" detection · tool    │
                          │  failure count · task      │
                          │  complexity class · user   │
                          │  said "are you sure?"      │
                          └─────────────┬──────────────┘
                                        │ low confidence / hard class
                          ┌─────────────▼──────────────┐
                          │ L2  CLOUD  (OpenRouter)    │  ~1–3 s, ~$0.002
                          │ mid-tier frontier model    │
                          └─────────────┬──────────────┘
                                        │ still hard / multi-step / research
                          ┌─────────────▼──────────────┐
                          │ L3  DEEP  (reasoning tier) │  ~10–60 s, ~$0.05
                          │ + sub-agent isolation      │  → run ASYNC, notify
                          │   (own clean context)      │    on completion
                          └────────────────────────────┘

    ─────────────── OFFLINE / ASYNC (never on the turn path) ───────────────
    L4 TEACHER (frontier, search-time only)
       · GEPA reflective prompt evolution
       · SEAL-style training-data synthesis from your traces
       · failure-cluster diagnosis → proposed edits across all 5 primitives
       · gated at 1% regression tolerance (OpenJarvis's "gate")

    L5 TRAINEE (Kaggle free T4, weekly)
       · QLoRA + Unsloth on Qwen3-4B → GGUF → gate → atomic swap into L1
```

### The router (L0) — train this yourself

This is your **best first fine-tuning target**: a tiny classifier trained on your own escalation
traces. Cheap, measurable, immediately useful, and it *is* a self-improving component.

```python
# Features the L0 gate emits (structured output, ~15 ms on qwen3:0.6b)
{
  "intent": "recall|action|chitchat|create|analyse|control|question",
  "complexity": 1-5,                 # 1 = canned, 5 = deep research
  "needs_memory": true,
  "needs_tools": ["calendar.read"],
  "needs_vision": false,
  "language": "ta|en|ta-en-mixed",
  "pii_present": false,
  "time_sensitive": false,
  "suggested_tier": "L1",
  "confidence": 0.87
}
```

Start with the LLM producing this JSON. After 500 turns you have a labelled dataset (because the
confidence gate records whether escalation was *needed*). Train a 135M–600M classifier on it.
Now routing costs **<5 ms and zero tokens.** That is a genuine, measurable, self-improving
component you built.

### Confidence gate — the signals that actually work

| Signal | How | Cost |
|---|---|---|
| **Token logprob entropy** | llama.cpp exposes logprobs; high mean entropy over the answer span = uncertainty | free |
| **Self-declared uncertainty** | regex/cheap-model detect "I'm not sure", "I think maybe", "I don't have" | free |
| **Tool failure count** | ≥2 failed tool calls in a turn → escalate | free |
| **Retrieval emptiness** | retrieval returned `[]` for a turn that `needs_memory=true` → escalate or ask | free |
| **Complexity class** | from L0's `complexity` ≥4 → skip straight to L2/L3 | free |
| **User correction history** | this task type was corrected before at L1 → escalate proactively | **this is learned** |
| **Semantic self-consistency** | sample 2 answers at temp 0.7, compare | 2× cost — use sparingly, offline only |

> ⚠️ **Do not use "ask the model if it's confident" as your primary signal.** Models are badly
> calibrated on verbalised confidence. Use logprob entropy and behavioural signals; use
> verbalised confidence as one weak feature among many.

### The economics

Assume 150 turns/day:
| Tier | Share | Turns | Cost/turn | Daily |
|---|---|---|---|---|
| L0/L0.5 | 15% | 22 | ₹0 | ₹0 |
| L1 local | 70% | 105 | ₹0 | ₹0 |
| L1.5 | 8% | 12 | ₹0 | ₹0 |
| L2 cloud | 6% | 9 | ~₹0.18 | ~₹1.6 |
| L3 deep | 1% | 1.5 | ~₹4.5 | ~₹7 |
| L4 teacher | nightly | 1 batch | — | ~₹2 |
| | | | **Total** | **~₹11/day ≈ ₹330/month** |

Compare: 100% cloud frontier ≈ ₹3,000–8,000/month. **The ladder is a ~10–20× cost reduction**,
and it's *also* faster (L1 local has no network round-trip) and *more private* (93% of turns
never leave the machine).

**Escalation is a feature, not a failure.** Say so in `SOUL.md`: FRIDAY should tell you
*"that one's beyond my local model, giving it to the big brain — 3 seconds"* rather than
silently producing a mediocre answer.

---

## 5. Windows 11 + WSL2 setup (the part that bites)

### Step 1 — WSL2 with mirrored networking
`.wslconfig` in `%USERPROFILE%`:
```ini
[wsl2]
# CRITICAL: WSL2 defaults to 50% of host RAM. On a 16 GB machine that's 8 GB
# for Linux + 8 GB for Windows, and Windows needs ~5. Cap WSL hard.
memory=6GB
processors=6
swap=8GB
swapFile=D:\\wsl-swap.vhdx      # put swap on the SSD, not the system drive if you can

networkingMode=mirrored          # Win11 22H2+ — localhost works BOTH ways.
                                 # This is what lets the WSL2 agent reach the
                                 # Windows-native llama-server at 127.0.0.1:8080
                                 # and your phone reach the WSL2 gateway.
dnsTunneling=true
autoProxy=true

[experimental]
autoMemoryReclaim=gradual        # gives memory back to Windows when idle
sparseVhd=true
```
Then `wsl --shutdown` and restart.

> **`networkingMode=mirrored` is the single most important line.** Without it you'll spend a day
> on `localhost` not resolving between Windows and WSL2.

### Step 2 — the split: inference on Windows, agent in WSL2

```
┌──────────────── WINDOWS 11 (native) ────────────────┐
│  llama-server.exe  (Vulkan build)                   │
│    → iGPU for prefill, CPU for decode               │
│    → binds 127.0.0.1:8080, OpenAI-compatible        │
│  Ollama.exe (optional, for model management)        │
│  screenpipe.exe (ambient capture, needs Windows     │
│    APIs for the accessibility tree)                 │
└──────────────────────┬──────────────────────────────┘
                       │ 127.0.0.1 (mirrored networking)
┌──────────────────────▼──────────────────────────────┐
│  WSL2 Ubuntu 24.04                                  │
│    FRIDAY core (Python 3.12, uv)                    │
│    SQLite + FTS5 + sqlite-vec                       │
│    Pipecat voice worker                             │
│    FastAPI + WSS gateway                            │
│    APScheduler (heartbeat / dreaming)               │
└─────────────────────────────────────────────────────┘
```

**Why this split:** llama.cpp Vulkan/ROCm runs **better natively on Windows** for the iGPU, and
screenpipe needs the **Windows accessibility APIs**. But the entire Python AI ecosystem
(DSPy, Unsloth, PEFT, sentence-transformers, faster-whisper, Pipecat) is **Linux-first**.
Mirrored networking makes the boundary invisible.

Alternative if the split annoys you: run llama.cpp **inside** WSL2 with GPU passthrough
(`/dev/dxg` for DirectML, or Mesa RADV for Vulkan). It works, but expect some prefill
performance loss vs native. **Benchmark both** — 20 minutes of measurement beats a day of
assumption.

### Step 3 — benchmark YOUR machine before writing a line of agent code

```bash
# 1. Get llama.cpp (Windows native, Vulkan build) or build it
#    Prebuilt Vulkan releases: github.com/ggml-org/llama.cpp/releases

# 2. Baseline: CPU-only vs iGPU
llama-bench -m Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf -ngl 0  -t 8 -fa 1 -p 512 -n 128   # CPU only
llama-bench -m Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf -ngl 99 -fa 1 -p 512 -n 128        # iGPU

# 3. Find your best MoE offload split (if you get 32 GB)
for n in 48 46 44 42 40 38; do
  llama-bench -m Qwen3-30B-A3B-IQ4_XS.gguf -ngl 99 --n-cpu-moe $n -fa 1 -p 512 -n 128
done

# 4. Test speculative decoding
llama-cli -m Qwen3-4B-...Q4_K_XL.gguf -hfrd Qwen3-0.6B-Q8_0.gguf --draft-max 16 -ngl 99 -fa 1 -p 0 -n 256

# 5. Derive your actual memory bandwidth
#    model_size_GB × tg_tokens_per_sec = effective bandwidth GB/s
#    compare against: memory_MT_per_sec × bus_bits / 8 = theoretical max
```

Record the results in `docs/architecture/BENCHMARKS.md` **with your exact CPU/RAM config**.
Every model-choice decision downstream should reference *your* numbers, not the ones in this doc.

### Step 4 — memory discipline (16 GB is unforgiving)

```python
# Unload idle models aggressively. On 16 GB this is not optional.
OLLAMA_KEEP_ALIVE = "2m"        # default is 5m; drop it
# or per-request:  {"keep_alive": "2m"}
# llama-server: restart the slot / use --no-warmup and a supervisor

# Cap context. 32K context on a 4B model costs real RAM for the KV cache.
# Start at 8K. Raise only when the Ledger says you're starving.
PARAMETER num_ctx 8192

# KV-cache quantisation — big RAM saving, small quality cost
--cache-type-k q8_0 --cache-type-v q8_0     # halves KV cache vs f16
```

**Rule of thumb:** `model_GB + (num_ctx × layers × 2 × hidden × 2 bytes / 1e9) + 5 GB for
Windows + 2 GB for WSL2 + 2 GB for browser < 16 GB`. Solve for `num_ctx`. On a 4B model you'll
land around **8K–12K**. That's fine — the Attention Ledger is designed for exactly this
constraint, and Law B/D (offload + JIT retrieval) means you don't need a big window.

> **This is a feature, not a limitation.** A 32K working budget forces the discipline that makes
> FRIDAY's answers clear. People with 1M-token windows stuff them and get worse results —
> accuracy loss begins around **50K tokens of genuinely relevant information**.

---

## 6. Model-swap discipline (OpenJarvis's "spec")

Adopt the five-primitive TOML spec so you can swap models **without touching a prompt**:

```toml
# friday.spec.toml
[intelligence]
model      = "qwen3:4b-instruct-2507"
quant      = "UD-Q4_K_XL"
path       = "artifacts/models/Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"
adapter    = "artifacts/adapters/friday-4b-v3.Q4_K_XL.gguf"   # null = base
temperature = 0.3
num_ctx    = 8192
top_p      = 0.9

[engine]
backend    = "llama.cpp"          # llama.cpp | ollama | openrouter | openai-compat
endpoint   = "http://127.0.0.1:8080/v1"
ngl        = 99
n_cpu_moe  = null                 # set for MoE models
flash_attn = true
cache_k    = "q8_0"
cache_v    = "q8_0"
draft_model = "artifacts/models/Qwen3-0.6B-Q8_0.gguf"   # speculative decoding
draft_max  = 16

[agents]
loop          = "react"           # react | codeact
max_turns     = 12
tool_policy   = "progressive"     # progressive | all | none
scopes        = ["interactive", "heartbeat", "dreaming", "eval"]

[tools_memory]
memory_backend = "markdown_sqlite"
mcp_spec       = "2026-07-28"
vector_dim     = 1024
retrieval      = { wide_k = 20, ship_k = 3, floor = 0.35, reranker = "bge-reranker-base" }

[learning]
optimizer   = "gepa"              # gepa | dspy-miprov2 | lora | none
teacher     = "openrouter:anthropic/claude-..."   # search-time only
train_target = "qwen3:4b"
gpu         = "kaggle-t4"
gate        = { target_min_gain = 0.05, heldout_max_regression = 0.01, latency_max_worse = 0.10 }
```

**Two specs can differ only in `[intelligence]` and `[engine]`** and run identical behaviour on
your VivoBook and (later) a 32 GB mini-PC or a GPU box — **without rewriting a prompt.** That
portability is the whole point, and it's what lets your Kaggle-fine-tuned adapter drop into
production with a one-line change and an atomic rollback.

---

## 7. If you buy hardware later (priority order)

| Purchase | ~Cost | Unlock |
|---|---|---|
| **32 GB SODIMM** | ₹4–6k | Qwen3-30B-A3B @ ~27 t/s. **Best value by a distance.** |
| **Used RTX 3090 24 GB + a cheap host** | ₹55–70k | Real local 14B–27B, local QLoRA, no Kaggle quota. Turns FRIDAY into a workstation agent |
| **Mac Mini M4 24 GB** | ~₹75k | Unified memory = whole 24 GB usable for models; <1 s whisper, <1 s 8B generation. The best appliance-style always-on host |
| **₹800/month VPS (8 vCPU/32 GB)** | ₹800/mo | Always-on heartbeat + presence fabric even when the laptop is shut. **The most underrated purchase** — proactivity requires an always-on host |
| **Raspberry Pi 5 / ESP32-S3** | ₹6–10k | Dedicated wake-word + mic node in each room. Whisper is too slow on a Pi (~3–4 s) but **speech-to-phrase** and wake-word work well |
| **A second phone / old Android** | free | Always-on voice node, camera, GPS sense |

**The always-on host matters more than the GPU.** A proactive assistant that only exists when
your laptop is open is not proactive. If you buy one thing after the RAM, buy the ₹800 VPS and
run the gateway + heartbeat there, with the VivoBook as a *client* and heavy-inference node.
