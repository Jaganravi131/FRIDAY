# Installing FRIDAY on your computer

Written for the actual target: **Windows 11, ASUS VivoBook, Ryzen 5, Radeon integrated
graphics, 16 GB RAM, 512 GB SSD, no CUDA, ₹0 cloud budget.** Every command below is
PowerShell. Total cost is zero and total download is roughly 1–2 GB.

If you only do one thing: **run `python -m friday doctor` at every step.** It tells you
what is wrong with *your* machine rather than describing mine.

---

## 1. Prerequisites

| Need | Why | Check |
|---|---|---|
| Python **3.11 or newer** | The codebase uses 3.10+ syntax throughout | `python --version` |
| ~4 GB free disk | One quantized model plus headroom | `friday doctor` |
| 16 GB RAM | Already what you have — this is the design target | `friday doctor` |
| No CUDA required | Your **Radeon iGPU still accelerates** via Vulkan — see §3 | `friday doctor` |
| No internet after setup | Everything is local by design | — |

Install Python from [python.org](https://www.python.org/downloads/windows/) and **tick
"Add python.exe to PATH"** in the installer. SQLite ships inside Python on Windows, so
there is nothing else to install — and upgrading Python later upgrades SQLite with it.

---

## 2. Get the code and create your memory

```powershell
git clone https://github.com/Jaganravi131/FRIDAY.git
cd FRIDAY
pip install pytest pytest-subtests        # the entire dependency list
python -m pytest tests/ -q                # ~455 tests, ~8 seconds, all should pass
```

Now decide where your **memory** lives. This is the important choice. The code and your
data are separate on purpose — `$FRIDAY_ROOT` is your data, the repository is code, and
conflating them means a `git clean` can eat your memory.

```powershell
# put this in your PowerShell profile so it survives every session
notepad $PROFILE
```

Add one line:

```powershell
$env:FRIDAY_ROOT = "$HOME\friday-data"
```

Then create and populate it:

```powershell
$env:FRIDAY_ROOT = "$HOME\friday-data"      # for this session too
python -m friday seed                        # writes soul/ and starter memory/facts/
python -m friday build                       # Markdown -> SQLite indices
python -m friday doctor                      # should say READY
```

`seed` writes five files you should read and then edit — they are FRIDAY's character:

```
%FRIDAY_ROOT%\soul\SOUL.md        voice, honesty rules, boundaries
%FRIDAY_ROOT%\soul\AGENTS.md      operating rules (keep it lean — it's on every turn)
%FRIDAY_ROOT%\soul\USER.md        who you are
%FRIDAY_ROOT%\soul\MEMORY.md      the core memory block
%FRIDAY_ROOT%\soul\HEARTBEAT.md   what to check when nobody is talking to it
```

These are Markdown, they are yours, and hand-edits are sacred: confidence 1.0, never
auto-overwritten without asking (Law 7). **Edit `USER.md` first** — the seeded one
describes the author of the project, not you.

---

## 3. Get a model (₹0)

Without this FRIDAY runs on its **mock client**: the memory, retrieval, provenance,
ledger and trust boundary all work, and you can prove every one of them — but it will
not converse. `friday doctor` warns about this and says which client got selected.

FRIDAY talks to anything with an **OpenAI-compatible HTTP API**. Two ways to get one:

### Option A — Ollama (easiest, five minutes)

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:1.5b           # ~1 GB, comfortable in 16 GB
$env:FRIDAY_LLM_URL   = "http://127.0.0.1:11434/v1"
$env:FRIDAY_LLM_MODEL = "qwen2.5:1.5b"
$env:FRIDAY_LLM_KEY   = "ollama"    # Ollama ignores it; the client requires a value
python -m friday doctor              # "Model server: reachable" and "ready"
```

### Option B — llama.cpp with the **Vulkan** backend (⭐ do this, not the CPU build)

> **Correction.** An earlier version of this file said the iGPU was not used and
> recommended staying at 1.2B–3B. That was CPU-only arithmetic and it was wrong.
> llama.cpp has a mature **Vulkan** backend that runs on AMD and Intel iGPUs **on
> Windows, with no ROCm and no extra drivers** — the Vulkan loader ships with every AMD
> graphics driver. Field reports put a 26B Q4 model at ~25 tok/s on a Radeon 780M that
> way, roughly a 5–6× uplift over the CPU path. `friday doctor` now detects this and
> tells you which case your machine is.

Download the **vulkan** Windows release, not the cpu one:

```powershell
# https://github.com/ggml-org/llama.cpp/releases
#   ->  llama-<ver>-bin-win-vulkan-x64.zip        ← vulkan, NOT -cpu-x64
# unzip somewhere stable, e.g. C:\tools\llama.cpp

C:\tools\llama.cpp\llama-server.exe `
  -hf Qwen/Qwen3-14B-GGUF `
  --n-gpu-layers 999 `        # offload everything to the iGPU
  -c 8192 --port 8080

$env:FRIDAY_LLM_URL   = "http://127.0.0.1:8080/v1"
$env:FRIDAY_LLM_MODEL = "Qwen/Qwen3-14B-GGUF"
```

Confirm Vulkan actually engaged — the startup log must contain a `ggml_vulkan: Found 1
Vulkan devices` line naming your Radeon. If it says `ggml_cpu` instead, you downloaded
the CPU build.

### Which model, and why the ceiling is higher than you'd guess

An iGPU has no dedicated VRAM — it shares your system RAM. On 16 GB, Windows and your
browser take ~4–5 GB, which leaves **~10–11 GB** for a model plus its KV cache. At
Q4_K_M that means:

| Model | Size at Q4_K_M | Verdict on 16 GB + Vulkan |
|---|---|---|
| **Qwen3 14B** | ~8.5 GB | The ceiling. Works, but close other tabs. Apache 2.0 |
| **Gemma 3 12B** | ~6.7 GB | ⭐ **The daily driver** — real headroom, strong all-round |
| **gpt-oss-20b** | ~11 GB (MoE, 3.6B active) | Excellent reasoning, fast because few params are active; tight fit |
| **Qwen3 8B** | ~5.0 GB | Comfortable; spend the savings on a longer context |
| Qwen3 4B / LFM2.5-1.2B | 2.6 / 0.8 GB | Only if you want an always-on background model |

**Use Q4_K_M.** Q3_K_M and below show measurable degradation, and Q8 does not fit at
these sizes. **Do not plan around Qwen3-30B-A3B** — ~18 GB at Q4, it does not fit, and
"it almost fits" is worse than "it definitely fits" because it fails under load rather
than at startup.

### Why Vulkan matters *more* to FRIDAY than to a normal chat app

Generation speed is the number people quote, but FRIDAY is **prefill-heavy**: every
single turn re-prefills the ledger block — identity, senses, user, core memory, the
recalled slots. On a CPU that prefill runs at roughly 45–50 tok/s; on an iGPU via
Vulkan it runs at **200–285 tok/s**. So the Vulkan win lands squarely on the part of
the workload FRIDAY actually spends its time in, and it compounds with the
byte-stable prefix the Attention Ledger enforces (doc 05): a stable prefix is what lets
llama.cpp's prompt cache skip that prefill entirely on follow-up turns.

That combination — a 12B model, a 5× faster prefill, and a prefix that caches — is
worth more to this project than any model swap, and it costs nothing.

What FRIDAY gives up in raw model size, it is designed to win back in context
discipline: the Attention Ledger keeps the prompt small and byte-stable (so the
provider's prompt cache actually hits), retrieval is reranked and just-in-time rather
than pre-packed, and the RSC state block replaces transcript growth. That is the whole
thesis — **a small model with excellent memory beats a large model with none.**

`FRIDAY_CTX` sets the context budget (default 8192). Lower it if the model struggles.

---

## 4. Talk to it

```powershell
python -m friday ask "what is my monthly rent"       # one question, with the ledger block
python -m friday ask "my rent is 18000"              # it records the fact
python -m friday why "what is my monthly rent"       # ⭐ the literal source, file:line
python -m friday chat                                # a REPL with hot-reload of your edits
python -m friday audit --today                       # every decision and denial
python -m friday history lease_amount_monthly        # the bi-temporal trail
```

Leave the ledger block on for the first week. It prints, per turn:

```
┌ LEDGER turn=4  budget=8192  spent=3276  cache_hit=true  prefix_stable=4
│ identity 1521/1792 · senses 28/256 · user 472/512 · core_mem 562/640
│ recalled 207/1200 · docs 0/3000 · transcript 406/2560 · scratch 0/768
│ overflow: none   reserve: 4916 ✓
└ hash=8c76da28 prefix=87cfb032
```

`cache_hit=true` and a `prefix` hash that does not change are what a well-behaved
context looks like. If `prefix` changes between turns, something per-turn leaked into
the stable prefix and you are paying to re-prefill every message.

---

## 5. Reach it from your phone

This is exit test #12, and it is what turns a laptop experiment into something you
carry. The route is **Tailscale**: free for personal use, encrypted, and your port never
appears on the internet.

```powershell
winget install Tailscale.Tailscale
tailscale up                                     # sign in, install on your phone too
python -m friday serve --host tailscale          # binds ONLY the tailnet address
```

It prints a bearer token and stores it at `%FRIDAY_ROOT%\config\serve.token` with
restricted permissions. Then:

- **Web UI:** open `http://<your-tailscale-ip>:8642/` on your phone. No build step, no
  CDN, works offline.
- **Any OpenAI-compatible client:** point its base URL at
  `http://<your-tailscale-ip>:8642/v1` and its API key at the token.
- **Health check (no token):** `curl http://<your-tailscale-ip>:8642/health`

```powershell
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8642/v1/models
```

Security notes, because this process can write to your memory:

- It binds **loopback by default**. `--host tailscale` binds the tailnet address only.
  `--host 0.0.0.0` exposes it to your local Wi-Fi and says so loudly — avoid it.
- Every request needs the token. No token, `401`, including on `/v1/*`.
- There is no CORS wildcard, so a web page on another origin cannot drive it.
- Remote turns are audited as `remote`, so `audit --today` can tell you which turns
  arrived over the network.
- The trust boundary still applies: a message from your phone is user input, and
  anything it causes to be *retrieved* is still fenced as data. See `SECURITY.md`.

---

## 6. Back it up

**Your Markdown is the only copy of your memory.** `artifacts/` is derived and can be
deleted at any time — `python -m friday rebuild` restores it — but `soul/` and `memory/`
cannot.

```powershell
cd $env:FRIDAY_ROOT
git init
# create a PRIVATE repository on GitHub, then:
git remote add origin git@github.com:<you>/friday-memory.git
git add soul memory
git commit -m "FRIDAY memory"
git push -u origin main
```

Two rules:

1. **Private.** A public memory repository is a public diary with your credentials in it.
2. **`memory/traces/` should stay gitignored.** Traces are raw ambient capture; they are
   the most sensitive files in the system and they are also the most redundant.

Secrets are now redacted at write time, so a newly recorded password becomes
`[REDACTED:CREDENTIAL]` before it reaches disk. Anything you wrote *before* that, or
pasted into a file by hand, is still there — `friday doctor` scans for it and will fail
loudly if it finds something.

---

## 7. Verifying the install

```powershell
python -m friday doctor                  # environment: READY, no ❌
python scripts/phase0_exit_test.py       # the 12-condition Phase 0 gate
python -m pytest tests/ -q               # ~455 tests
```

The exit gate runs in a throwaway directory, so it is safe to run against a real
install. Two of its twelve conditions are marked SKIP anywhere except your machine:

- **#11** — load a real model, idle an hour, compare RSS. On Windows use Task Manager or
  `psutil.Process().memory_info().rss`. Confirm `llama-server` actually *releases* the
  memory rather than keeping it mapped.
- **#12** — reach it from your phone over Tailscale (§5 above).

Run both by hand before calling Phase 0 done.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python is not recognized` | PATH not set | Re-run the installer, tick "Add to PATH" |
| `doctor` says Python 3.9 | Microsoft Store alias | `python.org` install, or `py -3.12` |
| Answers are `"lease amount monthly: ₹28,000"`-shaped and never vary | Mock client | No model server running — §3 |
| `Model server: nothing listening` | Ollama/llama.cpp not started | Start it, then `friday doctor` |
| Model loads then the machine crawls | Model too big for 16 GB | Drop to Gemma 3 12B, then Qwen3 8B |
| Log says `ggml_cpu`, not `ggml_vulkan` | Downloaded the CPU build | Get `llama-*-bin-win-vulkan-x64.zip` |
| Slow first token, fast after | Prefill on the CPU | `--n-gpu-layers 999`; check the Vulkan log line |
| `CHECK constraint failed: scope` | Database predates the `remote` scope | `python -m friday rebuild` |
| Every turn prints `⟳ memory: soul edited` | Fixed; if you see it, update | `git pull` — cold caches no longer report edits |
| `audit --today` is empty | Fixed; if you see it, update | `git pull` — allows are logged now |
| Phone cannot connect | Tailscale not up, or bound to loopback | `tailscale status`, then `--host tailscale` |
| `401` from the gateway | Missing or wrong token | `%FRIDAY_ROOT%\config\serve.token` |
| Lost everything in `artifacts/` | That is fine | `python -m friday rebuild` |
| Tests fail after `git pull` | Stale `.pyc` or an old database | `python -m friday rebuild`; delete `__pycache__` |

---

## 9. Where to go next

```
README.md                     what this is
docs/architecture/00-BLUEPRINT.md   the index; the eight Laws live here
docs/architecture/10-phase-0-implementation.md   what Phase 0 was required to prove
docs/architecture/14-competitive-analysis.md     Astra vs OpenClaw vs Dots vs FRIDAY
SECURITY.md                   the trust boundary, and what is NOT defended
CONTRIBUTING.md               how to change it without breaking it silently
```
