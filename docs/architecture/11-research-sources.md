# 11 — Research Sources

Everything in this blueprint was checked against current (2026) sources. Grouped by topic.
Items marked ⭐ are the ones that most changed FRIDAY's design.

---

## Reference projects (study these four first)

| Project | What to take from it |
|---|---|
| ⭐ **[Hermes Agent — NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent)** | The closed learning loop. Three-tier memory with **hard character budgets** (USER.md 1,375 / MEMORY.md 2,200). Autonomous skill creation + in-use revision. FTS5 cross-session recall. Honcho dialectic user modelling. **No external vector DB required.** MIT. |
| ⭐ **[roiguri/jarvis-agent](https://github.com/roiguri/jarvis-agent)** | The safety counterweight. *Sandboxed* (blast radius = tool set) + *deterministic* (hand-rolled StateGraph, per-scope whitelists). **Progressive tool disclosure.** **Two scopes sharing one memory** with bidirectional awareness. File-based sandboxed memory with hot-reload. Confirmation pattern for destructive actions. |
| ⭐ **[screenpipe/screenpipe](https://github.com/screenpipe/screenpipe)** | Don't build ambient capture. Event-driven capture + **accessibility tree** + OCR fallback. Local SQLite/FTS5. MCP server. Pipes. **Three-layer deterministic AI data permissions** — the model for FRIDAY's Sense Registry. |
| **[OpenJarvis — Stanford Hazy Research / Scaling Intelligence Lab](https://github.com/open-jarvis/OpenJarvis)** · [Ollama post](https://ollama.com/blog/openjarvis) · [MarkTechPost analysis](https://www.marktechpost.com/2026/06/03/meet-openjarvis-a-local-first-framework-for-on-device-personal-ai-agents-with-tools-memory-and-learning/) | **88.7% of single-turn queries handled locally.** Within 3.2 pp of best cloud at ~800× lower cost, ~4× lower latency. **Five primitives + one TOML spec.** **LLM-guided spec search with the 1% "gate"** → 13–32 pp recovered. **The swap test**: same model loses 25–39 pp in other frameworks vs 5.6–16.5 pp under a spec. |
| [Open-Jarvis (fedcal)](https://fedcal.github.io/open-jarvis/en/) | Multi-device framing: laptops, phones, watches, AR glasses, VR, holographic, medical wearables — one persistent identity. |
| [Jarvis (jarvis.pm)](https://jarvis.pm/) | Self-hosted FastAPI + Postgres/pgvector, SOUL.md personality, 15+ tools, 300+ models. A clean minimal reference. |
| [ramsbaby/jarvis](https://dev.to/ramsbaby/i-built-jarvis-a-self-hosted-ai-butler-on-discord-powered-by-claude-3eme) | Discord-as-UI, MCP tools, LaunchAgents scheduling, RAG memory, 62 E2E tests. Good pragmatic patterns. |
| [PersonalJarvis](https://github.com/slavakurilyak/awesome-ai-agents/issues/637) | Desktop app + wake word + local STT + background agents + computer use + an Agentic IDE running CLI coding agents side by side. Feature-checklist reference. |
| [What is OpenClaw? (DigitalOcean)](https://www.digitalocean.com/resources/articles/what-is-openclaw) · [How OpenClaw Works](https://bibek-poudel.medium.com/how-openclaw-works-understanding-ai-agents-through-a-real-architecture-5d59cc7a4764) · [2026 Guide](https://alphatechfinance.com/productivity-app/openclaw-ai-agent-2026-guide/) | ⭐ **The Heartbeat** (cron-triggered agentic loop, `HEARTBEAT_OK` suppression). The file-based memory layout (SOUL/AGENTS/MEMORY/HEARTBEAT.md + daily logs **not auto-injected**). 10 concrete security controls. |

---

## Memory frameworks

| Source | Key data |
|---|---|
| ⭐ [Mem0 vs Zep vs Letta vs Cognee tested (particula.tech)](https://particula.tech/blog/agent-memory-frameworks-tested-mem0-zep-letta-cognee-2026) | **Zep 63.8% vs Mem0 49.0% on LongMemEval (GPT-4o)** — a 15-pt gap entirely from temporal validity windows. Decision matrix by scenario. |
| [Best AI Agent Memory Frameworks 2026 (atlan)](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/) | Comparison table, self-host and pricing posture. |
| [Agent Memory in Production 2026 (agentmarketcap)](https://agentmarketcap.ai/blog/2026/04/11/agent-memory-architecture-production-2026) | ⭐ **Retrieval latency profiles**: vector 10–50 ms · graph 50–150 ms · multi-strategy 100–600 ms · **LLM synthesis 800–3,000 ms** · ingestion 500–2,000 ms. Memory scoring & decay. |
| [Best AI Agent Memory Frameworks (baeseokjae)](https://baeseokjae.github.io/posts/best-ai-agent-memory-frameworks-2026/) | Architecture taxonomy: vector / temporal-KG / tiered / hybrid. Latency by strategy. |
| [Mem0 vs Zep vs Letta vs LangMem (datapace)](https://datapace.ai/blog/ai-agent-memory-tools-2026) | ⭐ **Letta's sleep-time compute** — a background "sleeper" agent reorganises memory while idle, so consolidation cost never lands in user-facing latency. Zep invalidates with a timestamp and keeps the edge. |
| [Letta forum: Letta vs Mem0 vs Zep vs Cognee](https://forum.letta.com/t/agent-memory-letta-vs-mem0-vs-zep-vs-cognee/88) | Primary-source descriptions of each architecture. |
| [Mem0/Zep GPU deployment guide (Spheron)](https://www.spheron.network/blog/agent-memory-gpu-cloud-mem0-zep-guide/) | Three memory types: **semantic / procedural / episodic**. What GPU components each needs. |
| [Top 6 Agent Memory Frameworks (dev.to)](https://dev.to/thedailyagent/top-6-ai-agent-memory-frameworks-for-devs-2026-1fef) | Feature matrix incl. LlamaIndex. |
| ⭐ [r/AI_Agents — "Jarvis: Your Personal AI Companion"](https://www.reddit.com/r/AI_Agents/comments/1spmauz/jarvis_your_personal_ai_companion/) | The 3-layer DNI idea (physical MEMORIES.md / neural sqlite-vec shards / logical importance×recency ranking). Sliding-window compression. **Nightly "dreaming" reflection.** And the crucial rebuttal: *"the corpus that actually moves the needle is the browser profile, autofill tables, history, bookmarks — that's where your real preferences live."* |

---

## Context engineering

| Source | Key data |
|---|---|
| ⭐ [Anthropic — Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | **Just-in-time retrieval** with lightweight identifiers. Hybrid strategy (CLAUDE.md up front + glob/grep JIT). Compaction tuning: **maximise recall first, then precision**. |
| ⭐ [Context Engineering in 2026: Why We Stopped Compacting (Louis Bouchard)](https://www.louisbouchard.ai/context-engineering-2026/) | **Summarisation dropped recall 92% → ~33% and cost 2× more.** Capping tool output: **−38% cost/turn, won 14/15 trajectories, recall identical.** Offload-to-files-with-pointers + one index file → *context scales with task complexity, not session length.* **"Shrink the prefix, don't rewrite it."** |
| [Advanced Context Engineering (AgentSwarms)](https://agentswarms.fyi/blog/advanced-context-engineering-best-practices) | ⭐ The four-part JIT pipeline (intent classifier → query rewriter → retrieve/rerank → freshness gate) **and why the tool docstring contract is the most important detail.** Hard score floor. LangGraph typed-state assembly. |
| [Context Engineering for Coding Agents 2026](https://www.heyuan110.com/posts/ai/2026-06-16-context-engineering-2026/) | The token-budget diagram (fixed / working set / **reserved headroom**). *"An agent at 99% full has no room to think."* The failure→fix decision tree. Three additive pseudo-techniques to resist. |
| [Context Engineering: 2026 Playbook (cruxdigits)](https://cruxdigits.nl/blog/context-engineering-ai-agents-2026/) | Write/select/compress/isolate mapped across Anthropic, LangChain and GraphRAG framings. **Accuracy loss begins ~50K tokens of genuinely relevant info** even in 200K–1M windows. Retrieved knowledge should be 2–4K tokens. |
| [Context Engineering practical guide (happycapy)](https://happycapy.ai/blog/context-engineering-ai-agents) | The four operations, framework alignment table, what to measure. |
| [Agent Context Compaction for Long-Running Sessions (Zylos)](https://zylos.ai/research/2026-04-21-agent-context-compaction-long-running-sessions/) | ⭐ **Compact at ~70–75%, not 95–98%** (the "context anxiety" failure Devin documented). **Offload durable facts ON WRITE, not at compaction time.** Prompt caching and compaction are in fundamental tension. **Artifact tracking scores 2.19–2.45/5.0 across all production methods.** Full technique comparison table. |
| [Claude-Mem context engineering docs](https://docs.claude-mem.ai/context-engineering) | JIT vs pre-inference vs hybrid, with a decision framework by scenario. |
| [Agent Skills for Context Engineering (muratcankoylan)](https://github.com/muratcankoylan/agent-skills-for-context-engineering) | A skills library covering context-degradation, compression, **latent-briefing (KV-cache compaction for sub-agents)**, memory-systems, tool-design, harness-engineering, **self-improvement-loops with acceptance gates**. |

---

## Realtime voice

| Source | Key data |
|---|---|
| ⭐ [Voice AI agents in production 2026 (reactify)](https://www.reactify-solutions.com/articles/voice-ai-agents-production-2026) | LiveKit Agents 1.0 (Apr 2025), 1.5.x adaptive turn detection + native MCP. **OpenAI Realtime GA 28 Aug 2025** with gpt-realtime, SIP, remote MCP, image inputs. Working code for both paths. |
| ⭐ [Production Realtime Voice Agents: Relay, Barge-In, and the Sample-Rate Trap (Zylos)](https://zylos.ai/research/2026-07-21-realtime-voice-agent-relay-architecture/) | **Barge-in needs exact milliseconds** — cancelling leaves the full generated response in the model's history. `conversation.item.truncate` with `audio_end_ms` = *actual played duration*. Gemini's server-declarative `interrupted` flag instead. **Client-side VAD in an AudioWorklet** for zero-latency local mute. **The Realtime API lacks client-side auth → you need a relay or ephemeral tokens.** |
| ⭐ [Real-Time Voice AI Stacks 2026 (teachaitools)](https://teachaitools.blog/blog/real-time-voice-ai-agent-stacks-livekit-ultravox-cartesia-2026) | The measured latency table: **human turn-gap 200–250 ms** (Stivers et al., PNAS 2009) · LiveKit edge 15–25 ms · Nova-3 110–160 ms · Groq TTFT 65–95 ms · **Cartesia Sonic-2 / ElevenLabs Flash v2.5 TTFA 85–135 ms** · Ultravox 150–220 ms · full-duplex S2S 300–550 ms. `prewarm()` VAD (avoids ~1 s cold start). `preemptive_generation=True`. `min/max_endpointing_delay`. **Barge-in debounce 250–350 ms.** LiveKit's <40 ms mute+cancel+clear. |
| [OpenAI Realtime API: Production Voice Agents (Forasoft)](https://www.forasoft.com/blog/article/openai-realtime-api-voice-agent-production-guide-2026) | ⭐ Reference architecture: client → SFU → agent runtime → model API. **PCM16 24 kHz (48 kHz wastes bandwidth + adds resampling latency).** **Echo cancellation is non-negotiable — without it the agent barges in on itself.** *"Failing to clear the client buffer is the most common interruption bug we fix on prototypes."* |
| [LiveKit realtime models docs](https://docs.livekit.io/agents/models/realtime/) | ⭐ **Full-duplex vs half-cascade.** Half-cascade = realtime comprehension + a TTS you control. Loading extensive text history makes a realtime model *"more likely to respond in text only"* and *"limits their ability to interpret emotional context and other verbal cues that may not translate well to text transcription."* |
| [Gemini Live API overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api) | **Affective dialog** (adapts style/tone to your expression), barge-in, 24 languages, native bidirectional transcription. |
| [Cross-Provider Realtime Voice: Gemini Live on the OpenAI surface (Zylos)](https://zylos.ai/research/2026-07-19-realtime-voice-api-protocol-adaptation/) | ⭐ Protocol comparison table. **Session limits: OpenAI 60 min, no resumption; Gemini 15 min audio / 2 min audio+video**, extendable via `contextWindowCompression`, `SessionResumptionUpdate` token valid ~2 h, `GoAway` warning. **24 kHz ↔ 16 kHz resampling with anti-alias filtering.** `thinkingConfig` mis-set silently degrades barge-in. Framework comparison (Pipecat / LiveKit / TEN / purpose-built adapter). |
| [6 Best Open-Source Voice Agent Frameworks (techsy)](https://techsy.io/en/blog/best-open-source-voice-agent-frameworks) | LiveKit Agents 11,356 stars, Apache-2.0, **native SIP + phone numbers**, first-party OpenAI Realtime support, 1.5.x native MCP + adaptive interruption handling. |
| [Voice Agent API with LiveKit support (callmissed)](https://www.callmissed.com/blog/voice-agent-api-livekit-support-verified-2026) | **LiveKit $100M Series C at $1B valuation, Jan 2026.** Feature-by-feature comparison. |
| [Build Human-Like AI Voice Agents (metadesignsolutions)](https://metadesignsolutions.com/blog/building-human-like-ai-voice-agents-a-guide-to-using-livekit-with-gpt-4o-gemini) | Rooms/tracks/agents model. The 500 ms rule. Barge-in handling sequence. |

---

## Self-improvement (your research surface)

| Source | Key data |
|---|---|
| ⭐ [Self-Improving AI: What Actually Works in 2026 (morphllm)](https://www.morphllm.com/self-improving-ai) | **The full taxonomy** (prompt / agent / training / architecture) with numbers. **GEPA: +10% avg over GRPO, up to 20%, 35× fewer rollouts.** LATS 94.4% pass@1 HumanEval. **DGM 20%→50% SWE-bench.** SEAL **32.7%→47.0%** QA, beats GPT-4 data. Voyager 3.3× items / 15.3× faster. Eureka beat human experts on 83% of tasks. **The governance ladder.** |
| ⭐ [GEPA: Reflective Prompt Evolution Can Outperform RL (ICLR 2026 Oral)](https://arxiv.org/pdf/2507.19457) | The actual mechanism: sample trajectories → **reflect in natural language** → reflective mutation / crossover → **Pareto-front selection**. Why serialized natural-language traces beat scalar rewards. Working `gepa.optimize()` code. |
| ⭐ [Self-Improvements in Modern Agentic Systems: A Survey (arXiv 2607.13104)](https://selfimproving-agent.github.io/) | Ren, Chen, Guo, Rong, Li, Xiong, Lan, Wang, Guo, Zhuge, **Schmidhuber**. The curated paper library by category — your reading list. |
| ⭐ [AI Agent Self-Improvement: Reflection Loops (buildmvpfast)](https://www.buildmvpfast.com/blog/ai-agent-self-improvement-recursive-accuracy-production-2026) | **+34.2% breakdown**: runtime reflection (3 iter max) +12.1% in 2 days; failure-trace clustering +2.6% weekly. DSPy MIPROv2 **24%→51%** HotPotQA. Reflexion 80%→91%. **And the critical caveat: Huang et al., ICLR 2024 — LLMs cannot reliably self-correct reasoning without external feedback; performance sometimes DEGRADES.** |
| [Self-Improving LLM Agents at Test-Time (arXiv 2510.07841)](https://arxiv.org/html/2510.07841v1) | ⭐ **TT-SI on Qwen2.5-1.5B** — your size class. Uncertainty estimator → self-data augmentation → test-time LoRA (rank 8, α 16, all linear, 5 epochs, lr 1e-4, warmup 0.03, bs 1, cosine). |
| [Towards Self-Evolving Agents: Dual-Process Framework (Electronics 15(6):1232)](https://www.mdpi.com/2079-9292/15/6/1232) | ⭐ Fast-response vs deep-reflection separation. Evolvable long-term memory with a frozen backbone. Future work explicitly proposes **"combining memory evolution with lightweight parameter updates such as LoRA … internalizing high-confidence patterns into model weights while preserving the flexibility of external memory"** — literally FRIDAY's S2→S4. |
| [Experiential Reflective Learning (arXiv 2603.24639, ICLR 2026 MemAgents)](https://arxiv.org/pdf/2603.24639) | ERL: reflect on trajectories → generate **transferable heuristics**. Notes fine-tuning's per-turn overhead and that context-matching "provides no guidance when the current state fails to match any stored context." Implementation details (Reflexion w/ GPT-5-mini, ≤3 retries, L=3 trajectories, Qwen3-Embedding-0.6B, 3 few-shots). |
| [9 Open-Source Self-Improving Frameworks (futureagi)](https://futureagi.com/blog/self-improving-ai-agents-open-source-frameworks/) | ⭐ **Install-ready vs research-grade split.** DSPy/TextGrad/Trace = pip, MIT, stable. Reflexion/Voyager/SEAL/ADAS/DGM = "research code you adapt, not dependencies you add." TextGrad static since mid-2025. |
| [Self-Evolving Agents: A Developer's Guide (dev.to)](https://dev.to/chen115y/self-evolving-agents-a-developers-guide) | GEPA vs simple metaprompt table (population-based + Pareto + **train/validation split** vs greedy with no overfitting protection). DSPy self-distillation into smaller weights. Memento-Skills. AgentScope online fine-tuning. |
| [Hermes Agent deep guides](https://techjacksolutions.com/ai-tools/hermes/hermes-breakdown/) · [aibuilderclub](https://www.aibuilderclub.com/blog/hermes-nous-research-self-improving-agent) · [agentic-ai kb](https://agentic-ai.readthedocs.io/en/latest/AgentPlatforms/hermes-agent/) · [kie.ai](https://kie.ai/blog/what-is-hermes-agent) · [tosea](https://tosea.ai/blog/hermes-agent-self-improving-ai-guide) · [i-scoop](https://www.i-scoop.eu/hermes-agent-from-nous-research/) · [crabtalk](https://crabtalk.ai/blog/hermes-agent-survey) | The **40% faster claim, precisely**: 40% less token consumption and wall-clock, *not* 40% better output. Do/Learn/Improve loop. Three-layer memory internals. Honcho's 12 identity layers. **Render benchmark: same model (Opus 4.5) varies by 17 SWE-bench problems purely from scaffolding.** |

---

## MCP

| Source | Key data |
|---|---|
| ⭐ [MCP Specification Version Timeline](https://hidekazu-konishi.com/entry/mcp_specification_version_timeline.html) | All five revisions with headline changes. **Current: 2026-07-28** (RC published 2026-05-21). Roots/Sampling/Logging **deprecated** → use tool parameters, direct provider APIs, stderr/OpenTelemetry. |
| ⭐ [Scaling AI Agent Infrastructure with the MCP Stateless updates (Google Developers Blog)](https://developers.googleblog.com/scaling-ai-agent-infrastructure-with-the-mcp-stateless-updates/) | Handshake and `Mcp-Session-Id` **removed**. `_meta` carries protocol version + client info on every request. `Mcp-Method`/`Mcp-Name` headers for routing without DPI. **MRTR** (Multi Round-Trip Requests). JSON Schema 2020-12 with `oneOf`/`anyOf`/`allOf`/local `$ref`. Tier-1 SDKs (TS/Python/Go/C#) in beta. |
| ⭐ [The MCP Ecosystem in 2026 (ChatForest)](https://chatforest.com/guides/mcp-ecosystem-2026-state-of-the-standard/) | **30+ CVEs Jan–Feb 2026, incl. a CVSS 9.6 RCE.** **AgentSeal: security issues in 66% of scanned servers.** **Snyk: malicious payloads in 76 of 3,984 agent skills.** Registry counts by methodology (Glama 71K, PulseMCP 22K, Smithery 14K, ~101K combined). Pinterest's production architecture. The security tooling landscape. Protocol comparison MCP/A2A/ACP/UCP. |
| [MCP Adoption Statistics 2026 (digitalapplied)](https://www.digitalapplied.com/blog/mcp-adoption-statistics-2026-model-context-protocol) | **97M+ monthly SDK downloads. 10K+ active public servers. Official Registry 9,652 latest records / 28,959 server-versions. 15,926 GitHub topic repos. `modelcontextprotocol/servers` 86,148 stars.** Careful source-quality methodology — and the honest note that only 29% of large enterprises have *limited* production adoption, 12% broad. |
| [MCP Is Growing Up (AAIF)](https://aaif.io/blog/mcp-is-growing-up) | Why statelessness matters in practice for agentic systems. |
| [MCP Roadmap 2026 (a2a-mcp.org)](https://a2a-mcp.org/blog/mcp-2026-roadmap) | **MCP Server Cards** at `.well-known`. Extensions ecosystem / Skills primitive. Vercel Eve (filesystem-first durable agents). |
| [The 2026 Guide to the MCP Ecosystem (getknit)](https://www.getknit.dev/blog/the-guide-to-the-mcp-ecosystem) | What MCP is *and isn't* (not a framework, not orchestration, not a REST replacement). Resources/Tools/Prompts primitives. Client landscape. |
| [Complete Guide to MCP 2026 (dev.to)](https://dev.to/x4nent/complete-guide-to-mcp-model-context-protocol-in-2026-architecture-implementation-and-enterprise-4a11) | OAuth 2.1 + PKCE, SAML/OIDC. **MCP = USB-C, A2A = TCP/IP.** FastMCP 3.0 (Jan 2026). Enterprise roadmap table. |
| [MCP: Landscape, Security Threats (arXiv 2503.23278)](https://arxiv.org/pdf/2503.23278) | The academic threat model. Installer/deployment attack surface. |
| [Why MCP Became the Standard (NeuralCoreTech)](https://neuralcoretech.com/model-context-protocol-mcp-2026-agentic-ai-standard/) | Adoption timeline; the five client features (sampling, roots, elicitation) and five server primitives. |

---

## Local models & your hardware

| Source | Key data |
|---|---|
| ⭐ [r/LocalLLaMA — CPU-only LLM performance, t/s with llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1p90zzi/cpuonly_llm_performance_ts_with_llamacpp/) | **The table FRIDAY's model menu is built on.** Qwen3-0.6B Q8 **86** · gemma-3-1b Q8 **42** · LFM2-2.6B Q4 **30** · Llama-3.2-3B Q4 **25** · **Qwen3-4B UD-Q4_K_XL 20** · Qwen3-4B Q8 **13** · phi4-mini Q6 **18** · Llama-3.1-8B Q4 **11** · Qwen3-8B-128K Q4 **9** · gemma-3-12b Q4 **7** · **Huihui-Ling-mini-2.0 MXFP4_MOE 58** · **LFM2-8B-A1B Q4 38** · gemma-3n-E2B Q4 **28** · **Qwen3-30B-A3B IQ4_XS 27** · gpt-oss-20b mxfp4 **23**. Plus the key insight: *"Prompt processing is compute-bound; generation is only memory-bound if the compute is present — and it isn't, with CPU."* |
| ⭐ [Running a Local LLM on AMD Radeon 780M — gfx1103, ROCm, and the GPU That Wasn't Supposed to Work](https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/) | **The single most relevant benchmark for your machine.** Dense 7B/12B: ROCm+FA prefill **54 t/s**, Vulkan RADV **39**, native HIP **22**, CPU-only **14–16** — but **generation 4.5–5.3 t/s on ALL of them.** *"The Surprising Finding: CPU Beats GPU on Generation."* **The real bottleneck: single-channel RAM.** **The breakthrough: MoE on CPU** (26B-A4B reads ~2.5 GB/token vs 7 GB for dense 12B). ROCm gfx1102 spoofing for gfx1103 chips. |
| ⭐ [Qwen3-30B-A3B runs at 12–15 t/s on CPU (r/LocalLLaMA)](https://www.reddit.com/r/LocalLLaMA/comments/1kag4er/qwen330ba3b_runs_at_1215_tokenspersecond_on_cpu/) | **The 32 GB verdict.** `--cpu-moe` / `-ot ".ffn_.*_exps.=CPU"`. 10–14 t/s on an i7-1185G7 with dual-channel DDR4-3600. ~30 t/s on a 14700KF+3070. 70 t/s on an M4 Pro/48 GB with MLX 4-bit. And: ***"How much ram does it take? I have 16GB ram and Q4 can't be loaded."*** |
| [Running LLMs on CPU Only (localaimaster)](https://localaimaster.com/blog/run-llm-cpu-only) | **DDR4-2400 dual-channel laptop ≈ 38 GB/s → 7B at ~5–8 t/s.** The MoE loophole explained, with honest caveats (19 GB download, needs 32 GB+). Model picks by RAM tier. |
| [WHY Are Local LLMs So Slow On My Framework 13 AMD Strix Point](https://msf.github.io/blogpost/local-llm-performance-framework13.html) | ⭐ **How to derive your own bandwidth ceiling**: `model_size_GB × tg_t/s = effective_bandwidth`, vs `memory_MT_per_sec × bus_bits / 8 = theoretical_max`. Measured 13.4 t/s = 75% of a 15.2 t/s realistic ceiling. Vulkan vs ROCm vs CPU. Power-profile effects (146 vs 322 t/s prefill!). Speculative decoding command. Reproduction steps. |
| [Strix Halo Guide (Ryzen AI MAX+ 395)](https://github.com/hogeheer499-commits/strix-halo-guide) | What the *ceiling* AMD iGPU looks like (Qwen3.6-35B-A3B at 71.8 t/s generation, MTP speculative at 101–141 t/s). Useful for knowing what a 32 GB+ upgrade path could reach. |
| [Local LLM Inference on Windows 11 + AMD GPU using WSL and llama.cpp](https://dev.to/nicholaswiseman/local-llm-inference-on-windows-11-and-amd-gpu-using-wsl-and-llamacpp-36e7) | Ryzen 5 3600 CPU: **1.5 t/s prefill, 6.4 t/s generation** vs RX 7800 XT ROCm: **149 / 79.7**. The WSL2 GPU-passthrough path (`ggml_cuda_init: found 1 ROCm devices`). |
| [Local LLM on AMD RX 580 + Vulkan + Ollama](https://news.hamidun.com/en/news/9528/local-llm-on-a-2017-graphics-card-amd-rx-580-vulkan-ollama) | **Vulkan instead of ROCm** — no ROCm pain, works everywhere. 15–35 t/s on a 2017 card. |
| [Ollama System Requirements 2026](https://localaimaster.com/blog/ollama-system-requirements) | **AMD Ryzen 5 5600X (6/12): 7B Q4 = 5–8 t/s, 14B Q4 = 3–5 t/s.** VRAM by model size. **ROCm on Windows still experimental as of June 2026.** Ollama native on Windows since v0.3 (no WSL required). |
| [r/ollama — Running LLMs with AMD GPUs](https://www.reddit.com/r/ollama/comments/1go1e5z/running_llms_with_amd_gpus/) | Includes the sobering data point: *"On my 6-core Ryzen 5 I'm getting 0.5–2 toks/sec on large models via stock ollama."* Large models on your class of CPU are a dead end. |
| [r/LocalLLaMA — What's it like running AMD GPUs with AI?](https://www.reddit.com/r/LocalLLaMA/comments/1ln1a6u/whats_it_currently_like_for_people_here_running/) | *"llama.cpp Vulkan caught up with ROCm — no performance loss."* Framework 16 iGPU: 8–12B quants at ~30 t/s. VRAM is the most important number. |
| [Performance of llama.cpp on AMD ROCm (HIP) — ggml-org discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021) | ⭐ Real `llama-bench` output on **AMD Radeon Graphics (RADV RENOIR)** — a VivoBook-class iGPU: gemma4 26B-A4B Q4_0, Vulkan, 8 threads, mlock, FA on → **pp1090 144 t/s, tg128 14.7 t/s**. Plus: Vulkan tg ~8–23% faster than ROCm; ROCm pp more stable; Vulkan pp has cold-start shader-cache variance. Tuning flags (`GGML_VK_MAX_NODES_PER_SUBMIT`, `--ubatch-size`, `--load-mode mlock`). |
| [From OOM to 262K Context: Qwen3-Coder 30B on 8 GB VRAM](https://dev.to/upayanghosh/from-oom-to-262k-context-running-qwen3-coder-30b-locally-on-8gb-vram-1ej1e5) | ⭐ **The `--n-cpu-moe` sweep table.** `--cpu-moe` → 2.78 pp / 13.38 tg. `--n-cpu-moe 38` → **53.14 pp / 33.64 tg**. A 19× prefill improvement from one flag. Do this sweep on your machine. |
| [Best Local LLMs 2026 (layer3labs)](https://www.layer3labs.io/guides/best-local-llm-models) · [(kunalganglani)](https://www.kunalganglani.com/blog/local-ai-voice-assistant-whisper-piper-ollama) · [(klymentiev)](https://klymentiev.com/blog/best-local-llm) · [(codersera)](https://codersera.com/blog/best-small-llms-to-run-locally-a-comprehensive-guide/) · [(HF blog)](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally) · [(localaimaster SLMs)](https://localaimaster.com/blog/small-language-models-guide-2026) · [(promptquorum Ollama)](https://www.promptquorum.com/local-llms/top-open-source-models-ollama) | Model families by RAM tier. **16 GB machine → llama3.2:3b / qwen3:4b / gemma3:4b for voice latency.** Whisper latency by hardware tier (Pi 5 ~3–4 s; 16 GB mini-PC <1 s). Qwen3.5-4B (3.4 GB, text+image, 256K ctx), Qwen3.5-9B (6.6 GB). gpt-oss:20b as the best 16 GB reasoning model. |
| [Best Local Vision Language Models 2026 (tinyweights)](https://tinyweights.dev/posts/best-local-vision-language-models-2026/) | ⭐ **Qwen3-VL-4B: MMMU 67.4, DocVQA 95.3, 256K ctx, ~3.5 GB @ Q4, Apache 2.0.** Qwen3-VL-8B: MMMU 69.6, DocVQA 96.1, ~6 GB. Gemma 3 4B: easiest first-run. Moondream 3 for grounding/detection/pointing. |
| [Comparing LLMs on 16 GB VRAM (glukhov)](https://www.glukhov.org/llm-performance/benchmarks/choosing-best-llm-for-ollama-on-16gb-vram-gpu/) | Partial-offload behaviour: qwen3-vl:30b-a3b at 30% CPU / 70% GPU = 51 t/s using 22 GB. |

---

## Free GPU / fine-tuning

| Source | Key data |
|---|---|
| ⭐ [Google Colab Alternatives: 8 GPU Clouds Compared (Spheron)](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared/) | **Kaggle: ~30 GPU-hrs/week, VISIBLE quota, 2× T4 (16 GB each) or 1× P100, no credit card, 12 h/session.** Colab free: T4-only, **90-minute idle timeout**, 12-hour hard cap, GPU availability "varies over time" and premium hardware is "heavily restricted" for non-payers. |
| ⭐ [Fine-Tune Open-Source LLMs: LoRA & QLoRA 2026 Guide](https://www.kunalganglani.com/blog/fine-tune-open-source-llm-lora-qlora) | **VRAM table: 4B → ~5 GB with QLoRA+Unsloth; 7–8B → ~6 GB; 12B → ~10 GB.** *"A free Colab T4 (16 GB) works for models up to about 12B with QLoRA + Unsloth."* **Unsloth is 1.6× faster, 60% less VRAM, and patches the float16 tensor-core issue on T4 and older GPUs** that prevents gradient overflow. |
| ⭐ [aggreyeric/colab-finetune](https://github.com/aggreyeric/colab-finetune) | **The exact reference pipeline FRIDAY's S4 uses.** Qwen2.5-1.5B + LoRA r=16 on 106 examples → **70 seconds** on a free T4, loss **3.37 → 0.14** → **940 MB Q4_K_M GGUF** → Modelfile → `ollama create` → `ollama run`. **Cost ₹0.** Full `colab --auth / run --gpu T4 / download` CLI workflow. "Fits free T4" model list: SmolLM2 135M · Qwen3 0.6B · Qwen2.5 1.5B · Qwen3.5 2B · **Qwen3.5 4B ← sweet spot** · Gemma 4 E4B · Phi-4-mini · Qwen2.5 7B. |
| [Best Free Platforms to Build a Custom AI Model 2026 (sozee)](https://www.sozee.ai/resources/best-free-custom-ai-platforms/) | Quota-structure comparison. Unsloth's Colab notebooks QLoRA Llama-3.1-8B in **6 GB VRAM** with gradient checkpointing. Honest failure points of each free tier. |
| [Colab free-T4 checkpointing issue #96 (lexoliu/ml-ime)](https://github.com/lexoliu/ml-ime/issues/96) | ⭐ Operational lesson: **"Free Colab can reclaim the VM without warning. Checkpoint on a wall-clock interval, not only at the 5,000-step mark, and copy each checkpoint off the VM as soon as it is written."** Measured 2026-09-28: T4 = 15 GB VRAM, 2 vCPU, 12 GB RAM, 66 GB disk. |

---

## Ambient capture & proactive agents

| Source | Key data |
|---|---|
| ⭐ [screenpipe/screenpipe](https://github.com/screenpipe/screenpipe) | **The README is the spec.** Event-driven capture, accessibility tree + OCR fallback, speaker diarisation. **5–10% CPU, 0.5–3 GB RAM, ~20 GB/month** (elsewhere: ~5–10 GB). SQLite+FTS5, ~300 MB/8 hr. REST :3030. MCP server. Pipes. **Three-layer deterministic per-pipe AI data permissions** (skill gating / agent interception / server middleware with per-pipe cryptographic tokens). YAML frontmatter fields. **PostHog + Sentry telemetry ON by default** — disable in Settings → Privacy. |
| [screenpipe YC S26 — Local Work Memory (explainx)](https://explainx.ai/blog/screenpipe-yc-s26-local-work-memory-agents-july-2026) | The July 2026 repositioning: *"record how you work → searchable memory, SOPs, and AI agents."* screenpipe vs Limitless vs Recall comparison table. The privacy debate, both camps. |
| [Screen Assistant AI in 2026](https://screenpipe.com/blog/screen-assistant-ai-2026) | Tool comparison incl. data location and offline capability. |
| [CameraClaw (Show HN)](https://news.ycombinator.com/item?id=47417984) | ⭐ *"Chat logs tell what the agent SAID it did, not what it ACTUALLY did."* Isolated Docker workstation + KasmVNC screen capture + console streaming + network recording + VLM analysis → a scrubbable timeline. The right model for supervising autonomous code execution. |
| [OpenClaw v2026.8.1: Native Apps](https://docs.openclaw.ai/releases/2026.8.1/native-apps) | macOS Quick Chat from the menu bar / global shortcut over the current app, 5 most recent conversations, streams in place, dictation, model + reasoning-level picker. **Screen Recording permission → window/region capture; Accessibility permission → bounded text from the focused app + paste-back.** Captured context clears after send. Good UX reference for a desktop surface. |

---

## Two claims to treat carefully

1. **"40% better"** from self-created skills — the actual claim is **40% less token consumption
   and wall-clock time**, *not* 40% better output quality. Report your own numbers with that
   precision.
2. **LongMemEval / LOCOMO benchmark numbers** are vendor-published and contested. Mem0 publishes
   strong LOCOMO accuracy and ~90% token-savings claims; Letta argues LOCOMO is near-saturated;
   Zep publishes results claiming a lead over Mem0. Different sources report the *same* systems at
   26%, 49%, 93.4% and 94.8% depending on benchmark, model and methodology.
   **Use them for directional architecture choices (temporal > flat), not for precise ranking.**
   Build your own eval suite — that's the entire point of
   [06-INNOVATION-self-improvement-loop.md](./06-INNOVATION-self-improvement-loop.md).

---

## Primary papers worth reading in full

| Paper | Where |
|---|---|
| GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning | [arXiv 2507.19457](https://arxiv.org/pdf/2507.19457) — ICLR 2026 Oral |
| Self-Improvements in Modern Agentic Systems: A Survey | [arXiv 2607.13104](https://arxiv.org/abs/2607.13104) |
| Experiential Reflective Learning for Self-Improving LLM Agents | [arXiv 2603.24639](https://arxiv.org/pdf/2603.24639) |
| Self-Improving LLM Agents at Test-Time | [arXiv 2510.07841](https://arxiv.org/html/2510.07841v1) |
| Towards Self-Evolving Agents: A Dual-Process Framework | [Electronics 15(6):1232](https://www.mdpi.com/2079-9292/15/6/1232) |
| Model Context Protocol: Landscape, Security Threats | [arXiv 2503.23278](https://arxiv.org/pdf/2503.23278) |
| Reflexion: Language Agents with Verbal Reinforcement Learning | NeurIPS 2023 |
| MemGPT: Towards LLMs as Operating Systems | UC Berkeley (→ Letta) |
| Voyager: An Open-Ended Embodied Agent with LLMs | NVIDIA/Caltech 2023 |
| Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | Sakana AI 2025 |
| Cross-linguistic turn-taking (the 200–250 ms baseline) | Stivers et al., PNAS 2009, doi:10.1073/pnas.0903616106 |
