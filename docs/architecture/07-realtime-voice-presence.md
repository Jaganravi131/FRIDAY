# 07 — Realtime Voice & the Presence Fabric (Innovation #7)

> "Very natural and realistic" + "connect from mobile and laptop and feel empowered."
>
> These are two separate engineering problems. **Naturalness is a latency problem.**
> **Cross-device is a state-location problem.** This doc solves both.

---

## Part A — Voice that feels human

## A.1 The number you're aiming at

**Humans take turns at 200–250 ms average offset** (Stivers et al., PNAS 2009,
doi:10.1073/pnas.0903616106 — cross-linguistic, and the baseline every voice agent is judged
against).

| Perceived quality | End-to-end latency |
|---|---|
| Feels like a person in the room | **< 500 ms** |
| Feels natural, slightly deliberate | 500–800 ms |
| Feels like a good phone assistant | 800–1,500 ms |
| Feels like a phone tree | > 1,500 ms |
| You stop using it | > 3,000 ms |

### The measured component budget (2026)
| Component | Latency |
|---|---|
| LiveKit SFU edge audio (Opus 48 kHz, RED/FEC) | **15–25 ms** one-way |
| Silero VAD frame | **30–32 ms** |
| Deepgram Nova-3 streaming STT | **110–160 ms** |
| faster-whisper `small` on your Ryzen 5 (chunked) | ~300–600 ms |
| Groq LPU (Llama-3.3-70B) TTFT | **65–95 ms** |
| **Qwen3-4B local TTFT** | **~200–400 ms** ← your L1 |
| Cartesia Sonic-2 / ElevenLabs Flash v2.5 TTFA | **85–135 ms** |
| Piper TTS (local) TTFA | ~50–150 ms |
| Full-duplex speech-to-speech (Moshi / OpenAI Realtime) | **300–550 ms** total |

### Two viable local budgets for FRIDAY

**Cascade (recommended — you control every stage):**
```
VAD 30ms → whisper-small 400ms → Qwen3-4B TTFT 300ms → Piper 100ms → playback
                                                            = ~830 ms  ✅ "good assistant"
With Deepgram + Cartesia (cloud speech, local brain):
VAD 30ms → Nova-3 140ms → Qwen3-4B 300ms → Sonic-2 110ms  = ~580 ms  ✅ "natural"
With speculative decoding on Qwen3-4B (0.6B draft):
                                          TTFT → ~180ms     = ~460 ms  ✅ "very natural"
```

**Full-duplex speech-to-speech (simplest, best quality, cloud-only):**
```
OpenAI gpt-realtime-2.1  or  Gemini Live gemini-3.1-flash-live-preview
                          = 300–550 ms end-to-end  ✅✅ "person in the room"
```
One bidirectional stream: audio in, audio out. No separate STT/TTS to wire. Turn-taking,
prosody and tool calls all in one round trip.

> **FRIDAY's answer: implement BOTH behind one interface.**
> A `RealtimeModel` abstraction with a `CascadePipeline` implementation. Local cascade when
> offline / private / cheap; cloud full-duplex when you want the magic. LiveKit Agents already
> models exactly this (`RealtimeSession`/`RealtimeModel` per provider plugin), and Pipecat lets
> you mix both in one codebase.
>
> **Half-cascade is the sleeper option:** a realtime model for *comprehension* (it understands
> your speech, tone, interruptions natively) with a **separate TTS you control** for output.
> LiveKit documents this explicitly — you gain realtime speech understanding while keeping full
> control of the voice. Great middle ground: Gemini Live's comprehension + Cartesia's voice.

---

## A.2 Turn detection — the actual lever

**Do not use fixed silence timeouts.** They either clip people who pause to think or make you
wait 2 seconds after every sentence. Use **semantic turn detection** — a model judging
*linguistic completeness*.

```python
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import silero, deepgram, cartesia, openai, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

def prewarm(proc: JobProcess) -> None:
    # CRITICAL: instantiating Silero VAD inside the entrypoint adds ~1000ms cold start
    # to the FIRST turn. Load it once per worker process at container init.
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3", language="multi", interim_results=True),
        llm=openai.LLM(model="qwen3:4b", base_url="http://127.0.0.1:8080/v1"),  # local!
        tts=cartesia.TTS(model="sonic-2", voice="<your-friday-voice-id>"),

        # ── the lever ────────────────────────────────────────────────
        turn_detection=MultilingualModel(),   # semantic, multilingual (Tamil+English)
        min_endpointing_delay=0.4,   # floor: don't clip mid-sentence pauses
        max_endpointing_delay=3.0,   # ceiling: for trailing-off speakers
        preemptive_generation=True,  # ★ start the LLM on PARTIAL transcripts,
                                     #   cancel if the user keeps talking.
                                     #   This is what gets you under 400ms.
    )

    usage = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)   # EOU delay, TTFT, TTFA → your telemetry
        usage.collect(ev.metrics)

    await session.start(
        room=ctx.room, agent=FridayAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),   # BVCTelephony() for SIP
        ),
    )
    await session.generate_reply(instructions="Greet Jagan. One sentence. Ask nothing.")
```

**`preemptive_generation` is the single highest-value flag.** Speculatively begin inference on the
partial transcript; if the user continues, cancel and restart. You pay some wasted compute for
~300 ms of perceived latency. On a machine where compute is free and latency is everything, this
trade is obviously correct.

**Barge-in debounce physics:** with `interrupt_speech_duration = 0.25–0.35 s`, coughs and
background clicks don't false-trigger. LiveKit then mutes the media track, cancels LLM token
streaming, and clears the TTS buffer in **under 40 ms**.

---

## A.3 The three bugs that will eat your week

### Bug 1 — Barge-in needs exact milliseconds, not "cancel"

Documented repeatedly in OpenAI's own developer community:

> Cancelling the response stops *future* audio — but the model's conversation history still
> contains the **full, uninterrupted response it generated.** As far as the model's context is
> concerned, the user heard the complete answer, **including the part cut off and never played.**

**The correct sequence:**
1. On `input_audio_buffer.speech_started` → **stop local playback immediately**
2. **Measure exactly how many milliseconds of assistant audio were actually played**
3. Send `conversation.item.truncate` with that item's ID, content index, and
   `audio_end_ms` = **the actual played-back duration**
   — *not* the full response length, *not* zero

**Gemini Live does this differently:** the **server** unilaterally sets an `interrupted` flag on
`BidiGenerateContentServerContent`; the client's only job is to stop playing queued audio and
clear its buffer. There is **no upstream truncate equivalent** — an adapter must locally discard
queued output audio.

**Zero-latency variant:** run **client-side VAD in an AudioWorklet thread** so local playback can
be muted at effectively zero latency *before* waiting for a server-side interruption signal.
Trades some false-positive risk for eliminating the round-trip entirely. Worth doing.

**FRIDAY's test for this** (`eval/suites/voice-naturalness.yaml` `vn-005`): barge in at 40% of a
response, then ask *"what did you just say?"* It must report only what was **actually played**.
This bug is invisible until you test it explicitly, and it makes the agent feel gaslit.

### Bug 2 — Failing to clear the client playout buffer

> *"The most common interruption bug we fix on prototypes."*

The model has usually streamed **several hundred milliseconds ahead** of what the caller has
heard. `response.cancel` stops the server; your client must **also** flush everything already
received and queued. Two separate actions, and forgetting the second one means FRIDAY keeps
talking for half a second after you interrupt it.

### Bug 3 — No echo cancellation

**Non-negotiable.** Without AEC the agent hears its own voice and **barges in on itself** — an
infinite interrupt loop. Enable browser/OS AEC, and use the framework's noise cancellation
(`noise_cancellation.BVC()`, or `BVCTelephony()` on SIP trunks).

Also: **capture at 16-bit PCM 24 kHz** (OpenAI's native rate). 48 kHz wastes bandwidth and adds
resampling latency. Gemini Live is **fixed at 16 kHz input / 24 kHz output**, so a cross-provider
adapter needs anti-alias-filtered resampling between them.

---

## A.4 Voice instructions (small thing, huge effect)

```
You are FRIDAY. You are speaking aloud. Everything you say is converted directly to speech.

- Speak in short sentences. One or two per turn. Under 35 words unless asked for detail.
- NEVER use markdown, bullet points, numbered lists, headers, tables, or code fences.
  Do not say "dash dash" or "asterisk". If you must enumerate, say "first… second…".
- Never spell out a URL or a long number digit by digit unless asked.
- If the answer is longer than three sentences, say the headline, then offer:
  "want the detail?"
- Match Jagan's language. If he mixes Tamil and English, mix back — do not "correct" to English.
- Do not narrate your tool use. Never say "let me check my memory" — just check it, silently.
  If a lookup takes over two seconds, say "one sec".
- Interrupts are normal and welcome. If Jagan cuts you off, stop instantly, do not finish the
  sentence, and do not apologise.
- When you do not know, say so in four words. Do not pad.
```

That last-but-one rule matters more than it looks: **narrating tool use is the #1 reason voice
agents feel like software instead of a person.**

---

## A.5 Affect (what your blueprint called "Emotion Engine" — don't build one)

Your pasted blueprint had a separate "Emotion Engine" box with sentiment/tone/facial/voice-mood
classifiers. **Cut it.** A bolt-on sentiment classifier will be worse than the LLM's own
judgement and adds latency to the critical path.

Instead, do two things:

1. **Use a model with native affect.** Gemini Live has **affective dialog** — it adapts response
   style and tone to match the user's input expression, built in. Full-duplex speech-to-speech
   models hear prosody directly, which does not survive transcription. (LiveKit notes this
   explicitly as a limitation of loading text history into a realtime model: it *"limits their
   ability to interpret emotional context and other verbal cues that may not translate well to
   text transcription."*)

2. **Record affect as a FACT, not a module.** In the S0 trace:
   ```json
   "affect": {"valence": -0.2, "arousal": 0.4, "note": "mild frustration, self-reported tired"}
   ```
   and let the Memory Compiler promote durable patterns into S2:
   ```markdown
   - stress_trigger: "lease/broker interactions" (observed 6×, Sep 2026) [f_201 · observed · 0.75]
   - prefers_when_stressed: "short answers, no questions back, offer to handle it" [f_202 · stated · 0.95]
   - low_energy_hours: "before 09:00" (observed) [f_203 · observed · 0.60]
   ```
   Now "emotional intelligence" is *knowledge about you* rather than a real-time classifier — it
   compounds, it's editable, it's explainable, and it costs zero latency.

The episode's `affect_arc` field ("flat/frustrated AM → focused PM") does the rest.

---

## Part B — The Presence Fabric

## B.1 The core idea

> **One conversation is a ROOM. Devices are SURFACES that join and leave it.**
>
> The state lives in the room, not in any device.

Your pasted blueprint said "a unified WebSocket connection ensuring that if you ask a question on
your phone while walking, your laptop screen updates instantly when you sit down." That's the
right instinct but the wrong mechanism — a shared WebSocket is a *transport*, not a *presence
model*. You need the room abstraction or handoff will be janky.

```
                    ┌───────────────────────────────────────┐
                    │      SESSION  s_01J8ZK…               │
                    │  "the lease conversation"             │
                    │                                       │
                    │  authoritative state:                 │
                    │   · turn history (append-only)        │
                    │   · compiled context (ledger hash)    │
                    │   · in-flight generation              │
                    │   · audio playback position (ms!)     │
                    │   · active surfaces + their caps      │
                    │   · pending confirmations             │
                    └───────────────┬───────────────────────┘
                                    │ fan-out
        ┌───────────────┬───────────┴────────┬────────────────┐
        │               │                    │                │
   ┌────▼─────┐   ┌─────▼──────┐      ┌──────▼─────┐   ┌──────▼──────┐
   │ mobile   │   │ laptop     │      │ web PWA    │   │ voice node  │
   │          │   │            │      │ (tab)      │   │ (RPi/ESP32) │
   │ caps:    │   │ caps:      │      │ caps:      │   │ caps:       │
   │ mic ✓    │   │ mic ✓      │      │ mic ✓      │   │ mic ✓       │
   │ cam ✓    │   │ screen ✓   │      │ screen ✗   │   │ cam ✗       │
   │ gps ✓    │   │ files ✓    │      │ files ✗    │   │ speaker ✓   │
   │ speaker✓ │   │ shell ✓    │      │            │   │ display ✗   │
   │ display✓ │   │ display ✓  │      │            │   │             │
   └──────────┘   └────────────┘      └────────────┘   └─────────────┘

   Each surface declares CAPABILITIES. The session unions them.
   FRIDAY's available senses = ⋃(capabilities of attached surfaces) ∩ Sense Registry grants
```

### Why this gives you the magic moments

**Moment 1 — the walking handoff.**
You ask on your phone while walking. You sit at your laptop. The laptop joins `s_01J8ZK…`.
Because the state is in the room, the laptop immediately renders:
- the full transcript so far
- FRIDAY's in-progress answer, **continuing from the exact token**
- and if audio was playing, it can resume from `playback_position_ms`

No "let me repeat the question." No two separate conversations.

**Moment 2 — capability escalation.**
On the phone: *"what's this error on my screen?"* — phone has no screen capture.
FRIDAY notices the laptop is attached and has `screen ✓`:
> "I can't see your phone screen — but your laptop's open. Want me to look there, or send me a photo?"

That's the Sense Registry × Presence Fabric working together. This is the single most
"JARVIS" interaction you can build and it falls out of the architecture for free.

**Moment 3 — output routing.**
You're cooking, phone in your pocket, wake-word node on the kitchen counter.
FRIDAY answers **on the kitchen speaker** and simultaneously pushes the recipe **to the laptop
screen**. Input from the nearest mic; output to the best surface for the content type.

```python
ROUTING_POLICY = {
    "speech":        "nearest_active_mic_surface",
    "long_text":     "largest_display_surface",
    "code":          "surface_with_shell_capability",
    "image":         "largest_display_surface",
    "confirmation":  "surface_with_display AND most_recently_interacted",
    "urgent_alert":  "ALL surfaces (fan-out)",
    "private":       "surface_with_headphones OR single_attached_surface",
}
```

---

## B.2 Transport

| Channel | Use | Why |
|---|---|---|
| **WebRTC** (via LiveKit SFU or Pipecat/Daily) | audio + video + data tracks | UDP, 15–25 ms edge transit, NAT traversal via TURN/STUN, native barge-in track mute, room isolation |
| **WSS** (single multiplexed socket) | state, transcript, ledger records, notifications, confirmations, tool progress | Reliable ordered delivery for everything that isn't media |
| **Tailscale** | the network itself | **Private mesh. Your phone reaches your laptop with zero exposed ports, zero port-forwarding, zero Dynamic DNS.** Non-negotiable for a personal agent |
| MQTT | later, for IoT hardware | Only when you add real devices (Phase 6+) |
| gRPC | not yet | Law 8. You don't have internal services to connect |

**One WSS connection per surface**, multiplexed with a `type` field. Not one socket per feature.

```typescript
// surfaces/web/src/presence.ts
type ClientEvent =
  | { type: "join";   session: string; surface: SurfaceId; caps: Capabilities }
  | { type: "leave";  session: string }
  | { type: "text";   session: string; text: string }
  | { type: "audio";  session: string; pcm: ArrayBuffer; seq: number }
  | { type: "image";  session: string; blob: Blob; hint?: "screen"|"camera"|"document" }
  | { type: "barge_in"; session: string; played_ms: number }   // ← Bug 1's fix
  | { type: "confirm";  session: string; action_id: string; approved: boolean }
  | { type: "sense_grant"; sense: string; scope: object; ttl_days: number }
  | { type: "handoff_request"; from_session: string; to_surface: SurfaceId };

type ServerEvent =
  | { type: "turn_started";   session: string; turn: number; tier: string }
  | { type: "transcript_delta"; session: string; role: "user"|"friday"; text: string }
  | { type: "audio_delta";    session: string; pcm: ArrayBuffer; item_id: string }
  | { type: "audio_done";     session: string; item_id: string; total_ms: number }
  | { type: "tool_progress";  session: string; tool: string; state: string }
  | { type: "state_sync";     session: string; snapshot: SessionSnapshot }  // on join/handoff
  | { type: "confirmation_required"; session: string; action_id: string; summary: string; risk: string }
  | { type: "proactive";      session: string; text: string; urgency: "low"|"normal"|"high" }
  | { type: "ledger";         session: string; record: LedgerRecord }        // dev overlay
  | { type: "surface_joined"; surface: SurfaceId; caps: Capabilities }       // ← Moment 2 trigger
  | { type: "error"; code: string; message: string };
```

**`state_sync` on join is the whole handoff mechanism.** When a surface joins a session, it gets
the full snapshot. That's it. No migration protocol, no CRDT, no conflict resolution — because
there's exactly one authoritative state and one writer.

---

## B.3 The relay (security-critical)

> The OpenAI Realtime API **lacks client-side authentication** — it is insecure to connect
> directly from a user's browser. OpenAI's own WebRTC guide gives two mitigations:
> (a) WebRTC with a **short-lived ephemeral token minted server-side**, or
> (b) a **relay** — the browser talks to your server, your server holds the real key.

**FRIDAY is a relay by design.** No API key ever reaches a client. The Presence Fabric:
- authenticates surfaces (device keypair + Tailscale identity)
- mints **short-lived per-session tokens** for any cloud realtime provider
- holds all real credentials
- writes the audit log

```
phone ──WSS/WebRTC──► Presence Fabric ──► cognition core ──► llama-server (local)
      (device key)         │                             └─► OpenRouter/Realtime (ephemeral token)
                           ├── Sense Registry check
                           ├── redaction pass
                           └── audit log write
```

---

## B.4 Session limits & the always-on problem

| | OpenAI Realtime | Gemini Live |
|---|---|---|
| Max session | 60 min/connection; **no documented resumption protocol** | 15 min audio-only / **2 min audio+video** default |
| Extension | — | `contextWindowCompression`; **`SessionResumptionUpdate`** with a token valid **~2 h**; `GoAway` message warns of imminent termination |
| Reconnect | you rebuild | resumption token restores the session |

**Design consequence:** an always-on FRIDAY **must** have a session resumption layer. You cannot
hold a permanent video session — Gemini's 2-minute audio+video limit forbids it. So:

```python
# Vision is DUTY-CYCLED, not continuous.
VISION_POLICY = {
    "trigger": ["explicit_request", "wake_word + 'look at'", "document_detected"],
    "session_mode": "audio_only_persistent + video_on_demand",
    "video_max_seconds": 100,           # stay under Gemini's 120s audio+video limit
    "resumption_token_refresh": "on GoAway, or every 90 min",
    "compression": "contextWindowCompression at 70% of provider limit",
    "local_fallback": "qwen3-vl:4b on a single frame — DocVQA 95.3, no session limit at all",
}
```

**Prefer the local VLM for single frames.** Qwen3-VL-4B on one screenshot has **no session
limit, no cost, no network** and scores **95.3 on DocVQA**. Use cloud realtime video only for
*continuous* video understanding (e.g. "watch this whiteboard while I talk").

**Reconnection is not an edge case — it's the normal case** on mobile in India (network handover,
lifts, metro). Design for: socket drops mid-generation → client reconnects → `join` with the
session ID → `state_sync` → resumes with the partial answer intact. Test it by killing the
network mid-sentence 20 times.

---

## B.5 The wake-word node (the most empowering $10 you'll spend)

A dedicated always-listening node in each room is what turns FRIDAY from an app into a presence.

```
┌─────────────────────────────────────────────────────────┐
│  ESP32-S3 / Raspberry Pi Zero 2 W / old Android phone   │
│                                                         │
│  I2S MEMS mic → microWakeWord ("Friday")  [on-device]   │
│       │                                                 │
│       │  ONLY on wake-word detection:                   │
│       └──► stream audio → Presence Fabric (WSS/Tailscale)│
│                                                         │
│  speaker ◄── audio_delta from the session               │
│  LED ring: idle / listening / thinking / speaking       │
└─────────────────────────────────────────────────────────┘
```

**Key property: the wake word runs entirely on-device.** No audio ever leaves the node until
"Friday" is detected. That is a *far* stronger privacy posture than "we record everything and
filter server-side", and it's honest — you can verify it.

An **old Android phone is the best first node**: mic, speaker, screen, camera, GPS, always-on
power, and it costs you nothing. Use it before buying an ESP32.

**LED state is not a nicety.** An always-listening device with no visible state feels like
surveillance. Four states, physically visible, is what makes ambient AI feel like a companion.

---

## B.6 Latency checklist (do these in order, measure after each)

1. ☐ **Pre-warm the VAD** at process start (~1 s saved on turn one)
2. ☐ **Pre-warm the model** — keep L1 resident with `keep_alive` tuned, `--no-warmup` off
3. ☐ `preemptive_generation=True` — start LLM on partial transcripts (~300 ms saved)
4. ☐ **Stream LLM tokens straight into the TTS chunker** — never wait for the full response
5. ☐ **Speculative decoding** — 0.6B drafts for 4B (~30–50% TTFT reduction)
6. ☐ **Offload prefill to the iGPU** (`-ngl 99`), keep MoE experts on CPU (`--cpu-moe`)
7. ☐ `--flash-attn 1` + `--cache-type-k/v q8_0`
8. ☐ **Cap tool output** — a 4,000-token tool result is 4,000 tokens of prefill you pay for
9. ☐ Client-side VAD in an **AudioWorklet** for zero-latency local mute on barge-in
10. ☐ **AEC on, always** — otherwise the agent barges in on itself
11. ☐ PCM16 **24 kHz** capture (not 48 kHz) — no resampling latency
12. ☐ **Intent gate skips retrieval** for trivial turns (saves 100–600 ms entirely)
13. ☐ Tailscale, not a public tunnel — WireGuard beats ngrok/frp on RTT

Measure with the framework's own telemetry (`metrics_collected` → EOU delay, TTFT, TTFA) and
write it into the S0 trace. **You cannot optimise what you don't log, and voice latency drifts
silently.**
