# 03 — Memory & Knowledge Architecture

The concrete data layer. Read after [02-INNOVATION-memory-compiler.md](./02-INNOVATION-memory-compiler.md),
which explains *why* it's shaped this way.

---

## 1. Storage topology (all of it)

```
FRIDAY/                                  ← git repo = truth
├── soul/
│   ├── SOUL.md            persona, voice, boundaries          ALWAYS in context (~600 tok)
│   ├── AGENTS.md          operating rules — KEEP LEAN         ALWAYS in context (~300 tok)
│   ├── USER.md            who you are, ≤1,400 chars           ALWAYS in context (~250 tok)
│   ├── MEMORY.md          durable core facts, ≤2,200 chars    ALWAYS in context (~400 tok)
│   └── HEARTBEAT.md       proactive checklist                 heartbeat scope only
│
├── memory/
│   ├── traces/YYYY-MM-DD.jsonl       S0  append-only raw      NEVER in context
│   ├── episodes/YYYY-MM-DD.md        S1  daily narrative      JIT retrieved
│   ├── facts/*.md                    S2  bi-temporal facts    top-3 JIT retrieved
│   │     ├── identity.md   housing.md   work.md   health.md
│   │     ├── people.md     projects.md  money.md  devices.md
│   │     └── preferences.md habits.md   goals.md
│   ├── daily/YYYY-MM-DD.md           ephemeral log            JIT retrieved
│   ├── skills-index.md               compact skill directory  ALWAYS (~30 tok/skill)
│   └── index.json                    file → hash, for incremental recompile
│
├── skills/<name>/SKILL.md            S3  procedure            on activation only
├── eval/                             THE LOCKED SUITE         agent has NO write access
│   ├── suites/*.yaml
│   ├── graders/*.py
│   ├── heldout/*.jsonl               (agent cannot read these either)
│   └── gate-log.jsonl
│
└── artifacts/                        ← gitignored, rebuildable
    ├── friday.db                     SQLite: facts + FTS5 + sqlite-vec + sessions
    ├── adapters/                     S4 LoRA adapters + GGUFs (versioned separately)
    └── capture/                      screenpipe data (local only, never committed)
```

**Two hard rules:**
- **Everything in `FRIDAY/` (except `artifacts/`) is git-tracked.** Git is your backup, your
  diff, your rollback, your belief history.
- **Everything in `artifacts/` is a build output.** `rm -rf artifacts && friday compile` must
  fully restore function. If it can't, you've leaked truth into the DB — fix it.

---

## 2. The SQLite schema

One file, `artifacts/friday.db`. SQLite because: single-user, zero-ops, sub-ms keyword search,
no server, survives framework churn, and both Hermes and screenpipe validated it at scale.

```sql
PRAGMA journal_mode = WAL;          -- concurrent read during writes
PRAGMA synchronous  = NORMAL;       -- safe with WAL, much faster
PRAGMA mmap_size    = 268435456;    -- 256 MB mmap; big win on repeated reads
PRAGMA cache_size   = -64000;       -- 64 MB page cache
PRAGMA foreign_keys = ON;

-- ─── S2: bi-temporal facts ────────────────────────────────────────────────────
CREATE TABLE facts (
  id            TEXT PRIMARY KEY,               -- ULID: f_01J8ZK...
  subject       TEXT NOT NULL,
  predicate     TEXT NOT NULL,
  object        TEXT NOT NULL,
  object_type   TEXT DEFAULT 'str',
  predicate_class TEXT NOT NULL,                -- drives decay half-life
  single_valued INTEGER NOT NULL DEFAULT 0,     -- 1 = only one can be true at a time

  confidence    REAL NOT NULL DEFAULT 0.5,
  salience      REAL NOT NULL DEFAULT 0.5,

  valid_from    TEXT,                           -- world time: when it became true
  valid_to      TEXT,                           -- world time: NULL = still true
  asserted_at   TEXT NOT NULL,                  -- belief time: when FRIDAY learned it
  retracted_at  TEXT,                           -- belief time: NULL = still believed
  superseded_by TEXT REFERENCES facts(id) ON DELETE SET NULL,

  source_kind   TEXT NOT NULL CHECK (source_kind IN
                  ('user_edit','stated','imported','observed','inferred')),
  source_refs   TEXT NOT NULL,                  -- JSON array
  source_quote  TEXT,

  review_after  TEXT,
  access_count  INTEGER NOT NULL DEFAULT 0,
  last_accessed TEXT,

  origin_file   TEXT NOT NULL,                  -- memory/facts/housing.md
  origin_hash   TEXT NOT NULL,                  -- for incremental recompile
  origin_line   INTEGER
);
CREATE INDEX idx_facts_current ON facts(subject, predicate) WHERE retracted_at IS NULL;
CREATE INDEX idx_facts_valid   ON facts(subject, valid_from, valid_to);
CREATE INDEX idx_facts_file    ON facts(origin_file, origin_hash);
CREATE INDEX idx_facts_review  ON facts(review_after) WHERE retracted_at IS NULL;

-- ─── entity graph (SQLite, not Neo4j) ─────────────────────────────────────────
CREATE TABLE entities (
  id       TEXT PRIMARY KEY,
  kind     TEXT NOT NULL,      -- person|place|project|device|org|account|concept
  name     TEXT NOT NULL,
  aliases  TEXT,               -- JSON array — "Amma" = "Mrs. Lakshmi" = "+91 98…"
  notes    TEXT
);
CREATE TABLE relations (
  src        TEXT REFERENCES entities(id),
  predicate  TEXT NOT NULL,    -- works_at | lives_with | owns | part_of | reports_to
  dst        TEXT REFERENCES entities(id),
  valid_from TEXT, valid_to TEXT,
  confidence REAL, source_refs TEXT,
  PRIMARY KEY (src, predicate, dst, valid_from)
);
CREATE INDEX idx_rel_dst ON relations(dst, predicate);

-- ─── full-text (FTS5) over facts + episodes + skills ──────────────────────────
CREATE VIRTUAL TABLE memory_fts USING fts5(
  body,
  kind UNINDEXED,             -- fact|episode|skill|daily
  ref  UNINDEXED,             -- fact id / file path
  subject UNINDEXED,
  tokenize = 'porter unicode61 remove_diacritics 2'
);
-- porter stemming + unicode61 handles English well.
-- For Tamil: add a separate FTS5 table with tokenize='unicode61' and trigram fallback,
-- or index the romanised transliteration alongside. See §6.

-- ─── vectors (sqlite-vec) ─────────────────────────────────────────────────────
CREATE VIRTUAL TABLE memory_vec USING vec0(
  ref TEXT PRIMARY KEY,
  kind TEXT,
  embedding FLOAT[1024]       -- Qwen3-Embedding-0.6B → 1024-d
);

-- ─── sessions & turns ─────────────────────────────────────────────────────────
CREATE TABLE sessions (
  id           TEXT PRIMARY KEY,
  started_at   TEXT NOT NULL,
  ended_at     TEXT,
  scope        TEXT NOT NULL CHECK (scope IN ('interactive','heartbeat','dreaming','eval')),
  surfaces     TEXT,            -- JSON: which devices participated
  topic        TEXT,
  turn_count   INTEGER DEFAULT 0
);
CREATE TABLE turns (
  id           TEXT PRIMARY KEY,
  session_id   TEXT REFERENCES sessions(id),
  idx          INTEGER NOT NULL,
  ts           TEXT NOT NULL,
  role         TEXT NOT NULL,
  surface      TEXT,            -- mobile|desktop|web|cli|watch|voice-node
  mode         TEXT,            -- voice|text|image
  content      TEXT,            -- text as served (media stripped → path+hash)
  tier         TEXT,            -- L0..L5 escalation tier actually used
  tokens_in    INTEGER, tokens_out INTEGER,
  ttft_ms      INTEGER, total_ms INTEGER,
  cache_hit    INTEGER,
  ledger_hash  TEXT,
  trace_id     TEXT             -- → S0
);
CREATE INDEX idx_turns_session ON turns(session_id, idx);

-- ─── skills ───────────────────────────────────────────────────────────────────
CREATE TABLE skills (
  name         TEXT PRIMARY KEY,
  version      INTEGER NOT NULL,
  when_to_use  TEXT NOT NULL,   -- the ONLY part visible pre-activation
  triggers     TEXT,            -- JSON array
  path         TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('draft','shadow','active','retired')),
  shadow_pass  REAL,
  uses         INTEGER DEFAULT 0,
  last_used    TEXT,
  provenance   TEXT             -- JSON: distilled_from traces
);

-- ─── Sense Registry & audit (see 09) ──────────────────────────────────────────
CREATE TABLE senses (
  id           TEXT PRIMARY KEY,   -- 'screen.capture' | 'mic.listen' | 'calendar.read' …
  enabled      INTEGER NOT NULL DEFAULT 0,
  scope        TEXT,               -- JSON: allow/deny apps, windows, paths, time ranges
  retention_d  INTEGER,
  redact_rules TEXT,               -- JSON
  granted_at   TEXT, granted_by TEXT,
  token_hash   TEXT                -- per-scope capability token
);
CREATE TABLE audit (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ts         TEXT NOT NULL,
  actor      TEXT NOT NULL,        -- 'agent'|'heartbeat'|'dreaming'|'user'|'gate'
  action     TEXT NOT NULL,
  target     TEXT,
  sense_id   TEXT,
  decision   TEXT,                 -- allowed|denied|confirmation_required|redacted
  detail     TEXT
);
CREATE INDEX idx_audit_ts ON audit(ts);

-- ─── the gate log (your research artifact) ────────────────────────────────────
CREATE TABLE gate_log (
  id           TEXT PRIMARY KEY,
  ts           TEXT NOT NULL,
  gate         TEXT NOT NULL,      -- G1..G7
  candidate    TEXT NOT NULL,      -- prompt variant / skill / adapter id
  incumbent    TEXT,
  target_delta REAL, heldout_delta REAL, latency_delta REAL,
  decision     TEXT NOT NULL,      -- accepted|rejected|quarantined
  rationale    TEXT
);
```

---

## 3. The three memory scopes (and why they share one store)

Borrowed from `roiguri/jarvis-agent`, and it solves a real bug.

| Scope | Prompt style | Sees | May do |
|---|---|---|---|
| **interactive** | conversational, warm, expansive | today's proactive notifications + full JIT memory | anything (with confirmation gate) |
| **heartbeat** | terse, operational | today's chat summary + `HEARTBEAT.md` | notify or `HEARTBEAT_OK`; **never** repeat what interactive already handled |
| **dreaming** | analytical, no user-facing output | the day's traces | write S1/S2/S3, run GEPA, never message you |
| **eval** | locked, deterministic | the eval suite | measure only; **no write access to anything** |

All four share **one memory store and one tool registry**, but get **different prompts and
different tool whitelists by scope**. Awareness flows both ways (heartbeat reads what
interactive already told you; interactive reads today's notifications) so nothing gets said
twice.

This is the fix for the #2 project-killer: *FRIDAY telling you something it already told you.*

---

## 4. Context slot budgets (the always-resident part)

Hard character/token budgets. These are **constraints, not suggestions** — they force density.

| Slot | Source | Budget | Rationale |
|---|---|---|---|
| **Identity** | `SOUL.md` | ~600 tok | Persona, register, boundaries. Cached prefix. |
| **Rules** | `AGENTS.md` | ~300 tok | Durable operating rules ONLY. Every line is a recurring tax. |
| **User** | `USER.md` | **1,400 chars** (~250 tok) | Hermes's exact budget. Forces high signal. |
| **Core memory** | `MEMORY.md` | **2,200 chars** (~400 tok) | Hermes's exact budget. The 20 facts that matter most. |
| **Skill index** | `skills-index.md` | ~30 tok × N skills | name + when_to_use only. Bodies load on activation. |
| **Active senses** | Sense Registry | ~180 tok | What FRIDAY can currently see, and what it can't. |
| | | **≈ 1,750 tok fixed** | |
| **JIT: recalled memory** | retrieval path | 600–1,200 tok | top-3 reranked facts |
| **JIT: retrieved docs** | files/web | 2,000–4,000 tok | reranked, filtered |
| **Working: transcript** | recent turns | remainder | capped tool outputs |
| **Reserve** | — | **≥ 2,000 tok** | for the next tool result + reasoning. Never spend this. |

`MEMORY.md` promotion: when a fact's `salience × confidence × access_count` exceeds the weakest
resident fact, the Dreaming phase **proposes** swapping it into `MEMORY.md`. You can veto via
the git diff.

---

## 5. Multilingual: Tamil + English (your reality)

You're in Chennai; you will code-switch. Design for it from day 1 or it will never work.

| Component | Approach |
|---|---|
| **STT** | Whisper handles Tamil reasonably; **Deepgram Nova-3** is better if you allow cloud. Test both on *your* voice — accent matters more than language. |
| **Turn detection** | LiveKit's `MultilingualModel()` — explicitly built for this, not English-only silence timers. |
| **LLM** | Qwen3 family is genuinely multilingual (Qwen3-4B is strong here). Gemma 3 claims 140+ languages. Test Tamil output quality on the 4B **before** committing — small models degrade hardest in low-resource languages. This is a real reason to keep an L2 cloud tier. |
| **Embeddings** | `bge-m3` is explicitly multilingual/multi-function; Qwen3-Embedding also strong. Prefer these over English-only models. |
| **FTS5** | porter stemming is English-only. Add a **second FTS5 table** with `tokenize='unicode61'` for Tamil script, and store a **romanised transliteration** column for cross-script matching ("வீடு" ↔ "veedu" ↔ "home"). |
| **Facts** | Store `object` in the language you said it in, plus a normalised English gloss for retrieval. Never machine-translate your own memories — you lose the register. |
| **TTS** | Tamil TTS is the weak link. Test ElevenLabs/Cartesia Tamil quality early; local Piper Tamil voices are limited. **If natural Tamil voice matters to you, budget for a cloud TTS.** |

---

## 6. Sync across devices

You want phone + laptop + (later) watch, all seeing the same FRIDAY.

**Decision: server-authoritative, not CRDT.**

- The **laptop is the server** (or a VPS / home mini-PC if you want it always-on).
- All clients are **thin** — they render and capture, they don't hold memory.
- Sync = one WebSocket to the Presence Fabric. State lives in one place.
- Markdown/git gives you a **second, human-facing sync path** for free: `git push` to a private
  remote = offsite backup + you can read/edit FRIDAY's memory from any machine.

CRDTs (Automerge/Yjs) are for multi-writer peer-to-peer. You have **one writer** (FRIDAY) and
**one human editor** (you, rarely). Server-authoritative is simpler and correct.

**Offline tolerance:** clients cache the last N turns locally (encrypted) and replay on
reconnect. The agent never blocks on a client.

---

## 7. Backup & recovery

| What | How | Frequency |
|---|---|---|
| Truth (Markdown, git) | `git push` to a **private** remote | on every commit (post-Dreaming hook) |
| Traces (JSONL) | rclone → encrypted remote / external SSD | weekly |
| `friday.db` | **not backed up** — it's a build artifact | never |
| Adapters/GGUFs | rclone (large, but reproducible) | on promotion |
| screenpipe capture | local only, encrypted at rest, **never leaves the machine** | rolling 90-day delete |

**Recovery drill (run it once, in Week 4):** delete `artifacts/` entirely, run `friday compile`,
confirm retrieval still works. If it doesn't, you have truth in the DB — find it and move it out.

**Restore from a new machine:** `git clone` → `friday compile` → `friday doctor`. That's it.
This is the payoff of Markdown-as-truth: your AI's entire model of you is a git repo.

---

## 8. Sizing on 512 GB

| Component | Est. size |
|---|---|
| OS + apps | ~120 GB |
| Model GGUFs (1.7B + 4B + VL-4B + 0.6B + embed + rerank + whisper) | ~15 GB |
| Fine-tuned adapter versions (keep last 10) | ~20 GB |
| `friday.db` after 1 year | ~1–3 GB |
| Traces + episodes + facts (Markdown/JSONL), 1 year | ~2–5 GB |
| screenpipe capture, rolling 90 days | ~15–30 GB |
| Python + Node + WSL2 distro | ~25 GB |
| **Total** | **~200–220 GB** |
| **Free** | **~290 GB** |

Comfortable. You have room to keep *everything* — which is the point: at this scale,
**storage is not the constraint, RAM is.**
