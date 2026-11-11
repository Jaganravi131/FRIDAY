-- FRIDAY storage schema. Source: docs/architecture/03-memory-architecture.md §2
--
-- ⚠️ Everything here is a BUILD ARTIFACT. Truth lives in Markdown under memory/
--    and soul/. `rm -rf artifacts && python -m friday build` must restore all of
--    it. If a table here cannot be rebuilt from Markdown + traces, it is a bug.
--
-- Two additions over doc 03, both required by docs/architecture/13 §4.1:
--   supervision_spans  — free salience labels for the RSC write gate (w_t)
--   supervision_pairs  — free (key_old, key_new) labels for the RSC erase address
--                        (EDA's e_t). NEITHER CAN BE BACKFILLED. Collect from Day 1.

PRAGMA journal_mode = WAL;          -- concurrent read during writes
PRAGMA synchronous  = NORMAL;       -- safe with WAL, much faster
PRAGMA mmap_size    = 268435456;    -- 256 MB mmap; big win on repeated reads
PRAGMA cache_size   = -64000;       -- 64 MB page cache
PRAGMA foreign_keys = ON;

-- ─── S2: bi-temporal facts ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS facts (
  id            TEXT PRIMARY KEY,               -- f_<ulid-ish>
  subject       TEXT NOT NULL,
  predicate     TEXT NOT NULL,
  object        TEXT NOT NULL,
  object_type   TEXT DEFAULT 'str',
  predicate_class TEXT NOT NULL,                -- drives decay half-life
  single_valued INTEGER NOT NULL DEFAULT 0,     -- 1 = only one true at a time

  confidence    REAL NOT NULL DEFAULT 0.5,
  salience      REAL NOT NULL DEFAULT 0.5,

  valid_from    TEXT,                           -- WORLD time: when it became true
  valid_to      TEXT,                           -- WORLD time: NULL = still true
  asserted_at   TEXT NOT NULL,                  -- BELIEF time: when FRIDAY learned it
  retracted_at  TEXT,                           -- BELIEF time: NULL = still believed
  superseded_by TEXT REFERENCES facts(id) ON DELETE SET NULL,

  source_kind   TEXT NOT NULL CHECK (source_kind IN
                  ('user_edit','stated','imported','observed','inferred')),
  source_refs   TEXT NOT NULL DEFAULT '[]',     -- JSON array
  source_quote  TEXT,

  review_after  TEXT,
  access_count  INTEGER NOT NULL DEFAULT 0,
  last_accessed TEXT,

  origin_file   TEXT NOT NULL,                  -- memory/facts/housing.md
  origin_hash   TEXT NOT NULL,                  -- for incremental recompile
  origin_line   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_current ON facts(subject, predicate) WHERE retracted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_valid   ON facts(subject, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_file    ON facts(origin_file, origin_hash);
CREATE INDEX IF NOT EXISTS idx_facts_review  ON facts(review_after) WHERE retracted_at IS NULL;

-- ⭐ Compiled-state record for EVERY Markdown fact file, including the ones that
-- yield zero facts. Deriving staleness from `facts` alone
-- (COUNT(*) WHERE origin_file=? AND origin_hash=?) makes a file with no parseable
-- lines permanently "stale": no row is ever written, so the count stays 0, so it is
-- recompiled, re-audited and re-reported on every single turn forever. That fills
-- `audit --today` with phantom hand-edits and drowns the real ones — the opposite of
-- what the audit trail is for. A notes-only or half-written .md dropped into
-- memory/facts/ is an ordinary thing for a user to do.
CREATE TABLE IF NOT EXISTS origin_state (
    origin_file   TEXT PRIMARY KEY,
    origin_hash   TEXT NOT NULL,
    facts_derived INTEGER NOT NULL DEFAULT 0,
    compiled_at   TEXT NOT NULL
);

-- ─── entity graph (SQLite, not Neo4j) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
  id       TEXT PRIMARY KEY,
  kind     TEXT NOT NULL,      -- person|place|project|device|org|account|concept
  name     TEXT NOT NULL,
  aliases  TEXT NOT NULL DEFAULT '[]',   -- JSON: "Amma" = "Mrs. Lakshmi" = "+91 98…"
  notes    TEXT
);
CREATE TABLE IF NOT EXISTS relations (
  src        TEXT REFERENCES entities(id),
  predicate  TEXT NOT NULL,    -- works_at | lives_with | owns | part_of | reports_to
  dst        TEXT REFERENCES entities(id),
  valid_from TEXT, valid_to TEXT,
  confidence REAL, source_refs TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY (src, predicate, dst, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relations(dst, predicate);

-- ─── full-text (FTS5) over facts + episodes + skills + daily ──────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  body,
  kind    UNINDEXED,           -- fact|episode|skill|daily
  ref     UNINDEXED,           -- fact id / file path
  subject UNINDEXED,
  tokenize = 'porter unicode61 remove_diacritics 2'
);

-- ─── sessions & turns ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
  id           TEXT PRIMARY KEY,
  started_at   TEXT NOT NULL,
  ended_at     TEXT,
  -- ⭐ 'remote' added with the gateway. It is a USER-PRESENT scope, like
  -- 'interactive': the person is there, just on their phone over a tailnet rather than
  -- at the keyboard. It is deliberately NOT in policy.UNATTENDED_DENY's tuple. It gets
  -- its own value anyway because `audit --today` has to be able to answer "which turns
  -- came in over the network?" — a gateway that cannot answer that is a gateway you
  -- cannot trust. Existing databases carry the old CHECK and must be rebuilt; that is
  -- safe by construction, because artifacts/ is derived and `friday rebuild` restores
  -- it from the Markdown.
  scope        TEXT NOT NULL CHECK (scope IN ('interactive','remote','heartbeat','dreaming','eval')),
  surfaces     TEXT NOT NULL DEFAULT '[]',   -- JSON: which devices participated
  topic        TEXT,
  turn_count   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
  id           TEXT PRIMARY KEY,
  session_id   TEXT REFERENCES sessions(id),
  idx          INTEGER NOT NULL,
  ts           TEXT NOT NULL,
  role         TEXT NOT NULL,
  surface      TEXT,             -- mobile|desktop|web|cli|watch|voice-node
  mode         TEXT,             -- voice|text|image
  content      TEXT,
  tier         TEXT,             -- L0..L5 escalation tier actually used
  tokens_in    INTEGER, tokens_out INTEGER,
  ttft_ms      INTEGER, total_ms INTEGER,
  cache_hit    INTEGER,
  ledger_hash  TEXT,
  trace_id     TEXT              -- -> S0
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, idx);

-- ─── skills (S3) ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS skills (
  name         TEXT PRIMARY KEY,
  version      INTEGER NOT NULL,
  when_to_use  TEXT NOT NULL,    -- the ONLY part visible pre-activation
  triggers     TEXT NOT NULL DEFAULT '[]',
  path         TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('draft','shadow','active','retired')),
  shadow_pass  REAL,
  uses         INTEGER DEFAULT 0,
  last_used    TEXT,
  provenance   TEXT              -- JSON: distilled_from traces
);

-- ─── Sense Registry & audit (docs/architecture/09) ────────────────────────────
CREATE TABLE IF NOT EXISTS senses (
  id           TEXT PRIMARY KEY,   -- 'screen.capture' | 'mic.listen' | 'calendar.read'
  enabled      INTEGER NOT NULL DEFAULT 0,
  scope        TEXT NOT NULL DEFAULT '{}',   -- JSON allow/deny apps, windows, paths
  retention_d  INTEGER,
  redact_rules TEXT NOT NULL DEFAULT '[]',
  granted_at   TEXT, granted_by TEXT,
  token_hash   TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ts         TEXT NOT NULL,
  actor      TEXT NOT NULL,        -- agent|heartbeat|dreaming|user|gate
  action     TEXT NOT NULL,
  target     TEXT,
  sense_id   TEXT,
  decision   TEXT,                 -- allowed|denied|confirmation_required|redacted
  detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);

-- ─── the gate log (your research artifact, docs/architecture/06) ──────────────
CREATE TABLE IF NOT EXISTS gate_log (
  id           TEXT PRIMARY KEY,
  ts           TEXT NOT NULL,
  gate         TEXT NOT NULL,      -- G1..G7
  candidate    TEXT NOT NULL,      -- prompt variant / skill / adapter id
  incumbent    TEXT,
  target_delta REAL, heldout_delta REAL, latency_delta REAL,
  needle_delta REAL,               -- Law 2c: recall is a CHECKPOINT property
  decision     TEXT NOT NULL CHECK (decision IN ('accepted','rejected','quarantined')),
  rationale    TEXT
);

-- ══════════════════════════════════════════════════════════════════════════════
-- ⭐ RSC SUPERVISION — docs/architecture/13 §4.1
--
-- GDN-2/EDA expose three gate branches. The Memory Compiler already computes
-- three matching signals. These two tables are where that alignment is stored.
-- They are the ONLY thing in Phase 1 you cannot reconstruct later.
-- ══════════════════════════════════════════════════════════════════════════════

-- Salience labels -> supervise the channel-wise WRITE gate w_t.
-- A span is positive iff it produced an S2 fact. Labels come free from the
-- compiler's own output; no human annotation, no cloud teacher.
CREATE TABLE IF NOT EXISTS supervision_spans (
  id          TEXT PRIMARY KEY,
  trace_id    TEXT NOT NULL,
  -- trace_file is NULLABLE: a fact that arrives with a quote but no trace (CLI
  -- write, hand-typed Markdown line, import) has no file to anchor to. Those spans
  -- carry their text inline in `span_text` instead. Without this, every
  -- hand-authored fact — the highest-confidence population in the store — would
  -- silently contribute zero supervision to the RSC write gate.
  trace_file  TEXT,                   -- memory/traces/YYYY-MM-DD.jsonl
  char_start  INTEGER NOT NULL,
  char_end    INTEGER NOT NULL,
  label       INTEGER NOT NULL,       -- 1 = produced a fact, 0 = sampled negative
  fact_id     TEXT,                   -- which fact it produced (NULL for negatives)
  domain      TEXT,
  span_text   TEXT,                   -- inline mode only; NULL when trace-anchored
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_span_trace ON supervision_spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_span_label ON supervision_spans(label);

-- Retraction pairs -> supervise the INDEPENDENTLY-ADDRESSED erase e_t (EDA).
-- Every correction yields (key_old, key_new) with key_old != key_new. That
-- inequality is exactly why a write-anchored erase (GDN-2) cannot reach the
-- stale association, and exactly what L_erase trains.
CREATE TABLE IF NOT EXISTS supervision_pairs (
  id            TEXT PRIMARY KEY,
  fact_old_id   TEXT NOT NULL,
  fact_new_id   TEXT NOT NULL,
  key_old       TEXT NOT NULL,        -- "my address"
  key_new       TEXT NOT NULL,        -- "my new address"
  value_old     TEXT NOT NULL,
  value_new     TEXT NOT NULL,
  trace_old_id  TEXT,
  trace_new_id  TEXT,
  span_old_id   TEXT REFERENCES supervision_spans(id) ON DELETE SET NULL,
  span_new_id   TEXT REFERENCES supervision_spans(id) ON DELETE SET NULL,
  address_distinct INTEGER NOT NULL DEFAULT 0,  -- key_old != key_new (the EDA case)
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pair_old ON supervision_pairs(fact_old_id);
CREATE INDEX IF NOT EXISTS idx_pair_addr ON supervision_pairs(address_distinct);

-- ─── telemetry (docs/architecture/00 §8) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS ledger_telemetry (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  ts             TEXT NOT NULL,
  turn_id        TEXT,
  budget_tokens  INTEGER NOT NULL,
  spent_tokens   INTEGER NOT NULL,
  reserve_tokens INTEGER NOT NULL,
  cache_hit      INTEGER,
  prefix_stable  INTEGER,             -- consecutive turns with an identical prefix
  overflow       TEXT,                -- which rung fired, or NULL
  slots          TEXT NOT NULL,       -- JSON: {slot: [used, budget]}
  ledger_hash    TEXT NOT NULL
);
