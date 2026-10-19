"""SQLite-backed session store and long-term memory with FTS5 full-text search.

Local SQLite calls take microseconds, so this module is synchronous on purpose: the
agent awaits model and tool I/O, not the disk. A single connection is shared behind a
lock, which keeps the store correct under the server's concurrent sessions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from friday.types import Message

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New session',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT NOT NULL DEFAULT '[]',
    tool_call_id TEXT,
    name TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    created_at TEXT NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, session_id UNINDEXED, tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, tags, tokenize='porter unicode61'
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._fts = True
        self._init_schema()

    # ------------------------------------------------------------- lifecycle
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            current = self._meta("schema_version")
            if current is None:
                try:
                    self._conn.executescript(_FTS_SCHEMA)
                except sqlite3.OperationalError:
                    self._fts = False
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                if not self._fts:
                    self._set_meta("fts", "0")
            else:
                self._fts = self._meta("fts") != "0"
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # --------------------------------------------------------------- sessions
    def create_session(self, title: str = "New session", meta: dict[str, Any] | None = None) -> str:
        session_id = new_id()
        stamp = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at, meta) VALUES(?,?,?,?,?)",
                (session_id, title[:120] or "New session", stamp, stamp, json.dumps(meta or {})),
            )
            self._conn.commit()
        return session_id

    def ensure_session(self, session_id: str | None, title: str = "New session") -> str:
        if session_id and self.get_session(session_id):
            return session_id
        return self.create_session(title)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row:
                return None
            count = self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id=?", (session_id,)
            ).fetchone()["n"]
        data = dict(row)
        data.pop("meta", None)
        data["message_count"] = count
        return data

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count, "
                "(SELECT content FROM messages m WHERE m.session_id = s.id AND m.role='user' "
                " ORDER BY seq LIMIT 1) AS preview "
                "FROM sessions s ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item.pop("meta", None)
            preview = item.get("preview") or ""
            item["preview"] = " ".join(preview.split())[:140]
            out.append(item)
        return out

    def rename_session(self, session_id: str, title: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                (title[:120], _now(), session_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM messages WHERE session_id=?", (session_id,)).fetchall()
            if self._fts:
                for row in rows:
                    self._conn.execute("DELETE FROM messages_fts WHERE rowid=?", (row["id"],))
            cur = self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def touch_session(self, session_id: str, title: str | None = None) -> None:
        with self._lock:
            if title:
                self._conn.execute(
                    "UPDATE sessions SET updated_at=?, title=? WHERE id=?", (_now(), title[:120], session_id)
                )
            else:
                self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id))
            self._conn.commit()

    # --------------------------------------------------------------- messages
    def add_message(self, session_id: str, message: Message) -> int:
        stamp = _now()
        payload = json.dumps(
            [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls]
        )
        with self._lock:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages WHERE session_id=?",
                (session_id,),
            ).fetchone()["next"]
            cur = self._conn.execute(
                "INSERT INTO messages("
                "session_id, seq, role, content, tool_calls, tool_call_id, name, meta, created_at"
                ") VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    seq,
                    message.role,
                    message.content,
                    payload,
                    message.tool_call_id,
                    message.name,
                    json.dumps(message.meta),
                    stamp,
                ),
            )
            if self._fts and message.content.strip():
                self._conn.execute(
                    "INSERT INTO messages_fts(rowid, content, session_id) VALUES(?,?,?)",
                    (cur.lastrowid, message.content, session_id),
                )
            self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (stamp, session_id))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_messages(self, session_id: str, limit: int = 200) -> list[Message]:
        """Return the most recent `limit` messages, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id=? ORDER BY seq DESC LIMIT ?) "
                "ORDER BY seq ASC",
                (session_id, limit),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        message = Message.from_dict(
            {
                "role": row["role"],
                "content": row["content"],
                "tool_calls": json.loads(row["tool_calls"] or "[]"),
                "tool_call_id": row["tool_call_id"],
                "name": row["name"],
                "meta": json.loads(row["meta"] or "{}"),
            }
        )
        message.meta["created_at"] = row["created_at"]
        message.meta["db_id"] = row["id"]
        return message

    def search_messages(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        with self._lock:
            if self._fts:
                try:
                    rows = self._conn.execute(
                        "SELECT m.session_id, m.role, m.content, m.created_at, s.title, "
                        "snippet(messages_fts, 0, '[', ']', '…', 12) AS snippet "
                        "FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                        "JOIN sessions s ON s.id = m.session_id "
                        "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
                        (_fts_query(query), limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = self._like_search(query, limit)
            else:
                rows = self._like_search(query, limit)
        return [dict(row) for row in rows]

    def _like_search(self, query: str, limit: int) -> list[sqlite3.Row]:
        like = f"%{query}%"
        return self._conn.execute(
            "SELECT m.session_id, m.role, m.content, m.created_at, s.title, m.content AS snippet "
            "FROM messages m JOIN sessions s ON s.id = m.session_id "
            "WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT ?",
            (like, limit),
        ).fetchall()

    # ------------------------------------------------------------- long-term
    def remember(self, text: str, tags: str = "", session_id: str | None = None) -> int:
        text = text.strip()
        if not text:
            raise ValueError("cannot store an empty memory")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories(text, tags, session_id, created_at) VALUES(?,?,?,?)",
                (text[:4000], tags[:300], session_id, _now()),
            )
            if self._fts:
                self._conn.execute(
                    "INSERT INTO memories_fts(rowid, text, tags) VALUES(?,?,?)",
                    (cur.lastrowid, text, tags),
                )
            self._conn.commit()
            return int(cur.lastrowid)

    def recall(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        query = (query or "").strip()
        with self._lock:
            if query and self._fts:
                try:
                    rows = self._conn.execute(
                        "SELECT f.rowid AS id, m.text, m.tags, m.created_at, "
                        "snippet(memories_fts, 0, '[', ']', '…', 16) AS snippet "
                        "FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                        "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                        (_fts_query(query), limit),
                    ).fetchall()
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            rows = self._conn.execute(
                "SELECT id, text, tags, created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            if self._fts:
                self._conn.execute("DELETE FROM memories_fts WHERE rowid=?", (memory_id,))
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def memory_block(self, limit: int = 12) -> str:
        memories = self.recall(limit=limit)
        if not memories:
            return ""
        return "\n".join(f"- {item['text']}" for item in reversed(memories))

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            sessions = self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
            messages = self._conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
            memories = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "sessions": sessions,
            "messages": messages,
            "memories": memories,
            "db_bytes": size,
            "fts": self._fts,
        }


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH query (quoted terms, OR-joined)."""
    terms = [t for t in "".join(c if c.isalnum() or c.isspace() else " " for c in query).split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{term}"' for term in terms[:12])
