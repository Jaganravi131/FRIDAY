"""Storage layer: SQLite artifact DB, FTS5, vectors."""

from .db import (
    HAS_VEC_EXT,
    connect,
    apply_schema,
    fts_delete,
    fts_query,
    fts_search,
    fts_upsert,
    vec_delete,
    vec_search,
    vec_upsert,
)

__all__ = [
    "HAS_VEC_EXT",
    "connect",
    "apply_schema",
    "fts_delete",
    "fts_query",
    "fts_search",
    "fts_upsert",
    "vec_delete",
    "vec_search",
    "vec_upsert",
]
