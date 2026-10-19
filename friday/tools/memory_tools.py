"""Long-term memory and conversation-history tools."""

from __future__ import annotations

from typing import Any

from friday.tools.base import Tool, ToolContext, ToolError
from friday.types import ToolResult


class RememberTool(Tool):
    name = "remember"
    description = (
        "Store a durable fact about the user, their preferences, projects or decisions in long-term "
        "memory. Use when the user says 'remember…' or shares something worth carrying across sessions."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "A self-contained sentence to remember."},
            "tags": {"type": "string", "description": "Optional comma-separated tags."},
        },
        "required": ["fact"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        fact = " ".join(str(args["fact"]).split())
        if len(fact) < 3:
            raise ToolError("That fact is too short to be useful.")
        memory_id = ctx.memory.remember(fact, tags=str(args.get("tags") or ""), session_id=ctx.session_id)
        return ToolResult(ok=True, output=f"Noted (memory #{memory_id}): {fact}", data={"id": memory_id})


class RecallMemoryTool(Tool):
    name = "recall_memory"
    description = (
        "Search long-term memory for facts the user asked you to remember. Leave query empty to list "
        "the most recent memories."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for (optional)."},
            "limit": {"type": "integer", "description": "Max results, default 8."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        limit = max(1, min(int(args.get("limit", 8)), 30))
        query = str(args.get("query") or "").strip()
        memories = ctx.memory.recall(query, limit=limit)
        if not memories:
            scope = f"matching “{query}”" if query else "yet"
            return ToolResult(ok=True, output=f"No long-term memories {scope}.", data={"count": 0})
        lines = [
            f"#{item['id']} · {item['text']}" + (f"  [{item['tags']}]" if item.get("tags") else "")
            for item in memories
        ]
        return ToolResult(
            ok=True,
            output=f"{len(memories)} memory item(s):\n" + "\n".join(lines),
            data={"count": len(memories)},
        )


class ForgetMemoryTool(Tool):
    name = "forget_memory"
    description = "Delete a long-term memory by id (the id is shown by recall_memory)."
    risk = "medium"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"id": {"type": "integer", "description": "Memory id to delete."}},
        "required": ["id"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        memory_id = int(args["id"])
        if not ctx.memory.forget(memory_id):
            raise ToolError(f"No memory with id {memory_id}.")
        return ToolResult(ok=True, output=f"Forgotten: memory #{memory_id}.", data={"id": memory_id})


class SearchHistoryTool(Tool):
    name = "search_history"
    description = (
        "Search this assistant's own past conversations (full-text) and return the best matching "
        "snippets with session ids. Use when the user refers to something discussed earlier."
    )
    risk = "low"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text to search for."},
            "limit": {"type": "integer", "description": "Max hits, default 8."},
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args["query"]).strip()
        if len(query) < 2:
            raise ToolError("Query too short.")
        limit = max(1, min(int(args.get("limit", 8)), 30))
        hits = ctx.memory.search_messages(query, limit=limit)
        if not hits:
            return ToolResult(ok=True, output=f"No past messages matched “{query}”.", data={"count": 0})
        lines = [
            f"- [{hit['session_id']}] {hit.get('title') or 'session'} · {hit['role']}: "
            f"{' '.join(str(hit.get('snippet') or hit['content']).split())[:240]}"
            for hit in hits
        ]
        return ToolResult(
            ok=True,
            output=f"{len(hits)} matching message(s):\n" + "\n".join(lines),
            data={"count": len(hits)},
        )
