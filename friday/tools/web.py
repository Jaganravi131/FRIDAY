"""Web access with SSRF protection: page fetch and search."""

from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import socket
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from friday.tools.base import Tool, ToolContext, ToolError
from friday.types import ToolResult

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
MAX_BYTES = 1_500_000
MAX_TEXT = 12_000
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 FRIDAY/0.1"
)


def _allow_private() -> bool:
    return os.environ.get("FRIDAY_ALLOW_PRIVATE_FETCH", "").strip().lower() in {"1", "true", "yes"}


def assert_public_url(url: str) -> None:
    """Reject non-http(s), odd ports and any host resolving to a private range."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ToolError(f"Only http/https URLs are allowed (got '{parsed.scheme or 'none'}').")
    if not parsed.hostname:
        raise ToolError("URL has no hostname.")
    if parsed.port not in ALLOWED_PORTS:
        raise ToolError(f"Port {parsed.port} is not allowed; use 80 or 443.")
    if _allow_private():
        return

    host = parsed.hostname.lower()
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".local"):
        raise ToolError("Refusing to fetch a local/internal hostname.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ToolError(f"Could not resolve host '{host}': {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ToolError(f"Refusing to fetch internal address {address} ({host}).")


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch a public web page or JSON endpoint and return readable text. Follows up to three "
        "redirects, strips HTML, and blocks private/internal addresses."
    )
    risk = "low"
    timeout = 45.0
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Public http(s) URL."},
            "max_chars": {"type": "integer", "description": "Returned text limit (default 8000)."},
        },
        "required": ["url"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(args["url"]).strip()
        limit = max(500, min(int(args.get("max_chars", 8000)), MAX_TEXT))
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json,text/plain,*/*"}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False
        ) as client:
            current = url
            for _hop in range(4):
                assert_public_url(current)
                try:
                    response = await client.get(current, headers=headers)
                except httpx.HTTPError as exc:
                    raise ToolError(f"Fetch failed: {exc}") from exc
                if response.is_redirect and response.headers.get("location"):
                    current = str(response.url.join(response.headers["location"]))
                    continue
                break
            else:
                raise ToolError("Too many redirects.")

        if response.status_code >= 400:
            raise ToolError(f"HTTP {response.status_code} from {current}")

        content_type = response.headers.get("content-type", "")
        body = response.content[:MAX_BYTES]
        if "json" in content_type:
            try:
                text = json.dumps(response.json(), indent=2, ensure_ascii=False)
            except ValueError:
                text = body.decode("utf-8", "replace")
        else:
            text = body.decode(response.encoding or "utf-8", "replace")
            if "html" in content_type or text.lstrip()[:1] == "<":
                text = html_to_text(text)

        text = text.strip()
        truncated = len(text) > limit
        text = text[:limit]
        header = f"{current}\nHTTP {response.status_code} · {content_type.split(';')[0] or 'unknown'}"
        if truncated:
            header += f" · truncated to {limit} chars"
        return ToolResult(
            ok=True, output=f"{header}\n\n{text}", data={"url": current, "status": response.status_code}
        )


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and return titles, URLs and snippets. Uses Tavily when TAVILY_API_KEY is "
        "set, otherwise DuckDuckGo's HTML endpoint. Results can be fetched with fetch_url."
    )
    risk = "low"
    timeout = 45.0
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {"type": "integer", "description": "1-10, default 5."},
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("Empty search query.")
        count = max(1, min(int(args.get("max_results", 5)), 10))

        tavily_key = ctx.settings.tavily_api_key
        if tavily_key:
            results = await self._tavily(query, count, tavily_key)
            engine = "tavily"
        else:
            results = await self._duckduckgo(query, count)
            engine = "duckduckgo"

        if not results:
            raise ToolError(
                "Search returned no results. The network may be restricted, or the query was too narrow. "
                "Set TAVILY_API_KEY for a more reliable engine."
            )
        lines = [
            f"{index}. {item['title']}\n   {item['url']}\n   {item['snippet']}"
            for index, item in enumerate(results, 1)
        ]
        return ToolResult(
            ok=True,
            output=f'{engine} · {len(results)} result(s) for "{query}"\n\n' + "\n\n".join(lines),
            data={"engine": engine, "results": results},
        )

    async def _tavily(self, query: str, count: int, key: str) -> list[dict[str, str]]:
        payload = {"api_key": key, "query": query, "max_results": count, "search_depth": "basic"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post("https://api.tavily.com/search", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ToolError(f"Tavily search failed: {exc}") from exc
        data = response.json()
        return [
            {
                "title": str(item.get("title") or item.get("url") or "untitled"),
                "url": str(item.get("url") or ""),
                "snippet": " ".join(str(item.get("content") or "").split())[:300],
            }
            for item in data.get("results", [])
        ]

    async def _duckduckgo(self, query: str, count: int) -> list[dict[str, str]]:
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
            try:
                response = await client.post(
                    "https://html.duckduckgo.com/html/", data={"q": query, "kl": "us-en"}
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ToolError(f"DuckDuckGo search failed: {exc}") from exc

        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.S | re.I,
        )
        results: list[dict[str, str]] = []
        for href, title, snippet in pattern.findall(response.text):
            url = _unwrap_ddg(href)
            if not url:
                continue
            results.append(
                {
                    "title": _strip_tags(title)[:180] or url,
                    "url": url,
                    "snippet": _strip_tags(snippet)[:300],
                }
            )
            if len(results) >= count:
                break
        return results


def _unwrap_ddg(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    if parsed.scheme in ALLOWED_SCHEMES and parsed.netloc:
        return href
    return ""


def html_to_text(source: str) -> str:
    """A pragmatic HTML-to-text pass: no dependencies, good enough for model context."""
    source = re.sub(r"(?is)<(script|style|noscript|svg|iframe)[^>]*>.*?</\1>", " ", source)
    source = re.sub(r"(?is)<!--.*?-->", " ", source)
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", source)
    title = html.unescape(_strip_tags(title_match.group(1))).strip() if title_match else ""
    source = re.sub(
        r"(?i)</?(p|div|section|article|header|footer|li|tr|h[1-6]|br|table|ul|ol)[^>]*>", "\n", source
    )
    source = re.sub(r"(?s)<[^>]+>", " ", source)
    text = html.unescape(source)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return f"# {title}\n\n{text}" if title else text


def _strip_tags(source: str) -> str:
    return " ".join(html.unescape(re.sub(r"(?s)<[^>]+>", " ", source or "")).split())
