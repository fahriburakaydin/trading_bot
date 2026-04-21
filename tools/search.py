"""
Search tools for the Research Agent.

Two implementations:
- brave_search   — primary, uses Brave Search API (BRAVE_API_KEY env var required)
- duckduckgo_search — fallback, uses duckduckgo-search package (no API key needed)

Both are decorated with @tool so they plug directly into LangChain / LangGraph agents.
The module-level ``web_search`` tool automatically routes: Brave when the key is present,
DuckDuckGo otherwise.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from loguru import logger

_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_MAX_RESULTS = 10


# ── Brave Search ──────────────────────────────────────────────────────────────


def _call_brave(query: str, max_results: int) -> list[dict[str, str]]:
    """Call the Brave Search REST API and return normalised result dicts."""
    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("BRAVE_API_KEY environment variable is not set")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params: dict[str, Any] = {
        "q": query,
        "count": min(max_results, 20),  # Brave API caps at 20
        "text_decorations": False,
        "search_lang": "en",
    }

    response = requests.get(_BRAVE_API_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
        )
    return results[:max_results]


@tool
def brave_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Search the web using the Brave Search API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 10, max 20).

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    logger.debug(f"[brave_search] query={query!r} max_results={max_results}")
    return _call_brave(query, max_results)


# ── DuckDuckGo Search ────────────────────────────────────────────────────────


def _call_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    """Query DuckDuckGo and return normalised result dicts."""
    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
    return results


@tool
def duckduckgo_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Search the web using DuckDuckGo (no API key required).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    logger.debug(f"[duckduckgo_search] query={query!r} max_results={max_results}")
    return _call_duckduckgo(query, max_results)


# ── Auto-routing tool ─────────────────────────────────────────────────────────


@tool
def web_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Search the web, preferring Brave Search and falling back to DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    if os.getenv("BRAVE_API_KEY"):
        try:
            return _call_brave(query, max_results)
        except Exception as exc:
            logger.warning(f"[web_search] Brave failed ({exc}), falling back to DuckDuckGo")

    return _call_duckduckgo(query, max_results)
