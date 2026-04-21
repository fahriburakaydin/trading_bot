"""
Shared utilities for all LangGraph agents.

- extract_json:      Robust JSON extraction from LLM response text
- run_on_main_loop:  Dispatch an async coroutine onto the main event loop
                     from APScheduler executor threads
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger


# ── JSON extraction ──────────────────────────────────────────────────────────


def extract_json(text: str) -> Any:
    """Extract the first JSON object or array from an LLM response string.

    Strategy (fast path first):
      1. Try json.loads on the raw text (LLM sometimes returns pure JSON)
      2. Try extracting from ```json ... ``` fences
      3. Forward-scan for first { or [ and find the matching close bracket

    Raises ValueError if no valid JSON is found.
    """
    # 1. Fast path: entire response is valid JSON
    stripped = text.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 2. Fenced code block
    fence_match = re.search(r"```(?:json)?\s*([\[{].*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Forward-scan bracket finder
    pairs = {"{": "}", "[": "]"}
    for start_char, end_char in pairs.items():
        idx = text.find(start_char)
        if idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(idx, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[idx : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # malformed — try next pair type

    raise ValueError(f"No JSON found in LLM response: {text[:200]!r}")


# ── Async dispatch ───────────────────────────────────────────────────────────


def run_on_main_loop(coro, timeout: float = 30) -> Any:
    """Run an async coroutine on the main event loop from an executor thread.

    All LangGraph agent nodes run in APScheduler executor threads with no
    event loop.  IBKR and Telegram calls are async, so we need to dispatch
    them onto the main event loop that ``main.py`` spins up.

    Falls back to ``asyncio.run()`` when no main loop is available (dry-run,
    tests, or CLI scripts).

    Args:
        coro:    An awaitable coroutine object
        timeout: Max seconds to wait for the result (default 30)

    Returns:
        The coroutine's return value

    Raises:
        TimeoutError: If the coroutine doesn't complete within *timeout*
        RuntimeError: If the coroutine raises an exception
    """
    try:
        from tools.market_data import _main_loop
    except ImportError:
        _main_loop = None

    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
        return future.result(timeout=timeout)

    # Fallback: no shared loop (dry-run, tests, CLI scripts)
    return asyncio.run(coro)
