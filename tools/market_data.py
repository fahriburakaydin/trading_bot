"""
LangChain tool wrappers for market data fetching.

Wraps ``core.broker.market_data`` functions so they can be used as tools
inside LangGraph agent nodes.  The Analyst Agent uses these to pull OHLCV
data before running technical analysis.

These tools reuse the shared IB connection registered via ``set_ib()``.
All async IB calls are dispatched onto the main event loop from the
thread-executor context that LangGraph uses to run agent nodes.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
from langchain_core.tools import tool
from loguru import logger

from core.config import cfg

# ── Shared IB connection registry ─────────────────────────────────────────────

_ib = None          # ib_insync.IB instance
_main_loop = None   # the asyncio event loop running in main.py


def set_ib(ib, loop: asyncio.AbstractEventLoop) -> None:
    """Register the shared IB connection and its event loop.

    Call once at startup (after IBKRClient.connect()) before any agent runs.
    """
    global _ib, _main_loop
    _ib = ib
    _main_loop = loop
    logger.info("[market_data] Shared IB connection registered")


def _run_coro(coro):
    """Run a coroutine on the main event loop from any thread.

    Uses run_coroutine_threadsafe so we never create a second event loop or
    call asyncio.run() while one is already running.
    """
    if _main_loop is None or not _main_loop.is_running():
        raise RuntimeError(
            "No main event loop registered. Call tools.market_data.set_ib() at startup."
        )
    future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
    return future.result(timeout=120)


def _df_to_json(df: pd.DataFrame) -> str:
    """Serialise an OHLCV DataFrame to a compact JSON string."""
    return df.reset_index().to_json(orient="records", date_format="iso")


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool
def fetch_ohlcv(
    interval: str = "1h",
    lookback_days: int = 30,
) -> str:
    """Fetch OHLCV candlestick data for the active asset from IBKR.

    Args:
        interval: Bar granularity — one of '1m', '5m', '15m', '30m', '1h', '4h', '1d'.
        lookback_days: Number of calendar days of history to fetch.

    Returns:
        JSON string of OHLCV records with columns: datetime, open, high, low, close, volume.
    """
    from core.broker.market_data import build_contract, fetch_ohlcv as _fetch

    if _ib is None:
        raise RuntimeError("IB connection not registered. Call set_ib() at startup.")

    logger.debug(f"[fetch_ohlcv] interval={interval} lookback_days={lookback_days}")
    contract = build_contract()
    df = _run_coro(_fetch(_ib, contract, interval=interval, lookback_days=lookback_days))
    return _df_to_json(df)


@tool
def fetch_multi_timeframe(
    timeframes: list[str] | None = None,
) -> str:
    """Fetch OHLCV data across multiple timeframes simultaneously.

    Args:
        timeframes: List of interval strings. Defaults to ['1 hour', '4 hours', '1 day'].

    Returns:
        JSON object mapping each timeframe label to its OHLCV records list.
    """
    from core.broker.market_data import build_contract, fetch_ohlcv as _fetch

    if _ib is None:
        raise RuntimeError("IB connection not registered. Call set_ib() at startup.")

    if timeframes is None:
        timeframes = ["1 hour", "4 hours", "1 day"]

    # Map human-readable labels used by the analyst to bar-size strings
    _label_to_interval = {
        "1 hour": "1h",
        "4 hours": "4h",
        "1 day": "1d",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    logger.debug(f"[fetch_multi_timeframe] timeframes={timeframes}")

    async def _gather():
        contract = build_contract()
        tasks = [
            _fetch(_ib, contract, interval=_label_to_interval.get(tf, tf))
            for tf in timeframes
        ]
        dfs = await asyncio.gather(*tasks)
        return dict(zip(timeframes, dfs))

    results: dict[str, pd.DataFrame] = _run_coro(_gather())
    return json.dumps(
        {tf: json.loads(_df_to_json(df)) for tf, df in results.items()}
    )
