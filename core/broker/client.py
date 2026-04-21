"""
IBKR connection client.

Manages async connection to IB Gateway via ib_insync, with:
- Exponential-backoff auto-reconnect (10 attempts by default)
- 60-second heartbeat coroutine that detects stale connections
- Telegram alert on disconnect/reconnect
- Paper port 7497 / live port 7496 controlled by config

Usage:
    client = IBKRClient()
    await client.connect()
    # ... use client.ib for ib_insync calls ...
    await client.disconnect()
"""

from __future__ import annotations

import asyncio
import random

from ib_insync import IB
from loguru import logger

from core.config import cfg
from notifications.telegram import get_notifier


class IBKRClient:
    """
    Thin async wrapper around ib_insync.IB with resilience features.

    Public attributes:
        ib:        The underlying IB instance (use for all ib_insync API calls)
        connected: True after a successful connection is established
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        reconnect_attempts: int | None = None,
        backoff_base: int | None = None,
    ) -> None:
        self.host = host or cfg.ibkr.host
        self.port = port or cfg.ibkr.port
        # Use a random clientId to avoid "competing session" errors from stale
        # connections left behind by unclean shutdowns of previous bot runs.
        self.client_id = client_id or random.randint(10, 99)
        self.reconnect_attempts = reconnect_attempts or cfg.ibkr.reconnect_attempts
        self.backoff_base = backoff_base or cfg.ibkr.reconnect_backoff_base

        self.ib = IB()
        self.connected = False
        self._heartbeat_task: asyncio.Task | None = None
        self._notifier = get_notifier()

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to IB Gateway and start the heartbeat loop."""
        await self.connect_with_retry()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def connect_with_retry(
        self,
        max_attempts: int | None = None,
        backoff_base: int | None = None,
        _is_reconnect: bool = False,
    ) -> None:
        """
        Attempt to connect to IB Gateway with exponential backoff.

        Raises:
            ConnectionError: if all attempts are exhausted
        """
        max_attempts = max_attempts or self.reconnect_attempts
        backoff_base = backoff_base or self.backoff_base

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    f"Connecting to IB Gateway {self.host}:{self.port} "
                    f"(clientId={self.client_id}, attempt {attempt}/{max_attempts})"
                )
                await self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=20,
                )
                self.connected = True
                account_type = 'paper' if self.port == 7497 else 'LIVE'
                logger.success(
                    f"Connected to IB Gateway — {account_type} account"
                )
                # Only send Telegram on first connect or when recovering from a disconnect.
                # Suppress duplicate messages if the heartbeat reconnects multiple times
                # in quick succession (IB Gateway paper-account idle timeouts).
                if _is_reconnect:
                    await self._notifier.send(
                        f"✅ Trading bot reconnected to IB Gateway ({account_type} port {self.port})"
                    )
                else:
                    await self._notifier.send(
                        f"🤖 Trading bot connected to IB Gateway ({account_type} port {self.port})"
                    )
                return

            except Exception as exc:
                wait = backoff_base * (2 ** (attempt - 1))  # 5, 10, 20, 40 ...
                logger.warning(
                    f"Connection attempt {attempt} failed: {exc}. "
                    f"{'Retrying in ' + str(wait) + 's' if attempt < max_attempts else 'No more retries.'}"
                )
                if attempt < max_attempts:
                    await asyncio.sleep(wait)

        self.connected = False
        msg = (
            f"IB Gateway unreachable after {max_attempts} attempts "
            f"({self.host}:{self.port}). Bot is halted."
        )
        logger.error(msg)
        await self._notifier.send(f"ALERT: {msg}")
        raise ConnectionError(msg)

    async def disconnect(self) -> None:
        """Graceful disconnect — cancels heartbeat and closes the IB connection."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False
        logger.info("Disconnected from IB Gateway")

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """
        Ping IB Gateway every 60 seconds.
        On failure: send Telegram alert and attempt reconnect.
        """
        _disconnect_notified = False
        while True:
            await asyncio.sleep(60)
            if not self.ib.isConnected():
                if not _disconnect_notified:
                    msg = "IB Gateway heartbeat lost — attempting reconnect..."
                    logger.warning(msg)
                    await self._notifier.send(f"⚠️ {msg}")
                    _disconnect_notified = True
                else:
                    logger.warning("IB Gateway still disconnected — retrying...")
                try:
                    await self.connect_with_retry(_is_reconnect=True)
                    _disconnect_notified = False  # reset after successful reconnect
                except ConnectionError:
                    # connect_with_retry already sent the alert; just stop the heartbeat
                    logger.error("Heartbeat: reconnect exhausted, stopping heartbeat loop")
                    return

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> IBKRClient:
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()
