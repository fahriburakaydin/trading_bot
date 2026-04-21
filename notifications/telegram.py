"""
Telegram notification sender.

Thin wrapper around python-telegram-bot's Bot.send_message.
Designed to be imported by any module that needs to send alerts.

Usage:
    from notifications.telegram import TelegramNotifier
    tg = TelegramNotifier()
    await tg.send("Trade opened: LONG BTC @ 95,000")
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from loguru import logger
from telegram import Bot
from telegram.error import TelegramError

from core.config import secrets


class TelegramNotifier:
    """Async Telegram message sender. Safe to call without credentials (logs a warning)."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        self._token = bot_token or secrets.telegram_bot_token
        self._chat_id = chat_id or secrets.telegram_chat_id
        self._bot: Bot | None = None

        if not self._token or not self._chat_id:
            logger.warning(
                "Telegram credentials not set — notifications will be logged only. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable."
            )

    def _get_bot(self) -> Bot | None:
        if not self._token:
            return None
        if self._bot is None:
            self._bot = Bot(token=self._token)
        return self._bot

    async def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to the configured Telegram chat.

        Args:
            text:       Message text (HTML markup supported by default)
            parse_mode: "HTML" or "Markdown"

        Returns:
            True if message was sent, False if credentials missing or error occurred
        """
        bot = self._get_bot()
        if bot is None or not self._chat_id:
            logger.info(f"[Telegram stub] {text}")
            return False

        try:
            await bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            return True
        except TelegramError as exc:
            logger.error(f"Telegram send failed: {exc}")
            return False

    def send_sync(self, text: str) -> bool:
        """Synchronous wrapper for non-async contexts.

        Uses run_coroutine_threadsafe when a main loop exists (reliable delivery
        with timeout), falls back to asyncio.run() when no loop is running.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self.send(text), loop)
                try:
                    return future.result(timeout=10)
                except Exception as exc:
                    logger.error(f"Telegram send_sync future failed: {exc}")
                    return False
            # No running loop (e.g. executor thread) — spin up a fresh one
            return asyncio.run(self.send(text))
        except Exception as exc:
            logger.error(f"Telegram send_sync failed: {exc}")
            return False


@lru_cache(maxsize=1)
def get_notifier() -> TelegramNotifier:
    """Return the process-wide singleton notifier."""
    return TelegramNotifier()
