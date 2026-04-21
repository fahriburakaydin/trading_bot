"""
Telegram command listener for manual bot control.

Registers three slash commands via python-telegram-bot's Application polling:

  /improve     — trigger the Evaluator Agent (strategy reflection + KB update)
  /show-alarms — display all active alarms from the database
  /set-alarms  — run the Analyst Agent (cancels old alarms, sets new ones)

Designed to run alongside the existing asyncio event loop in main.py.
If TELEGRAM_BOT_TOKEN is not set the handler initialises as a no-op.

Usage:
    handler = TelegramCommandHandler()
    await handler.start()   # call once at boot
    ...
    await handler.stop()    # call on shutdown
"""

from __future__ import annotations

import asyncio
from datetime import timezone

from loguru import logger
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from core.config import cfg, secrets
from core.memory.database import get_session
from core.memory.models import Alarm


class TelegramCommandHandler:
    """Polls Telegram for slash commands and dispatches them to bot agents."""

    def __init__(self) -> None:
        self._app: Application | None = None
        self._improve_lock = asyncio.Lock()
        self._set_alarms_lock = asyncio.Lock()

        token = secrets.telegram_bot_token
        if not token:
            logger.warning(
                "[telegram_commands] TELEGRAM_BOT_TOKEN not set — command listener disabled"
            )
            return

        self._app = (
            ApplicationBuilder()
            .token(token)
            .build()
        )

        self._app.add_handler(CommandHandler("improve", self._cmd_improve))
        self._app.add_handler(CommandHandler("show_alarms", self._cmd_show_alarms))
        self._app.add_handler(CommandHandler("set_alarms", self._cmd_set_alarms))

        logger.info("[telegram_commands] Command handlers registered: /improve /show_alarms /set_alarms")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise and start polling. Safe to call when token is missing."""
        if self._app is None:
            return
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("[telegram_commands] Polling started")

    async def stop(self) -> None:
        """Stop polling and shut down gracefully."""
        if self._app is None:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("[telegram_commands] Polling stopped")

    # ── Command handlers ──────────────────────────────────────────────────────

    async def _cmd_improve(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/improve — run the Evaluator Agent."""
        if self._improve_lock.locked():
            await update.message.reply_text(
                "Strategy evaluation is already running. Please wait for it to finish."
            )
            return

        async with self._improve_lock:
            await update.message.reply_text(
                "Running strategy evaluation\u2026 this may take a few minutes."
            )
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, _run_evaluator)
                kb_added = result.get("kb_entries_added", 0)
                await update.message.reply_text(
                    f"Evaluation complete. {kb_added} new knowledge base rule(s) added."
                )
            except Exception as exc:
                logger.error(f"[telegram_commands] /improve failed: {exc}", exc_info=True)
                await update.message.reply_text("Evaluation failed. Check the bot logs for details.")

    async def _cmd_show_alarms(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/show-alarms — list active alarms from the database."""
        try:
            with get_session() as session:
                alarms: list[Alarm] = (
                    session.query(Alarm)
                    .filter(
                        Alarm.asset == cfg.asset.symbol,
                        Alarm.status == "active",
                    )
                    .order_by(Alarm.trigger_price)
                    .all()
                )

            if not alarms:
                await update.message.reply_text(
                    f"No active alarms for {cfg.asset.symbol}."
                )
                return

            lines = [f"<b>Active Alarms — {cfg.asset.symbol}</b> ({len(alarms)} total)\n"]
            for a in alarms:
                expires = (
                    a.expires_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    if a.expires_at
                    else "no expiry"
                )
                direction_emoji = "\U0001f7e2" if a.action == "LONG" else "\U0001f534"
                lines.append(
                    f"{direction_emoji} <b>{a.action}</b> @ <code>{a.trigger_price:.5f}</code>\n"
                    f"  SL: <code>{a.stop_loss:.5f}</code> | TP: <code>{a.target_price:.5f}</code> | R:R <code>{a.risk_reward:.1f}</code>\n"
                    f"  Conf: <code>{a.confidence:.0%}</code> | {a.timeframe} | expires {expires}"
                )

            await update.message.reply_text("\n\n".join(lines), parse_mode="HTML")
        except Exception as exc:
            logger.error(f"[telegram_commands] /show_alarms failed: {exc}", exc_info=True)
            await update.message.reply_text("Failed to fetch alarms. Check the bot logs for details.")

    async def _cmd_set_alarms(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/set-alarms — run the Analyst Agent to generate fresh alarms."""
        if self._set_alarms_lock.locked():
            await update.message.reply_text(
                "Analysis is already running. Please wait for it to finish."
            )
            return

        async with self._set_alarms_lock:
            await update.message.reply_text(
                "Running technical analysis\u2026 setting new alarms."
            )
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, _run_analyst)
                saved = result.get("alarms_saved", 0)
                await update.message.reply_text(
                    f"Analysis complete. {saved} alarm(s) set for {cfg.asset.symbol}."
                )
            except Exception as exc:
                logger.error(f"[telegram_commands] /set_alarms failed: {exc}", exc_info=True)
                await update.message.reply_text("Analysis failed. Check the bot logs for details.")


# ── Agent runners (blocking — called via run_in_executor) ─────────────────────


def _run_evaluator() -> dict:
    from agents.evaluator.agent import EvaluatorAgent
    return EvaluatorAgent().run()


def _run_analyst() -> dict:
    from agents.analyst.agent import AnalystAgent
    return AnalystAgent().run()
