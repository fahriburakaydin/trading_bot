#!/usr/bin/env python3
"""
Trading Bot — main entrypoint.

Boot sequence:
  1. Initialise database (create tables)
  2. Connect to IB Gateway (paper port 7497)
  3. Start APScheduler (Research 10:00, Analyst 10:30, Evaluator Sun 20:00)
  4. Start Price Monitor (polls live quote, fires Trader Agent on alarm hits)
  5. Block until SIGINT / SIGTERM

Usage:
    python main.py
    python main.py --dry-run   # skips IBKR connection; useful for testing scheduler
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from loguru import logger

from core.config import cfg
from core.memory.database import init_db, engine
from core.scheduler.jobs import start_scheduler, stop_scheduler


# ── Logging setup ─────────────────────────────────────────────────────────────


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.logging.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        "logs/trading_bot.log",
        level=cfg.logging.level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
        compression="zip",
    )


# ── Startup notification ──────────────────────────────────────────────────────


async def _send_startup_notification(monitor) -> None:
    """Send a Telegram message summarising bot state on startup."""
    try:
        from notifications.telegram import get_notifier
        from core.memory.models import Alarm, Trade
        from sqlalchemy.orm import Session
        from datetime import datetime, timezone

        notifier = get_notifier()

        # Current price
        price = monitor.current_price()
        price_str = f"{price:.5f}" if price else "fetching..."

        # Active alarms
        with Session(engine) as s:
            alarms = s.query(Alarm).filter(Alarm.status == "active").all()
            recent_trades = s.query(Trade).order_by(Trade.id.desc()).limit(3).all()

        alarm_lines = []
        for a in alarms:
            alarm_lines.append(
                f"  {'🟢' if a.action == 'LONG' else '🔴'} {a.action} @ {a.trigger_price:.5f} "
                f"(SL {a.stop_loss:.5f} / TP {a.target_price:.5f}) RR {a.risk_reward:.1f}"
            )

        trade_lines = []
        for t in recent_trades:
            pnl = f"+{t.pnl:.2f}" if t.pnl and t.pnl > 0 else (f"{t.pnl:.2f}" if t.pnl else "open")
            trade_lines.append(f"  {t.direction} {t.entry_price:.5f} → {pnl}")

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"🤖 <b>Trading Bot Online</b> — {now_utc}",
            f"Asset: <b>{cfg.asset.symbol}</b> | Price: <code>{price_str}</code>",
            "",
        ]
        if alarm_lines:
            lines.append(f"<b>Active alarms ({len(alarms)}):</b>")
            lines.extend(alarm_lines)
        else:
            lines.append("📭 No active alarms — Analyst runs at 10:30 IST to set levels")

        if trade_lines:
            lines.append("")
            lines.append("<b>Recent trades:</b>")
            lines.extend(trade_lines)

        lines += [
            "",
            "⏰ Research: 10:00 IST | Analyst: 10:30 IST | Evaluator: Sun 20:00 IST",
        ]

        await notifier.send("\n".join(lines), parse_mode="HTML")
        logger.info("[main] Startup notification sent to Telegram")
    except Exception as exc:
        logger.warning(f"[main] Startup notification failed: {exc}")


# ── Analyst bootstrap ─────────────────────────────────────────────────────────


async def _bootstrap_analyst_if_needed() -> None:
    """
    Run the Analyst Agent on startup when alarms are absent or stale.

    Stale means: the newest active alarm was created more than
    `alarm_expiry_hours` ago — i.e. the bot was down long enough that the
    alarm levels are no longer relevant to current market structure.
    """
    from core.memory.models import Alarm
    from sqlalchemy.orm import Session
    from datetime import datetime, timezone

    stale_threshold_hours = cfg.trading.price_monitor.alarm_expiry_hours
    now = datetime.now(timezone.utc)

    with Session(engine) as s:
        newest_alarm = (
            s.query(Alarm)
            .filter(Alarm.status == "active", Alarm.asset == cfg.asset.symbol)
            .order_by(Alarm.created_at.desc())
            .first()
        )

    if newest_alarm is not None:
        created_at = newest_alarm.created_at
        # created_at may be naive (UTC) — normalise for comparison
        if created_at.tzinfo is None:
            from datetime import timezone as tz
            created_at = created_at.replace(tzinfo=tz.utc)
        age_hours = (now - created_at).total_seconds() / 3600
        if age_hours <= stale_threshold_hours:
            logger.info(
                f"[main] Active alarm(s) are fresh ({age_hours:.1f}h old, "
                f"threshold={stale_threshold_hours}h) — skipping Analyst bootstrap"
            )
            return
        logger.warning(
            f"[main] Active alarm(s) are stale ({age_hours:.1f}h old, "
            f"threshold={stale_threshold_hours}h) — re-running Analyst to refresh levels"
        )
    else:
        logger.info("[main] No active alarms — running Analyst Agent to populate levels...")

    try:
        from agents.analyst.agent import AnalystAgent
        from notifications.telegram import get_notifier

        await get_notifier().send(
            "🔍 <b>Running Analyst Agent</b> to set price alarms... "
            "(this takes 1–3 minutes)",
            parse_mode="HTML",
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: AnalystAgent().run(),
        )
        n = result.get("alarms_saved", 0)
        logger.info(f"[main] Analyst bootstrap complete — {n} alarm(s) set")
    except Exception as exc:
        logger.exception(f"[main] Analyst bootstrap failed: {exc}")


# ── Trader Agent callback ─────────────────────────────────────────────────────


def _make_trader_callback(ib):
    """
    Return an async callback suitable for PriceMonitor.on_trigger().

    Instantiates a fresh TraderAgent per alarm so each execution is isolated.
    """
    async def _on_alarm_triggered(alarm_id: int, price: float) -> None:
        logger.info(f"[main] Alarm {alarm_id} triggered @ {price:.2f} — invoking TraderAgent")
        try:
            from agents.trader.agent import TraderAgent
            agent = TraderAgent(ib=ib)
            # Run in executor so blocking LangGraph call doesn't stall the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: agent.run(alarm_id=alarm_id, trigger_price=price),
            )
            logger.info(
                f"[main] TraderAgent done — "
                f"approved={result.get('approved')} "
                f"trade_id={result.get('trade_id')}"
            )
        except Exception as exc:
            logger.error(f"[main] TraderAgent error for alarm {alarm_id}: {exc}")

    return _on_alarm_triggered


# ── Main async loop ───────────────────────────────────────────────────────────


async def run(dry_run: bool = False) -> None:
    """
    Full async boot sequence.

    dry_run=True skips IBKR connection and price monitoring — useful for
    verifying scheduler registration and DB init without IB Gateway running.
    """
    _configure_logging()
    logger.info("=" * 60)
    logger.info(f"Trading Bot starting — asset={cfg.asset.symbol} dry_run={dry_run}")
    logger.info("=" * 60)

    # ── 1. Database ───────────────────────────────────────────────────────────
    init_db()

    # ── 2. Scheduler ─────────────────────────────────────────────────────────
    scheduler = start_scheduler()

    # ── 3. Telegram command listener ──────────────────────────────────────────
    from notifications.telegram_commands import TelegramCommandHandler
    cmd_handler = TelegramCommandHandler()
    await cmd_handler.start()

    monitor = None
    client = None

    if not dry_run:
        # ── 3. IBKR connection ────────────────────────────────────────────────
        from core.broker.client import IBKRClient
        client = IBKRClient()
        await client.connect()

        # ── 4. Register shared IB connection for tool use ─────────────────────
        from tools.market_data import set_ib as _set_ib
        _set_ib(client.ib, asyncio.get_event_loop())

        # ── 5. Price Monitor ──────────────────────────────────────────────────
        from core.monitor.monitor import PriceMonitor
        monitor = PriceMonitor(client.ib)
        monitor.on_trigger(_make_trader_callback(client.ib))
        await monitor.start()

        # ── 6. Startup Telegram notification (send before bootstrap so user
        #        knows the bot is online even if Analyst takes minutes) ──────
        await asyncio.sleep(2)   # let first quote tick arrive
        await _send_startup_notification(monitor)

        # ── 7. Bootstrap: run Analyst if no active alarms exist ──────────────
        await _bootstrap_analyst_if_needed()

    logger.info("[main] Bot is running. Press Ctrl+C to stop.")

    # ── 5. Wait for shutdown signal ───────────────────────────────────────────
    stop_event = asyncio.Event()

    def _signal_handler(*_):
        logger.info("[main] Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("[main] Shutting down...")

    if monitor is not None:
        await monitor.stop()

    stop_scheduler(scheduler)
    await cmd_handler.stop()

    if client is not None:
        await client.disconnect()

    logger.info("[main] Bot stopped cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start scheduler only — skip IBKR connection and price monitoring",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(dry_run=args.dry_run))
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
