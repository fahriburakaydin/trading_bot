"""
Price Monitor — polls IBKR every N seconds and fires alarms when price
touches a trigger level within the configured tolerance.

Flow per poll tick:
  1. Fetch current price from IBKR live quote
  2. Load all active, non-expired alarms from DB
  3. For each alarm, check if |current_price - trigger_price| / trigger_price
     <= trigger_tolerance_pct for two consecutive polls (prevents false fires)
  4. On confirmed trigger: mark alarm as triggered, emit callback, expire others

Usage:
    monitor = PriceMonitor(ib_client)
    monitor.on_trigger(my_callback)   # async callback(alarm_id, price)
    await monitor.start()             # runs until cancelled
    await monitor.stop()
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Awaitable, Callable

from loguru import logger

from core.broker.market_data import build_contract, subscribe_quote, unsubscribe_quote
from core.config import cfg
from core.memory.database import get_session
from core.memory.models import Alarm


# Type alias for the trigger callback
TriggerCallback = Callable[[int, float], Awaitable[None]]


class PriceMonitor:
    """
    Async price monitor that polls a live IBKR quote and fires registered
    callbacks when an alarm's trigger level is touched.

    Two-consecutive-poll confirmation prevents single-tick noise from firing.
    """

    def __init__(self, ib) -> None:
        """
        Args:
            ib: Connected ib_insync.IB instance (from IBKRClient.ib)
        """
        self._ib = ib
        self._poll_interval = cfg.trading.price_monitor.poll_interval_seconds
        self._tolerance = cfg.trading.price_monitor.trigger_tolerance_pct
        self._callbacks: list[TriggerCallback] = []
        self._running = False
        self._task: asyncio.Task | None = None

        # Track which alarm IDs were "near" on the previous poll
        # alarm_id → count of consecutive polls within tolerance
        self._near_counts: dict[int, int] = {}

        self._contract = build_contract()
        self._quote = None
        self._tick_count = 0         # total ticks since start
        self._null_price_streak = 0  # consecutive ticks with no price

    # ── Callback registration ─────────────────────────────────────────────────

    def on_trigger(self, callback: TriggerCallback) -> None:
        """Register an async callback to call when an alarm fires.

        Callback signature: async def callback(alarm_id: int, price: float) -> None
        """
        self._callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to live quote and begin the polling loop."""
        if self._running:
            logger.warning("[monitor] Already running")
            return

        self._quote = subscribe_quote(self._ib, self._contract)
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"[monitor] Started — polling every {self._poll_interval}s "
            f"tolerance={self._tolerance:.1%}"
        )

    async def stop(self) -> None:
        """Stop the polling loop and unsubscribe from live quote."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._quote is not None:
            unsubscribe_quote(self._ib, self._contract)
            self._quote = None

        logger.info("[monitor] Stopped")

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> PriceMonitor:
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    # ── Current price ─────────────────────────────────────────────────────────

    def current_price(self) -> float | None:
        """Return the latest mid price from the live quote, or None if not yet populated."""
        if self._quote is None:
            return None
        return self._quote.price

    # ── Poll loop ─────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main polling loop. Runs until stop() is called."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[monitor] Unexpected error in poll tick: {exc}")
            await asyncio.sleep(self._poll_interval)

    def _is_within_trading_hours(self) -> bool:
        """Check if current time is within configured trading hours."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

        tz = ZoneInfo(cfg.trading.hours.timezone)
        now = datetime.now(tz)
        day_abbr = now.strftime("%a")  # Mon, Tue, etc.

        if day_abbr not in cfg.trading.hours.days:
            return False
        if not (cfg.trading.hours.start_hour <= now.hour < cfg.trading.hours.end_hour):
            return False
        return True

    async def _tick(self) -> None:
        """Single poll tick: get price, load alarms, check triggers."""
        self._tick_count += 1

        # D2: Trading hours enforcement
        if not self._is_within_trading_hours():
            if self._tick_count % 20 == 1:  # log once every ~10 minutes
                logger.info("[monitor] Outside trading hours — monitoring paused")
            return

        price = self.current_price()
        if price is None or price <= 0:
            # B4: Track consecutive null-price ticks
            self._null_price_streak += 1
            if self._null_price_streak >= 20:  # ~10 min at 30s poll
                logger.error(
                    f"[monitor] No live price for {self._null_price_streak} consecutive ticks "
                    f"({self._null_price_streak * self._poll_interval}s). "
                    "Quote subscription may have failed."
                )
                # Send alert once, then every 60 ticks (~30 min)
                if self._null_price_streak == 20 or self._null_price_streak % 60 == 0:
                    try:
                        from notifications.telegram import get_notifier
                        from agents.utils import run_on_main_loop
                        notifier = get_notifier()
                        run_on_main_loop(notifier.send(
                            f"⚠️ Price monitor: no live quote for "
                            f"{self._null_price_streak * self._poll_interval // 60} min. "
                            f"Check IB Gateway connection."
                        ), timeout=10)
                    except Exception:
                        pass  # best-effort alert
            elif self._null_price_streak < 20:
                logger.debug("[monitor] No live price yet — skipping tick")
            return

        self._null_price_streak = 0  # reset on valid price

        # B1: Run alarm expiry every 10th tick (~5 minutes at 30s poll)
        if self._tick_count % 10 == 0:
            self.expire_stale_alarms()

        alarms = self._load_active_alarms()
        if not alarms:
            self._near_counts.clear()
            return

        logger.debug(f"[monitor] tick price={price:.2f} active_alarms={len(alarms)}")

        triggered_ids: set[int] = set()

        for alarm in alarms:
            alarm_id = alarm.id
            distance_pct = abs(price - alarm.trigger_price) / alarm.trigger_price

            if distance_pct <= self._tolerance:
                # Price is within tolerance — increment consecutive count
                self._near_counts[alarm_id] = self._near_counts.get(alarm_id, 0) + 1
                logger.debug(
                    f"[monitor] Alarm {alarm_id} near "
                    f"(price={price:.2f} trigger={alarm.trigger_price:.2f} "
                    f"dist={distance_pct:.3%} count={self._near_counts[alarm_id]})"
                )

                if self._near_counts[alarm_id] >= 2:
                    # Confirmed trigger — fire
                    triggered_ids.add(alarm_id)
                    await self._fire_alarm(alarm, price)
            else:
                # Price moved away — reset count
                self._near_counts.pop(alarm_id, None)

        # Clean up counts for alarms that are no longer active
        active_ids = {a.id for a in alarms}
        stale = [aid for aid in self._near_counts if aid not in active_ids]
        for aid in stale:
            del self._near_counts[aid]

    async def _fire_alarm(self, alarm: Alarm, price: float) -> None:
        """Mark alarm as triggered in DB and invoke all registered callbacks."""
        alarm_id = alarm.id
        logger.info(
            f"[monitor] ALARM TRIGGERED id={alarm_id} "
            f"asset={alarm.asset} action={alarm.action} "
            f"trigger={alarm.trigger_price:.2f} current={price:.2f}"
        )

        # Persist triggered status
        try:
            with get_session() as session:
                db_alarm = session.query(Alarm).filter(Alarm.id == alarm_id).first()
                if db_alarm and db_alarm.status == "active":
                    db_alarm.status = "triggered"
                    db_alarm.triggered_at = datetime.utcnow()
        except Exception as exc:
            logger.error(f"[monitor] Could not update alarm {alarm_id} status: {exc}")
            return

        # Remove from near_counts so it won't fire again
        self._near_counts.pop(alarm_id, None)

        # Invoke callbacks
        for cb in self._callbacks:
            try:
                await cb(alarm_id, price)
            except Exception as exc:
                logger.error(f"[monitor] Callback error for alarm {alarm_id}: {exc}")

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _load_active_alarms(self) -> list[Alarm]:
        """Load all active, non-expired alarms for the configured asset."""
        now = datetime.utcnow()
        try:
            with get_session() as session:
                alarms = (
                    session.query(Alarm)
                    .filter(
                        Alarm.asset == cfg.asset.symbol,
                        Alarm.status == "active",
                        (Alarm.expires_at == None) | (Alarm.expires_at > now),  # noqa: E711
                    )
                    .all()
                )
                # Detach from session so they can be used outside the context
                session.expunge_all()
                return alarms
        except Exception as exc:
            logger.error(f"[monitor] Could not load alarms: {exc}")
            return []

    # ── Alarm expiry maintenance ──────────────────────────────────────────────

    def expire_stale_alarms(self) -> int:
        """
        Mark any alarms past their expires_at as 'expired'.
        Called internally; can also be called externally for maintenance.
        Returns count of expired alarms.
        """
        now = datetime.utcnow()
        count = 0
        try:
            with get_session() as session:
                count = (
                    session.query(Alarm)
                    .filter(
                        Alarm.status == "active",
                        Alarm.expires_at != None,  # noqa: E711
                        Alarm.expires_at <= now,
                    )
                    .update({"status": "expired"})
                )
            if count:
                logger.info(f"[monitor] Expired {count} stale alarms")
        except Exception as exc:
            logger.error(f"[monitor] Could not expire alarms: {exc}")
        return count
