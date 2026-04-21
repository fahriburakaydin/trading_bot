"""
APScheduler job registration for the trading bot.

All recurring agent runs are defined here and registered with a single
BackgroundScheduler instance.  Call ``start_scheduler()`` once at boot.

Schedule (all times in Europe/Istanbul):
  - Research Agent: 10:00 every day
  - Analyst Agent:  10:30 every day
  - Evaluator Agent: 20:00 every Sunday  (Phase 4)

Usage:
    from core.scheduler.jobs import start_scheduler, stop_scheduler
    scheduler = start_scheduler()
    ...
    stop_scheduler(scheduler)
"""

from __future__ import annotations

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from core.config import cfg


# ── Job functions ─────────────────────────────────────────────────────────────


def _run_research_agent() -> None:
    """Instantiate and run the Research Agent (import deferred to avoid circular deps)."""
    logger.info("[scheduler] Starting Research Agent job")
    try:
        from agents.research.agent import ResearchAgent
        agent = ResearchAgent()
        result = agent.run()
        logger.info(
            f"[scheduler] Research Agent complete — "
            f"risk_level={result.get('risk_environment', {}).get('risk_level', 'unknown')}"
        )
    except Exception as exc:
        logger.error(f"[scheduler] Research Agent job failed: {exc}")


def _run_analyst_agent() -> None:
    """Instantiate and run the Analyst Agent."""
    logger.info("[scheduler] Starting Analyst Agent job")
    try:
        from agents.analyst.agent import AnalystAgent
        agent = AnalystAgent()
        result = agent.run()
        logger.info(
            f"[scheduler] Analyst Agent complete — "
            f"alarms_saved={result.get('alarms_saved', 0)}"
        )
    except Exception as exc:
        logger.error(f"[scheduler] Analyst Agent job failed: {exc}")


def _run_evaluator_agent() -> None:
    """Instantiate and run the Evaluator Agent."""
    logger.info("[scheduler] Starting Evaluator Agent job")
    try:
        from agents.evaluator.agent import EvaluatorAgent
        agent = EvaluatorAgent()
        result = agent.run()
        logger.info(
            f"[scheduler] Evaluator Agent complete — "
            f"kb_entries_added={result.get('kb_entries_added', 0)}"
        )
    except Exception as exc:
        logger.error(f"[scheduler] Evaluator Agent job failed: {exc}")


# ── Scheduler factory ──────────────────────────────────────────────────────────


def build_scheduler() -> BackgroundScheduler:
    """Build and configure a BackgroundScheduler with all agent jobs registered."""
    tz = cfg.schedule.research.timezone

    executors = {"default": ThreadPoolExecutor(max_workers=2)}
    job_defaults = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 300}

    scheduler = BackgroundScheduler(
        executors=executors,
        job_defaults=job_defaults,
        timezone=tz,
    )

    # Research Agent — 10:00 Istanbul daily
    research_cfg = cfg.schedule.research
    scheduler.add_job(
        _run_research_agent,
        trigger="cron",
        hour=research_cfg.hour,
        minute=research_cfg.minute,
        timezone=research_cfg.timezone,
        id="research_agent",
        name="Research Agent — daily macro report",
        replace_existing=True,
    )
    logger.info(
        f"[scheduler] Research Agent scheduled at "
        f"{research_cfg.hour:02d}:{research_cfg.minute:02d} {research_cfg.timezone}"
    )

    # Analyst Agent — 10:30 Istanbul daily
    analyst_cfg = cfg.schedule.analyst
    scheduler.add_job(
        _run_analyst_agent,
        trigger="cron",
        hour=analyst_cfg.hour,
        minute=analyst_cfg.minute,
        timezone=analyst_cfg.timezone,
        id="analyst_agent",
        name="Analyst Agent — daily TA + alarms",
        replace_existing=True,
    )
    logger.info(
        f"[scheduler] Analyst Agent scheduled at "
        f"{analyst_cfg.hour:02d}:{analyst_cfg.minute:02d} {analyst_cfg.timezone}"
    )

    # Evaluator Agent — 20:00 Istanbul every Sunday
    evaluator_cfg = cfg.schedule.evaluator
    scheduler.add_job(
        _run_evaluator_agent,
        trigger="cron",
        day_of_week=evaluator_cfg.day_of_week or "sun",
        hour=evaluator_cfg.hour,
        minute=evaluator_cfg.minute,
        timezone=evaluator_cfg.timezone,
        id="evaluator_agent",
        name="Evaluator Agent — weekly performance review",
        replace_existing=True,
    )
    logger.info(
        f"[scheduler] Evaluator Agent scheduled at "
        f"{evaluator_cfg.hour:02d}:{evaluator_cfg.minute:02d} "
        f"{evaluator_cfg.timezone} on {evaluator_cfg.day_of_week or 'sunday'}"
    )

    return scheduler


def start_scheduler() -> BackgroundScheduler:
    """Build, start, and return the scheduler. Call once at application boot."""
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("[scheduler] Scheduler started")
    return scheduler


def stop_scheduler(scheduler: BackgroundScheduler) -> None:
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("[scheduler] Scheduler stopped")
