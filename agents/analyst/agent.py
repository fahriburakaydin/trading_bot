"""
Analyst Agent — multi-timeframe TA analysis and price alarm generation.

LangGraph flow:
  load_context → fetch_ohlcv_multi_timeframe → calculate_all_indicators
      → detect_key_levels → score_confluence → filter_by_confidence
      → set_alarms → notify → END

Runs each morning at 10:30 Istanbul time (configured in config.yaml),
after the Research Agent has completed its report.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from agents.base import BaseAgent
from agents.utils import extract_json, run_on_main_loop
from agents.analyst.prompts import (
    CALCULATE_INDICATORS_PROMPT,
    DETECT_LEVELS_PROMPT,
    FETCH_DATA_PROMPT,
    FILTER_SETUPS_PROMPT,
    SYSTEM_PROMPT,
    TELEGRAM_ALARM_TEMPLATE,
    TELEGRAM_NO_ALARMS,
    WRITE_SUMMARY_PROMPT,
)
from agents.research.prompts import RISK_LEVEL_EMOJI, sentiment_label
from core.config import cfg
from core.memory.database import get_session
from core.memory.models import Alarm, AnalysisRun, ResearchReport
from notifications.telegram import get_notifier
from tools.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_vwap,
    detect_support_resistance,
    score_confluence_levels,
)
from tools.market_data import fetch_multi_timeframe, fetch_ohlcv


# ── State definition ──────────────────────────────────────────────────────────


class AnalystState(TypedDict, total=False):
    """Shared state threaded through every node in the graph."""

    date: str
    current_price: float
    context: str
    research_context: str          # latest ResearchReport summary
    risk_level: str                # from latest research report
    ohlcv_data: dict[str, str]     # timeframe → JSON string
    indicators_summary: dict       # computed indicator values
    scored_levels: list[dict]      # from confluence scoring
    approved_alarms: list[dict]    # filtered setups
    alarms_saved: int              # count of alarms saved to DB
    summary_text: str
    error: str | None


# ── Agent class ───────────────────────────────────────────────────────────────


class AnalystAgent(BaseAgent):
    """Runs the daily technical analysis workflow and sets price alarms."""

    agent_name = "analyst"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model=model)
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(AnalystState)

        g.add_node("load_context", self._node_load_context)
        g.add_node("fetch_ohlcv_multi_timeframe", self._node_fetch_ohlcv)
        g.add_node("calculate_all_indicators", self._node_calculate_indicators)
        g.add_node("detect_key_levels", self._node_detect_levels)
        g.add_node("score_confluence", self._node_score_confluence)
        g.add_node("filter_by_confidence", self._node_filter_by_confidence)
        g.add_node("set_alarms", self._node_set_alarms)
        g.add_node("notify", self._node_notify)

        g.set_entry_point("load_context")
        g.add_edge("load_context", "fetch_ohlcv_multi_timeframe")
        g.add_edge("fetch_ohlcv_multi_timeframe", "calculate_all_indicators")
        g.add_edge("calculate_all_indicators", "detect_key_levels")
        g.add_edge("detect_key_levels", "score_confluence")
        g.add_edge("score_confluence", "filter_by_confidence")
        g.add_edge("filter_by_confidence", "set_alarms")
        g.add_edge("set_alarms", "notify")
        g.add_edge("notify", END)

        return g.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_load_context(self, state: AnalystState) -> AnalystState:
        today = datetime.utcnow().date().isoformat()
        context = self.load_context()

        # Load latest research report for context and risk level
        research_context = ""
        risk_level = "GREEN"
        try:
            with get_session() as session:
                report: ResearchReport | None = (
                    session.query(ResearchReport)
                    .order_by(ResearchReport.created_at.desc())
                    .first()
                )
                if report:
                    research_context = report.summary or ""
                    risk_map = {"low": "GREEN", "medium": "YELLOW", "high": "RED"}
                    risk_level = risk_map.get(report.risk_level or "medium", "YELLOW")
        except Exception as exc:
            logger.warning(f"[analyst] Could not load research report: {exc}")

        logger.info(f"[analyst] load_context date={today} risk_level={risk_level}")
        return {
            **state,
            "date": today,
            "context": context,
            "research_context": research_context,
            "risk_level": risk_level,
            "current_price": 0.0,
        }

    def _node_fetch_ohlcv(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] fetch_ohlcv_multi_timeframe")

        # Short-circuit if risk is RED — no point fetching data
        if state.get("risk_level") == "RED":
            logger.warning("[analyst] Risk is RED — skipping data fetch")
            return {**state, "ohlcv_data": {}, "approved_alarms": []}

        try:
            raw = fetch_multi_timeframe.invoke(
                {"timeframes": ["1 hour", "4 hours", "1 day"]}
            )
            ohlcv_map: dict[str, str] = json.loads(raw)  # {"1 hour": [...], ...}

            # Extract current price from most recent 1h candle
            current_price = 0.0
            if "1 hour" in ohlcv_map:
                candles = ohlcv_map["1 hour"]
                if candles:
                    last = candles[-1] if isinstance(candles, list) else []
                    current_price = float(last.get("close", 0.0)) if last else 0.0

            if current_price <= 0.0:
                logger.error("[analyst] current_price is 0.0 after OHLCV fetch — aborting")
                return {**state, "ohlcv_data": {}, "current_price": 0.0, "error": "current_price is 0.0"}

            logger.info(f"[analyst] fetched OHLCV current_price={current_price}")
            return {
                **state,
                "ohlcv_data": {tf: json.dumps(data) for tf, data in ohlcv_map.items()},
                "current_price": current_price,
            }
        except Exception as exc:
            logger.exception(f"[analyst] fetch_ohlcv failed: {exc}")
            return {**state, "ohlcv_data": {}, "current_price": 0.0, "error": str(exc)}

    def _node_calculate_indicators(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] calculate_all_indicators")
        ohlcv_data = state.get("ohlcv_data", {})
        if not ohlcv_data:
            return {**state, "indicators_summary": {}}

        summary: dict = {}
        timeframe_map = {"1 hour": "1h", "4 hours": "4h", "1 day": "1d"}

        for tf_raw, tf_label in timeframe_map.items():
            ohlcv_json = ohlcv_data.get(tf_raw)
            if not ohlcv_json:
                continue
            try:
                rsi = json.loads(calculate_rsi.invoke({"ohlcv_json": ohlcv_json}))
                macd = json.loads(calculate_macd.invoke({"ohlcv_json": ohlcv_json}))
                ema = json.loads(calculate_ema.invoke({"ohlcv_json": ohlcv_json}))
                atr = json.loads(calculate_atr.invoke({"ohlcv_json": ohlcv_json}))
                summary[tf_label] = {
                    "rsi": rsi.get("rsi"),
                    "rsi_signal": rsi.get("signal"),
                    "macd_direction": macd.get("direction"),
                    "trend": ema.get("trend"),
                    "atr": atr.get("atr"),
                    "atr_pct": atr.get("atr_pct"),
                }
            except Exception as exc:
                logger.warning(f"[analyst] indicator calc failed for {tf_label}: {exc}")

        # VWAP from 1h
        if "1 hour" in ohlcv_data:
            try:
                vwap = json.loads(calculate_vwap.invoke({"ohlcv_json": ohlcv_data["1 hour"]}))
                summary["vwap"] = vwap
            except Exception as exc:
                logger.warning(f"[analyst] VWAP calc failed: {exc}")

        logger.info(f"[analyst] indicators computed for {list(summary.keys())}")
        return {**state, "indicators_summary": summary}

    def _node_detect_levels(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] detect_key_levels")
        ohlcv_data = state.get("ohlcv_data", {})
        if not ohlcv_data:
            return {**state, "scored_levels": []}

        levels: list[dict] = []
        # Detect raw S/R from 4h (higher-timeframe structure)
        ohlcv_4h = ohlcv_data.get("4 hours")
        if ohlcv_4h:
            try:
                raw = detect_support_resistance.invoke({"ohlcv_json": ohlcv_4h})
                levels.extend(json.loads(raw))
            except Exception as exc:
                logger.warning(f"[analyst] 4h S/R detection failed: {exc}")

        logger.info(f"[analyst] detect_key_levels — {len(levels)} raw levels")
        return {**state, "scored_levels": levels}

    def _node_score_confluence(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] score_confluence")
        ohlcv_data = state.get("ohlcv_data", {})
        current_price = state.get("current_price", 0.0)
        existing_levels = state.get("scored_levels", [])  # 4h S/R from detect_levels

        if not ohlcv_data or not current_price:
            return state

        ohlcv_1h = ohlcv_data.get("1 hour")
        if not ohlcv_1h:
            return state

        # Score 1h confluence levels
        new_scored: list[dict] = []
        try:
            raw = score_confluence_levels.invoke(
                {"ohlcv_json": ohlcv_1h, "current_price": current_price}
            )
            new_scored = json.loads(raw)
        except Exception as exc:
            logger.warning(f"[analyst] 1h score_confluence failed: {exc}")

        # Merge with existing 4h levels instead of overwriting.
        # Deduplicate by proximity: if a 1h level is within 0.3% of a 4h level,
        # keep the one with higher confidence/score.
        merged = list(new_scored)
        proximity_pct = 0.003

        for existing in existing_levels:
            ex_price = existing.get("price", 0)
            if ex_price <= 0:
                continue
            is_duplicate = False
            for new_level in new_scored:
                new_price = new_level.get("price", 0)
                if new_price > 0 and abs(ex_price - new_price) / new_price <= proximity_pct:
                    is_duplicate = True
                    break
            if not is_duplicate:
                # Convert raw 4h S/R level format to scored format for downstream
                merged.append({
                    "price": ex_price,
                    "action": "LONG" if existing.get("type") == "support" else "SHORT",
                    "confidence": existing.get("score", 0.3),
                    "stop_loss": 0,     # will be set by LLM in filter step
                    "target_price": 0,  # will be set by LLM in filter step
                    "risk_reward": 0,
                    "confluence_factors": [f"4h S/R ({existing.get('type', 'unknown')}, {existing.get('touches', 0)} touches)"],
                    "timeframe": "4h",
                })

        if not merged:
            logger.warning("[analyst] score_confluence — 0 levels survived scoring from both 1h and 4h data")

        logger.info(
            f"[analyst] score_confluence — {len(new_scored)} from 1h, "
            f"{len(existing_levels)} from 4h, {len(merged)} merged total"
        )
        return {**state, "scored_levels": merged}

    def _node_filter_by_confidence(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] filter_by_confidence")
        scored_levels = state.get("scored_levels", [])
        risk_level = state.get("risk_level", "GREEN")
        indicators = state.get("indicators_summary", {})
        research_context = state.get("research_context", "")

        if risk_level == "RED":
            logger.warning("[analyst] Risk RED — no alarms will be set")
            return {**state, "approved_alarms": []}

        max_alarms = cfg.trading.price_monitor.queue_max_size
        if risk_level == "YELLOW":
            max_alarms = 1

        # Use LLM to make final filtering decision with all context.
        # Keep the prompt compact — local models have small context windows.
        context = state.get("context", "")
        system = SYSTEM_PROMPT.format(
            symbol=cfg.asset.symbol,
            asset_class=cfg.asset.type.lower(),
            current_price=state.get("current_price", 0.0),
            context=context,
        )

        # Compact indicator summary — only key fields, no pretty-print
        compact_indicators = {
            tf: {k: v for k, v in vals.items() if k in ("rsi", "rsi_signal", "macd_direction", "trend")}
            for tf, vals in indicators.items()
            if tf != "vwap"
        }

        # Trim research context to first 300 chars to save tokens
        research_snippet = (research_context or "")[:300]

        prompt = FILTER_SETUPS_PROMPT.format(
            min_confidence=cfg.risk.min_confidence,
            min_risk_reward=cfg.risk.min_risk_reward,
            risk_level=risk_level,
            indicators_summary=json.dumps(compact_indicators, separators=(",", ":")),
            scored_levels=json.dumps(scored_levels, separators=(",", ":")),
            research_context=research_snippet,
            max_alarms=max_alarms,
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        logger.info(
            f"[analyst] filter_by_confidence — calling LLM ({len(scored_levels)} levels to evaluate, "
            f"may take 1-3 min with local models)"
        )
        response = self.llm.invoke(messages)
        logger.info("[analyst] filter_by_confidence — LLM response received")

        approved: list[dict] = []
        try:
            approved = extract_json(response.content)
            if not isinstance(approved, list):
                approved = []
        except Exception:
            logger.warning("[analyst] Could not parse approved alarms JSON")

        logger.info(f"[analyst] filter_by_confidence — {len(approved)} alarms approved")
        return {**state, "approved_alarms": approved}

    def _node_set_alarms(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] set_alarms")
        approved = state.get("approved_alarms", [])
        indicators = state.get("indicators_summary", {})
        saved_count = 0

        # Cancel existing active alarms for this asset before setting new ones
        try:
            with get_session() as session:
                session.query(Alarm).filter(
                    Alarm.asset == cfg.asset.symbol,
                    Alarm.status == "active",
                ).update({"status": "cancelled"})
            logger.info("[analyst] Cancelled previous active alarms")
        except Exception as exc:
            logger.warning(f"[analyst] Could not cancel old alarms: {exc}")

        # Validate and save new alarms
        expires_at = datetime.utcnow() + timedelta(
            hours=cfg.trading.price_monitor.alarm_expiry_hours
        )
        validated = []
        for setup in approved:
            try:
                tp = setup.get("trigger_price", 0)
                sl = setup.get("stop_loss", 0)
                target = setup.get("target_price", 0)
                action = setup.get("action", "")
                conf = setup.get("confidence", 0)
                rr = setup.get("risk_reward", 0)

                if tp <= 0 or sl <= 0 or target <= 0:
                    logger.warning(f"[analyst] Alarm dropped: invalid prices tp={tp} sl={sl} target={target}")
                    continue
                if action == "LONG" and sl >= tp:
                    logger.warning(f"[analyst] Alarm dropped: LONG but SL ({sl}) >= entry ({tp})")
                    continue
                if action == "SHORT" and sl <= tp:
                    logger.warning(f"[analyst] Alarm dropped: SHORT but SL ({sl}) <= entry ({tp})")
                    continue
                if action == "LONG" and target <= tp:
                    logger.warning(f"[analyst] Alarm dropped: LONG but target ({target}) <= entry ({tp})")
                    continue
                if action == "SHORT" and target >= tp:
                    logger.warning(f"[analyst] Alarm dropped: SHORT but target ({target}) >= entry ({tp})")
                    continue
                if rr < cfg.risk.min_risk_reward:
                    logger.warning(f"[analyst] Alarm dropped: R:R {rr:.2f} < min {cfg.risk.min_risk_reward}")
                    continue
                if conf < cfg.risk.min_confidence:
                    logger.warning(f"[analyst] Alarm dropped: confidence {conf:.2f} < min {cfg.risk.min_confidence}")
                    continue
                validated.append(setup)
            except Exception as exc:
                logger.warning(f"[analyst] Alarm validation error: {exc}")

        if len(validated) < len(approved):
            logger.info(f"[analyst] Validation: {len(approved)} → {len(validated)} alarms passed")
        approved = validated

        for setup in approved:
            try:
                with get_session() as session:
                    alarm = Alarm(
                        asset=cfg.asset.symbol,
                        trigger_price=setup["trigger_price"],
                        direction=setup["direction"],
                        action=setup["action"],
                        confidence=setup["confidence"],
                        stop_loss=setup["stop_loss"],
                        target_price=setup["target_price"],
                        risk_reward=setup["risk_reward"],
                        timeframe=setup.get("timeframe", "1h"),
                        reasoning=setup.get("reasoning", ""),
                        expires_at=expires_at,
                        status="active",
                    )
                    alarm.set_confluence_factors(setup.get("confluence_factors", []))
                    session.add(alarm)
                saved_count += 1
                logger.info(
                    f"[analyst] Alarm set: {setup['action']} ({setup.get('entry_type', '?')}) "
                    f"@ {setup['trigger_price']} (current={state.get('current_price', 0.0):.5f})"
                )
            except Exception as exc:
                logger.exception(f"[analyst] Could not save alarm: {exc}")

        # Save AnalysisRun record
        try:
            with get_session() as session:
                run = AnalysisRun(
                    asset=cfg.asset.symbol,
                    alarms_set=saved_count,
                    alarms_skipped=len(approved) - saved_count,
                    avg_confidence=(
                        sum(a["confidence"] for a in approved) / len(approved)
                        if approved
                        else None
                    ),
                    summary=f"{saved_count} alarms set for {cfg.asset.symbol}",
                )
                run.set_timeframes(["1h", "4h", "1d"])
                run.indicators_snapshot = json.dumps(indicators)
                session.add(run)
        except Exception as exc:
            logger.warning(f"[analyst] Could not save AnalysisRun: {exc}")

        # Write summary text
        summary = self._format_summary(state, saved_count)
        logger.info(f"[analyst] set_alarms complete — {saved_count} saved")
        return {**state, "alarms_saved": saved_count, "summary_text": summary}

    def _node_notify(self, state: AnalystState) -> AnalystState:
        logger.info("[analyst] notify")
        approved = state.get("approved_alarms", [])
        indicators = state.get("indicators_summary", {})
        risk_level = state.get("risk_level", "GREEN")

        if not approved:
            message = TELEGRAM_NO_ALARMS.format(
                date=state["date"],
                symbol=cfg.asset.symbol,
                risk_level_emoji=RISK_LEVEL_EMOJI.get(risk_level, "⚪"),
                risk_level=risk_level,
                reason="No setups met the confidence threshold today.",
            )
        else:
            alarm_lines = "\n".join(
                f"• {a['action']} @ ${a['trigger_price']:,.2f} "
                f"(conf={a['confidence']:.0%}, R:R={a['risk_reward']:.1f})"
                for a in approved
            )
            i1h = indicators.get("1h", {})
            message = TELEGRAM_ALARM_TEMPLATE.format(
                date=state["date"],
                symbol=cfg.asset.symbol,
                current_price=state.get("current_price", 0.0),
                alarm_count=len(approved),
                alarm_lines=alarm_lines,
                trend_1d=indicators.get("1d", {}).get("trend", "unknown"),
                rsi_1h=i1h.get("rsi") or 50.0,
                rsi_signal=i1h.get("rsi_signal", "neutral"),
                macd_direction=i1h.get("macd_direction", "unknown"),
            )

        try:
            notifier = get_notifier()
            run_on_main_loop(notifier.send(message, parse_mode="HTML"), timeout=15)
            logger.info("[analyst] Telegram notification sent")
        except Exception as exc:
            logger.warning(f"[analyst] Telegram notify failed: {exc}")

        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_summary(self, state: AnalystState, alarm_count: int) -> str:
        approved = state.get("approved_alarms", [])
        indicators = state.get("indicators_summary", {})
        alarm_bullets = "\n".join(
            f"- {a['action']} @ {a['trigger_price']:.2f} "
            f"SL={a['stop_loss']:.2f} TP={a['target_price']:.2f} "
            f"conf={a['confidence']:.0%}"
            for a in approved
        ) or "- None"

        i1d = indicators.get("1d", {})
        i4h = indicators.get("4h", {})
        i1h = indicators.get("1h", {})
        vwap = indicators.get("vwap", {})

        return (
            f"ANALYST SUMMARY — {state['date']}\n"
            f"Asset: {cfg.asset.symbol} @ {state.get('current_price', 0.0):.2f}\n"
            f"Alarms Set: {alarm_count}\n"
            f"{alarm_bullets}\n\n"
            f"Market Structure:\n"
            f"- Trend (1D): {i1d.get('trend', 'unknown')}\n"
            f"- Trend (4H): {i4h.get('trend', 'unknown')}\n"
            f"- RSI (1H): {i1h.get('rsi', 'N/A')} ({i1h.get('rsi_signal', 'N/A')})\n"
            f"- MACD (1H): {i1h.get('macd_direction', 'unknown')}\n"
            f"- VWAP: {vwap.get('price_vs_vwap', 'unknown')}\n"
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the full Analyst Agent workflow and return the final state."""
        logger.info("[analyst] Starting daily analysis run")
        run_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()
        initial_state: AnalystState = {}
        final_state = self._graph.invoke(initial_state)
        duration_ms = int((time.monotonic() - t0) * 1000)
        result = dict(final_state)
        self._log_run(run_id, duration_ms, result)
        alarms_saved = result.get("alarms_saved", 0)
        error = result.get("error")
        if error:
            logger.warning(f"[analyst] Analysis run completed with error: {error}")
        logger.info(
            f"[analyst] Analysis run complete ({duration_ms}ms) — "
            f"alarms_saved={alarms_saved} "
            f"levels_detected={len(result.get('scored_levels', []))} "
            f"approved={len(result.get('approved_alarms', []))}"
        )
        return result


