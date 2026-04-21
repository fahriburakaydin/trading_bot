"""
Evaluator Agent — weekly performance review and knowledge extraction.

LangGraph flow:
  load_trades → calculate_metrics → analyse_performance
      → extract_knowledge → update_knowledge_base → write_report → notify → END

Runs every Sunday at 20:00 Istanbul time (configured in config.yaml).
Writes extracted rules into the knowledge_base table so future agents
benefit from accumulated trading experience.
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
from agents.evaluator.prompts import (
    ANALYSE_PERFORMANCE_PROMPT,
    EXTRACT_KNOWLEDGE_PROMPT,
    NO_TRADES_MSG,
    SYSTEM_PROMPT,
    WEEKLY_REPORT_MSG,
)
from core.config import cfg
from core.memory.database import get_session
from core.memory.models import EvaluatorReport, KnowledgeEntry, Trade
from notifications.telegram import get_notifier


# ── State definition ──────────────────────────────────────────────────────────


class EvaluatorState(TypedDict, total=False):
    """Shared state threaded through the Evaluator graph."""

    week_start: str           # ISO date string
    week_end: str             # ISO date string
    context: str
    trades: list[dict]        # serialised Trade records
    metrics: dict             # calculated performance metrics
    analysis: dict            # LLM performance analysis
    new_knowledge: list[dict] # extracted rules/insights
    kb_entries_added: int
    report_text: str
    error: str | None
    dry_run: bool             # if True, skip all DB writes and Telegram notify
    trades_override: list[dict]  # if set, load_trades uses this instead of querying DB


# ── Agent class ───────────────────────────────────────────────────────────────


class EvaluatorAgent(BaseAgent):
    """Runs the weekly performance review and knowledge extraction workflow."""

    agent_name = "evaluator"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model=model)
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(EvaluatorState)

        g.add_node("load_trades", self._node_load_trades)
        g.add_node("calculate_metrics", self._node_calculate_metrics)
        g.add_node("analyse_performance", self._node_analyse_performance)
        g.add_node("extract_knowledge", self._node_extract_knowledge)
        g.add_node("update_knowledge_base", self._node_update_knowledge_base)
        g.add_node("write_report", self._node_write_report)
        g.add_node("notify", self._node_notify)

        g.set_entry_point("load_trades")
        g.add_edge("load_trades", "calculate_metrics")
        g.add_edge("calculate_metrics", "analyse_performance")
        g.add_edge("analyse_performance", "extract_knowledge")
        g.add_edge("extract_knowledge", "update_knowledge_base")
        g.add_edge("update_knowledge_base", "write_report")
        g.add_edge("write_report", "notify")
        g.add_edge("notify", END)

        return g.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_load_trades(self, state: EvaluatorState) -> EvaluatorState:
        # Allow caller to inject trades directly (used for dry-run comparison)
        if state.get("trades_override") is not None:
            now = datetime.utcnow()
            logger.info(
                f"[evaluator] load_trades — using {len(state['trades_override'])} "
                "injected trades (trades_override)"
            )
            return {
                **state,
                "week_start": (now - timedelta(days=7)).date().isoformat(),
                "week_end": now.date().isoformat(),
                "context": self.load_context(),
                "trades": state["trades_override"],
            }

        now = datetime.utcnow()
        week_end = now
        week_start = now - timedelta(days=7)
        context = self.load_context()

        logger.info(
            f"[evaluator] load_trades "
            f"{week_start.date()} → {week_end.date()}"
        )

        trades: list[dict] = []
        try:
            with get_session() as session:
                db_trades = (
                    session.query(Trade)
                    .filter(
                        Trade.asset == cfg.asset.symbol,
                        Trade.status == "closed",
                        Trade.closed_at >= week_start,
                        Trade.closed_at <= week_end,
                    )
                    .order_by(Trade.closed_at)
                    .all()
                )
                for t in db_trades:
                    trades.append({
                        "id": t.id,
                        "direction": t.direction,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "notional": t.notional,
                        "stop_loss": t.stop_loss,
                        "target_price": t.target_price,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                        "pnl_r": t.pnl_r,
                        "exit_reason": t.exit_reason,
                        "opened_at": t.opened_at.isoformat() if t.opened_at else None,
                        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                    })
        except Exception as exc:
            logger.error(f"[evaluator] Could not load trades: {exc}")

        logger.info(f"[evaluator] Loaded {len(trades)} closed trades")
        return {
            **state,
            "week_start": week_start.date().isoformat(),
            "week_end": week_end.date().isoformat(),
            "context": context,
            "trades": trades,
        }

    def _node_calculate_metrics(self, state: EvaluatorState) -> EvaluatorState:
        logger.info("[evaluator] calculate_metrics")
        trades = state.get("trades", [])

        if not trades:
            return {**state, "metrics": {}}

        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        win_rate = len(wins) / total_trades if total_trades else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        total_pnl = sum(pnls)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Max drawdown (sequential)
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd

        # Best and worst trade IDs
        best_trade = max(trades, key=lambda t: t.get("pnl") or 0)
        worst_trade = min(trades, key=lambda t: t.get("pnl") or 0)

        metrics = {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "best_trade_id": best_trade.get("id"),
            "worst_trade_id": worst_trade.get("id"),
        }

        logger.info(
            f"[evaluator] metrics: trades={total_trades} "
            f"win_rate={win_rate:.1%} pnl={total_pnl:.2f}"
        )
        return {**state, "metrics": metrics}

    def _node_analyse_performance(self, state: EvaluatorState) -> EvaluatorState:
        logger.info("[evaluator] analyse_performance")
        trades = state.get("trades", [])
        metrics = state.get("metrics", {})

        if not trades:
            return {**state, "analysis": {"performance_summary": "No trades this week."}}

        system = SYSTEM_PROMPT.format(
            symbol=cfg.asset.symbol,
            asset_class=cfg.asset.type.lower(),
            week_start=state["week_start"],
            week_end=state["week_end"],
            context=state.get("context", ""),
        )
        prompt = ANALYSE_PERFORMANCE_PROMPT.format(
            trades_json=json.dumps(trades, indent=2),
            total_trades=metrics.get("total_trades", 0),
            win_rate=metrics.get("win_rate", 0),
            avg_win=metrics.get("avg_win", 0),
            avg_loss=metrics.get("avg_loss", 0),
            total_pnl=metrics.get("total_pnl", 0),
            profit_factor=metrics.get("profit_factor", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        try:
            response = self.llm.invoke(messages)
            analysis = extract_json(response.content)
            # Model occasionally returns a list wrapping the object — unwrap it
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}
            if not isinstance(analysis, dict):
                raise ValueError(f"Expected dict, got {type(analysis)}")
        except Exception as exc:
            logger.warning(f"[evaluator] Could not parse analysis JSON: {exc}")
            analysis = {
                "performance_summary": "Analysis parsing failed.",
                "strengths": [],
                "weaknesses": [],
                "notable_patterns": [],
            }

        logger.info("[evaluator] analyse_performance complete")
        return {**state, "analysis": analysis}

    def _node_extract_knowledge(self, state: EvaluatorState) -> EvaluatorState:
        logger.info("[evaluator] extract_knowledge")
        analysis = state.get("analysis", {})
        trades = state.get("trades", [])

        if not trades:
            return {**state, "new_knowledge": []}

        system = SYSTEM_PROMPT.format(
            symbol=cfg.asset.symbol,
            asset_class=cfg.asset.type.lower(),
            week_start=state["week_start"],
            week_end=state["week_end"],
            context=state.get("context", ""),
        )
        prompt = EXTRACT_KNOWLEDGE_PROMPT.format(
            analysis_json=json.dumps(analysis, indent=2),
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        try:
            response = self.llm.invoke(messages)
            knowledge = extract_json(response.content)
            if not isinstance(knowledge, list):
                knowledge = []
        except Exception as exc:
            logger.warning(f"[evaluator] Could not parse knowledge JSON: {exc}")
            knowledge = []

        logger.info(f"[evaluator] extract_knowledge — {len(knowledge)} entries")
        return {**state, "new_knowledge": knowledge}

    def _node_update_knowledge_base(self, state: EvaluatorState) -> EvaluatorState:
        logger.info("[evaluator] update_knowledge_base")
        if state.get("dry_run"):
            knowledge = state.get("new_knowledge", [])
            logger.info("[evaluator] dry_run — skipping KB writes")
            return {**state, "kb_entries_added": len(knowledge)}
        knowledge = state.get("new_knowledge", [])
        added = 0

        for entry_data in knowledge:
            try:
                with get_session() as session:
                    entry = KnowledgeEntry(
                        source="evaluator",
                        category=entry_data.get("category", "insight"),
                        title=entry_data.get("title", "Untitled"),
                        content=entry_data.get("content", ""),
                        applies_to=entry_data.get("applies_to", "all"),
                        active=True,
                        performance_impact=entry_data.get("performance_impact"),
                    )
                    session.add(entry)
                added += 1
                logger.info(f"[evaluator] KB entry added: {entry_data.get('title')}")
            except Exception as exc:
                logger.error(f"[evaluator] Could not save KB entry: {exc}")

        logger.info(f"[evaluator] update_knowledge_base — {added} entries added")
        return {**state, "kb_entries_added": added}

    def _node_write_report(self, state: EvaluatorState) -> EvaluatorState:
        logger.info("[evaluator] write_report")
        metrics = state.get("metrics", {})
        analysis = state.get("analysis", {})
        knowledge = state.get("new_knowledge", [])
        trades = state.get("trades", [])

        report_lines = [
            f"WEEKLY EVALUATOR REPORT",
            f"Period: {state['week_start']} → {state['week_end']}",
            f"Asset: {cfg.asset.symbol}",
            "",
        ]

        if not trades:
            report_lines.append("No completed trades this week.")
        else:
            report_lines += [
                f"Performance:",
                f"  Trades: {metrics.get('total_trades', 0)} "
                f"({metrics.get('winning_trades', 0)}W / {metrics.get('losing_trades', 0)}L)",
                f"  Win Rate: {metrics.get('win_rate', 0):.1%}",
                f"  Total P&L: ${metrics.get('total_pnl', 0):+,.2f}",
                f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}",
                f"  Expectancy: ${metrics.get('expectancy', 0):+.2f}/trade",
                f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1%}",
                "",
                f"Analysis: {analysis.get('performance_summary', '')}",
                "",
            ]

            if analysis.get("strengths"):
                report_lines.append("Strengths:")
                for s in analysis["strengths"]:
                    report_lines.append(f"  + {s}")
                report_lines.append("")

            if analysis.get("weaknesses"):
                report_lines.append("Weaknesses:")
                for w in analysis["weaknesses"]:
                    report_lines.append(f"  - {w}")
                report_lines.append("")

        if knowledge:
            report_lines.append(f"Knowledge Base ({len(knowledge)} new entries):")
            for k in knowledge:
                report_lines.append(f"  [{k.get('category', '?')}] {k.get('title', '')}")

        report_text = "\n".join(report_lines)

        # Persist EvaluatorReport (skipped in dry-run mode)
        if state.get("dry_run"):
            logger.info("[evaluator] dry_run — skipping EvaluatorReport DB write")
            return {**state, "report_text": report_text}

        try:
            from strategies import get_strategy
            try:
                strategy_id = get_strategy(cfg.asset.symbol).config.strategy_id
            except KeyError:
                strategy_id = None

            week_start_dt = datetime.fromisoformat(state["week_start"])
            week_end_dt = datetime.fromisoformat(state["week_end"])
            with get_session() as session:
                report = EvaluatorReport(
                    week_start=week_start_dt,
                    week_end=week_end_dt,
                    strategy_id=strategy_id,
                    total_trades=metrics.get("total_trades", 0),
                    winning_trades=metrics.get("winning_trades", 0),
                    losing_trades=metrics.get("losing_trades", 0),
                    total_pnl=metrics.get("total_pnl", 0.0),
                    win_rate=metrics.get("win_rate"),
                    avg_win=metrics.get("avg_win"),
                    avg_loss=metrics.get("avg_loss"),
                    expectancy=metrics.get("expectancy"),
                    profit_factor=metrics.get("profit_factor"),
                    max_drawdown=metrics.get("max_drawdown"),
                    best_trade_id=metrics.get("best_trade_id"),
                    worst_trade_id=metrics.get("worst_trade_id"),
                    full_report=report_text,
                )
                notes = [k.get("title", "") for k in knowledge]
                report.set_improvement_notes(notes)
                session.add(report)
            logger.info("[evaluator] EvaluatorReport saved to DB")
        except Exception as exc:
            logger.warning(f"[evaluator] Could not save EvaluatorReport: {exc}")

        return {**state, "report_text": report_text}

    def _node_notify(self, state: EvaluatorState) -> EvaluatorState:
        logger.info("[evaluator] notify")
        if state.get("dry_run"):
            logger.info("[evaluator] dry_run — skipping Telegram notify")
            return state
        metrics = state.get("metrics", {})
        analysis = state.get("analysis", {})
        knowledge = state.get("new_knowledge", [])
        trades = state.get("trades", [])
        notifier = get_notifier()

        if not trades:
            msg = NO_TRADES_MSG.format(
                week_start=state["week_start"],
                week_end=state["week_end"],
                reason=analysis.get("performance_summary", "Bot may not have been running."),
            )
        else:
            insight_bullets = "\n".join(
                f"• {k.get('title', '')}"
                for k in knowledge[:5]  # cap at 5 in Telegram
            ) or "• No new insights this week."

            msg = WEEKLY_REPORT_MSG.format(
                week_start=state["week_start"],
                week_end=state["week_end"],
                total_trades=metrics.get("total_trades", 0),
                winning_trades=metrics.get("winning_trades", 0),
                losing_trades=metrics.get("losing_trades", 0),
                win_rate=metrics.get("win_rate", 0.0),
                total_pnl=metrics.get("total_pnl", 0.0),
                profit_factor=metrics.get("profit_factor", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                expectancy=metrics.get("expectancy", 0.0),
                insight_bullets=insight_bullets,
                kb_entries_added=state.get("kb_entries_added", 0),
            )

        try:
            run_on_main_loop(notifier.send(msg, parse_mode="Markdown"), timeout=15)
        except Exception as exc:
            logger.warning(f"[evaluator] Telegram notify failed: {exc}")

        return state

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        dry_run: bool = False,
        trades_override: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the weekly Evaluator workflow and return the final state.

        Args:
            dry_run:         If True, skip all DB writes and Telegram notifications.
            trades_override: Inject a trade list directly instead of querying the DB.
                             Implies dry_run behaviour for the load_trades node.
        """
        logger.info(f"[evaluator] Starting weekly evaluation run (dry_run={dry_run})")
        run_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()
        initial_state: EvaluatorState = {}
        if dry_run:
            initial_state["dry_run"] = True
        if trades_override is not None:
            initial_state["trades_override"] = trades_override
        final_state = self._graph.invoke(initial_state)
        duration_ms = int((time.monotonic() - t0) * 1000)
        result = dict(final_state)
        if not dry_run:
            self._log_run(run_id, duration_ms, result)
        logger.info(f"[evaluator] Evaluation complete ({duration_ms}ms)")
        return result


