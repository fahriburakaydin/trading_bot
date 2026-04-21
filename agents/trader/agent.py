"""
Trader Agent — executes a single trade when the Price Monitor fires an alarm.

LangGraph flow:
  load_alarm → check_risk → size_position → place_order → notify → persist → END

Called by the Price Monitor callback; one invocation per triggered alarm.
The agent is stateless between runs — all context is loaded fresh each call.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from agents.base import BaseAgent
from agents.trader.prompts import (
    RISK_CHECK_PROMPT,
    SYSTEM_PROMPT,
    TRADE_ERROR_MSG,
    TRADE_OPENED_MSG,
    TRADE_REJECTED_MSG,
)
from agents.utils import extract_json, run_on_main_loop
from core.broker.market_data import build_contract
from core.broker.orders import BracketResult, check_slippage, place_bracket_order
from core.broker.portfolio import count_open_positions, get_account_summary, get_daily_pnl
from core.config import cfg
from core.memory.database import get_session
from core.memory.models import Alarm, Trade
from core.risk.position_sizing import calculate_position_size
from notifications.telegram import get_notifier


# ── State definition ──────────────────────────────────────────────────────────


class TraderState(TypedDict, total=False):
    """Shared state for one Trader Agent run."""

    alarm_id: int
    trigger_price: float           # price at which alarm fired
    alarm: dict                    # serialised Alarm fields
    context: str                   # knowledge-base context block
    risk_level: str                # GREEN | YELLOW | RED
    account_value: float
    daily_pnl: float
    open_positions: int
    sizing: dict | None            # result of calculate_position_size
    approved: bool
    reject_reason: str
    bracket: dict | None           # serialised BracketResult fields
    trade_id: int | None
    error: str | None


# ── Agent class ───────────────────────────────────────────────────────────────


class TraderAgent(BaseAgent):
    """
    Executes one trade per invocation.

    Args:
        ib:  Connected ib_insync.IB instance (from IBKRClient.ib)
    """

    agent_name = "trader"

    def __init__(self, ib, model: str | None = None) -> None:
        super().__init__(model=model)
        self._ib = ib
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(TraderState)

        g.add_node("load_alarm", self._node_load_alarm)
        g.add_node("check_risk", self._node_check_risk)
        g.add_node("size_position", self._node_size_position)
        g.add_node("place_order", self._node_place_order)
        g.add_node("notify", self._node_notify)
        g.add_node("persist", self._node_persist)

        g.set_entry_point("load_alarm")
        g.add_edge("load_alarm", "check_risk")
        g.add_edge("check_risk", "size_position")
        g.add_edge("size_position", "place_order")
        g.add_edge("place_order", "notify")
        g.add_edge("notify", "persist")
        g.add_edge("persist", END)

        return g.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_load_alarm(self, state: TraderState) -> TraderState:
        alarm_id = state["alarm_id"]
        logger.info(f"[trader] load_alarm id={alarm_id}")
        context = self.load_context()

        # Load alarm from DB
        alarm_dict: dict = {}
        try:
            with get_session() as session:
                alarm = session.query(Alarm).filter(Alarm.id == alarm_id).first()
                if alarm:
                    alarm_dict = {
                        "id": alarm.id,
                        "asset": alarm.asset,
                        "action": alarm.action,
                        "direction": alarm.direction,
                        "trigger_price": alarm.trigger_price,
                        "stop_loss": alarm.stop_loss,
                        "target_price": alarm.target_price,
                        "confidence": alarm.confidence,
                        "risk_reward": alarm.risk_reward,
                        "timeframe": alarm.timeframe,
                        "reasoning": alarm.reasoning or "",
                        "confluence_factors": alarm.get_confluence_factors(),
                        "status": alarm.status,
                    }
        except Exception as exc:
            logger.error(f"[trader] Could not load alarm {alarm_id}: {exc}")
            return {**state, "alarm": {}, "error": str(exc), "approved": False}

        # Load latest research risk level
        risk_level = "GREEN"
        try:
            from core.memory.models import ResearchReport
            with get_session() as session:
                report = (
                    session.query(ResearchReport)
                    .order_by(ResearchReport.created_at.desc())
                    .first()
                )
                if report:
                    risk_map = {"low": "GREEN", "medium": "YELLOW", "high": "RED"}
                    risk_level = risk_map.get(report.risk_level or "medium", "YELLOW")
        except Exception as exc:
            logger.warning(f"[trader] Could not load research risk level: {exc}")

        logger.info(f"[trader] alarm loaded action={alarm_dict.get('action')} risk={risk_level}")
        return {
            **state,
            "alarm": alarm_dict,
            "context": context,
            "risk_level": risk_level,
            "approved": True,  # presumed until check_risk overrides
            "error": None,
        }

    def _node_check_risk(self, state: TraderState) -> TraderState:
        logger.info("[trader] check_risk")
        alarm = state.get("alarm", {})

        # Hard gate: no alarm data
        if not alarm:
            return {**state, "approved": False, "reject_reason": "Alarm not found in DB"}

        # Hard gate: RED risk
        if state.get("risk_level") == "RED":
            reason = "Risk level is RED — no new trades allowed"
            logger.warning(f"[trader] check_risk REJECTED: {reason}")
            return {**state, "approved": False, "reject_reason": reason}

        # Fetch live account state
        account_value = cfg.trading.capital.paper_account_size
        daily_pnl = 0.0
        open_positions = 0

        try:
            summary = run_on_main_loop(get_account_summary(self._ib))
            account_value = summary.net_liquidation or account_value
            daily_pnl = run_on_main_loop(get_daily_pnl(self._ib))
            open_positions = count_open_positions(self._ib)
        except Exception as exc:
            logger.warning(f"[trader] Could not fetch live account state: {exc}")

        max_daily_loss = account_value * cfg.trading.capital.max_daily_loss_pct
        max_concurrent = cfg.trading.capital.max_concurrent_positions

        # Hard gate: daily loss breached
        if daily_pnl <= -max_daily_loss:
            reason = (
                f"Daily loss limit reached: PnL={daily_pnl:.2f} limit=-{max_daily_loss:.2f}"
            )
            logger.warning(f"[trader] check_risk REJECTED: {reason}")
            return {
                **state,
                "approved": False,
                "reject_reason": reason,
                "account_value": account_value,
                "daily_pnl": daily_pnl,
                "open_positions": open_positions,
            }

        # Hard gate: too many open positions
        if open_positions >= max_concurrent:
            reason = f"Max concurrent positions reached ({open_positions}/{max_concurrent})"
            logger.warning(f"[trader] check_risk REJECTED: {reason}")
            return {
                **state,
                "approved": False,
                "reject_reason": reason,
                "account_value": account_value,
                "daily_pnl": daily_pnl,
                "open_positions": open_positions,
            }

        # Pre-calculate sizing so LLM can see it
        risk_pct = cfg.trading.capital.risk_per_trade_pct
        sizing = calculate_position_size(
            portfolio_value=account_value,
            risk_pct=risk_pct,
            entry_price=alarm["trigger_price"],
            stop_loss=alarm["stop_loss"],
            min_quantity=cfg.asset.min_quantity,
            max_notional_pct=cfg.trading.capital.max_notional_pct,
        )

        # Ask LLM for final sanity check
        system = SYSTEM_PROMPT.format(
            symbol=cfg.asset.symbol,
            asset_class=cfg.asset.type.lower(),
            current_price=state.get("trigger_price", alarm["trigger_price"]),
            account_value=account_value,
            context=state.get("context", ""),
        )
        prompt = RISK_CHECK_PROMPT.format(
            alarm_json=json.dumps(alarm, indent=2),
            net_liq=account_value,
            daily_pnl=daily_pnl,
            max_daily_loss=max_daily_loss,
            open_positions=open_positions,
            max_concurrent=max_concurrent,
            risk_level=state.get("risk_level", "GREEN"),
            sizing_json=json.dumps(sizing, indent=2),
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        try:
            response = self.llm.invoke(messages)
            result = extract_json(response.content)
            approved = bool(result.get("approved", False))
            reason = result.get("reason", "LLM did not provide a reason")
        except Exception as exc:
            logger.error(f"[trader] LLM risk check failed — REJECTING trade: {exc}")
            approved = False
            reason = f"LLM risk check failed: {exc}"

        logger.info(f"[trader] check_risk approved={approved} reason={reason}")
        return {
            **state,
            "approved": approved,
            "reject_reason": "" if approved else reason,
            "account_value": account_value,
            "daily_pnl": daily_pnl,
            "open_positions": open_positions,
            "sizing": sizing,
        }

    def _node_size_position(self, state: TraderState) -> TraderState:
        logger.info("[trader] size_position")

        if not state.get("approved"):
            return state

        # Sizing was already calculated in check_risk; use it or recalculate
        sizing = state.get("sizing")
        if sizing is None:
            alarm = state.get("alarm", {})
            sizing = calculate_position_size(
                portfolio_value=state.get("account_value", cfg.trading.capital.paper_account_size),
                risk_pct=cfg.trading.capital.risk_per_trade_pct,
                entry_price=alarm.get("trigger_price", 0),
                stop_loss=alarm.get("stop_loss", 0),
                min_quantity=cfg.asset.min_quantity,
                max_notional_pct=cfg.trading.capital.max_notional_pct,
            )

        if sizing is None:
            reason = "Position sizing returned None (guardrail triggered)"
            logger.warning(f"[trader] size_position REJECTED: {reason}")
            return {**state, "approved": False, "reject_reason": reason, "sizing": None}

        logger.info(
            f"[trader] size_position qty={sizing['quantity']} "
            f"notional={sizing['notional']:.2f} risk={sizing['risk_amount']:.2f}"
        )
        return {**state, "sizing": sizing}

    def _node_place_order(self, state: TraderState) -> TraderState:
        logger.info("[trader] place_order")

        if not state.get("approved"):
            return {**state, "bracket": None}

        alarm = state["alarm"]
        sizing = state["sizing"]

        action_map = {"LONG": "BUY", "SHORT": "SELL"}
        ib_action = action_map.get(alarm["action"], "BUY")

        try:
            contract = build_contract()
            bracket: BracketResult = run_on_main_loop(
                place_bracket_order(
                    ib=self._ib,
                    contract=contract,
                    action=ib_action,
                    quantity=sizing["quantity"],
                    entry_price=alarm["trigger_price"],
                    stop_loss=alarm["stop_loss"],
                    take_profit=alarm["target_price"],
                )
            )

            # Check fill slippage (entry is a limit order; actual fill may differ slightly)
            fill_price = alarm["trigger_price"]  # limit order — fill at trigger or better
            _, slippage_pct = check_slippage(alarm["trigger_price"], fill_price)

            bracket_dict = {
                "entry_order_id": bracket.entry_order_id,
                "fill_price": fill_price,
                "slippage_pct": slippage_pct,
            }
            logger.info(
                f"[trader] Order placed: {ib_action} {sizing['quantity']} "
                f"{cfg.asset.symbol} @ {alarm['trigger_price']} "
                f"orderId={bracket.entry_order_id}"
            )
            return {**state, "bracket": bracket_dict, "error": None}

        except Exception as exc:
            logger.error(f"[trader] place_order failed: {exc}")
            return {**state, "bracket": None, "error": str(exc), "approved": False}

    def _node_notify(self, state: TraderState) -> TraderState:
        logger.info("[trader] notify")
        alarm = state.get("alarm", {})
        notifier = get_notifier()

        if state.get("error"):
            msg = TRADE_ERROR_MSG.format(
                symbol=cfg.asset.symbol,
                direction=alarm.get("action", "?"),
                trigger_price=alarm.get("trigger_price", 0),
                error=state["error"],
            )
        elif not state.get("approved"):
            msg = TRADE_REJECTED_MSG.format(
                symbol=cfg.asset.symbol,
                direction=alarm.get("action", "?"),
                trigger_price=alarm.get("trigger_price", 0),
                reason=state.get("reject_reason", "Unknown reason"),
            )
        else:
            sizing = state.get("sizing", {})
            bracket = state.get("bracket", {})
            msg = TRADE_OPENED_MSG.format(
                symbol=cfg.asset.symbol,
                direction=alarm.get("action", "?"),
                entry_price=alarm.get("trigger_price", 0),
                stop_loss=alarm.get("stop_loss", 0),
                target_price=alarm.get("target_price", 0),
                quantity=sizing.get("quantity", 0),
                notional=sizing.get("notional", 0),
                risk_reward=alarm.get("risk_reward", 0),
                confidence=alarm.get("confidence", 0),
                order_id=bracket.get("entry_order_id", "N/A"),
            )

        try:
            run_on_main_loop(notifier.send(msg, parse_mode="Markdown"))
        except Exception as exc:
            logger.warning(f"[trader] Telegram notify failed: {exc}")

        return state

    def _node_persist(self, state: TraderState) -> TraderState:
        logger.info("[trader] persist")

        if not state.get("approved") or not state.get("bracket"):
            return state

        alarm = state["alarm"]
        sizing = state["sizing"]
        bracket = state["bracket"]

        try:
            with get_session() as session:
                trade = Trade(
                    alarm_id=alarm["id"],
                    asset=cfg.asset.symbol,
                    direction=alarm["action"],
                    entry_price=alarm["trigger_price"],
                    quantity=sizing["quantity"],
                    notional=sizing["notional"],
                    stop_loss=alarm["stop_loss"],
                    target_price=alarm["target_price"],
                    ibkr_order_id=str(bracket.get("entry_order_id", "")),
                    fill_price=bracket.get("fill_price"),
                    slippage_pct=bracket.get("slippage_pct"),
                    status="open",
                    opened_at=datetime.utcnow(),
                )
                session.add(trade)
                session.flush()
                trade_id = trade.id
            logger.info(f"[trader] Trade persisted id={trade_id}")
            return {**state, "trade_id": trade_id}
        except Exception as exc:
            logger.error(f"[trader] Could not persist trade: {exc}")
            return {**state, "trade_id": None}

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, alarm_id: int, trigger_price: float, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the Trader Agent for one triggered alarm.

        Args:
            alarm_id:      DB id of the triggered Alarm
            trigger_price: Actual price at which the alarm was fired

        Returns:
            Final state dict with 'approved', 'trade_id', 'error' keys.
        """
        logger.info(f"[trader] Starting trade execution alarm_id={alarm_id} price={trigger_price}")
        run_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()
        initial_state: TraderState = {
            "alarm_id": alarm_id,
            "trigger_price": trigger_price,
        }
        final_state = self._graph.invoke(initial_state)
        duration_ms = int((time.monotonic() - t0) * 1000)
        result = dict(final_state)
        self._log_run(run_id, duration_ms, result)
        logger.info(
            f"[trader] Execution complete ({duration_ms}ms) — "
            f"approved={result.get('approved')} "
            f"trade_id={result.get('trade_id')}"
        )
        return result


