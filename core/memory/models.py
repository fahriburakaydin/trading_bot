"""
SQLAlchemy ORM models for all 7 database tables.

Tables:
    alarms           — Analyst price alarms
    trades           — Executed trades with full lifecycle
    research_reports — Daily research summaries
    analysis_runs    — Analyst TA run snapshots
    agent_logs       — Every agent action with reasoning
    evaluator_reports — Weekly performance reports
    knowledge_base   — Agent-learned insights + user-injected rules
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.utcnow()


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj)


# ── Models ────────────────────────────────────────────────────────────────────


class Alarm(Base):
    """
    Price alarm set by the Analyst Agent.
    Status flow: active → triggered | expired | cancelled
    """

    __tablename__ = "alarms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    asset = Column(String(20), nullable=False)
    trigger_price = Column(Float, nullable=False)
    direction = Column(String(10), nullable=False)  # "above" | "below"
    action = Column(String(10), nullable=False)      # "LONG" | "SHORT"
    confidence = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    risk_reward = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    confluence_factors = Column(Text)  # JSON list of strings
    reasoning = Column(Text)
    status = Column(String(20), default="active", nullable=False)
    triggered_at = Column(DateTime)
    expires_at = Column(DateTime)

    # Relationship
    trades = relationship("Trade", back_populates="alarm")

    def set_confluence_factors(self, factors: list[str]) -> None:
        self.confluence_factors = _json_dumps(factors)

    def get_confluence_factors(self) -> list[str]:
        if not self.confluence_factors:
            return []
        return json.loads(self.confluence_factors)

    def __repr__(self) -> str:
        return (
            f"<Alarm id={self.id} {self.asset} {self.action} @ "
            f"{self.trigger_price} [{self.status}]>"
        )


class Trade(Base):
    """
    Executed trade with full entry/exit lifecycle.
    exit_reason values: target_hit | stop_hit | trader_closed | slippage_abort | manual
    """

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alarm_id = Column(Integer, ForeignKey("alarms.id"), nullable=True)
    opened_at = Column(DateTime, default=_now, nullable=False)
    closed_at = Column(DateTime)
    asset = Column(String(20), nullable=False)
    strategy_id = Column(String(50), nullable=True)   # e.g. "btc_sr_bounce_v1"
    direction = Column(String(10), nullable=False)   # "LONG" | "SHORT"
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    notional = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    pnl = Column(Float)                              # USD P&L
    pnl_pct = Column(Float)                          # % P&L
    pnl_r = Column(Float)                            # P&L in multiples of R
    exit_reason = Column(String(30))
    ibkr_order_id = Column(String(50))
    fill_price = Column(Float)
    slippage_pct = Column(Float)
    status = Column(String(10), default="open", nullable=False)  # open | closed

    # Relationship
    alarm = relationship("Alarm", back_populates="trades")

    def close(
        self,
        exit_price: float,
        exit_reason: str,
        fill_price: float | None = None,
    ) -> None:
        """Mark trade as closed and calculate P&L."""
        self.exit_price = exit_price
        self.closed_at = _now()
        self.exit_reason = exit_reason
        self.fill_price = fill_price or exit_price
        self.status = "closed"

        if self.direction == "LONG":
            raw_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            raw_pnl = (self.entry_price - exit_price) * self.quantity

        self.pnl = round(raw_pnl, 2)
        self.pnl_pct = round(raw_pnl / self.notional, 6)

        risk_per_unit = abs(self.entry_price - self.stop_loss)
        if risk_per_unit > 0:
            self.pnl_r = round(raw_pnl / (risk_per_unit * self.quantity), 3)

    def __repr__(self) -> str:
        return (
            f"<Trade id={self.id} {self.asset} {self.direction} "
            f"{self.quantity} @ {self.entry_price} [{self.status}]>"
        )


class ResearchReport(Base):
    """Daily research summary produced by the Research Agent."""

    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    asset = Column(String(20), nullable=False)
    sentiment = Column(String(20))          # bullish | bearish | neutral
    risk_level = Column(String(20))         # low | medium | high
    key_events = Column(Text)               # JSON list of strings
    trading_recommendation = Column(String(20))  # proceed | caution | avoid
    summary = Column(Text)
    full_report = Column(Text)

    def set_key_events(self, events: list[str]) -> None:
        self.key_events = _json_dumps(events)

    def get_key_events(self) -> list[str]:
        if not self.key_events:
            return []
        return json.loads(self.key_events)

    def __repr__(self) -> str:
        return (
            f"<ResearchReport id={self.id} {self.asset} "
            f"{self.sentiment} [{self.trading_recommendation}]>"
        )


class AnalysisRun(Base):
    """Snapshot of an Analyst Agent TA run."""

    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    asset = Column(String(20), nullable=False)
    timeframes_used = Column(Text)          # JSON list e.g. ["1H", "15m", "1D"]
    alarms_set = Column(Integer, default=0)
    alarms_skipped = Column(Integer, default=0)
    avg_confidence = Column(Float)
    summary = Column(Text)
    indicators_snapshot = Column(Text)      # JSON dict of key indicator values

    def set_timeframes(self, tfs: list[str]) -> None:
        self.timeframes_used = _json_dumps(tfs)

    def get_timeframes(self) -> list[str]:
        if not self.timeframes_used:
            return []
        return json.loads(self.timeframes_used)

    def __repr__(self) -> str:
        return (
            f"<AnalysisRun id={self.id} {self.asset} "
            f"alarms={self.alarms_set}>"
        )


class AgentLog(Base):
    """
    Detailed log of every agent action for Evaluator analysis.
    One row per agent decision step.
    """

    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_now, nullable=False)
    agent = Column(String(30), nullable=False)    # research | analyst | trader | evaluator
    run_id = Column(String(50))                   # UUID grouping one agent run
    action = Column(String(100), nullable=False)
    input_data = Column(Text)                     # JSON
    output_data = Column(Text)                    # JSON
    reasoning = Column(Text)
    duration_ms = Column(Integer)
    llm_tokens_used = Column(Integer)

    def set_input(self, data: Any) -> None:
        self.input_data = _json_dumps(data)

    def set_output(self, data: Any) -> None:
        self.output_data = _json_dumps(data)

    def __repr__(self) -> str:
        return f"<AgentLog id={self.id} agent={self.agent} action={self.action}>"


class EvaluatorReport(Base):
    """Weekly performance report produced by the Evaluator Agent."""

    __tablename__ = "evaluator_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week_start = Column(DateTime, nullable=False)
    week_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)
    strategy_id = Column(String(50), nullable=True)   # strategy active during this period
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    expectancy = Column(Float)
    profit_factor = Column(Float)
    sharpe = Column(Float)
    max_drawdown = Column(Float)
    calmar = Column(Float)
    best_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    worst_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    improvement_notes = Column(Text)   # JSON list of strings
    full_report = Column(Text)

    def set_improvement_notes(self, notes: list[str]) -> None:
        self.improvement_notes = _json_dumps(notes)

    def get_improvement_notes(self) -> list[str]:
        if not self.improvement_notes:
            return []
        return json.loads(self.improvement_notes)

    def __repr__(self) -> str:
        return (
            f"<EvaluatorReport id={self.id} "
            f"week={self.week_start.date()}–{self.week_end.date()} "
            f"trades={self.total_trades} pnl={self.total_pnl}>"
        )


class KnowledgeEntry(Base):
    """
    Agent-learned insights and user-injected rules.
    source: evaluator | user_input | system
    category: pattern | rule | insight | algorithm
    applies_to: research | analyst | trader | evaluator | all
    """

    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    source = Column(String(30), nullable=False)
    category = Column(String(30), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    applies_to = Column(String(30), nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    times_applied = Column(Integer, default=0)
    last_applied_at = Column(DateTime)
    performance_impact = Column(Float)   # optional score set by Evaluator

    def __repr__(self) -> str:
        return (
            f"<KnowledgeEntry id={self.id} [{self.category}] "
            f"'{self.title}' → {self.applies_to}>"
        )
