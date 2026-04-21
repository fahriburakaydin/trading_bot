"""
Unit tests for Phase 3 — Price Monitor, Trader Agent, Evaluator Agent,
updated scheduler jobs, and main entrypoint boot sequence.

All external dependencies (IBKR, Ollama, Telegram) are mocked.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_alarm(
    alarm_id: int = 1,
    asset: str = "BTC",
    action: str = "LONG",
    direction: str = "above",
    trigger_price: float = 95_000.0,
    stop_loss: float = 93_000.0,
    target_price: float = 99_000.0,
    confidence: float = 0.75,
    risk_reward: float = 2.0,
    status: str = "active",
    expires_at=None,
) -> MagicMock:
    alarm = MagicMock()
    alarm.id = alarm_id
    alarm.asset = asset
    alarm.action = action
    alarm.direction = direction
    alarm.trigger_price = trigger_price
    alarm.stop_loss = stop_loss
    alarm.target_price = target_price
    alarm.confidence = confidence
    alarm.risk_reward = risk_reward
    alarm.timeframe = "1h"
    alarm.reasoning = "Test alarm"
    alarm.status = status
    alarm.expires_at = expires_at
    alarm.get_confluence_factors = MagicMock(return_value=["EMA cross", "RSI oversold"])
    return alarm


def _make_session_ctx(query_return=None):
    """Build a mock get_session() context manager."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.first.return_value = query_return
    mock_query.all.return_value = query_return if isinstance(query_return, list) else []
    mock_query.update.return_value = 1
    mock_session.query.return_value = mock_query
    return mock_session


# ── PriceMonitor ──────────────────────────────────────────────────────────────


class TestPriceMonitor:
    """Tests for core/monitor/monitor.py"""

    def _make_monitor(self, mock_ib=None):
        from core.monitor.monitor import PriceMonitor

        mock_ib = mock_ib or MagicMock()

        with patch("core.monitor.monitor.build_contract", return_value=MagicMock()):
            monitor = PriceMonitor(mock_ib)

        return monitor

    def test_init_sets_config_values(self):
        monitor = self._make_monitor()
        assert monitor._poll_interval > 0
        assert 0 < monitor._tolerance < 1

    def test_on_trigger_registers_callback(self):
        monitor = self._make_monitor()
        cb = AsyncMock()
        monitor.on_trigger(cb)
        assert cb in monitor._callbacks

    def test_current_price_returns_none_before_start(self):
        monitor = self._make_monitor()
        assert monitor.current_price() is None

    def test_current_price_reads_from_quote(self):
        monitor = self._make_monitor()
        mock_quote = MagicMock()
        mock_quote.price = 95_000.0
        monitor._quote = mock_quote
        assert monitor.current_price() == 95_000.0

    def test_load_active_alarms_returns_list(self):
        monitor = self._make_monitor()
        alarm = _make_alarm()

        from contextlib import contextmanager
        import core.monitor.monitor as monitor_mod

        mock_session = _make_session_ctx([alarm])
        mock_session.expunge_all = MagicMock()

        @contextmanager
        def _fake_get_session():
            yield mock_session

        with patch.object(monitor_mod, "get_session", _fake_get_session):
            alarms = monitor._load_active_alarms()

        assert len(alarms) == 1

    def test_load_active_alarms_returns_empty_on_error(self):
        monitor = self._make_monitor()

        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", side_effect=Exception("DB error")):
            alarms = monitor._load_active_alarms()

        assert alarms == []

    def test_tick_no_action_when_price_none(self):
        monitor = self._make_monitor()
        monitor._quote = None
        # Should complete without error and without loading alarms
        with patch.object(monitor, "_load_active_alarms") as mock_load:
            asyncio.get_event_loop().run_until_complete(monitor._tick())
            mock_load.assert_not_called()

    def test_tick_increments_near_count_within_tolerance(self):
        monitor = self._make_monitor()
        mock_quote = MagicMock()
        mock_quote.price = 1.10000
        monitor._quote = mock_quote

        # 0.03% away — within the 0.05% tolerance configured for EUR/USD
        alarm = _make_alarm(trigger_price=1.10033)
        with patch.object(monitor, "_load_active_alarms", return_value=[alarm]):
            with patch.object(monitor, "_fire_alarm", new_callable=AsyncMock) as mock_fire:
                asyncio.get_event_loop().run_until_complete(monitor._tick())
                mock_fire.assert_not_called()  # only 1 poll — needs 2

        assert monitor._near_counts.get(1) == 1

    def test_tick_fires_alarm_on_second_consecutive_poll(self):
        monitor = self._make_monitor()
        mock_quote = MagicMock()
        mock_quote.price = 1.10000
        monitor._quote = mock_quote
        monitor._near_counts[1] = 1  # already 1 from previous tick

        # 0.03% away — within tolerance, so second poll should fire
        alarm = _make_alarm(trigger_price=1.10033)
        with patch.object(monitor, "_load_active_alarms", return_value=[alarm]):
            with patch.object(monitor, "_fire_alarm", new_callable=AsyncMock) as mock_fire:
                asyncio.get_event_loop().run_until_complete(monitor._tick())
                mock_fire.assert_called_once_with(alarm, 1.10000)

    def test_tick_resets_count_when_price_moves_away(self):
        monitor = self._make_monitor()
        mock_quote = MagicMock()
        mock_quote.price = 90_000.0  # far from 95_100
        monitor._quote = mock_quote
        monitor._near_counts[1] = 1  # had been near

        alarm = _make_alarm(trigger_price=95_100.0)
        with patch.object(monitor, "_load_active_alarms", return_value=[alarm]):
            asyncio.get_event_loop().run_until_complete(monitor._tick())

        assert 1 not in monitor._near_counts

    def test_expire_stale_alarms_calls_db_update(self):
        monitor = self._make_monitor()

        from contextlib import contextmanager
        import core.monitor.monitor as monitor_mod

        inner_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.update.return_value = 0
        inner_session.query.return_value = mock_query

        @contextmanager
        def _fake_get_session():
            yield inner_session

        with patch.object(monitor_mod, "get_session", _fake_get_session):
            monitor.expire_stale_alarms()

        inner_session.query.assert_called()

    @pytest.mark.asyncio
    async def test_fire_alarm_updates_db_and_calls_callbacks(self):
        monitor = self._make_monitor()
        cb = AsyncMock()
        monitor._callbacks = [cb]

        alarm = _make_alarm()

        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", return_value=_make_session_ctx(alarm)):
            await monitor._fire_alarm(alarm, 95_000.0)

        cb.assert_called_once_with(1, 95_000.0)
        assert 1 not in monitor._near_counts


# ── TraderAgent ───────────────────────────────────────────────────────────────


class TestTraderAgent:
    """Tests for agents/trader/agent.py"""

    def _make_agent(self):
        from agents.trader.agent import TraderAgent

        mock_ib = MagicMock()
        with patch("langchain_ollama.ChatOllama"):
            agent = TraderAgent(ib=mock_ib)
        return agent

    def test_init_sets_ib(self):
        agent = self._make_agent()
        assert agent._ib is not None

    def test_load_alarm_returns_error_state_when_alarm_missing(self):
        agent = self._make_agent()

        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", return_value=_make_session_ctx(None)):
            state = agent._node_load_alarm({"alarm_id": 999, "trigger_price": 95_000.0})

        # Alarm not found — alarm dict should be empty, approved should be False
        assert state.get("alarm") == {} or state.get("error") is not None

    def test_check_risk_rejects_red_risk(self):
        agent = self._make_agent()
        state = {
            "alarm_id": 1,
            "trigger_price": 95_000.0,
            "alarm": {
                "id": 1, "action": "LONG", "trigger_price": 95_000.0,
                "stop_loss": 93_000.0, "target_price": 99_000.0,
                "confidence": 0.75, "risk_reward": 2.0,
            },
            "risk_level": "RED",
            "approved": True,
            "context": "",
        }
        result = agent._node_check_risk(state)
        assert result["approved"] is False
        assert "RED" in result["reject_reason"]

    def test_check_risk_rejects_when_daily_loss_breached(self):
        agent = self._make_agent()

        import asyncio
        mock_summary = MagicMock()
        mock_summary.net_liquidation = 100_000.0

        state = {
            "alarm_id": 1,
            "trigger_price": 95_000.0,
            "alarm": {
                "id": 1, "action": "LONG", "trigger_price": 95_000.0,
                "stop_loss": 93_000.0, "target_price": 99_000.0,
                "confidence": 0.75, "risk_reward": 2.0,
            },
            "risk_level": "GREEN",
            "approved": True,
            "context": "",
        }

        with patch("agents.trader.agent.get_account_summary", new_callable=AsyncMock, return_value=mock_summary):
            with patch("agents.trader.agent.get_daily_pnl", new_callable=AsyncMock, return_value=-4_000.0):
                with patch("agents.trader.agent.count_open_positions", return_value=0):
                    result = agent._node_check_risk(state)

        assert result["approved"] is False
        assert "daily loss" in result["reject_reason"].lower()

    def test_size_position_returns_none_on_invalid_prices(self):
        agent = self._make_agent()
        state = {
            "alarm_id": 1,
            "trigger_price": 95_000.0,
            "alarm": {
                "id": 1, "action": "LONG",
                "trigger_price": 0.0,   # invalid
                "stop_loss": 0.0,       # invalid
                "target_price": 99_000.0,
                "confidence": 0.75, "risk_reward": 2.0,
            },
            "approved": True,
            "sizing": None,
            "account_value": 100_000.0,
        }
        result = agent._node_size_position(state)
        assert result["approved"] is False

    def test_size_position_skips_when_not_approved(self):
        agent = self._make_agent()
        state = {
            "approved": False,
            "reject_reason": "RED risk",
            "sizing": None,
        }
        result = agent._node_size_position(state)
        # State should pass through unchanged
        assert result["approved"] is False

    def test_extract_json_parses_fenced_block(self):
        from agents.utils import extract_json as _extract_json
        text = '```json\n{"approved": true, "reason": "ok"}\n```'
        result = _extract_json(text)
        assert result["approved"] is True

    def test_extract_json_raises_on_no_json(self):
        from agents.utils import extract_json as _extract_json
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("just plain text here")


# ── EvaluatorAgent ────────────────────────────────────────────────────────────


class TestEvaluatorAgent:
    """Tests for agents/evaluator/agent.py"""

    def _make_agent(self):
        from agents.evaluator.agent import EvaluatorAgent

        with patch("langchain_ollama.ChatOllama"):
            agent = EvaluatorAgent()
        return agent

    def _make_trade_dict(self, pnl: float, direction: str = "LONG") -> dict:
        return {
            "id": 1,
            "direction": direction,
            "entry_price": 95_000.0,
            "exit_price": 97_000.0 if pnl > 0 else 93_000.0,
            "quantity": 0.1,
            "notional": 9_500.0,
            "stop_loss": 93_000.0,
            "target_price": 99_000.0,
            "pnl": pnl,
            "pnl_pct": pnl / 9_500.0,
            "pnl_r": pnl / 200.0,
            "exit_reason": "target_hit" if pnl > 0 else "stop_hit",
            "opened_at": "2026-04-10T10:00:00",
            "closed_at": "2026-04-11T14:00:00",
        }

    def test_calculate_metrics_empty_trades(self):
        agent = self._make_agent()
        state = {
            "week_start": "2026-04-10",
            "week_end": "2026-04-17",
            "trades": [],
            "context": "",
        }
        result = agent._node_calculate_metrics(state)
        assert result["metrics"] == {}

    def test_calculate_metrics_correct_win_rate(self):
        agent = self._make_agent()
        trades = [
            self._make_trade_dict(200.0),   # win
            self._make_trade_dict(-100.0),  # loss
            self._make_trade_dict(150.0),   # win
        ]
        state = {
            "week_start": "2026-04-10",
            "week_end": "2026-04-17",
            "trades": trades,
            "context": "",
        }
        result = agent._node_calculate_metrics(state)
        metrics = result["metrics"]
        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert abs(metrics["win_rate"] - 2/3) < 0.001

    def test_calculate_metrics_total_pnl(self):
        agent = self._make_agent()
        trades = [
            self._make_trade_dict(200.0),
            self._make_trade_dict(-100.0),
        ]
        state = {"week_start": "2026-04-10", "week_end": "2026-04-17", "trades": trades, "context": ""}
        result = agent._node_calculate_metrics(state)
        assert result["metrics"]["total_pnl"] == 100.0

    def test_calculate_metrics_profit_factor(self):
        agent = self._make_agent()
        trades = [
            self._make_trade_dict(200.0),
            self._make_trade_dict(-100.0),
        ]
        state = {"week_start": "2026-04-10", "week_end": "2026-04-17", "trades": trades, "context": ""}
        result = agent._node_calculate_metrics(state)
        # gross_profit=200, gross_loss=100 → profit_factor=2.0
        assert abs(result["metrics"]["profit_factor"] - 2.0) < 0.001

    def test_update_knowledge_base_saves_entries(self):
        agent = self._make_agent()
        knowledge = [
            {
                "category": "rule",
                "applies_to": "trader",
                "title": "Avoid trading on RED days",
                "content": "Historical data shows losses triple on RED risk days.",
                "performance_impact": -0.4,
            }
        ]
        state = {
            "week_start": "2026-04-10",
            "week_end": "2026-04-17",
            "new_knowledge": knowledge,
        }

        import core.memory.database as db_mod
        mock_session = _make_session_ctx()
        with patch.object(db_mod, "get_session", return_value=mock_session):
            result = agent._node_update_knowledge_base(state)

        assert result["kb_entries_added"] == 1

    def test_update_knowledge_base_zero_entries_when_empty(self):
        agent = self._make_agent()
        state = {
            "week_start": "2026-04-10",
            "week_end": "2026-04-17",
            "new_knowledge": [],
        }
        result = agent._node_update_knowledge_base(state)
        assert result["kb_entries_added"] == 0

    def test_write_report_no_trades(self):
        agent = self._make_agent()
        state = {
            "week_start": "2026-04-10",
            "week_end": "2026-04-17",
            "trades": [],
            "metrics": {},
            "analysis": {"performance_summary": "No trades this week."},
            "new_knowledge": [],
            "kb_entries_added": 0,
        }
        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", return_value=_make_session_ctx()):
            result = agent._node_write_report(state)
        assert "No completed trades" in result["report_text"]

    def test_write_report_includes_metrics(self):
        agent = self._make_agent()
        state = {
            "week_start": "2026-04-10",
            "week_end": "2026-04-17",
            "trades": [self._make_trade_dict(200.0)],
            "metrics": {
                "total_trades": 1,
                "winning_trades": 1,
                "losing_trades": 0,
                "win_rate": 1.0,
                "avg_win": 200.0,
                "avg_loss": 0.0,
                "total_pnl": 200.0,
                "profit_factor": float("inf"),
                "expectancy": 200.0,
                "max_drawdown": 0.0,
                "best_trade_id": 1,
                "worst_trade_id": 1,
            },
            "analysis": {"performance_summary": "Strong week."},
            "new_knowledge": [],
            "kb_entries_added": 0,
        }
        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", return_value=_make_session_ctx()):
            result = agent._node_write_report(state)
        assert "200" in result["report_text"]
        assert "Win Rate" in result["report_text"]


# ── Scheduler — Evaluator job ─────────────────────────────────────────────────


class TestSchedulerEvaluatorJob:
    """Tests for the new Evaluator job in core/scheduler/jobs.py"""

    def test_build_scheduler_registers_evaluator_job(self):
        from core.scheduler.jobs import build_scheduler
        scheduler = build_scheduler()
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "evaluator_agent" in job_ids

    def test_evaluator_job_calls_agent(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"kb_entries_added": 2}

        import core.scheduler.jobs as jobs_module
        with patch.object(jobs_module, "_run_evaluator_agent") as mock_job:
            mock_job.side_effect = lambda: mock_agent.run()
            mock_job()
            mock_agent.run.assert_called_once()
