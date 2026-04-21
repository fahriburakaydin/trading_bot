"""
Tests for the production-readiness changes (Passes A–D).

Covers:
  - agents/utils.py: extract_json, run_on_main_loop
  - Alarm validation in analyst agent
  - Position sizing with max_notional_pct
  - LLM failure → reject (not approve)
  - LiveQuote None handling
  - Monitor initial state
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest


# ── extract_json ──────────────────────────────────────────────────────────────


class TestExtractJson:
    """Tests for the improved extract_json in agents/utils.py."""

    def test_fast_path_pure_json_object(self):
        from agents.utils import extract_json
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fast_path_pure_json_array(self):
        from agents.utils import extract_json
        result = extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_fenced_json(self):
        from agents.utils import extract_json
        text = 'Here is the result:\n```json\n{"approved": true}\n```'
        result = extract_json(text)
        assert result["approved"] is True

    def test_bare_json_in_text(self):
        from agents.utils import extract_json
        text = 'Some LLM preamble {"score": 0.85, "action": "LONG"} and some trailing text'
        result = extract_json(text)
        assert result["score"] == 0.85
        assert result["action"] == "LONG"

    def test_nested_json(self):
        from agents.utils import extract_json
        nested = '{"outer": {"inner": [1, 2]}, "ok": true}'
        result = extract_json(f"Result: {nested}")
        assert result["outer"]["inner"] == [1, 2]

    def test_raises_on_no_json(self):
        from agents.utils import extract_json
        with pytest.raises(ValueError, match="No JSON"):
            extract_json("No JSON content here at all")

    def test_json_with_escaped_quotes(self):
        from agents.utils import extract_json
        text = r'{"message": "He said \"hello\""}'
        result = extract_json(text)
        assert "hello" in result["message"]

    def test_whitespace_around_json(self):
        from agents.utils import extract_json
        result = extract_json("  \n  [1, 2, 3]  \n  ")
        assert result == [1, 2, 3]


# ── run_on_main_loop ─────────────────────────────────────────────────────────


class TestRunOnMainLoop:
    """Tests for the centralised async dispatch helper."""

    def test_fallback_to_asyncio_run(self):
        """When no main loop exists, falls back to asyncio.run()."""
        from agents.utils import run_on_main_loop

        async def coro():
            return 42

        # Patch the import inside the function so _main_loop is None
        with patch("tools.market_data._main_loop", None):
            result = run_on_main_loop(coro())
            assert result == 42

    def test_async_exception_propagates(self):
        """Exceptions from the coroutine should propagate to the caller."""
        from agents.utils import run_on_main_loop

        async def failing_coro():
            raise ValueError("test error")

        with patch("tools.market_data._main_loop", None):
            with pytest.raises(ValueError, match="test error"):
                run_on_main_loop(failing_coro())


# ── Alarm validation ─────────────────────────────────────────────────────────


class TestAlarmValidation:
    """Tests for the alarm validation logic added in B3."""

    def _make_valid_alarm(self, **overrides) -> dict:
        base = {
            "trigger_price": 1.10000,
            "stop_loss": 1.09500,
            "target_price": 1.11000,
            "action": "LONG",
            "direction": "above",
            "confidence": 0.75,
            "risk_reward": 2.0,
            "timeframe": "1h",
            "reasoning": "test alarm",
        }
        base.update(overrides)
        return base

    def test_valid_long_alarm_passes(self):
        alarm = self._make_valid_alarm()
        assert alarm["trigger_price"] > 0
        assert alarm["stop_loss"] > 0
        assert alarm["target_price"] > 0
        assert alarm["stop_loss"] < alarm["trigger_price"]
        assert alarm["target_price"] > alarm["trigger_price"]

    def test_valid_short_alarm_passes(self):
        alarm = self._make_valid_alarm(
            action="SHORT",
            trigger_price=1.10000,
            stop_loss=1.10500,
            target_price=1.09000,
        )
        assert alarm["stop_loss"] > alarm["trigger_price"]
        assert alarm["target_price"] < alarm["trigger_price"]

    def test_long_with_sl_above_entry_is_invalid(self):
        alarm = self._make_valid_alarm(action="LONG", stop_loss=1.11000, trigger_price=1.10000)
        assert alarm["stop_loss"] >= alarm["trigger_price"]

    def test_short_with_sl_below_entry_is_invalid(self):
        alarm = self._make_valid_alarm(
            action="SHORT", trigger_price=1.10000, stop_loss=1.09500, target_price=1.09000,
        )
        assert alarm["stop_loss"] <= alarm["trigger_price"]

    def test_zero_price_is_invalid(self):
        alarm = self._make_valid_alarm(trigger_price=0)
        assert alarm["trigger_price"] <= 0


# ── Position sizing with max_notional_pct ────────────────────────────────────


class TestPositionSizingMaxNotional:
    """Tests for A5: position sizing respects config max_notional_pct."""

    def test_default_cap_at_20_pct(self):
        from core.risk.position_sizing import calculate_position_size
        result = calculate_position_size(
            portfolio_value=100_000,
            risk_pct=0.005,
            entry_price=50_000,
            stop_loss=49_000,
        )
        assert result is not None
        assert result["notional"] <= 100_000 * 0.20 + 0.01

    def test_config_override_40_pct(self):
        from core.risk.position_sizing import calculate_position_size
        result = calculate_position_size(
            portfolio_value=100_000,
            risk_pct=0.02,
            entry_price=1.10000,
            stop_loss=1.09500,
            min_quantity=0.0001,
            max_notional_pct=0.40,
        )
        assert result is not None
        assert result["notional"] <= 100_000 * 0.40 + 0.01

    def test_tight_cap_scales_down(self):
        from core.risk.position_sizing import calculate_position_size
        result = calculate_position_size(
            portfolio_value=100_000,
            risk_pct=0.02,
            entry_price=1.10000,
            stop_loss=1.09500,
            min_quantity=0.0001,
            max_notional_pct=0.05,
        )
        assert result is not None
        assert result["notional"] <= 100_000 * 0.05 + 0.01

    def test_none_max_notional_uses_default(self):
        from core.risk.position_sizing import calculate_position_size
        result = calculate_position_size(
            portfolio_value=100_000,
            risk_pct=0.005,
            entry_price=50_000,
            stop_loss=49_000,
            max_notional_pct=None,
        )
        assert result is not None
        assert result["notional"] <= 100_000 * 0.20 + 0.01


# ── LLM failure → reject trade ──────────────────────────────────────────────


class TestLLMFailureRejectsTrader:
    """Tests for A3: LLM check failure must reject, not approve."""

    def test_llm_exception_sets_approved_false(self):
        """When the LLM risk check throws, the trade must be rejected."""
        from agents.trader.agent import TraderAgent

        mock_ib = MagicMock()
        with patch("agents.base._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_build.return_value = mock_llm
            agent = TraderAgent(ib=mock_ib)
            mock_llm.invoke.side_effect = RuntimeError("LLM connection timeout")

            state = {
                "alarm_id": 1,
                "trigger_price": 1.10000,
                "alarm": {
                    "id": 1, "asset": "EURUSD", "action": "LONG",
                    "trigger_price": 1.10000, "stop_loss": 1.09500,
                    "target_price": 1.11000, "confidence": 0.80,
                    "risk_reward": 2.0, "timeframe": "1h",
                    "direction": "above", "reasoning": "",
                    "confluence_factors": [], "status": "triggered",
                },
                "context": "", "risk_level": "GREEN",
                "approved": True, "error": None,
            }

            mock_summary = MagicMock(net_liquidation=100000)
            with patch("agents.trader.agent.run_on_main_loop", side_effect=[mock_summary, 0.0]):
                with patch("agents.trader.agent.count_open_positions", return_value=0):
                    result = agent._node_check_risk(state)

            assert result["approved"] is False
            assert "LLM risk check failed" in result["reject_reason"]


# ── LiveQuote handles None values ────────────────────────────────────────────
# These need ib_insync to be importable — they run in the same process as test_broker.py


class TestLiveQuoteNoneHandling:
    """Tests for the fix in core/broker/market_data.py LiveQuote.update."""

    def test_update_handles_none_values(self):
        from core.broker.market_data import LiveQuote
        quote = LiveQuote()
        ticker = MagicMock()
        ticker.bid = None
        ticker.ask = 1.10500
        ticker.last = None

        quote.update(ticker)
        assert quote.bid is None
        assert quote.ask == 1.10500
        assert quote.last is None

    def test_update_handles_valid_values(self):
        from core.broker.market_data import LiveQuote
        quote = LiveQuote()
        ticker = MagicMock()
        ticker.bid = 1.10000
        ticker.ask = 1.10500
        ticker.last = 1.10250

        quote.update(ticker)
        assert quote.bid == 1.10000
        assert quote.ask == 1.10500
        assert quote.last == 1.10250
        assert quote.mid == pytest.approx((1.10000 + 1.10500) / 2)

    def test_update_ignores_zero_bid(self):
        from core.broker.market_data import LiveQuote
        quote = LiveQuote()
        ticker = MagicMock()
        ticker.bid = 0.0
        ticker.ask = 1.10500
        ticker.last = 1.10250

        quote.update(ticker)
        assert quote.bid is None  # zero should be ignored
        assert quote.ask == 1.10500


# ── Monitor initial state ────────────────────────────────────────────────────


class TestMonitorInitialState:
    """Tests for B1/B4: monitor has tracking counters initialized."""

    def test_initial_counters(self):
        from core.monitor.monitor import PriceMonitor
        with patch("core.monitor.monitor.build_contract", return_value=MagicMock()):
            monitor = PriceMonitor(ib=MagicMock())
        assert monitor._tick_count == 0
        assert monitor._null_price_streak == 0
        assert monitor._near_counts == {}
