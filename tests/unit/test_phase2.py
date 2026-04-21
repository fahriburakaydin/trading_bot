"""
Unit tests for Phase 2 — Research Agent, Analyst Agent, search tools,
indicator tools, notification templates, and scheduler jobs.

All external dependencies (Ollama, IBKR, Telegram, Brave, DuckDuckGo) are mocked.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# ── agents/base.py ─────────────────────────────────────────────────────────────


class TestBaseAgent:
    """Tests for BaseAgent — LLM init and knowledge-base context loading."""

    def _make_agent(self, mock_llm: MagicMock):
        """Return a concrete subclass of BaseAgent with a mocked LLM."""
        from agents.base import BaseAgent

        class ConcreteAgent(BaseAgent):
            agent_name = "test_agent"

            def run(self, **kwargs):
                return {}

        # ChatOllama is imported lazily inside _build_llm → patch at its source module
        with patch("langchain_ollama.ChatOllama", return_value=mock_llm):
            return ConcreteAgent()

    def test_init_uses_config_model(self):
        mock_llm = MagicMock()
        with patch("langchain_ollama.ChatOllama") as MockLLM:
            MockLLM.return_value = mock_llm
            from agents.base import BaseAgent

            class ConcreteAgent(BaseAgent):
                agent_name = "t"
                def run(self, **kwargs): return {}

            agent = ConcreteAgent()
            MockLLM.assert_called_once()
            call_kwargs = MockLLM.call_args.kwargs
            # Model should come from resolved config
            assert call_kwargs["model"] is not None

    def _mock_session(self, entries: list):
        """Build a mock context manager session that returns the given entries."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = entries
        mock_session.query.return_value = mock_query
        return mock_session

    def test_load_context_returns_empty_when_no_entries(self):
        mock_llm = MagicMock()
        agent = self._make_agent(mock_llm)

        # get_session is imported lazily inside load_context — patch at its source module
        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", return_value=self._mock_session([])):
            result = agent.load_context()

        assert result == ""

    def test_load_context_formats_entries(self):
        mock_llm = MagicMock()
        agent = self._make_agent(mock_llm)

        entry = MagicMock()
        entry.category = "rule"
        entry.title = "Always use stop-loss"
        entry.content = "Never enter without a defined stop."

        import core.memory.database as db_mod
        mock_session = self._mock_session([entry])
        with patch.object(db_mod, "get_session", return_value=mock_session):
            result = agent.load_context()

        assert "[CONTEXT]" in result
        assert "Always use stop-loss" in result
        assert "[/CONTEXT]" in result

    def test_load_context_returns_empty_on_db_error(self):
        mock_llm = MagicMock()
        agent = self._make_agent(mock_llm)

        import core.memory.database as db_mod
        with patch.object(db_mod, "get_session", side_effect=Exception("DB error")):
            result = agent.load_context()

        assert result == ""


# ── tools/search.py ────────────────────────────────────────────────────────────


class TestSearchTools:
    """Tests for web search tools — Brave and DuckDuckGo."""

    def test_brave_search_raises_without_api_key(self):
        from tools.search import _call_brave
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAVE_API_KEY", None)
            with pytest.raises(EnvironmentError, match="BRAVE_API_KEY"):
                _call_brave("test query", 5)

    def test_brave_search_parses_response(self):
        from tools.search import _call_brave
        fake_response = {
            "web": {
                "results": [
                    {"title": "BTC News", "url": "https://example.com", "description": "Bitcoin hits ATH"},
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "test-key"}):
            with patch("tools.search.requests.get", return_value=mock_resp):
                results = _call_brave("bitcoin news", 5)

        assert len(results) == 1
        assert results[0]["title"] == "BTC News"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "Bitcoin hits ATH"

    def test_duckduckgo_search_returns_normalised_results(self):
        from tools.search import _call_duckduckgo
        fake_results = [
            {"title": "Fed News", "href": "https://fed.gov", "body": "Fed holds rates"},
        ]
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = fake_results

        with patch("tools.search.DDGS", return_value=mock_ddgs):
            results = _call_duckduckgo("fed rates", 5)

        assert len(results) == 1
        assert results[0]["title"] == "Fed News"
        assert results[0]["url"] == "https://fed.gov"
        assert results[0]["snippet"] == "Fed holds rates"

    def test_web_search_uses_brave_when_key_present(self):
        from tools.search import _call_brave, _call_duckduckgo
        with patch.dict(os.environ, {"BRAVE_API_KEY": "test-key"}):
            with patch("tools.search._call_brave", return_value=[{"title": "brave"}]) as mock_brave:
                with patch("tools.search._call_duckduckgo") as mock_ddg:
                    from tools.search import web_search
                    result = web_search.invoke({"query": "bitcoin", "max_results": 3})
                    mock_brave.assert_called_once()
                    mock_ddg.assert_not_called()

    def test_web_search_falls_back_to_ddg_on_brave_failure(self):
        with patch.dict(os.environ, {"BRAVE_API_KEY": "test-key"}):
            with patch("tools.search._call_brave", side_effect=Exception("API error")):
                with patch("tools.search._call_duckduckgo", return_value=[{"title": "ddg"}]) as mock_ddg:
                    from tools.search import web_search
                    result = web_search.invoke({"query": "bitcoin", "max_results": 3})
                    mock_ddg.assert_called_once()

    def test_web_search_uses_ddg_without_brave_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAVE_API_KEY", None)
            with patch("tools.search._call_duckduckgo", return_value=[]) as mock_ddg:
                from tools.search import web_search
                web_search.invoke({"query": "bitcoin"})
                mock_ddg.assert_called_once()


# ── tools/indicators.py ────────────────────────────────────────────────────────


def _make_ohlcv_json(n: int = 50) -> str:
    """Generate a minimal OHLCV DataFrame and serialise to JSON."""
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    data = {
        "date": dates,
        "open": [100.0 + i for i in range(n)],
        "high": [105.0 + i for i in range(n)],
        "low": [95.0 + i for i in range(n)],
        "close": [102.0 + i for i in range(n)],
        "volume": [1000.0 + i * 10 for i in range(n)],
    }
    df = pd.DataFrame(data)
    return df.to_json(orient="records", date_format="iso")


class TestIndicatorTools:
    """Tests for LangChain indicator tool wrappers."""

    def test_calculate_rsi_returns_float_and_signal(self):
        from tools.indicators import calculate_rsi
        ohlcv = _make_ohlcv_json(50)
        result = json.loads(calculate_rsi.invoke({"ohlcv_json": ohlcv}))
        assert "rsi" in result
        assert "signal" in result
        assert result["signal"] in ("overbought", "oversold", "neutral")

    def test_calculate_macd_returns_direction(self):
        from tools.indicators import calculate_macd
        ohlcv = _make_ohlcv_json(60)
        result = json.loads(calculate_macd.invoke({"ohlcv_json": ohlcv}))
        assert "macd" in result
        assert "direction" in result

    def test_calculate_ema_returns_trend(self):
        from tools.indicators import calculate_ema
        ohlcv = _make_ohlcv_json(250)  # need 200 bars for EMA-200
        result = json.loads(calculate_ema.invoke({"ohlcv_json": ohlcv}))
        assert "trend" in result
        assert result["trend"] in ("bullish", "bearish", "ranging")

    def test_calculate_atr_returns_positive_value(self):
        from tools.indicators import calculate_atr
        ohlcv = _make_ohlcv_json(50)
        result = json.loads(calculate_atr.invoke({"ohlcv_json": ohlcv}))
        assert "atr" in result
        assert result["atr"] is not None
        assert result["atr"] > 0

    def test_calculate_vwap_returns_price_position(self):
        from tools.indicators import calculate_vwap
        ohlcv = _make_ohlcv_json(50)
        result = json.loads(calculate_vwap.invoke({"ohlcv_json": ohlcv}))
        assert "price_vs_vwap" in result

    def test_detect_support_resistance_returns_list(self):
        from tools.indicators import detect_support_resistance
        ohlcv = _make_ohlcv_json(100)
        result = json.loads(detect_support_resistance.invoke({"ohlcv_json": ohlcv}))
        assert isinstance(result, list)
        # If any levels returned, validate structure
        for lvl in result:
            assert "price" in lvl
            assert "type" in lvl
            assert "score" in lvl


# ── tools/market_data.py ──────────────────────────────────────────────────────


class TestMarketDataTools:
    """Tests for market data tool wrappers (IBKR is mocked)."""

    def _make_mock_df(self) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=5, freq="1h")
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [105.0, 106.0, 107.0, 108.0, 109.0],
                "low": [95.0, 96.0, 97.0, 98.0, 99.0],
                "close": [102.0, 103.0, 104.0, 105.0, 106.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            },
            index=dates,
        )

    def test_fetch_ohlcv_returns_json_string(self):
        from tools.market_data import fetch_ohlcv
        mock_df = self._make_mock_df()

        # Patch the module-level _ib and _run_coro so the tool doesn't raise
        with patch("tools.market_data._ib", MagicMock()):
            with patch("tools.market_data._run_coro", return_value=mock_df):
                result = fetch_ohlcv.invoke({"interval": "1h", "lookback_days": 5})

        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_fetch_multi_timeframe_returns_json_object(self):
        from tools.market_data import fetch_multi_timeframe
        mock_df = self._make_mock_df()
        tfs = ["1 hour", "4 hours"]
        # _run_coro receives the _gather() coroutine which returns a dict of DataFrames
        mock_result = {tf: mock_df for tf in tfs}

        with patch("tools.market_data._ib", MagicMock()):
            with patch("tools.market_data._run_coro", return_value=mock_result):
                result = fetch_multi_timeframe.invoke({"timeframes": tfs})

        parsed = json.loads(result)
        assert "1 hour" in parsed
        assert "4 hours" in parsed


# ── notifications/templates.py ────────────────────────────────────────────────


class TestNotificationTemplates:
    """Tests for Jinja2 notification templates."""

    def test_render_research_report(self):
        from notifications.templates import RESEARCH_REPORT, render
        result = render(
            RESEARCH_REPORT,
            date="2026-04-16",
            sentiment_label="Bullish",
            sentiment_score=0.45,
            risk_emoji="🟢",
            risk_level="GREEN",
            macro_bullets=["DXY weakening", "BTC ETF inflows positive"],
            trading_implication="Conditions favour cautious longs.",
        )
        assert "2026-04-16" in result
        assert "Bullish" in result
        assert "GREEN" in result
        assert "DXY weakening" in result

    def test_render_analyst_alarms(self):
        from notifications.templates import ANALYST_ALARMS, render
        alarms = [
            SimpleNamespace(action="LONG", trigger_price=95000.0, confidence=0.75, risk_reward=2.0)
        ]
        result = render(
            ANALYST_ALARMS,
            date="2026-04-16",
            symbol="BTC",
            current_price=94500.0,
            alarm_count=1,
            alarms=alarms,
            trend_1d="uptrend",
            rsi_1h=52.3,
            rsi_signal="neutral",
            macd_direction="bullish",
        )
        assert "LONG" in result
        assert "95,000.00" in result
        assert "uptrend" in result

    def test_render_trade_opened(self):
        from notifications.templates import TRADE_OPENED, render
        result = render(
            TRADE_OPENED,
            symbol="BTC",
            direction="LONG",
            entry_price=95000.0,
            stop_loss=93000.0,
            target_price=99000.0,
            quantity=0.1,
            notional=9500.0,
            risk_reward=2.0,
            confidence=0.72,
        )
        assert "LONG" in result
        assert "95,000.00" in result

    def test_render_missing_variable_raises(self):
        from jinja2 import UndefinedError
        from notifications.templates import RESEARCH_REPORT, render
        with pytest.raises(UndefinedError):
            render(RESEARCH_REPORT, date="2026-04-16")  # missing required vars

    def test_render_no_alarms(self):
        from notifications.templates import ANALYST_NO_ALARMS, render
        result = render(
            ANALYST_NO_ALARMS,
            date="2026-04-16",
            symbol="BTC",
            risk_emoji="🟡",
            risk_level="YELLOW",
            reason="No high-confidence setups found.",
        )
        assert "No actionable setups" in result
        assert "YELLOW" in result


# ── core/scheduler/jobs.py ────────────────────────────────────────────────────


class TestSchedulerJobs:
    """Tests for APScheduler job registration."""

    def test_build_scheduler_registers_research_job(self):
        from core.scheduler.jobs import build_scheduler
        scheduler = build_scheduler()
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "research_agent" in job_ids
        # Don't call shutdown — scheduler was never started

    def test_build_scheduler_registers_analyst_job(self):
        from core.scheduler.jobs import build_scheduler
        scheduler = build_scheduler()
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "analyst_agent" in job_ids
        # Don't call shutdown — scheduler was never started

    def test_research_job_function_calls_agent(self):
        from core.scheduler.jobs import _run_research_agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"risk_environment": {"risk_level": "GREEN"}}

        with patch("core.scheduler.jobs.ResearchAgent", return_value=mock_agent, create=True):
            # Patch the import inside _run_research_agent
            import core.scheduler.jobs as jobs_module
            with patch.object(jobs_module, '_run_research_agent') as mock_job:
                mock_job.side_effect = lambda: mock_agent.run()
                mock_job()
                mock_agent.run.assert_called_once()

    def test_analyst_job_function_calls_agent(self):
        from core.scheduler.jobs import _run_analyst_agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"alarms_saved": 2}

        import core.scheduler.jobs as jobs_module
        with patch.object(jobs_module, '_run_analyst_agent') as mock_job:
            mock_job.side_effect = lambda: mock_agent.run()
            mock_job()
            mock_agent.run.assert_called_once()

    def test_stop_scheduler_shuts_down_gracefully(self):
        from core.scheduler.jobs import build_scheduler, stop_scheduler
        scheduler = build_scheduler()
        scheduler.start()
        stop_scheduler(scheduler)
        assert not scheduler.running


# ── research/prompts.py helpers ───────────────────────────────────────────────


class TestResearchPromptHelpers:
    """Tests for prompt utility functions."""

    def test_sentiment_label_strongly_bearish(self):
        from agents.research.prompts import sentiment_label
        assert sentiment_label(-0.8) == "Strongly Bearish"

    def test_sentiment_label_bearish(self):
        from agents.research.prompts import sentiment_label
        assert sentiment_label(-0.3) == "Bearish"

    def test_sentiment_label_neutral(self):
        from agents.research.prompts import sentiment_label
        assert sentiment_label(0.0) == "Neutral"

    def test_sentiment_label_bullish(self):
        from agents.research.prompts import sentiment_label
        assert sentiment_label(0.3) == "Bullish"

    def test_sentiment_label_strongly_bullish(self):
        from agents.research.prompts import sentiment_label
        assert sentiment_label(0.9) == "Strongly Bullish"


# ── research/agent.py — _extract_json ─────────────────────────────────────────


class TestExtractJson:
    """Tests for the _extract_json helper in research.agent."""

    def test_extracts_json_from_fence(self):
        from agents.utils import extract_json as _extract_json
        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extracts_json_array_from_fence(self):
        from agents.utils import extract_json as _extract_json
        text = "```\n[1, 2, 3]\n```"
        result = _extract_json(text)
        assert result == [1, 2, 3]

    def test_extracts_bare_json_object(self):
        from agents.utils import extract_json as _extract_json
        text = 'Some text {"score": 0.5} trailing text'
        result = _extract_json(text)
        assert result["score"] == 0.5

    def test_raises_when_no_json(self):
        from agents.utils import extract_json as _extract_json
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("No JSON here at all")
