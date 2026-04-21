"""
Research Agent — daily macro and sentiment analysis.

LangGraph flow:
  load_context → gather_news → scrape_economic_calendar
      → analyze_sentiment → assess_risk_environment
      → write_report → notify → END

Runs each morning at 10:00 Istanbul time (configured in config.yaml).
Output is stored in the ``research_reports`` SQLite table and a Telegram
summary is sent to the configured chat.
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
from agents.utils import extract_json, run_on_main_loop
from agents.research.prompts import (
    ECONOMIC_CALENDAR_PROMPT,
    GATHER_NEWS_PROMPT,
    RISK_ENVIRONMENT_PROMPT,
    SENTIMENT_PROMPT,
    SYSTEM_PROMPT,
    TELEGRAM_SUMMARY_TEMPLATE,
    WRITE_REPORT_PROMPT,
    RISK_LEVEL_EMOJI,
    sentiment_label,
)
from core.memory.database import get_session
from core.memory.models import ResearchReport
from notifications.telegram import get_notifier
from tools.search import web_search


# ── State definition ──────────────────────────────────────────────────────────


class ResearchState(TypedDict, total=False):
    """Shared state threaded through every node in the graph."""

    date: str                         # ISO date string "YYYY-MM-DD"
    context: str                      # knowledge-base context block
    news_findings: list[dict]         # raw news results
    economic_calendar: list[dict]     # calendar events
    sentiment: dict                   # sentiment scores
    risk_environment: dict            # risk assessment
    report_text: str                  # final formatted report
    error: str | None                 # set if any node fails


# ── Agent class ───────────────────────────────────────────────────────────────


class ResearchAgent(BaseAgent):
    """Runs the daily macro intelligence workflow."""

    agent_name = "research"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model=model)
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(ResearchState)

        g.add_node("load_context", self._node_load_context)
        g.add_node("gather_news", self._node_gather_news)
        g.add_node("scrape_economic_calendar", self._node_scrape_economic_calendar)
        g.add_node("analyze_sentiment", self._node_analyze_sentiment)
        g.add_node("assess_risk_environment", self._node_assess_risk_environment)
        g.add_node("write_report", self._node_write_report)
        g.add_node("notify", self._node_notify)

        g.set_entry_point("load_context")
        g.add_edge("load_context", "gather_news")
        g.add_edge("gather_news", "scrape_economic_calendar")
        g.add_edge("scrape_economic_calendar", "analyze_sentiment")
        g.add_edge("analyze_sentiment", "assess_risk_environment")
        g.add_edge("assess_risk_environment", "write_report")
        g.add_edge("write_report", "notify")
        g.add_edge("notify", END)

        return g.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_load_context(self, state: ResearchState) -> ResearchState:
        today = datetime.utcnow().date().isoformat()
        context = self.load_context()
        symbol = self.cfg.asset.symbol
        asset_class = self.cfg.asset.type.lower()  # "forex", "crypto", "stk"
        logger.info(f"[research] load_context — date={today} symbol={symbol}")
        return {**state, "date": today, "context": context, "symbol": symbol, "asset_class": asset_class}

    def _node_gather_news(self, state: ResearchState) -> ResearchState:
        logger.info("[research] gather_news — fetching web search results")
        today = state["date"]
        context = state.get("context", "")
        symbol = state.get("symbol", self.cfg.asset.symbol)
        asset_class = state.get("asset_class", self.cfg.asset.type.lower())

        system = SYSTEM_PROMPT.format(context=context, symbol=symbol, asset_class=asset_class)
        prompt = GATHER_NEWS_PROMPT.format(date=today, symbol=symbol, asset_class=asset_class)

        llm_with_tools = self.llm.bind_tools([web_search])
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        findings: list[dict] = []
        # Run tool-use loop
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Execute any tool calls the LLM requested
        for tool_call in getattr(response, "tool_calls", []):
            try:
                result = web_search.invoke(tool_call["args"])
                findings.extend(result if isinstance(result, list) else [])
            except Exception as exc:
                logger.warning(f"[research] gather_news tool call failed: {exc}")

        # C3: If LLM didn't call any tools, re-prompt once with explicit instruction
        if not getattr(response, "tool_calls", []):
            logger.warning("[research] LLM did not call web_search — retrying with explicit prompt")
            retry_msg = HumanMessage(
                content=f"You MUST call the web_search tool now to find news about {symbol}. "
                f"Search for: '{symbol} {asset_class} news {today}'"
            )
            messages.append(retry_msg)
            retry_response = llm_with_tools.invoke(messages)
            messages.append(retry_response)
            for tool_call in getattr(retry_response, "tool_calls", []):
                try:
                    result = web_search.invoke(tool_call["args"])
                    findings.extend(result if isinstance(result, list) else [])
                except Exception as exc:
                    logger.warning(f"[research] retry tool call failed: {exc}")
            if not getattr(retry_response, "tool_calls", []):
                logger.warning("[research] LLM still did not call tools after retry — proceeding with 0 findings")

        # Ask LLM to parse and score the raw findings
        if findings:
            parse_prompt = (
                f"Here are the raw search results:\n{json.dumps(findings, indent=2)}\n\n"
                "Return a JSON array of findings with keys: "
                "source_title, url, summary, relevance_score"
            )
            messages.append(HumanMessage(content=parse_prompt))
            parse_response = self.llm.invoke(messages)
            try:
                findings = extract_json(parse_response.content)
            except Exception:
                logger.warning("[research] Could not parse LLM JSON for findings, using raw")

        logger.info(f"[research] gather_news — {len(findings)} findings")
        return {**state, "news_findings": findings}

    def _node_scrape_economic_calendar(self, state: ResearchState) -> ResearchState:
        logger.info("[research] scrape_economic_calendar")
        today = state["date"]
        context = state.get("context", "")
        symbol = state.get("symbol", self.cfg.asset.symbol)
        asset_class = state.get("asset_class", self.cfg.asset.type.lower())

        system = SYSTEM_PROMPT.format(context=context, symbol=symbol, asset_class=asset_class)
        prompt = ECONOMIC_CALENDAR_PROMPT.format(date=today, symbol=symbol, asset_class=asset_class)

        llm_with_tools = self.llm.bind_tools([web_search])
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        raw_events: list[dict] = []
        for tool_call in getattr(response, "tool_calls", []):
            try:
                result = web_search.invoke(tool_call["args"])
                raw_events.extend(result if isinstance(result, list) else [])
            except Exception as exc:
                logger.warning(f"[research] calendar tool call failed: {exc}")

        # Ask LLM to extract structured calendar from results
        if raw_events:
            parse_prompt = (
                f"Search results:\n{json.dumps(raw_events, indent=2)}\n\n"
                + prompt
            )
            messages.append(HumanMessage(content=parse_prompt))
            parse_response = self.llm.invoke(messages)
            try:
                calendar = extract_json(parse_response.content)
            except Exception:
                calendar = []
        else:
            calendar = []

        logger.info(f"[research] scrape_economic_calendar — {len(calendar)} events")
        return {**state, "economic_calendar": calendar}

    def _node_analyze_sentiment(self, state: ResearchState) -> ResearchState:
        logger.info("[research] analyze_sentiment")
        context = state.get("context", "")
        news_findings = state.get("news_findings", [])
        symbol = state.get("symbol", self.cfg.asset.symbol)
        asset_class = state.get("asset_class", self.cfg.asset.type.lower())

        system = SYSTEM_PROMPT.format(context=context, symbol=symbol, asset_class=asset_class)
        prompt = SENTIMENT_PROMPT.format(
            news_findings=json.dumps(news_findings, indent=2),
            symbol=symbol,
            asset_class=asset_class,
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        try:
            sentiment = extract_json(response.content)
        except Exception:
            logger.warning("[research] Could not parse sentiment JSON, using neutral defaults")
            sentiment = {
                "macro_sentiment": 0.0,
                "risk_appetite": 0.0,
                "crypto_specific": 0.0,
                "overall_sentiment": 0.0,
                "reasoning": "Sentiment parsing failed — defaulting to neutral.",
            }

        logger.info(f"[research] sentiment overall={sentiment.get('overall_sentiment', 0):.2f}")
        return {**state, "sentiment": sentiment}

    def _node_assess_risk_environment(self, state: ResearchState) -> ResearchState:
        logger.info("[research] assess_risk_environment")
        context = state.get("context", "")
        symbol = state.get("symbol", self.cfg.asset.symbol)
        asset_class = state.get("asset_class", self.cfg.asset.type.lower())
        market_context = {
            "news_findings": state.get("news_findings", []),
            "economic_calendar": state.get("economic_calendar", []),
            "sentiment": state.get("sentiment", {}),
        }

        system = SYSTEM_PROMPT.format(context=context, symbol=symbol, asset_class=asset_class)
        prompt = RISK_ENVIRONMENT_PROMPT.format(
            market_context=json.dumps(market_context, indent=2),
            symbol=symbol,
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        try:
            risk = extract_json(response.content)
        except Exception:
            logger.warning("[research] Could not parse risk JSON, defaulting to YELLOW")
            risk = {
                "risk_level": "YELLOW",
                "primary_risk_factor": "Risk assessment parsing failed",
                "secondary_risk_factors": [],
                "trading_implication": "Use caution until data can be reassessed.",
            }

        logger.info(f"[research] risk_level={risk.get('risk_level', 'UNKNOWN')}")
        return {**state, "risk_environment": risk}

    def _node_write_report(self, state: ResearchState) -> ResearchState:
        logger.info("[research] write_report")
        context = state.get("context", "")
        symbol = state.get("symbol", self.cfg.asset.symbol)
        asset_class = state.get("asset_class", self.cfg.asset.type.lower())
        system = SYSTEM_PROMPT.format(context=context, symbol=symbol, asset_class=asset_class)
        prompt = WRITE_REPORT_PROMPT.format(
            date=state["date"],
            symbol=symbol,
            news_findings=json.dumps(state.get("news_findings", []), indent=2),
            economic_calendar=json.dumps(state.get("economic_calendar", []), indent=2),
            sentiment=json.dumps(state.get("sentiment", {}), indent=2),
            risk_environment=json.dumps(state.get("risk_environment", {}), indent=2),
        )

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        report_text = response.content.strip()

        # Persist to DB
        try:
            sentiment_data = state.get("sentiment", {})
            risk_data = state.get("risk_environment", {})
            overall_score = sentiment_data.get("overall_sentiment", 0.0)
            risk_level_raw = risk_data.get("risk_level", "YELLOW")

            # Fix stale key name from old BTC-centric sentiment prompt
            sentiment_data["asset_specific"] = sentiment_data.pop("crypto_specific", sentiment_data.get("asset_specific", 0.0))

            # Map numeric sentiment to label expected by the DB model
            if overall_score >= 0.1:
                sentiment_str = "bullish"
            elif overall_score <= -0.1:
                sentiment_str = "bearish"
            else:
                sentiment_str = "neutral"

            # Map risk level to DB expected values
            risk_map = {"GREEN": "low", "YELLOW": "medium", "RED": "high"}
            risk_str = risk_map.get(risk_level_raw, "medium")

            # Map risk level to trading recommendation
            rec_map = {"GREEN": "proceed", "YELLOW": "caution", "RED": "avoid"}
            rec_str = rec_map.get(risk_level_raw, "caution")

            with get_session() as session:
                report = ResearchReport(
                    asset=self.cfg.asset.symbol,
                    sentiment=sentiment_str,
                    risk_level=risk_str,
                    trading_recommendation=rec_str,
                    summary=risk_data.get("trading_implication", ""),
                    full_report=report_text,
                    created_at=datetime.utcnow(),
                )
                key_events = [
                    e.get("event_name", "") for e in state.get("economic_calendar", [])
                ]
                report.set_key_events([e for e in key_events if e])
                session.add(report)
            logger.info("[research] report saved to DB")
        except Exception as exc:
            logger.warning(f"[research] Could not save report to DB: {exc}")

        return {**state, "report_text": report_text}

    def _node_notify(self, state: ResearchState) -> ResearchState:
        logger.info("[research] notify — sending Telegram summary")
        sentiment = state.get("sentiment", {})
        risk = state.get("risk_environment", {})

        overall_score = sentiment.get("overall_sentiment", 0.0)
        risk_level = risk.get("risk_level", "YELLOW")

        # Build macro bullet points from the first 3 news findings
        news = state.get("news_findings", [])
        macro_bullets = "\n".join(
            f"• {item.get('summary', item.get('snippet', ''))}"
            for item in news[:3]
            if item.get("summary") or item.get("snippet")
        ) or "• No major macro news found."

        message = TELEGRAM_SUMMARY_TEMPLATE.format(
            date=state["date"],
            overall_sentiment_label=sentiment_label(overall_score),
            overall_sentiment_score=overall_score,
            risk_level_emoji=RISK_LEVEL_EMOJI.get(risk_level, "⚪"),
            risk_level=risk_level,
            macro_bullets=macro_bullets,
            trading_implication=risk.get("trading_implication", "No specific implication."),
        )

        try:
            notifier = get_notifier()
            run_on_main_loop(notifier.send(message, parse_mode="Markdown"), timeout=15)
        except Exception as exc:
            logger.warning(f"[research] Telegram notify failed: {exc}")

        return state

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the full Research Agent workflow and return the final state."""
        logger.info("[research] Starting daily research run")
        run_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()
        initial_state: ResearchState = {}
        final_state = self._graph.invoke(initial_state)
        duration_ms = int((time.monotonic() - t0) * 1000)
        result = dict(final_state)
        self._log_run(run_id, duration_ms, result)
        logger.info(f"[research] Research run complete ({duration_ms}ms)")
        return result


