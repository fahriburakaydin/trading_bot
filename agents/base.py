"""
Base agent class providing shared infrastructure for all LangGraph agents.

Every concrete agent subclasses BaseAgent and implements ``run()``.
The base class handles:
- LLM initialisation with provider routing (ollama | anthropic | openai)
- Per-agent model resolution: explicit arg > agents.<name> config > llm global default
- Knowledge-base context loading from SQLite
- Structured logging
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from loguru import logger

from core.config import LLMConfig, cfg as _cfg, secrets as _secrets


def _build_llm(llm_cfg: LLMConfig) -> BaseChatModel:
    """Instantiate the correct LangChain chat model for the given provider config."""
    provider = llm_cfg.provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic

        return ChatAnthropic(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            api_key=_secrets.anthropic_api_key or None,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai

        return ChatOpenAI(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            api_key=_secrets.openai_api_key or None,
        )

    # Default: Ollama (local)
    from langchain_ollama import ChatOllama

    # Hard-cap the context window to prevent runaway token consumption.
    # qwen3 models enable extended thinking by default in Ollama, which can
    # fill the entire context before producing a single output token.
    # num_ctx=8192 keeps each agent call fast and predictable.
    kwargs: dict = {
        "model": llm_cfg.model,
        "base_url": llm_cfg.base_url,
        "temperature": llm_cfg.temperature,
        "num_ctx": 8192,
    }

    # Disable thinking mode explicitly for qwen3 models
    if "qwen3" in llm_cfg.model.lower():
        kwargs["think"] = False

    return ChatOllama(**kwargs)


def _resolve_llm_cfg(agent_name: str, model_override: str | None) -> LLMConfig:
    """
    Resolve the effective LLMConfig for an agent.

    Resolution order (first match wins):
      1. Explicit ``model`` constructor arg  — runtime override, uses global provider/base_url
      2. ``cfg.agents.<agent_name>``          — per-agent YAML config
      3. ``cfg.llm``                          — global default
    """
    if model_override:
        # Explicit override: keep provider/base_url from global config, just swap model
        base = _cfg.llm.model_copy(update={"model": model_override})
        return base

    per_agent: LLMConfig | None = getattr(_cfg.agents, agent_name, None)
    if per_agent is not None:
        return per_agent

    return _cfg.llm


class BaseAgent(ABC):
    """Abstract base for all trading-bot agents."""

    #: Override in subclasses — used for log labels, DB queries, and config key lookup.
    agent_name: str = "base"

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        llm_cfg = _resolve_llm_cfg(self.agent_name, model)

        # Allow temperature to be overridden at construction time without replacing the
        # full config (useful for tests and one-off runs)
        if temperature is not None:
            llm_cfg = llm_cfg.model_copy(update={"temperature": temperature})

        self.llm: BaseChatModel = _build_llm(llm_cfg)
        self.cfg = _cfg
        logger.info(
            f"[{self.agent_name}] initialised — "
            f"provider={llm_cfg.provider} model={llm_cfg.model} "
            f"temperature={llm_cfg.temperature}"
        )

    # ── Knowledge-base context ─────────────────────────────────────────────────

    def load_context(self, limit: int = 20) -> str:
        """Return a formatted [CONTEXT] block from the knowledge_base table.

        Fetches entries that apply to this agent (``applies_to IN ('all', agent_name)``)
        ordered newest-first, limited to *limit* rows.  Returns an empty string when
        the table has no matching rows so callers can safely append it to any prompt.
        """
        from core.memory.database import get_session
        from core.memory.models import KnowledgeEntry

        try:
            with get_session() as session:
                entries: list[KnowledgeEntry] = (
                    session.query(KnowledgeEntry)
                    .filter(
                        KnowledgeEntry.applies_to.in_(["all", self.agent_name]),
                        KnowledgeEntry.active.is_(True),
                    )
                    .order_by(KnowledgeEntry.created_at.desc())
                    .limit(limit)
                    .all()
                )
        except Exception as exc:
            logger.warning(f"[{self.agent_name}] Could not load context: {exc}")
            return ""

        if not entries:
            return ""

        lines = ["[CONTEXT]"]
        for entry in entries:
            lines.append(f"- [{entry.category.upper()}] {entry.title}: {entry.content}")
        lines.append("[/CONTEXT]")
        return "\n".join(lines)

    # ── Agent logging ──────────────────────────────────────────────────────────

    def _log_run(self, run_id: str, duration_ms: int, result: dict[str, Any]) -> None:
        """Persist a log entry for this agent run to the agent_logs table."""
        try:
            from core.memory.database import get_session
            from core.memory.models import AgentLog

            error = result.get("error")
            with get_session() as session:
                log = AgentLog(
                    agent=self.agent_name,
                    run_id=run_id,
                    action="run",
                    reasoning=error if error else "completed",
                    duration_ms=duration_ms,
                )
                # Compact summary — avoid serialising full state to save space
                summary = {
                    k: v for k, v in result.items()
                    if k in (
                        "alarms_saved", "approved", "trade_id", "error",
                        "kb_entries_added", "risk_level", "report_text",
                    ) and v is not None
                }
                log.set_output(summary)
                session.add(log)
        except Exception as exc:
            logger.debug(f"[{self.agent_name}] Could not save agent log: {exc}")

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the agent's full workflow and return a result dict."""
