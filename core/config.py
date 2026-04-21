"""
Central configuration system.

Loads config/config.yaml, then deep-merges the active asset overlay
(config/assets/{active_asset}.yaml).  Environment variables from .env
override any value via pydantic-settings.

Usage:
    from core.config import cfg
    print(cfg.ibkr.port)
    print(cfg.asset.symbol)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"


# ── Sub-models ───────────────────────────────────────────────────────────────


class AssetConfig(BaseModel):
    symbol: str = "BTC"
    currency: str = "USD"
    exchange: str = "PAXOS"
    type: str = "CRYPTO"
    min_quantity: float = 0.0001
    price_precision: int = 2
    quantity_precision: int = 4


class CapitalConfig(BaseModel):
    paper_account_size: float = 100_000
    risk_per_trade_pct: float = 0.005
    max_concurrent_positions: int = 3
    max_daily_loss_pct: float = 0.03
    max_notional_pct: float = 0.20


class HoursConfig(BaseModel):
    days: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    start_hour: int = 0
    end_hour: int = 24
    timezone: str = "Europe/Istanbul"


class PriceMonitorConfig(BaseModel):
    poll_interval_seconds: int = 30
    alarm_expiry_hours: int = 48
    trigger_tolerance_pct: float = 0.003
    queue_max_size: int = 10


class OrdersConfig(BaseModel):
    type: str = "limit"
    max_slippage_pct: float = 0.005
    sl_type: str = "stop_limit"
    sl_limit_offset_pct: float = 0.002
    trail_stop: bool = False


class TradingConfig(BaseModel):
    asset: AssetConfig = Field(default_factory=AssetConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    hours: HoursConfig = Field(default_factory=HoursConfig)
    price_monitor: PriceMonitorConfig = Field(default_factory=PriceMonitorConfig)
    orders: OrdersConfig = Field(default_factory=OrdersConfig)


class ScheduleJobConfig(BaseModel):
    hour: int = 10
    minute: int = 0
    timezone: str = "Europe/Istanbul"
    day_of_week: str | None = None  # for Evaluator


class ScheduleConfig(BaseModel):
    research: ScheduleJobConfig = Field(default_factory=ScheduleJobConfig)
    analyst: ScheduleJobConfig = Field(
        default_factory=lambda: ScheduleJobConfig(hour=10, minute=30)
    )
    evaluator: ScheduleJobConfig = Field(
        default_factory=lambda: ScheduleJobConfig(hour=20, day_of_week="sunday")
    )


class LLMConfig(BaseModel):
    provider: str = "ollama"          # ollama | anthropic | openai
    model: str = "qwen2.5:32b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 4096


class AgentsConfig(BaseModel):
    """Per-agent LLM overrides. Any field left as None falls back to the global llm config."""

    research: LLMConfig | None = None
    analyst: LLMConfig | None = None
    trader: LLMConfig | None = None
    evaluator: LLMConfig | None = None


class IBKRConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    reconnect_attempts: int = 10
    reconnect_backoff_base: int = 5


class RiskConfig(BaseModel):
    appetite: str = "moderate"
    min_confidence: float = 0.65
    min_risk_reward: float = 1.5


class IndicatorsConfig(BaseModel):
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_periods: list[int] = [20, 50, 200]
    vwap_session: bool = True
    volume_profile_sessions: int = 20
    atr_period: int = 14


class SearchConfig(BaseModel):
    primary: str = "brave"
    fallback: str = "duckduckgo"
    max_results: int = 10


class DashboardConfig(BaseModel):
    port: int = 8501
    auto_refresh_seconds: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "30 days"


# ── Root config ──────────────────────────────────────────────────────────────


class AppConfig(BaseModel):
    active_asset: str = "btc"
    trading: TradingConfig = Field(default_factory=TradingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Convenience shortcut
    @property
    def asset(self) -> AssetConfig:
        return self.trading.asset


# ── Env overrides (secrets only — never put them in YAML) ────────────────────


class EnvSecrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    ibkr_host: str | None = None
    ibkr_port: int | None = None
    ibkr_client_id: int | None = None
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    brave_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str | None = None
    llm_base_url: str | None = None


# ── Loader ───────────────────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_config() -> tuple[AppConfig, EnvSecrets]:
    """Load and merge config + asset overlay + env secrets. Cached after first call."""
    base = _load_yaml(CONFIG_DIR / "config.yaml")
    active_asset = base.get("active_asset", "btc")
    overlay = _load_yaml(CONFIG_DIR / "assets" / f"{active_asset}.yaml")
    merged = _deep_merge(base, overlay)
    config = AppConfig.model_validate(merged)
    secrets = EnvSecrets()

    # Apply env overrides to IBKR connection
    if secrets.ibkr_host:
        config.ibkr.host = secrets.ibkr_host
    if secrets.ibkr_port:
        config.ibkr.port = secrets.ibkr_port
    if secrets.ibkr_client_id:
        config.ibkr.client_id = secrets.ibkr_client_id

    # Apply env overrides to LLM
    if secrets.llm_model:
        config.llm.model = secrets.llm_model
    if secrets.llm_base_url:
        config.llm.base_url = secrets.llm_base_url

    return config, secrets


# Module-level singletons for ergonomic imports
cfg, secrets = load_config()
