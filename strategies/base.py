"""
Strategy base definitions.

Each asset strategy is a StrategyConfig instance that carries:
  - Signal generation parameters (stops, confidence thresholds, filters)
  - Confluence scoring weights (how much each indicator matters for this asset)

The weights replace the hardcoded constants in confluence.py, so BTC and
EUR/USD can have completely different indicator importance profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class StrategyWeights:
    """
    Confluence scoring weights for a single strategy.

    These control how much each indicator contributes to the confidence score.
    They do NOT need to sum to 1.0 — the scorer caps the final score at 1.0.

    Design intent:
      - Assets with reliable volume data  → higher weight_volume_profile
      - Assets that trend strongly        → higher weight_daily_trend, weight_ema
      - Assets with sharp RSI extremes   → higher weight_rsi
      - Forex (sparse volume)            → lower weight_volume_profile, weight_vwap
    """

    weight_ema: float = 0.15          # EMA 20/50/200 proximity
    weight_vwap: float = 0.10         # VWAP alignment
    weight_volume_profile: float = 0.20  # Volume POC/VAH/VAL
    weight_sr_touch: float = 0.15     # Historical S/R touches (per touch, capped)
    weight_rsi: float = 0.10          # RSI overbought/oversold extreme
    weight_macd: float = 0.10         # MACD direction agreement
    weight_daily_trend: float = 0.05  # Daily EMA trend alignment


@dataclass
class StrategyConfig:
    """
    Full strategy definition for one asset.

    Stored in the strategy registry and passed to signal generators and
    the confluence scorer. Adding a new strategy = creating a new instance
    of this class — no code changes needed elsewhere.
    """

    # Identity
    strategy_id: str          # e.g. "btc_sr_bounce_v1"
    asset: str                # e.g. "BTC", "EURUSD"
    version: str              # semantic version string e.g. "1.0"
    description: str          # human-readable one-liner

    # Confluence scoring weights
    weights: StrategyWeights = field(default_factory=StrategyWeights)

    # Signal generation parameters
    min_confidence: float = 0.65
    min_rr: float = 1.5
    atr_stop_multiplier: float = 1.5   # stop = entry ± ATR * this
    atr_target_multiplier: float | None = None  # if None → stop_mult * min_rr
    trend_filter: bool = False          # if True, skip counter-trend signals
    max_signals_per_day: int = 4

    # S/R detection parameters
    lookback_sr: int = 200             # bars of history for S/R detection
    lookback_vp: int = 500             # bars of history for volume profile
    cluster_pct: float = 0.015         # S/R cluster merge threshold
    swing_window: int = 5              # half-window for swing high/low detection
    top_n_sr: int = 8                  # max S/R levels to score per bar

    # Tolerance parameters
    vwap_tolerance: float = 0.003      # proximity threshold for VWAP alignment
    ema_tolerance: float = 0.003       # proximity threshold for EMA alignment
    vol_tolerance: float = 0.005       # proximity threshold for volume levels

    def target_multiplier(self) -> float:
        """Return the ATR target multiplier (stop_mult * min_rr if not overridden)."""
        if self.atr_target_multiplier is not None:
            return self.atr_target_multiplier
        return self.atr_stop_multiplier * self.min_rr


class SignalGeneratorProtocol(Protocol):
    """
    Duck-typing interface for strategy signal generators.

    Each strategy module must expose a function matching this signature.
    The backtesting dispatcher calls it via the registry.
    """

    def __call__(
        self,
        df_1h: pd.DataFrame,
        df_daily: pd.DataFrame,
        strategy: StrategyConfig,
        ema_periods: list[int],
        rsi_period: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal_period: int,
        atr_period: int,
    ) -> list: ...
