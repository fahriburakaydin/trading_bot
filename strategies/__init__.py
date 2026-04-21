"""
Strategy registry.

Maps asset symbols to their active strategy config + signal generator.
To add a new asset strategy:
  1. Create strategies/{asset}.py with a StrategyConfig instance and generate_signals()
  2. Add an entry here in REGISTRY.

The backtesting dispatcher and live Analyst agent both use get_strategy() to
look up the correct strategy for the currently configured asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from strategies.base import StrategyConfig
from strategies.btc import BTC_SR_BOUNCE_V1
from strategies.btc import generate_signals as btc_generate_signals
from strategies.eurusd import EURUSD_PIVOT_SCALP_V2
from strategies.eurusd import generate_signals as eurusd_generate_signals


@dataclass
class StrategyEntry:
    config: StrategyConfig
    generate_signals: Callable[..., list]


# ── Registry ──────────────────────────────────────────────────────────────────
# Key: asset symbol (matches cfg.asset.symbol, uppercase)

REGISTRY: dict[str, StrategyEntry] = {
    "BTC": StrategyEntry(
        config=BTC_SR_BOUNCE_V1,
        generate_signals=btc_generate_signals,
    ),
    "EURUSD": StrategyEntry(
        config=EURUSD_PIVOT_SCALP_V2,
        generate_signals=eurusd_generate_signals,
    ),
}


def get_strategy(asset: str) -> StrategyEntry:
    """
    Look up the strategy for a given asset symbol.

    Raises KeyError with a helpful message if the asset has no strategy defined.
    """
    key = asset.upper()
    if key not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise KeyError(
            f"No strategy defined for asset '{asset}'. "
            f"Available: {available}. "
            f"Add a new entry in strategies/__init__.py to support this asset."
        )
    return REGISTRY[key]


def list_strategies() -> list[StrategyConfig]:
    """Return all registered strategy configs (useful for the Evaluator report)."""
    return [entry.config for entry in REGISTRY.values()]
