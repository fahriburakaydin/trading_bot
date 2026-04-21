"""
Deterministic OHLCV fixtures for unit tests.

All fixtures produce known-value outputs that can be verified against
independent TA calculations (e.g. TradingView manual calculations).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def make_trending_up(n: int = 300) -> pd.DataFrame:
    """
    Uptrending OHLCV series.
    Price rises from 100 to ~200 with small noise.
    RSI should be elevated, EMA20 > EMA50, trend = bullish.
    """
    np.random.seed(42)
    base = np.linspace(100, 200, n)
    noise = np.random.normal(0, 0.5, n)
    close = base + noise
    high = close + np.abs(np.random.normal(0, 0.8, n))
    low = close - np.abs(np.random.normal(0, 0.8, n))
    open_ = close - np.random.normal(0, 0.3, n)
    volume = np.random.uniform(1000, 5000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_trending_down(n: int = 300) -> pd.DataFrame:
    """
    Downtrending OHLCV series.
    Price falls from 200 to ~100.
    RSI should be depressed, EMA20 < EMA50, trend = bearish.
    """
    np.random.seed(99)
    base = np.linspace(200, 100, n)
    noise = np.random.normal(0, 0.5, n)
    close = base + noise
    high = close + np.abs(np.random.normal(0, 0.8, n))
    low = close - np.abs(np.random.normal(0, 0.8, n))
    open_ = close - np.random.normal(0, 0.3, n)
    volume = np.random.uniform(1000, 5000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_ranging(n: int = 300) -> pd.DataFrame:
    """
    Sideways/ranging OHLCV series.
    Price oscillates around 150 with regular highs/lows.
    Good for S/R detection test.
    """
    np.random.seed(7)
    t = np.linspace(0, 6 * np.pi, n)
    close = 150 + 10 * np.sin(t) + np.random.normal(0, 0.5, n)
    high = close + np.abs(np.random.normal(0, 0.8, n))
    low = close - np.abs(np.random.normal(0, 0.8, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.uniform(1000, 5000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_simple_rsi_known() -> pd.DataFrame:
    """
    14 identical up-days then 14 identical down-days.
    After 14 up-days: RSI should be 100 (no losses).
    After full 28 bars: RSI calculable with known expected value.
    """
    closes = [100.0 + i for i in range(14)] + [113.0 - i for i in range(14)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    opens = [c - 0.1 for c in closes]
    volumes = [1000.0] * 28

    idx = pd.date_range("2024-01-01", periods=28, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )
