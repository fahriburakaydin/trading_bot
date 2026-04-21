"""
Candlestick pattern detection.

Pure functions — take a DataFrame slice and return pattern classifications.
All patterns are detected on the most recent completed bar(s).

Patterns implemented:
    - Bearish engulfing
    - Bullish engulfing
    - Push exhaustion (trend move + retracement)
"""

from __future__ import annotations

import pandas as pd


def is_bearish_engulfing(df: pd.DataFrame, idx: int = -1) -> bool:
    """
    Detect a bearish engulfing candlestick pattern.

    A bearish engulfing occurs when:
      - The prior candle (idx-1) is bullish (close > open)
      - The current candle (idx) is bearish (close < open)
      - The current candle's body fully engulfs the prior candle's body
        (current open >= prior close AND current close <= prior open)

    This signals a potential reversal from up to down — used by the video
    trader as confirmation to enter a SHORT at a pivot supply zone.

    Args:
        df:   OHLCV DataFrame (at least 2 rows)
        idx:  Index of the current (engulfing) bar (-1 = last bar)

    Returns:
        True if bearish engulfing pattern detected
    """
    if len(df) < 2:
        return False

    current = df.iloc[idx]
    prior = df.iloc[idx - 1]

    prior_bullish = float(prior["close"]) > float(prior["open"])
    current_bearish = float(current["close"]) < float(current["open"])

    if not (prior_bullish and current_bearish):
        return False

    engulfs = (
        float(current["open"]) >= float(prior["close"])
        and float(current["close"]) <= float(prior["open"])
    )
    return engulfs


def is_bullish_engulfing(df: pd.DataFrame, idx: int = -1) -> bool:
    """
    Detect a bullish engulfing candlestick pattern.

    A bullish engulfing occurs when:
      - The prior candle (idx-1) is bearish (close < open)
      - The current candle (idx) is bullish (close > open)
      - The current candle's body fully engulfs the prior candle's body

    Used to confirm LONG entries at a pivot demand zone.

    Args:
        df:   OHLCV DataFrame (at least 2 rows)
        idx:  Index of the current (engulfing) bar (-1 = last bar)

    Returns:
        True if bullish engulfing pattern detected
    """
    if len(df) < 2:
        return False

    current = df.iloc[idx]
    prior = df.iloc[idx - 1]

    prior_bearish = float(prior["close"]) < float(prior["open"])
    current_bullish = float(current["close"]) > float(current["open"])

    if not (prior_bearish and current_bullish):
        return False

    engulfs = (
        float(current["open"]) <= float(prior["close"])
        and float(current["close"]) >= float(prior["open"])
    )
    return engulfs


def is_strong_bearish_bar(df: pd.DataFrame, idx: int = -1, min_body_ratio: float = 0.6) -> bool:
    """
    Detect a strong bearish bar (large body, small wicks).

    Used as a looser alternative to bearish engulfing — catches bars where
    momentum is clearly to the downside even without full engulfment.

    Args:
        df:             OHLCV DataFrame
        idx:            Bar index (-1 = last bar)
        min_body_ratio: Minimum ratio of body to total range (default 0.6 = 60%)

    Returns:
        True if bearish bar with large body relative to range
    """
    bar = df.iloc[idx]
    body = abs(float(bar["close"]) - float(bar["open"]))
    total_range = float(bar["high"]) - float(bar["low"])
    if total_range == 0:
        return False
    return float(bar["close"]) < float(bar["open"]) and (body / total_range) >= min_body_ratio


def is_strong_bullish_bar(df: pd.DataFrame, idx: int = -1, min_body_ratio: float = 0.6) -> bool:
    """
    Detect a strong bullish bar (large body, small wicks).

    Args:
        df:             OHLCV DataFrame
        idx:            Bar index (-1 = last bar)
        min_body_ratio: Minimum ratio of body to total range

    Returns:
        True if bullish bar with large body relative to range
    """
    bar = df.iloc[idx]
    body = abs(float(bar["close"]) - float(bar["open"]))
    total_range = float(bar["high"]) - float(bar["low"])
    if total_range == 0:
        return False
    return float(bar["close"]) > float(bar["open"]) and (body / total_range) >= min_body_ratio


def candle_signal(df: pd.DataFrame, idx: int = -1) -> str:
    """
    Return the strongest candlestick signal at a given bar.

    Checks patterns in priority order:
      1. Bearish engulfing  → "bearish_engulfing"
      2. Bullish engulfing  → "bullish_engulfing"
      3. Strong bearish bar → "strong_bearish"
      4. Strong bullish bar → "strong_bullish"
      5. None              → "neutral"

    Args:
        df:  OHLCV DataFrame
        idx: Bar index

    Returns:
        Pattern name string
    """
    if is_bearish_engulfing(df, idx):
        return "bearish_engulfing"
    if is_bullish_engulfing(df, idx):
        return "bullish_engulfing"
    if is_strong_bearish_bar(df, idx):
        return "strong_bearish"
    if is_strong_bullish_bar(df, idx):
        return "strong_bullish"
    return "neutral"
