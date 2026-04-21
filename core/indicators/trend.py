"""
Trend indicators: EMA 20/50/200 and ATR.
"""

from __future__ import annotations

import pandas as pd


def add_ema(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    col: str = "close",
) -> pd.DataFrame:
    """
    Add EMA columns for each requested period.

    Args:
        df:      OHLCV DataFrame
        periods: List of EMA periods. Defaults to [20, 50, 200].
        col:     Price column (default 'close')

    Returns:
        New DataFrame with columns: ema_20, ema_50, ema_200 (for default periods)
    """
    if periods is None:
        periods = [20, 50, 200]

    result = df.copy()
    for p in periods:
        result[f"ema_{p}"] = result[col].ewm(span=p, adjust=False).mean()
    return result


def trend_direction(df: pd.DataFrame, periods: list[int] | None = None) -> str:
    """
    Determine overall trend from EMA alignment on the most recent bar.

    Bullish:  EMA20 > EMA50 > EMA200
    Bearish:  EMA20 < EMA50 < EMA200
    Ranging:  mixed / neither

    Args:
        df:      DataFrame with ema_20, ema_50, ema_200 columns
        periods: Override periods (must match what's in df)

    Returns:
        "bullish" | "bearish" | "ranging"
    """
    if periods is None:
        periods = [20, 50, 200]

    cols = [f"ema_{p}" for p in periods]
    if not all(c in df.columns for c in cols):
        df = add_ema(df, periods=periods)

    last = df.iloc[-1]
    ema_vals = [last[c] for c in cols]

    if all(ema_vals[i] > ema_vals[i + 1] for i in range(len(ema_vals) - 1)):
        return "bullish"
    if all(ema_vals[i] < ema_vals[i + 1] for i in range(len(ema_vals) - 1)):
        return "bearish"
    return "ranging"


def price_vs_ema(price: float, df: pd.DataFrame, period: int) -> str:
    """
    Return whether current price is above or below a given EMA.

    Returns:
        "above" | "below" | "at" (within 0.1%)
    """
    col = f"ema_{period}"
    if col not in df.columns:
        df = add_ema(df, periods=[period])

    ema_val = df[col].iloc[-1]
    diff_pct = (price - ema_val) / ema_val

    if abs(diff_pct) <= 0.001:
        return "at"
    return "above" if price > ema_val else "below"


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) column.

    ATR measures volatility — used for stop loss placement and
    abnormal price movement detection.

    Args:
        df:     OHLCV DataFrame with high, low, close columns
        period: ATR period (default 14)

    Returns:
        New DataFrame with column: atr_{period}
    """
    result = df.copy()
    high = result["high"]
    low = result["low"]
    prev_close = result["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return result


def current_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the most recent ATR value."""
    col = f"atr_{period}"
    if col not in df.columns:
        df = add_atr(df, period=period)
    return float(df[col].iloc[-1])


def is_near_ema(price: float, df: pd.DataFrame, period: int, tolerance_pct: float = 0.005) -> bool:
    """Return True if price is within tolerance_pct of the EMA."""
    col = f"ema_{period}"
    if col not in df.columns:
        df = add_ema(df, periods=[period])
    ema_val = df[col].iloc[-1]
    return abs(price - ema_val) / ema_val <= tolerance_pct
