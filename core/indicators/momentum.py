"""
Momentum indicators: RSI and MACD.

All functions are pure — they take a DataFrame and return a new DataFrame
with added columns. The input DataFrame is never mutated.
"""

from __future__ import annotations

import pandas as pd


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    """
    Add RSI column to DataFrame.

    Uses Wilder's smoothing (exponential method), which matches TradingView
    and most professional platforms.

    Args:
        df:     OHLCV DataFrame with a 'close' column
        period: Lookback period (default 14)
        col:    Column to use as price (default 'close')

    Returns:
        New DataFrame with added column: rsi_{period}
    """
    result = df.copy()
    delta = result[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return result


def rsi_signal(rsi_value: float, overbought: float = 70, oversold: float = 30) -> str:
    """
    Classify RSI value.

    Returns:
        "overbought" | "oversold" | "neutral"
    """
    if rsi_value >= overbought:
        return "overbought"
    if rsi_value <= oversold:
        return "oversold"
    return "neutral"


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """
    Add MACD, Signal line, and Histogram columns.

    Args:
        df:     OHLCV DataFrame
        fast:   Fast EMA period (default 12)
        slow:   Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        col:    Price column (default 'close')

    Returns:
        New DataFrame with added columns:
            macd_{fast}_{slow}       — MACD line
            macd_signal_{signal}     — Signal line
            macd_hist                — Histogram (MACD - Signal)
    """
    result = df.copy()
    ema_fast = result[col].ewm(span=fast, adjust=False).mean()
    ema_slow = result[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result[f"macd_{fast}_{slow}"] = macd_line
    result[f"macd_signal_{signal}"] = signal_line
    result["macd_hist"] = histogram
    return result


def macd_signal_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Detect MACD crossovers.

    Returns:
        Series with values:
            1  = bullish crossover (MACD crossed above signal)
           -1  = bearish crossover (MACD crossed below signal)
            0  = no crossover
    """
    macd_col = f"macd_{fast}_{slow}"
    sig_col = f"macd_signal_{signal}"

    if macd_col not in df.columns:
        df = add_macd(df, fast=fast, slow=slow, signal=signal)

    diff = df[macd_col] - df[sig_col]
    prev_diff = diff.shift(1)

    crossover = pd.Series(0, index=df.index, dtype=int)
    crossover[diff > 0] = 1
    crossover[(diff > 0) & (prev_diff <= 0)] = 1   # confirmed bullish cross
    crossover[(diff < 0) & (prev_diff >= 0)] = -1  # confirmed bearish cross
    return crossover


def add_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
    col: str = "close",
) -> pd.DataFrame:
    """
    Add Stochastic RSI columns.

    Stochastic RSI applies the Stochastic oscillator formula to RSI values
    instead of price. This makes it more sensitive than plain RSI — it reaches
    overbought/oversold levels more frequently, which is why the video trader
    uses it for scalping EUR/USD on the 15m timeframe.

    Overbought: stoch_rsi_k >= 80
    Oversold:   stoch_rsi_k <= 20

    Args:
        df:           OHLCV DataFrame
        rsi_period:   RSI lookback period (default 14)
        stoch_period: Stochastic lookback over RSI (default 14)
        k_period:     %K smoothing period (default 3)
        d_period:     %D signal smoothing period (default 3)
        col:          Price column (default 'close')

    Returns:
        New DataFrame with added columns:
            stoch_rsi_k  — %K line (smoothed)
            stoch_rsi_d  — %D signal line (smoothed %K)
    """
    result = add_rsi(df, period=rsi_period, col=col)
    rsi_col = f"rsi_{rsi_period}"
    rsi = result[rsi_col]

    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min

    raw_k = ((rsi - rsi_min) / rsi_range.replace(0, float("nan"))) * 100
    k = raw_k.rolling(k_period).mean()
    d = k.rolling(d_period).mean()

    result["stoch_rsi_k"] = k.round(2)
    result["stoch_rsi_d"] = d.round(2)
    return result


def stoch_rsi_signal(k: float, d: float, overbought: float = 80, oversold: float = 20) -> str:
    """
    Classify Stochastic RSI state.

    Returns:
        "overbought" | "oversold" | "neutral"
    """
    if k >= overbought and d >= overbought:
        return "overbought"
    if k <= oversold and d <= oversold:
        return "oversold"
    return "neutral"


def macd_momentum_direction(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> str:
    """
    Return MACD momentum direction for the most recent bar.

    Returns:
        "bullish" | "bearish" | "neutral"
    """
    macd_col = f"macd_{fast}_{slow}"
    sig_col = f"macd_signal_{signal}"

    if macd_col not in df.columns:
        df = add_macd(df, fast=fast, slow=slow, signal=signal)

    last_macd = df[macd_col].iloc[-1]
    last_sig = df[sig_col].iloc[-1]
    last_hist = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else last_macd - last_sig

    if last_macd > last_sig and last_hist > 0:
        return "bullish"
    if last_macd < last_sig and last_hist < 0:
        return "bearish"
    return "neutral"
