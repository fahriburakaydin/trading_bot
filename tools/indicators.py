"""
LangChain tool wrappers for technical analysis indicators.

These tools accept serialised OHLCV data (JSON string as produced by
``tools.market_data``) and return computed indicator values as JSON.

The Analyst Agent uses these tools to calculate signals before scoring
confluence and setting price alarms.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from loguru import logger

from core.config import cfg
from core.indicators.confluence import score_all_levels
from core.indicators.levels import detect_sr_levels
from core.indicators.momentum import add_macd, add_rsi, macd_momentum_direction, rsi_signal
from core.indicators.trend import add_atr, add_ema, current_atr, trend_direction
from core.indicators.volume import add_vwap, has_reliable_volume, volume_profile


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_ohlcv(ohlcv_json: str) -> pd.DataFrame:
    """Parse a JSON-serialised OHLCV string into a DataFrame with a DatetimeIndex."""
    records = json.loads(ohlcv_json)
    df = pd.DataFrame(records)
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    return df


def _safe_float(value: object) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool
def calculate_rsi(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv or fetch_multi_timeframe"],
    period: Annotated[int, "RSI period (default 14)"] = 14,
) -> str:
    """Calculate RSI and return the current value with its signal.

    Args:
        ohlcv_json: Serialised OHLCV data (JSON string).
        period: RSI look-back period.

    Returns:
        JSON: {"rsi": float, "signal": "overbought|oversold|neutral"}
    """
    df = _parse_ohlcv(ohlcv_json)
    df = add_rsi(df, period=period)
    col = f"rsi_{period}"
    rsi_val = _safe_float(df[col].iloc[-1]) if col in df.columns else None
    signal = rsi_signal(rsi_val) if rsi_val is not None else "unknown"
    logger.debug(f"[calculate_rsi] rsi={rsi_val} signal={signal}")
    return json.dumps({"rsi": rsi_val, "signal": signal})


@tool
def calculate_macd(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv"],
    fast: Annotated[int, "Fast EMA period"] = 12,
    slow: Annotated[int, "Slow EMA period"] = 26,
    signal: Annotated[int, "Signal line period"] = 9,
) -> str:
    """Calculate MACD and return line, signal, histogram and momentum direction.

    Args:
        ohlcv_json: Serialised OHLCV data.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        JSON: {"macd": float, "signal_line": float, "histogram": float, "direction": str}
    """
    df = _parse_ohlcv(ohlcv_json)
    df = add_macd(df, fast=fast, slow=slow, signal=signal)
    direction = macd_momentum_direction(df, fast=fast, slow=slow, signal=signal)
    last = df.iloc[-1]
    result = {
        "macd": _safe_float(last.get("macd")),
        "signal_line": _safe_float(last.get("macd_signal")),
        "histogram": _safe_float(last.get("macd_hist")),
        "direction": direction,
    }
    logger.debug(f"[calculate_macd] direction={direction}")
    return json.dumps(result)


@tool
def calculate_ema(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv"],
    periods: Annotated[list[int], "EMA periods to compute"] = None,
) -> str:
    """Calculate EMAs and return their current values along with trend direction.

    Args:
        ohlcv_json: Serialised OHLCV data.
        periods: EMA periods list (default from config: [20, 50, 200]).

    Returns:
        JSON: {"ema_<period>": float, ..., "trend": "uptrend|downtrend|sideways"}
    """
    if periods is None:
        periods = cfg.indicators.ema_periods
    df = _parse_ohlcv(ohlcv_json)
    df = add_ema(df, periods=periods)
    trend = trend_direction(df, periods=periods)
    result: dict = {"trend": trend}
    for p in periods:
        col = f"ema_{p}"
        result[col] = _safe_float(df[col].iloc[-1]) if col in df.columns else None
    logger.debug(f"[calculate_ema] trend={trend}")
    return json.dumps(result)


@tool
def calculate_atr(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv"],
    period: Annotated[int, "ATR period (default 14)"] = 14,
) -> str:
    """Calculate Average True Range (ATR) for volatility assessment.

    Args:
        ohlcv_json: Serialised OHLCV data.
        period: ATR look-back period.

    Returns:
        JSON: {"atr": float, "atr_pct": float (ATR as % of last close)}
    """
    df = _parse_ohlcv(ohlcv_json)
    df = add_atr(df, period=period)
    atr_val = current_atr(df, period=period)
    last_close = _safe_float(df["close"].iloc[-1])
    atr_pct = (atr_val / last_close * 100) if last_close else None
    result = {"atr": atr_val, "atr_pct": round(atr_pct, 4) if atr_pct else None}
    logger.debug(f"[calculate_atr] atr={atr_val:.2f}")
    return json.dumps(result)


@tool
def calculate_vwap(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv"],
) -> str:
    """Calculate VWAP and report price position relative to it.

    Args:
        ohlcv_json: Serialised OHLCV data.

    Returns:
        JSON: {"vwap": float, "last_close": float, "price_vs_vwap": "above|below|at"}
    """
    df = _parse_ohlcv(ohlcv_json)
    if not has_reliable_volume(df):
        return json.dumps({"vwap": None, "price_vs_vwap": "no_volume_data"})
    df = add_vwap(df)
    vwap_val = _safe_float(df["vwap"].iloc[-1]) if "vwap" in df.columns else None
    last_close = _safe_float(df["close"].iloc[-1])
    if vwap_val and last_close:
        diff_pct = (last_close - vwap_val) / vwap_val
        pos = "above" if diff_pct > 0.001 else ("below" if diff_pct < -0.001 else "at")
    else:
        pos = "unknown"
    result = {"vwap": vwap_val, "last_close": last_close, "price_vs_vwap": pos}
    logger.debug(f"[calculate_vwap] vwap={vwap_val} pos={pos}")
    return json.dumps(result)


@tool
def detect_support_resistance(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv"],
    swing_window: Annotated[int, "Swing detection window in bars"] = 5,
    cluster_pct: Annotated[float, "Clustering tolerance as fraction (e.g. 0.005 = 0.5%)"] = 0.005,
) -> str:
    """Detect key support and resistance levels from price action.

    Args:
        ohlcv_json: Serialised OHLCV data.
        swing_window: Number of bars on each side to confirm a swing high/low.
        cluster_pct: Tolerance for merging nearby levels.

    Returns:
        JSON list of levels: [{"price": float, "type": "support|resistance",
        "strength": int, "touches": int}, ...]
    """
    df = _parse_ohlcv(ohlcv_json)
    levels = detect_sr_levels(df, swing_window=swing_window, cluster_pct=cluster_pct)
    result = [
        {
            "price": lvl.price,
            "type": lvl.level_type,
            "score": lvl.score,
            "touches": lvl.touch_count,
        }
        for lvl in levels
    ]
    logger.debug(f"[detect_support_resistance] found {len(result)} levels")
    return json.dumps(result)


@tool
def score_confluence_levels(
    ohlcv_json: Annotated[str, "JSON string from fetch_ohlcv (preferably 1h timeframe)"],
    current_price: Annotated[float, "Current market price of the asset"],
) -> str:
    """Detect S/R levels and score them by confluence with indicators.

    Internally computes RSI, MACD, EMA trend, volume profile, and ATR from
    the provided OHLCV data before scoring each level.

    Args:
        ohlcv_json: Serialised OHLCV data (1h bars recommended for level detection).
        current_price: Latest traded price used for trade-direction assessment.

    Returns:
        JSON list of high-confidence scored levels sorted by confidence descending:
        [{"price": float, "action": "LONG|SHORT", "confidence": float,
          "stop_loss": float, "target_price": float, "risk_reward": float,
          "confluence_factors": [str, ...]}, ...]
    """
    df = _parse_ohlcv(ohlcv_json)

    # Compute all required indicators
    ema_periods = cfg.indicators.ema_periods
    df = add_ema(df, periods=ema_periods)
    df = add_rsi(df, period=cfg.indicators.rsi_period)
    df = add_macd(
        df,
        fast=cfg.indicators.macd_fast,
        slow=cfg.indicators.macd_slow,
        signal=cfg.indicators.macd_signal,
    )
    df = add_atr(df, period=cfg.indicators.atr_period)

    rsi_col = f"rsi_{cfg.indicators.rsi_period}"
    rsi_val = (_safe_float(df[rsi_col].iloc[-1]) if rsi_col in df.columns else None) or 50.0
    macd_dir = macd_momentum_direction(
        df,
        fast=cfg.indicators.macd_fast,
        slow=cfg.indicators.macd_slow,
        signal=cfg.indicators.macd_signal,
    )
    daily_trend = trend_direction(df, periods=ema_periods)
    atr_val = current_atr(df, period=cfg.indicators.atr_period)

    # Volume profile
    vp: dict = {}
    if has_reliable_volume(df):
        vp = volume_profile(df, sessions=cfg.indicators.volume_profile_sessions)

    # Detect raw S/R levels
    sr_levels = detect_sr_levels(df)

    # Load strategy weights for the active asset
    strategy_weights = None
    try:
        from strategies import get_strategy
        strategy_entry = get_strategy(cfg.asset.symbol)
        strategy_weights = strategy_entry.config.weights
    except (KeyError, ImportError):
        pass  # fall back to default weights

    # Score and filter
    scored = score_all_levels(
        sr_levels=sr_levels,
        df_1h=df,
        vp=vp,
        rsi_val=rsi_val,
        macd_direction=macd_dir,
        daily_trend=daily_trend,
        min_confidence=cfg.risk.min_confidence,
        min_rr=cfg.risk.min_risk_reward,
        atr_val=atr_val,
        weights=strategy_weights,
    )

    result = [
        {
            "price": sl.price,
            "action": sl.action,
            "confidence": sl.confidence,
            "stop_loss": sl.stop_loss,
            "target_price": sl.target_price,
            "risk_reward": sl.risk_reward,
            "confluence_factors": sl.confluence_factors,
        }
        for sl in scored
    ]
    logger.debug(f"[score_confluence_levels] {len(result)} high-confidence levels")
    return json.dumps(result)
