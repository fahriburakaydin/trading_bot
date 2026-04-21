"""
EUR/USD Pivot Scalp Strategy — v2.0

Based on the strategy described in the trading video:
  "EUR/USD scalping SL day trading strategy" (20/20 profitable trades claimed)

Core logic:
  1. Calculate daily floor pivot points (PP, R1, S1, R2, S2) from previous day OHLC
  2. On the 15-minute timeframe, wait for price to reach a pivot zone
  3. Require CONFLUENCE:
       a. Pivot zone is also near EMA 20/50 or the EMA cross confirms direction
       b. Stochastic RSI is overbought (SHORT near resistance) or oversold (LONG near support)
       c. Bearish or bullish engulfing candlestick at the pivot zone
  4. Enter with tight stop (just beyond the pivot zone wick, ~9 pips typical)
  5. Partial take-profits: TP1 = 1R (same distance as stop), TP2 = next pivot

Key differences from previous strategies:
  - Uses pivot points (formula-based) instead of swing high/low S/R detection
  - 15-minute timeframe for precise entry timing
  - Engulfing candle = required confirmation (not optional)
  - Stochastic RSI (not plain RSI) — more sensitive, reaches extremes on 15m
  - No daily trend filter needed — pivot + engulfing + stoch-RSI is self-contained
  - Tight stops: 1×ATR on 15m (typically 5–10 pips on EUR/USD)

Indicator weights reflect the pivot strategy approach:
  - Pivot zone proximity: highest weight (cannot be in strategy weights directly,
    but manifests as the mandatory pre-condition — no pivot, no trade)
  - EMA proximity: high (0.25) — EMAs act as additional S/R confluence
  - Stochastic RSI: high (0.25) — the key momentum filter
  - Engulfing candle: embedded in signal generation (mandatory, not weighted)
  - Daily trend: low (0.10) — strategy works both ways, trend is confirmation only
  - MACD: low (0.10) — secondary confirmation
  - Volume profile: very low (0.05) — forex volume unreliable
"""

from __future__ import annotations

import pandas as pd

from core.indicators.candles import candle_signal
from core.indicators.momentum import (
    add_macd,
    add_rsi,
    add_stoch_rsi,
    macd_momentum_direction,
    stoch_rsi_signal,
)
from core.indicators.pivots import (
    calculate_daily_pivots,
    get_pivot_levels_for_bar,
    nearest_pivot_above,
    nearest_pivot_below,
    pivot_levels_as_list,
)
from core.indicators.trend import add_atr, add_ema, trend_direction
from core.indicators.volume import add_vwap, volume_profile
from strategies.base import StrategyConfig, StrategyWeights

# ── Strategy definition ───────────────────────────────────────────────────────

EURUSD_PIVOT_SCALP_V2 = StrategyConfig(
    strategy_id="eurusd_pivot_scalp_v2",
    asset="EURUSD",
    version="2.0",
    description=(
        "Daily pivot point scalp strategy for EUR/USD on 15m timeframe. "
        "Enters at pivot S/R zones confirmed by engulfing candle + Stochastic RSI extreme. "
        "Based on the babypips pivot + EMA confluence method."
    ),

    weights=StrategyWeights(
        weight_ema=0.25,             # EMA as additional S/R confluence at pivot zones
        weight_vwap=0.02,            # not relevant for this strategy
        weight_volume_profile=0.05,  # forex volume unreliable
        weight_sr_touch=0.08,        # secondary — pivot points are the primary zones
        weight_rsi=0.25,             # Stochastic RSI confirmation — key filter
        weight_macd=0.10,            # secondary confirmation
        weight_daily_trend=0.10,     # soft confirmation only — strategy works both ways
    ),

    min_confidence=0.50,             # lower — engulfing + stoch_rsi is mandatory pre-filter
    min_rr=1.5,
    atr_stop_multiplier=1.2,         # 1.2×ATR on 15m ≈ 8–12 pips stop for EUR/USD
    trend_filter=False,              # strategy is self-contained; both directions valid
    max_signals_per_day=4,           # scalp — more trades per day acceptable

    lookback_sr=96,                  # 96 × 15m = 24 hours of S/R history
    lookback_vp=192,                 # 48 hours volume profile
    cluster_pct=0.002,               # very tight for 15m
    swing_window=3,                  # shorter window for 15m swings
    top_n_sr=6,

    vwap_tolerance=0.001,
    ema_tolerance=0.002,
    vol_tolerance=0.002,
)

# Pivot zone proximity tolerance — price must be within this fraction of a pivot level
PIVOT_ZONE_TOLERANCE = 0.0008     # 0.08% ≈ 8–9 pips on EUR/USD at 1.10


# ── Signal generator ──────────────────────────────────────────────────────────

def generate_signals(
    df_1h: pd.DataFrame,        # kept for interface compatibility; not used directly
    df_daily: pd.DataFrame,
    strategy: StrategyConfig,
    ema_periods: list[int],
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal_period: int,
    atr_period: int,
    df_15m: pd.DataFrame | None = None,
) -> list:
    """
    Walk-forward pivot scalp signal generation for EUR/USD on 15m data.

    For each 15m bar, checks:
      SHORT setup (at pivot resistance):
        1. Current price is near a daily pivot resistance level (R1/R2/PP acting as R)
        2. Stochastic RSI is overbought (k >= 80, d >= 80)
        3. Bearish engulfing (or strong bearish) candle at the pivot
        4. EMA 20 or 50 is nearby (optional bonus — raises confidence)
        → Short at the pivot level, stop above the pivot wick, target next pivot below

      LONG setup (at pivot support):
        1. Current price is near a daily pivot support level (S1/S2/PP acting as S)
        2. Stochastic RSI is oversold (k <= 20, d <= 20)
        3. Bullish engulfing (or strong bullish) candle at the pivot
        4. EMA 20 or 50 is nearby (optional bonus)
        → Long at the pivot level, stop below the pivot wick, target next pivot above

    Falls back to df_1h if df_15m is not provided (reduced precision).
    No look-ahead bias — each bar only sees past data.
    """
    from backtesting.strategy import Signal

    # Use 15m if provided, else fall back to 1H
    df = df_15m if df_15m is not None else df_1h

    if len(df) < 100:
        return []

    signals: list[Signal] = []
    warm_up = max(ema_periods) + 50
    last_signal_date: str | None = None
    daily_signal_count = 0

    # Pre-compute pivot points for all days in the daily data
    pivots = calculate_daily_pivots(df_daily)

    for i in range(warm_up, len(df)):
        window = df.iloc[: i + 1]
        current_bar = window.iloc[-1]
        bar_date = str(current_bar.name.date())

        if bar_date != last_signal_date:
            daily_signal_count = 0
            last_signal_date = bar_date

        if daily_signal_count >= strategy.max_signals_per_day:
            continue

        # ── Get today's pivot levels ──────────────────────────────────────────
        pivot = get_pivot_levels_for_bar(pivots, current_bar.name)
        if pivot is None:
            continue

        pivot_list = pivot_levels_as_list(pivot)   # sorted high → low

        # ── Indicators on the current window ─────────────────────────────────
        w = add_rsi(window, period=rsi_period)
        w = add_stoch_rsi(w, rsi_period=rsi_period)
        w = add_macd(w, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)
        w = add_ema(w, periods=ema_periods)
        w = add_atr(w, period=atr_period)

        atr_col = f"atr_{atr_period}"
        if atr_col not in w.columns or pd.isna(w[atr_col].iloc[-1]):
            continue
        atr_val = float(w[atr_col].iloc[-1])

        stoch_k = float(w["stoch_rsi_k"].iloc[-1]) if not pd.isna(w["stoch_rsi_k"].iloc[-1]) else 50.0
        stoch_d = float(w["stoch_rsi_d"].iloc[-1]) if not pd.isna(w["stoch_rsi_d"].iloc[-1]) else 50.0
        stoch_state = stoch_rsi_signal(stoch_k, stoch_d)

        macd_dir = macd_momentum_direction(w, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)

        candle = candle_signal(w, idx=-1)

        current_close = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        # ── Check each pivot level for a setup ───────────────────────────────
        for label, pivot_price in pivot_list:
            if daily_signal_count >= strategy.max_signals_per_day:
                break

            # Is current price near this pivot level?
            distance_pct = abs(current_close - pivot_price) / pivot_price
            if distance_pct > PIVOT_ZONE_TOLERANCE * 3:
                continue   # too far away

            near_pivot = distance_pct <= PIVOT_ZONE_TOLERANCE

            # ── SHORT setup: price near resistance pivot ───────────────────
            # Resistances: R1, R2, R3, and PP when price approaches from below
            is_resistance_pivot = (
                label in ("R1", "R2", "R3")
                or (label == "PP" and current_close < pivot_price)
            )

            if is_resistance_pivot and near_pivot:
                # Required: Stochastic RSI overbought
                if stoch_state != "overbought":
                    continue
                # Required: bearish candle confirmation
                bearish_candle = candle in ("bearish_engulfing", "strong_bearish")
                if not bearish_candle:
                    continue

                entry_price = round(pivot_price, 5)
                stop_loss = round(current_high + atr_val * strategy.atr_stop_multiplier, 5)

                # Target: next pivot below
                next_below = nearest_pivot_below(pivot, entry_price * 0.9995)
                if next_below:
                    target_price = round(next_below[1], 5)
                    target_label = next_below[0]
                else:
                    target_price = round(entry_price - atr_val * 2.0, 5)
                    target_label = "2×ATR"

                sl_dist = stop_loss - entry_price
                tp_dist = entry_price - target_price
                if sl_dist <= 0 or tp_dist <= 0:
                    continue
                rr = tp_dist / sl_dist
                if rr < strategy.min_rr:
                    continue

                # Confidence: base from stoch + candle, bonus for EMA proximity
                confidence = _pivot_confidence(
                    pivot_price=pivot_price,
                    w=w,
                    ema_periods=ema_periods,
                    stoch_state=stoch_state,
                    candle=candle,
                    macd_dir=macd_dir,
                    weights=strategy.weights,
                    ema_tolerance=strategy.ema_tolerance,
                )
                if confidence < strategy.min_confidence:
                    continue

                factors = [
                    f"Pivot {label} @ {pivot_price:.5f}",
                    f"Stoch RSI overbought (K={stoch_k:.0f})",
                    f"{candle.replace('_', ' ').title()}",
                ]
                if macd_dir == "bearish":
                    factors.append("MACD bearish")

                signals.append(Signal(
                    bar_index=i,
                    timestamp=current_bar.name,
                    action="SHORT",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    confidence=min(confidence, 1.0),
                    risk_reward=round(rr, 2),
                    confluence_factors=factors,
                    entry_type="limit",
                ))
                daily_signal_count += 1

            # ── LONG setup: price near support pivot ───────────────────────
            # Supports: S1, S2, S3, and PP when price approaches from above
            is_support_pivot = (
                label in ("S1", "S2", "S3")
                or (label == "PP" and current_close > pivot_price)
            )

            if is_support_pivot and near_pivot:
                if stoch_state != "oversold":
                    continue
                bullish_candle = candle in ("bullish_engulfing", "strong_bullish")
                if not bullish_candle:
                    continue

                entry_price = round(pivot_price, 5)
                stop_loss = round(current_low - atr_val * strategy.atr_stop_multiplier, 5)

                next_above = nearest_pivot_above(pivot, entry_price * 1.0005)
                if next_above:
                    target_price = round(next_above[1], 5)
                    target_label = next_above[0]
                else:
                    target_price = round(entry_price + atr_val * 2.0, 5)
                    target_label = "2×ATR"

                sl_dist = entry_price - stop_loss
                tp_dist = target_price - entry_price
                if sl_dist <= 0 or tp_dist <= 0:
                    continue
                rr = tp_dist / sl_dist
                if rr < strategy.min_rr:
                    continue

                confidence = _pivot_confidence(
                    pivot_price=pivot_price,
                    w=w,
                    ema_periods=ema_periods,
                    stoch_state=stoch_state,
                    candle=candle,
                    macd_dir=macd_dir,
                    weights=strategy.weights,
                    ema_tolerance=strategy.ema_tolerance,
                )
                if confidence < strategy.min_confidence:
                    continue

                factors = [
                    f"Pivot {label} @ {pivot_price:.5f}",
                    f"Stoch RSI oversold (K={stoch_k:.0f})",
                    f"{candle.replace('_', ' ').title()}",
                ]
                if macd_dir == "bullish":
                    factors.append("MACD bullish")

                signals.append(Signal(
                    bar_index=i,
                    timestamp=current_bar.name,
                    action="LONG",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    confidence=min(confidence, 1.0),
                    risk_reward=round(rr, 2),
                    confluence_factors=factors,
                    entry_type="limit",
                ))
                daily_signal_count += 1

    return signals


def _pivot_confidence(
    pivot_price: float,
    w: pd.DataFrame,
    ema_periods: list[int],
    stoch_state: str,
    candle: str,
    macd_dir: str,
    weights,
    ema_tolerance: float,
) -> float:
    """
    Calculate confidence score for a pivot-based setup.

    Mandatory factors (already confirmed before calling this):
      - Stochastic RSI extreme
      - Engulfing or strong candle

    Bonus factors scored here:
      - EMA proximity at the pivot level
      - MACD direction alignment
    """
    # Base: stoch RSI extreme (mandatory — already confirmed before calling)
    score = weights.weight_rsi

    # Engulfing candle (mandatory — always add its weight)
    # Bearish engulfing > strong bearish; bullish engulfing > strong bullish
    if candle in ("bearish_engulfing", "bullish_engulfing"):
        score += weights.weight_macd        # full momentum weight for engulfing
    elif candle in ("strong_bearish", "strong_bullish"):
        score += weights.weight_macd * 0.6  # partial for strong-but-not-engulfing

    # EMA proximity at the pivot level (bonus)
    ema_score = 0.0
    for period in ema_periods:
        col = f"ema_{period}"
        if col in w.columns:
            ema_val = float(w[col].iloc[-1])
            if abs(pivot_price - ema_val) / ema_val <= ema_tolerance:
                ema_score += weights.weight_ema / len(ema_periods)
    score += min(ema_score, weights.weight_ema)

    # MACD direction alignment (extra bonus when trend agrees)
    if (candle in ("bearish_engulfing", "strong_bearish") and macd_dir == "bearish") or \
       (candle in ("bullish_engulfing", "strong_bullish") and macd_dir == "bullish"):
        score += weights.weight_daily_trend  # use daily_trend weight as MACD bonus slot

    return round(score, 3)
