"""
BTC S/R Bounce Strategy — v1.0

Logic: detect high-scoring Support/Resistance levels and trade bounces off them.

Why this works for BTC:
  - BTC has sharp, volatile price bounces off key S/R levels
  - Volume data from exchanges is highly reliable (liquid 24/7 market)
  - RSI extremes (overbought/oversold) are common and meaningful
  - Volume Profile (POC/VAH/VAL) is a strong edge with real exchange volume
  - Daily trend is less critical — BTC bounces both with and against trend

Backtest result (Sep 2024 – Mar 2025):
  Win rate: 56.4% | Profit factor: 1.81 | Max drawdown: 1.8%
  Phase 0 gate: PASSED
"""

from __future__ import annotations

import pandas as pd

from core.indicators.confluence import score_all_levels
from core.indicators.levels import detect_sr_levels
from core.indicators.momentum import add_macd, add_rsi, macd_momentum_direction
from core.indicators.trend import add_atr, add_ema, trend_direction
from core.indicators.volume import add_vwap, volume_profile
from strategies.base import StrategyConfig, StrategyWeights

# ── Strategy definition ───────────────────────────────────────────────────────

BTC_SR_BOUNCE_V1 = StrategyConfig(
    strategy_id="btc_sr_bounce_v1",
    asset="BTC",
    version="1.0",
    description="S/R bounce strategy for BTC. Trades sharp reversals at high-touch levels.",

    weights=StrategyWeights(
        weight_ema=0.15,            # useful but secondary
        weight_vwap=0.10,           # reliable on BTC (liquid, 24/7)
        weight_volume_profile=0.20, # strong edge — real exchange volume
        weight_sr_touch=0.15,       # per touch, capped at 0.30
        weight_rsi=0.10,            # RSI extremes are common and meaningful on BTC
        weight_macd=0.10,           # MACD confirms momentum
        weight_daily_trend=0.05,    # BTC bounces both ways — trend is secondary
    ),

    min_confidence=0.65,
    min_rr=1.5,
    atr_stop_multiplier=1.5,
    trend_filter=False,             # trade both with and against trend
    max_signals_per_day=4,

    lookback_sr=200,
    lookback_vp=500,
    cluster_pct=0.015,              # 1.5% — BTC moves in large clusters
    swing_window=5,
    top_n_sr=8,

    vwap_tolerance=0.003,
    ema_tolerance=0.003,
    vol_tolerance=0.005,
)


# ── Signal generator ──────────────────────────────────────────────────────────

def generate_signals(
    df_1h: pd.DataFrame,
    df_daily: pd.DataFrame,
    strategy: StrategyConfig,
    ema_periods: list[int],
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal_period: int,
    atr_period: int,
) -> list:
    """
    Walk-forward S/R bounce signal generation for BTC.

    For each bar, uses only past data to:
    1. Apply indicator stack (RSI, MACD, EMA, ATR, VWAP)
    2. Detect S/R levels via swing highs/lows
    3. Score confluence with BTC-tuned weights
    4. Emit signals for levels meeting the threshold

    No look-ahead bias — each bar only sees data[0:i].
    """
    from backtesting.strategy import Signal  # local import to avoid circular

    signals: list[Signal] = []
    warm_up = max(ema_periods) + 50
    last_signal_date: str | None = None
    daily_signal_count = 0

    df_daily_full = add_ema(df_daily, periods=ema_periods)

    for i in range(warm_up, len(df_1h)):
        window = df_1h.iloc[: i + 1]
        current_bar = window.iloc[-1]
        bar_date = str(current_bar.name.date())

        if bar_date != last_signal_date:
            daily_signal_count = 0
            last_signal_date = bar_date

        if daily_signal_count >= strategy.max_signals_per_day:
            continue

        w = add_rsi(window, period=rsi_period)
        w = add_macd(w, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)
        w = add_ema(w, periods=ema_periods)
        w = add_atr(w, period=atr_period)
        w = add_vwap(w)

        rsi_col = f"rsi_{rsi_period}"
        rsi_val = float(w[rsi_col].iloc[-1])
        if pd.isna(rsi_val):
            continue

        macd_dir = macd_momentum_direction(w, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)

        daily_before = df_daily_full[df_daily_full.index < current_bar.name]
        daily_trend = (
            trend_direction(daily_before)
            if len(daily_before) >= max(ema_periods)
            else "ranging"
        )

        sr_window = window.tail(strategy.lookback_sr)
        sr_levels = detect_sr_levels(
            sr_window,
            swing_window=strategy.swing_window,
            cluster_pct=strategy.cluster_pct,
            top_n=strategy.top_n_sr,
        )
        if not sr_levels:
            continue

        vp_window = window.tail(strategy.lookback_vp)
        vp = volume_profile(vp_window)

        atr_col = f"atr_{atr_period}"
        atr_val = (
            float(w[atr_col].iloc[-1])
            if atr_col in w.columns and not pd.isna(w[atr_col].iloc[-1])
            else None
        )

        scored = score_all_levels(
            sr_levels=sr_levels,
            df_1h=w,
            vp=vp,
            rsi_val=rsi_val,
            macd_direction=macd_dir,
            daily_trend=daily_trend,
            min_confidence=strategy.min_confidence,
            min_rr=strategy.min_rr,
            max_alarms=strategy.max_signals_per_day - daily_signal_count,
            atr_val=atr_val,
            atr_stop_multiplier=strategy.atr_stop_multiplier,
            weights=strategy.weights,
            vwap_tolerance=strategy.vwap_tolerance,
            ema_tolerance=strategy.ema_tolerance,
            vol_tolerance=strategy.vol_tolerance,
        )

        for level in scored:
            signals.append(
                Signal(
                    bar_index=i,
                    timestamp=current_bar.name,
                    action=level.action,
                    entry_price=level.price,
                    stop_loss=level.stop_loss,
                    target_price=level.target_price,
                    confidence=level.confidence,
                    risk_reward=level.risk_reward,
                    confluence_factors=level.confluence_factors,
                )
            )
            daily_signal_count += 1

    return signals
