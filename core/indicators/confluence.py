"""
Confluence scoring for price levels.

Scores each candidate level based on how many independent TA signals
align at that price. Higher score = higher probability level.

Scoring weights are now passed in via StrategyWeights rather than being
hardcoded constants — this allows each asset strategy to weight indicators
differently based on what actually works for that asset.

Default weights (used when no StrategyWeights is supplied):
    EMA proximity       (+0.15 per period, cap 0.15)
    VWAP alignment      (+0.10)
    Volume POC/VAH/VAL  (+0.20)
    Historical S/R      (+0.15/touch, max 0.30)
    RSI extreme         (+0.10)
    MACD alignment      (+0.10)
    Trend alignment     (+0.05)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.indicators.levels import SRLevel
from core.indicators.momentum import rsi_signal
from core.indicators.trend import is_near_ema, trend_direction
from core.indicators.volume import has_reliable_volume, is_near_level


@dataclass
class ScoredLevel:
    price: float
    action: str                     # "LONG" | "SHORT"
    confidence: float               # 0.0 – 1.0
    stop_loss: float
    target_price: float
    risk_reward: float
    confluence_factors: list[str]
    reasoning: str
    sr_level: SRLevel | None = None

    def __repr__(self) -> str:
        return (
            f"<ScoredLevel {self.action} @ {self.price:.5f} "
            f"conf={self.confidence:.2f} R:R={self.risk_reward:.2f}>"
        )


def score_level(
    sr_level: SRLevel,
    df_1h: pd.DataFrame,
    vp: dict[str, float],
    rsi_val: float,
    macd_direction: str,
    daily_trend: str,
    ema_periods: list[int] | None = None,
    vwap_tolerance: float = 0.003,
    ema_tolerance: float = 0.003,
    vol_tolerance: float = 0.005,
    min_rr: float = 1.5,
    atr_val: float | None = None,
    atr_stop_multiplier: float = 1.5,
    weights=None,   # StrategyWeights | None — avoid circular import with type hint
) -> ScoredLevel | None:
    """
    Score a single S/R level and build a ScoredLevel if strong enough.

    Args:
        sr_level:             The S/R level being scored
        df_1h:                1H OHLCV with all indicator columns populated
        vp:                   Volume profile dict {poc, vah, val}
        rsi_val:              Current RSI value
        macd_direction:       "bullish" | "bearish" | "neutral"
        daily_trend:          "bullish" | "bearish" | "ranging" from Daily EMA
        ema_periods:          EMA periods to check (default [20, 50, 200])
        vwap_tolerance:       Proximity threshold for VWAP
        ema_tolerance:        Proximity threshold for EMA
        vol_tolerance:        Proximity threshold for volume levels
        min_rr:               Minimum Risk:Reward to accept the level
        atr_val:              ATR value for stop/target sizing
        atr_stop_multiplier:  Multiplier for ATR stop distance (strategy-specific)
        weights:              StrategyWeights instance with per-asset factor weights.
                              If None, uses default weights matching original behaviour.

    Returns:
        ScoredLevel if the level has any confluence, else None
    """
    if ema_periods is None:
        ema_periods = [20, 50, 200]

    # Default weights (original hardcoded values) when no strategy weights supplied
    if weights is None:
        from strategies.base import StrategyWeights
        weights = StrategyWeights()

    price = sr_level.price
    factors: list[str] = []
    score = 0.0

    # Determine intended trade direction from level type
    if sr_level.level_type == "support":
        action = "LONG"
    elif sr_level.level_type == "resistance":
        action = "SHORT"
    else:
        # "both" — use trend to decide
        action = "LONG" if daily_trend == "bullish" else "SHORT"

    # ── Factor 1: EMA proximity ───────────────────────────────────────────────
    ema_score = 0.0
    for period in ema_periods:
        col = f"ema_{period}"
        if col in df_1h.columns:
            ema_val = float(df_1h[col].iloc[-1])
            if abs(price - ema_val) / ema_val <= ema_tolerance:
                ema_score += weights.weight_ema / len(ema_periods)
                factors.append(f"EMA{period} @ {ema_val:.5f}")

    score += min(ema_score, weights.weight_ema)  # cap at full weight

    # ── Factor 2: VWAP alignment ──────────────────────────────────────────────
    if "vwap" in df_1h.columns:
        vwap_val = float(df_1h["vwap"].iloc[-1])
        if is_near_level(price, vwap_val, tolerance_pct=vwap_tolerance):
            score += weights.weight_vwap
            factors.append(f"VWAP @ {vwap_val:.5f}")

    # ── Factor 3: Volume Profile ──────────────────────────────────────────────
    if has_reliable_volume(df_1h):
        for vp_key, label in [("poc", "Volume POC"), ("vah", "Volume VAH"), ("val", "Volume VAL")]:
            if is_near_level(price, vp[vp_key], tolerance_pct=vol_tolerance):
                score += weights.weight_volume_profile
                factors.append(f"{label} @ {vp[vp_key]:.5f}")
                break  # count only once even if near multiple

    # ── Factor 4: Historical S/R touches ─────────────────────────────────────
    touch_cap = weights.weight_sr_touch * 2   # cap at 2x the per-touch weight
    touch_contribution = min(sr_level.touch_count * weights.weight_sr_touch, touch_cap)
    if sr_level.touch_count >= 2:
        score += touch_contribution
        factors.append(f"{sr_level.touch_count} historical touches")

    # ── Factor 5: RSI extreme ─────────────────────────────────────────────────
    rsi_state = rsi_signal(rsi_val)
    if action == "LONG" and rsi_state == "oversold":
        score += weights.weight_rsi
        factors.append(f"RSI oversold ({rsi_val:.1f})")
    elif action == "SHORT" and rsi_state == "overbought":
        score += weights.weight_rsi
        factors.append(f"RSI overbought ({rsi_val:.1f})")

    # ── Factor 6: MACD alignment ──────────────────────────────────────────────
    if (action == "LONG" and macd_direction == "bullish") or (
        action == "SHORT" and macd_direction == "bearish"
    ):
        score += weights.weight_macd
        factors.append(f"MACD {macd_direction}")

    # ── Factor 7: Daily trend alignment ───────────────────────────────────────
    if (action == "LONG" and daily_trend == "bullish") or (
        action == "SHORT" and daily_trend == "bearish"
    ):
        score += weights.weight_daily_trend
        factors.append(f"Daily trend {daily_trend}")

    if not factors:
        return None

    confidence = min(round(score, 3), 1.0)

    # ── Stop loss and target calculation ──────────────────────────────────────
    if atr_val and atr_val > 0:
        sl_distance = atr_val * atr_stop_multiplier
        tp_distance = sl_distance * min_rr
    else:
        # Fallback: 1% stop, scaled by min_rr for target
        sl_distance = price * 0.01
        tp_distance = price * 0.01 * min_rr

    # Preserve enough decimal places: forex needs 5, stocks need 2
    price_decimals = 5 if price < 100 else 2

    if action == "LONG":
        stop_loss = round(price - sl_distance, price_decimals)
        target_price = round(price + tp_distance, price_decimals)
    else:
        stop_loss = round(price + sl_distance, price_decimals)
        target_price = round(price - tp_distance, price_decimals)

    rr = tp_distance / sl_distance if sl_distance > 0 else 0
    if rr < min_rr:
        return None  # filter: R:R too low

    reasoning = (
        f"{action} at {price:.{price_decimals}f}. "
        f"Confluence factors: {', '.join(factors)}. "
        f"Confidence: {confidence:.0%}. "
        f"Stop: {stop_loss:.{price_decimals}f}, Target: {target_price:.{price_decimals}f}, R:R: {rr:.2f}."
    )

    return ScoredLevel(
        price=round(price, price_decimals),
        action=action,
        confidence=confidence,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_reward=round(rr, 2),
        confluence_factors=factors,
        reasoning=reasoning,
        sr_level=sr_level,
    )


def score_all_levels(
    sr_levels: list[SRLevel],
    df_1h: pd.DataFrame,
    vp: dict[str, float],
    rsi_val: float,
    macd_direction: str,
    daily_trend: str,
    min_confidence: float = 0.65,
    min_rr: float = 1.5,
    max_alarms: int = 4,
    atr_val: float | None = None,
    atr_stop_multiplier: float = 1.5,
    weights=None,   # StrategyWeights | None
    vwap_tolerance: float = 0.003,
    ema_tolerance: float = 0.003,
    vol_tolerance: float = 0.005,
) -> list[ScoredLevel]:
    """
    Score all S/R levels and return those meeting the confidence threshold.

    Args:
        sr_levels:            List of raw S/R levels from detect_sr_levels()
        min_confidence:       Minimum score to include
        max_alarms:           Maximum alarms to set per session
        atr_stop_multiplier:  Strategy-specific ATR stop multiplier
        weights:              Strategy-specific indicator weights

    Returns:
        List of ScoredLevel, sorted by confidence descending, capped at max_alarms
    """
    scored = []
    for level in sr_levels:
        result = score_level(
            sr_level=level,
            df_1h=df_1h,
            vp=vp,
            rsi_val=rsi_val,
            macd_direction=macd_direction,
            daily_trend=daily_trend,
            min_rr=min_rr,
            atr_val=atr_val,
            atr_stop_multiplier=atr_stop_multiplier,
            weights=weights,
            vwap_tolerance=vwap_tolerance,
            ema_tolerance=ema_tolerance,
            vol_tolerance=vol_tolerance,
        )
        if result and result.confidence >= min_confidence:
            scored.append(result)

    scored.sort(key=lambda x: x.confidence, reverse=True)
    return scored[:max_alarms]
