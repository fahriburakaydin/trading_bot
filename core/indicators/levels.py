"""
Support and Resistance (S/R) detection.

Algorithm:
  1. Find swing highs and swing lows using a local extrema method
  2. Cluster nearby levels (within cluster_pct of each other)
  3. Score each cluster by:
     a. Number of touches (each touch = +1 score)
     b. Volume at touch candles (higher volume = stronger level)
     c. Recency (more recent touches weighted higher)
  4. Return top-N levels sorted by score descending
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SRLevel:
    price: float
    level_type: str        # "support" | "resistance" | "both"
    touch_count: int
    avg_touch_volume: float
    last_touched_at: pd.Timestamp
    score: float
    constituent_prices: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<SRLevel {self.level_type} @ {self.price:.2f} "
            f"touches={self.touch_count} score={self.score:.2f}>"
        )


def _find_swing_highs(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Find local maxima (swing highs) using a rolling window.
    A swing high is a candle whose high is the maximum in a 2*window+1 bar range.
    """
    highs = df["high"]
    is_swing_high = (
        highs == highs.rolling(2 * window + 1, center=True, min_periods=window).max()
    )
    return df[is_swing_high][["high", "volume"]].rename(columns={"high": "price"})


def _find_swing_lows(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Find local minima (swing lows) using a rolling window.
    A swing low is a candle whose low is the minimum in a 2*window+1 bar range.
    """
    lows = df["low"]
    is_swing_low = (
        lows == lows.rolling(2 * window + 1, center=True, min_periods=window).min()
    )
    return df[is_swing_low][["low", "volume"]].rename(columns={"low": "price"})


def _cluster_levels(
    prices_with_meta: list[tuple[float, float, pd.Timestamp]],  # (price, volume, timestamp)
    cluster_pct: float = 0.015,
) -> list[dict]:
    """
    Group nearby price levels into clusters.

    Two levels are in the same cluster if they are within cluster_pct of each other.
    The cluster's representative price is the volume-weighted mean.
    """
    if not prices_with_meta:
        return []

    # Sort by price
    sorted_levels = sorted(prices_with_meta, key=lambda x: x[0])
    clusters = []
    current_cluster: list[tuple[float, float, pd.Timestamp]] = [sorted_levels[0]]

    for price, volume, ts in sorted_levels[1:]:
        ref_price = current_cluster[0][0]
        if abs(price - ref_price) / ref_price <= cluster_pct:
            current_cluster.append((price, volume, ts))
        else:
            clusters.append(current_cluster)
            current_cluster = [(price, volume, ts)]

    clusters.append(current_cluster)

    result = []
    for cluster in clusters:
        prices = [c[0] for c in cluster]
        volumes = [c[1] for c in cluster]
        timestamps = [c[2] for c in cluster]
        total_vol = sum(volumes)
        # Fall back to simple mean when volume is zero/unreliable (e.g. forex data)
        if total_vol > 0:
            weighted_price = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        else:
            weighted_price = sum(prices) / len(prices)
        result.append(
            {
                "price": weighted_price,
                "touch_count": len(cluster),
                "avg_volume": total_vol / len(cluster),
                "last_touched_at": max(timestamps),
                "constituent_prices": prices,
            }
        )

    return result


def detect_sr_levels(
    df: pd.DataFrame,
    swing_window: int = 5,
    cluster_pct: float = 0.003,
    top_n: int = 8,
    recency_weight: float = 0.3,
) -> list[SRLevel]:
    """
    Detect Support and Resistance levels from OHLCV data.

    Args:
        df:             OHLCV DataFrame (1H preferred, last 100–500 bars)
        swing_window:   Half-window for swing detection (default 5 → 11-bar window)
        cluster_pct:    Max distance to merge levels (default 1.5%)
        top_n:          Return top N levels by score
        recency_weight: Weight given to recent touches vs. old ones (0–1)

    Returns:
        List of SRLevel objects, sorted by score descending
    """
    if len(df) < 2 * swing_window + 1:
        return []

    # Collect swing highs and lows
    swing_highs = _find_swing_highs(df, window=swing_window)
    swing_lows = _find_swing_lows(df, window=swing_window)

    high_meta = [
        (row["price"], row["volume"], idx)
        for idx, row in swing_highs.iterrows()
    ]
    low_meta = [
        (row["price"], row["volume"], idx)
        for idx, row in swing_lows.iterrows()
    ]
    all_meta = high_meta + low_meta

    if not all_meta:
        return []

    clusters = _cluster_levels(all_meta, cluster_pct=cluster_pct)
    if not clusters:
        return []

    # Score each cluster
    most_recent = df.index[-1]
    oldest = df.index[0]
    time_range = (most_recent - oldest).total_seconds() or 1.0

    scored = []
    for cluster in clusters:
        touch_score = cluster["touch_count"]
        vol_score = cluster["avg_volume"] / (df["volume"].mean() or 1.0)

        # Recency: 1.0 for most recent, 0.0 for oldest
        age_seconds = (most_recent - cluster["last_touched_at"]).total_seconds()
        recency = max(0.0, 1.0 - age_seconds / time_range)

        score = touch_score * (1 - recency_weight) + recency * recency_weight * touch_score

        # Level type based on position vs current price
        current_price = df["close"].iloc[-1]
        if cluster["price"] < current_price * 0.998:
            level_type = "support"
        elif cluster["price"] > current_price * 1.002:
            level_type = "resistance"
        else:
            level_type = "both"

        # Use enough decimal places: forex needs 5, stocks need 2
        price_decimals = 5 if cluster["price"] < 100 else 2
        scored.append(
            SRLevel(
                price=round(cluster["price"], price_decimals),
                level_type=level_type,
                touch_count=cluster["touch_count"],
                avg_touch_volume=cluster["avg_volume"],
                last_touched_at=cluster["last_touched_at"],
                score=round(score, 3),
                constituent_prices=cluster["constituent_prices"],
            )
        )

    # Sort by score descending, return top N
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]


def nearest_support(levels: list[SRLevel], price: float) -> SRLevel | None:
    """Return the nearest support level below the current price."""
    supports = [lv for lv in levels if lv.price < price]
    if not supports:
        return None
    return max(supports, key=lambda lv: lv.price)


def nearest_resistance(levels: list[SRLevel], price: float) -> SRLevel | None:
    """Return the nearest resistance level above the current price."""
    resistances = [lv for lv in levels if lv.price > price]
    if not resistances:
        return None
    return min(resistances, key=lambda lv: lv.price)
