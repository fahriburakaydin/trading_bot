"""
Daily Pivot Points indicator.

Calculates classic floor pivot points from the previous day's OHLC.
These are the same values used on babypips.com and respected by
institutional forex traders.

Formula (Standard/Floor pivots):
    PP  = (High + Low + Close) / 3
    R1  = 2 * PP - Low
    S1  = 2 * PP - High
    R2  = PP + (High - Low)
    S2  = PP - (High - Low)
    R3  = High + 2 * (PP - Low)
    S3  = Low - 2 * (High - PP)

Usage:
    pivots = calculate_daily_pivots(df_daily)
    # Returns: {date: {"pp": ..., "r1": ..., "s1": ..., "r2": ..., "s2": ...}}

    levels = get_pivot_levels_for_bar(pivots, bar_timestamp)
    # Returns the pivot dict for the day containing bar_timestamp
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def calculate_daily_pivots(df_daily: pd.DataFrame) -> dict[date, dict[str, float]]:
    """
    Calculate daily pivot points for each trading day.

    Each day's pivots are based on the PREVIOUS day's High, Low, Close.
    The pivots are valid from the start of the new day until the end of it.

    Args:
        df_daily: Daily OHLCV DataFrame with columns: open, high, low, close
                  Index must be DatetimeIndex (UTC preferred)

    Returns:
        Dict mapping each date → pivot level dict with keys:
            pp, r1, s1, r2, s2, r3, s3
    """
    result: dict[date, dict[str, float]] = {}

    # Shift by 1 row: today's pivots = yesterday's OHLC
    prev = df_daily.shift(1)

    for idx, row in df_daily.iterrows():
        prev_row = prev.loc[idx]
        if pd.isna(prev_row["high"]):
            continue

        h = float(prev_row["high"])
        l = float(prev_row["low"])
        c = float(prev_row["close"])

        pp = (h + l + c) / 3
        r1 = 2 * pp - l
        s1 = 2 * pp - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        r3 = h + 2 * (pp - l)
        s3 = l - 2 * (h - pp)

        day = idx.date() if hasattr(idx, "date") else idx
        result[day] = {
            "pp": round(pp, 5),
            "r1": round(r1, 5),
            "s1": round(s1, 5),
            "r2": round(r2, 5),
            "s2": round(s2, 5),
            "r3": round(r3, 5),
            "s3": round(s3, 5),
        }

    return result


def get_pivot_levels_for_bar(
    pivots: dict[date, dict[str, float]],
    bar_timestamp: pd.Timestamp,
) -> dict[str, float] | None:
    """
    Return the pivot levels active at a given bar timestamp.

    Pivot levels are set at the start of each new day and remain valid
    for the entire day. The "day" uses UTC date.

    Args:
        pivots:         Output of calculate_daily_pivots()
        bar_timestamp:  Timestamp of the current bar

    Returns:
        Dict with pp/r1/s1/r2/s2/r3/s3, or None if no pivots available
    """
    bar_date = bar_timestamp.date() if hasattr(bar_timestamp, "date") else bar_timestamp
    return pivots.get(bar_date)


def pivot_levels_as_list(pivot: dict[str, float]) -> list[tuple[str, float]]:
    """
    Return all pivot levels as a sorted list of (label, price) tuples.

    Sorted from highest price to lowest.

    Args:
        pivot: Pivot dict from calculate_daily_pivots()

    Returns:
        List of (label, price) tuples sorted by price descending
    """
    levels = [
        ("R3", pivot["r3"]),
        ("R2", pivot["r2"]),
        ("R1", pivot["r1"]),
        ("PP", pivot["pp"]),
        ("S1", pivot["s1"]),
        ("S2", pivot["s2"]),
        ("S3", pivot["s3"]),
    ]
    return sorted(levels, key=lambda x: x[1], reverse=True)


def nearest_pivot_above(pivot: dict[str, float], price: float) -> tuple[str, float] | None:
    """Return the nearest pivot level above the current price."""
    candidates = [(label, lvl) for label, lvl in pivot_levels_as_list(pivot) if lvl > price]
    return candidates[-1] if candidates else None   # list is desc, so last = closest above


def nearest_pivot_below(pivot: dict[str, float], price: float) -> tuple[str, float] | None:
    """Return the nearest pivot level below the current price."""
    candidates = [(label, lvl) for label, lvl in pivot_levels_as_list(pivot) if lvl < price]
    return candidates[0] if candidates else None    # list is desc, so first = closest below
