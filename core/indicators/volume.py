"""
Volume indicators: VWAP and Volume Profile (POC, VAH, VAL).
"""

from __future__ import annotations

import pandas as pd


def add_vwap(df: pd.DataFrame, session_aware: bool = True) -> pd.DataFrame:
    """
    Add VWAP (Volume Weighted Average Price).

    In session-aware mode the VWAP resets at the start of each UTC day,
    which approximates session VWAP for crypto (no official session).
    For stocks, intraday VWAP resets at market open.

    Args:
        df:             OHLCV DataFrame with DatetimeIndex (UTC)
        session_aware:  Reset VWAP daily (True) or cumulative (False)

    Returns:
        New DataFrame with column: vwap
    """
    result = df.copy()
    typical_price = (result["high"] + result["low"] + result["close"]) / 3

    if session_aware and hasattr(result.index, "date"):
        # Reset cumulative sums at each new UTC day
        result["_date"] = result.index.date
        result["_tp_vol"] = typical_price * result["volume"]
        result["_cum_tp_vol"] = result.groupby("_date")["_tp_vol"].cumsum()
        result["_cum_vol"] = result.groupby("_date")["volume"].cumsum()
        result["vwap"] = result["_cum_tp_vol"] / result["_cum_vol"].replace(0, float("nan"))
        result.drop(columns=["_date", "_tp_vol", "_cum_tp_vol", "_cum_vol"], inplace=True)
    else:
        cum_tp_vol = (typical_price * result["volume"]).cumsum()
        cum_vol = result["volume"].cumsum()
        result["vwap"] = cum_tp_vol / cum_vol.replace(0, float("nan"))

    return result


def has_reliable_volume(df: pd.DataFrame, min_nonzero_pct: float = 0.5) -> bool:
    """
    Return True if the DataFrame has reliable volume data.

    yfinance often returns zero volume for crypto hourly bars.
    If more than (1 - min_nonzero_pct) of bars have zero volume,
    the data is unreliable for Volume Profile scoring.
    """
    if df.empty or "volume" not in df.columns:
        return False
    nonzero = (df["volume"] > 0).sum()
    return (nonzero / len(df)) >= min_nonzero_pct


def volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
) -> dict[str, float]:
    """
    Calculate Volume Profile for the given OHLCV window.

    Uses close price binned into num_bins to find:
        POC  — Point of Control: price level with highest volume
        VAH  — Value Area High: upper bound of 70% value area
        VAL  — Value Area Low: lower bound of 70% value area

    Args:
        df:       OHLCV DataFrame (typically last N sessions)
        num_bins: Number of price bins (default 50)

    Returns:
        {"poc": float, "vah": float, "val": float}
    """
    if df.empty or df["volume"].sum() == 0:
        mid = float(df["close"].mean()) if not df.empty else 0.0
        return {"poc": mid, "vah": mid, "val": mid}

    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_min == price_max:
        mid = float(price_min)
        return {"poc": mid, "vah": mid, "val": mid}

    bins = pd.cut(df["close"], bins=num_bins, labels=False, include_lowest=True)
    bin_edges = pd.cut(df["close"], bins=num_bins, include_lowest=True, retbins=True)[1]

    vol_per_bin = df.groupby(bins, observed=False)["volume"].sum()
    total_vol = vol_per_bin.sum()

    # Reset to positional index so iloc works reliably
    vol_series = vol_per_bin.reset_index(drop=True)

    # POC: bin with maximum volume (positional index)
    poc_pos = int(vol_series.idxmax())
    poc = float((bin_edges[poc_pos] + bin_edges[poc_pos + 1]) / 2)

    # Value Area: expand from POC until 70% of total volume is captured
    value_area_target = total_vol * 0.70
    captured_vol = float(vol_series.iloc[poc_pos])
    low_bin = poc_pos
    high_bin = poc_pos
    n_bins = len(vol_series)

    while captured_vol < value_area_target:
        can_expand_low = low_bin > 0
        can_expand_high = high_bin < n_bins - 1

        if not can_expand_low and not can_expand_high:
            break

        next_low_vol = float(vol_series.iloc[low_bin - 1]) if can_expand_low else 0
        next_high_vol = float(vol_series.iloc[high_bin + 1]) if can_expand_high else 0

        if next_high_vol >= next_low_vol and can_expand_high:
            high_bin += 1
            captured_vol += next_high_vol
        elif can_expand_low:
            low_bin -= 1
            captured_vol += next_low_vol
        else:
            high_bin += 1
            captured_vol += next_high_vol

    val = float(bin_edges[low_bin])
    vah = float(bin_edges[high_bin + 1])

    return {"poc": poc, "vah": vah, "val": val}


def price_in_value_area(price: float, vp: dict[str, float]) -> bool:
    """Return True if price is between VAL and VAH."""
    return vp["val"] <= price <= vp["vah"]


def is_near_level(price: float, level: float, tolerance_pct: float = 0.005) -> bool:
    """Return True if price is within tolerance_pct of a volume level."""
    return abs(price - level) / level <= tolerance_pct
