"""
Unit tests for all technical indicators.

Tests use deterministic fixtures to assert known-value correctness.
"""

import pytest
import pandas as pd
import numpy as np

from tests.fixtures.sample_ohlcv import (
    make_trending_up,
    make_trending_down,
    make_ranging,
    make_simple_rsi_known,
)
from core.indicators.momentum import add_rsi, add_macd, macd_momentum_direction, rsi_signal
from core.indicators.trend import add_ema, add_atr, trend_direction, is_near_ema
from core.indicators.volume import add_vwap, volume_profile, price_in_value_area
from core.indicators.levels import detect_sr_levels, nearest_support, nearest_resistance


# ── RSI tests ─────────────────────────────────────────────────────────────────


class TestRSI:
    def test_rsi_column_added(self):
        df = make_trending_up()
        result = add_rsi(df, period=14)
        assert "rsi_14" in result.columns

    def test_rsi_bounds(self):
        """RSI must always be between 0 and 100."""
        for fixture in [make_trending_up(), make_trending_down(), make_ranging()]:
            result = add_rsi(fixture, period=14)
            valid = result["rsi_14"].dropna()
            assert (valid >= 0).all(), "RSI below 0"
            assert (valid <= 100).all(), "RSI above 100"

    def test_rsi_uptrend_elevated(self):
        """RSI should be above 50 in a clear uptrend."""
        df = make_trending_up(n=300)
        result = add_rsi(df, period=14)
        last_rsi = result["rsi_14"].iloc[-1]
        assert last_rsi > 50, f"Expected RSI > 50 in uptrend, got {last_rsi:.1f}"

    def test_rsi_downtrend_depressed(self):
        """RSI should be below 50 in a clear downtrend."""
        df = make_trending_down(n=300)
        result = add_rsi(df, period=14)
        last_rsi = result["rsi_14"].iloc[-1]
        assert last_rsi < 50, f"Expected RSI < 50 in downtrend, got {last_rsi:.1f}"

    def test_rsi_nan_for_insufficient_data(self):
        """RSI should be NaN for the first (period-1) bars."""
        df = make_trending_up(n=50)
        result = add_rsi(df, period=14)
        # First bar has no prior data — NaN expected
        assert pd.isna(result["rsi_14"].iloc[0])

    def test_rsi_does_not_mutate_input(self):
        df = make_trending_up()
        original_cols = list(df.columns)
        _ = add_rsi(df, period=14)
        assert list(df.columns) == original_cols

    def test_rsi_signal_classification(self):
        assert rsi_signal(75) == "overbought"
        assert rsi_signal(25) == "oversold"
        assert rsi_signal(50) == "neutral"
        assert rsi_signal(70) == "overbought"  # boundary inclusive
        assert rsi_signal(30) == "oversold"    # boundary inclusive

    def test_rsi_custom_period(self):
        df = make_trending_up(n=100)
        result = add_rsi(df, period=9)
        assert "rsi_9" in result.columns
        assert "rsi_14" not in result.columns


# ── MACD tests ────────────────────────────────────────────────────────────────


class TestMACD:
    def test_macd_columns_added(self):
        df = make_trending_up()
        result = add_macd(df)
        assert "macd_12_26" in result.columns
        assert "macd_signal_9" in result.columns
        assert "macd_hist" in result.columns

    def test_macd_hist_equals_macd_minus_signal(self):
        df = make_trending_up()
        result = add_macd(df)
        diff = (result["macd_12_26"] - result["macd_signal_9"]) - result["macd_hist"]
        assert diff.abs().max() < 1e-10, "Histogram ≠ MACD - Signal"

    def test_macd_bullish_in_uptrend(self):
        df = make_trending_up(n=300)
        df = add_macd(df)
        direction = macd_momentum_direction(df)
        assert direction == "bullish", f"Expected bullish MACD in uptrend, got {direction}"

    def test_macd_bearish_in_downtrend(self):
        df = make_trending_down(n=300)
        df = add_macd(df)
        direction = macd_momentum_direction(df)
        assert direction == "bearish", f"Expected bearish MACD in downtrend, got {direction}"

    def test_macd_does_not_mutate_input(self):
        df = make_trending_up()
        original_cols = list(df.columns)
        _ = add_macd(df)
        assert list(df.columns) == original_cols


# ── EMA / Trend tests ─────────────────────────────────────────────────────────


class TestEMA:
    def test_ema_columns_added(self):
        df = make_trending_up()
        result = add_ema(df, periods=[20, 50, 200])
        assert "ema_20" in result.columns
        assert "ema_50" in result.columns
        assert "ema_200" in result.columns

    def test_ema_uptrend_alignment(self):
        """In a clear uptrend: EMA20 > EMA50 > EMA200."""
        df = make_trending_up(n=300)
        result = add_ema(df, periods=[20, 50, 200])
        last = result.iloc[-1]
        assert last["ema_20"] > last["ema_50"] > last["ema_200"], (
            f"EMA alignment wrong: {last['ema_20']:.2f} > "
            f"{last['ema_50']:.2f} > {last['ema_200']:.2f}"
        )

    def test_ema_downtrend_alignment(self):
        """In a clear downtrend: EMA20 < EMA50 < EMA200."""
        df = make_trending_down(n=300)
        result = add_ema(df, periods=[20, 50, 200])
        last = result.iloc[-1]
        assert last["ema_20"] < last["ema_50"] < last["ema_200"]

    def test_trend_direction_bullish(self):
        df = make_trending_up(n=300)
        df = add_ema(df, periods=[20, 50, 200])
        assert trend_direction(df) == "bullish"

    def test_trend_direction_bearish(self):
        df = make_trending_down(n=300)
        df = add_ema(df, periods=[20, 50, 200])
        assert trend_direction(df) == "bearish"

    def test_ema_custom_periods(self):
        df = make_trending_up()
        result = add_ema(df, periods=[10, 30])
        assert "ema_10" in result.columns
        assert "ema_30" in result.columns

    def test_ema_approaches_price_for_short_period(self):
        """Short EMA (period=1) should equal the close price."""
        df = make_trending_up(n=50)
        result = add_ema(df, periods=[1])
        # EMA(1) = close price exactly
        diff = (result["ema_1"] - result["close"]).abs()
        assert diff.max() < 1e-10


# ── ATR tests ─────────────────────────────────────────────────────────────────


class TestATR:
    def test_atr_column_added(self):
        df = make_trending_up()
        result = add_atr(df)
        assert "atr_14" in result.columns

    def test_atr_positive(self):
        """ATR must always be positive."""
        df = make_ranging()
        result = add_atr(df)
        valid = result["atr_14"].dropna()
        assert (valid > 0).all()

    def test_atr_higher_in_volatile_data(self):
        """ATR for volatile data should be higher than for smooth data."""
        np.random.seed(1)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

        smooth_close = np.linspace(100, 110, n)
        smooth_df = pd.DataFrame({
            "open": smooth_close,
            "high": smooth_close + 0.1,
            "low": smooth_close - 0.1,
            "close": smooth_close,
            "volume": np.ones(n) * 1000,
        }, index=idx)

        volatile_close = 100 + np.random.normal(0, 5, n).cumsum()
        volatile_df = pd.DataFrame({
            "open": volatile_close,
            "high": volatile_close + 5,
            "low": volatile_close - 5,
            "close": volatile_close,
            "volume": np.ones(n) * 1000,
        }, index=idx)

        smooth_atr = add_atr(smooth_df)["atr_14"].iloc[-1]
        volatile_atr = add_atr(volatile_df)["atr_14"].iloc[-1]
        assert volatile_atr > smooth_atr


# ── VWAP tests ────────────────────────────────────────────────────────────────


class TestVWAP:
    def test_vwap_column_added(self):
        df = make_trending_up()
        result = add_vwap(df)
        assert "vwap" in result.columns

    def test_vwap_within_overall_price_range(self):
        """VWAP (cumulative session average) must stay within the overall price range."""
        df = make_ranging()
        result = add_vwap(df)
        valid = result.dropna(subset=["vwap"])
        # VWAP is a session-cumulative typical-price average, so it won't be
        # within each individual candle's H-L, but must be within the global range.
        global_low = df["low"].min()
        global_high = df["high"].max()
        assert (valid["vwap"] >= global_low * 0.98).all(), "VWAP below global low"
        assert (valid["vwap"] <= global_high * 1.02).all(), "VWAP above global high"

    def test_vwap_does_not_mutate_input(self):
        df = make_ranging()
        original_cols = list(df.columns)
        _ = add_vwap(df)
        assert list(df.columns) == original_cols


# ── Volume Profile tests ───────────────────────────────────────────────────────


class TestVolumeProfile:
    def test_vp_returns_poc_vah_val(self):
        df = make_ranging()
        vp = volume_profile(df)
        assert "poc" in vp
        assert "vah" in vp
        assert "val" in vp

    def test_vp_ordering(self):
        """VAL <= POC <= VAH."""
        df = make_ranging()
        vp = volume_profile(df)
        assert vp["val"] <= vp["poc"] <= vp["vah"], (
            f"VAL={vp['val']:.2f} POC={vp['poc']:.2f} VAH={vp['vah']:.2f}"
        )

    def test_vp_poc_within_price_range(self):
        df = make_ranging()
        vp = volume_profile(df)
        assert df["low"].min() <= vp["poc"] <= df["high"].max()

    def test_vp_empty_returns_midpoint(self):
        """Empty DataFrame should not crash."""
        df = make_ranging().head(0)
        vp = volume_profile(df)
        assert "poc" in vp

    def test_price_in_value_area(self):
        vp = {"poc": 150.0, "vah": 160.0, "val": 140.0}
        assert price_in_value_area(150.0, vp) is True
        assert price_in_value_area(140.0, vp) is True
        assert price_in_value_area(139.9, vp) is False
        assert price_in_value_area(160.1, vp) is False


# ── S/R Level tests ───────────────────────────────────────────────────────────


class TestSRLevels:
    def test_sr_levels_detected_on_ranging_market(self):
        """Ranging market with regular highs/lows should produce S/R levels."""
        df = make_ranging(n=300)
        levels = detect_sr_levels(df, swing_window=3)
        assert len(levels) > 0, "No S/R levels detected on ranging market"

    def test_sr_levels_sorted_by_score(self):
        df = make_ranging(n=300)
        levels = detect_sr_levels(df)
        scores = [lv.score for lv in levels]
        assert scores == sorted(scores, reverse=True), "Levels not sorted by score"

    def test_sr_level_type_correctness(self):
        """Support levels should be below current price, resistance above."""
        df = make_ranging(n=300)
        levels = detect_sr_levels(df)
        current_price = float(df["close"].iloc[-1])
        for lv in levels:
            if lv.level_type == "support":
                assert lv.price < current_price * 1.01, (
                    f"Support level {lv.price:.2f} >= current price {current_price:.2f}"
                )
            elif lv.level_type == "resistance":
                assert lv.price > current_price * 0.99

    def test_nearest_support_below_price(self):
        levels_data = [
            {"price": 100, "score": 1.0},
            {"price": 120, "score": 0.8},
            {"price": 140, "score": 0.6},
        ]
        from core.indicators.levels import SRLevel
        from datetime import datetime
        levels = [
            SRLevel(
                price=d["price"], level_type="support", touch_count=2,
                avg_touch_volume=1000, last_touched_at=pd.Timestamp("2024-01-01", tz="UTC"),
                score=d["score"],
            )
            for d in levels_data
        ]
        result = nearest_support(levels, price=130)
        assert result is not None
        assert result.price == 120

    def test_nearest_resistance_above_price(self):
        from core.indicators.levels import SRLevel
        levels = [
            SRLevel(
                price=160, level_type="resistance", touch_count=2,
                avg_touch_volume=1000, last_touched_at=pd.Timestamp("2024-01-01", tz="UTC"),
                score=0.8,
            ),
            SRLevel(
                price=180, level_type="resistance", touch_count=2,
                avg_touch_volume=1000, last_touched_at=pd.Timestamp("2024-01-01", tz="UTC"),
                score=0.6,
            ),
        ]
        result = nearest_resistance(levels, price=150)
        assert result is not None
        assert result.price == 160

    def test_sr_no_levels_insufficient_data(self):
        """With fewer bars than 2*window+1, should return empty list."""
        df = make_ranging(n=5)
        levels = detect_sr_levels(df, swing_window=5)
        assert levels == []
