"""
Historical OHLCV data loader.

Downloads and caches data from yfinance for backtesting.
For live trading, IBKR provides the data via core.broker.market_data.

Cache: data/cache/{symbol}_{interval}_{start}_{end}.parquet
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    key = f"{symbol}_{interval}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{symbol}_{interval}_{h}.parquet"


def _yf_symbol(symbol: str, currency: str = "USD") -> str:
    """Convert our symbol to yfinance format."""
    crypto_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "LTC": "LTC-USD"}
    forex_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
                 "AUDUSD": "AUDUSD=X", "USDCHF": "CHF=X", "USDCAD": "CAD=X"}
    sym = symbol.upper()
    if sym in crypto_map and currency == "USD":
        return crypto_map[sym]
    if sym in forex_map:
        return forex_map[sym]
    return symbol  # stocks already in correct format


def load_ohlcv(
    symbol: str,
    interval: str,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    currency: str = "USD",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from yfinance with local parquet cache.

    Args:
        symbol:   Asset symbol e.g. "BTC", "AAPL"
        interval: yfinance interval: "1h", "15m", "1d" etc.
        start:    Start date (inclusive)
        end:      End date (exclusive); defaults to today
        currency: Quote currency (for crypto lookup)
        use_cache: Whether to use local cache

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: UTC DatetimeIndex
    """
    if end is None:
        end = datetime.utcnow()

    start_str = str(start)[:10]
    end_str = str(end)[:10]
    cache_file = _cache_path(symbol, interval, start_str, end_str)

    if use_cache and cache_file.exists():
        logger.debug(f"Loading {symbol} {interval} from cache: {cache_file.name}")
        df = pd.read_parquet(cache_file)
        return df

    yf_sym = _yf_symbol(symbol, currency)
    logger.info(f"Downloading {yf_sym} {interval} from {start_str} to {end_str}")

    ticker = yf.Ticker(yf_sym)
    df = ticker.history(interval=interval, start=start_str, end=end_str, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No data returned for {yf_sym} {interval} "
            f"({start_str} → {end_str}). "
            "Check symbol, date range, and internet connection."
        )

    # Normalise columns
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Cache to disk
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.info(f"Cached {len(df)} rows → {cache_file.name}")

    return df


def load_multi_timeframe(
    symbol: str,
    start: str,
    end: str | None = None,
    currency: str = "USD",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load 1H and 15m data (the two timeframes used by the Analyst Agent).
    Also loads 1D for trend context.

    Returns:
        {"1h": df_1h, "15m": df_15m, "1d": df_1d}
    """
    # yfinance uses "1h" not "1H"
    intervals = {"1h": "1h", "15m": "15m", "1d": "1d"}
    result = {}
    for key, interval in intervals.items():
        df = load_ohlcv(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            currency=currency,
            use_cache=use_cache,
        )
        result[key] = df
    return result


def default_backtest_range() -> tuple[str, str]:
    """Return default 12-month backtest window (trading period only)."""
    end = datetime.utcnow()
    start = end - timedelta(days=365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def default_daily_warmup_start(trading_start: str, warmup_days: int = 250) -> str:
    """
    Return a daily data start date that provides enough history to warm up EMA-200.

    The daily data loads an extra `warmup_days` of history before the trading
    period begins, so that trend_direction() has a valid EMA-200 from bar 1
    of the trading window.

    Args:
        trading_start: The first date of the trading window (YYYY-MM-DD)
        warmup_days:   Extra days of daily history to prepend (default 250 ≈ 1yr)

    Returns:
        Earlier start date string in YYYY-MM-DD format
    """
    start_dt = datetime.strptime(trading_start[:10], "%Y-%m-%d")
    warmup_start = start_dt - timedelta(days=warmup_days)
    return warmup_start.strftime("%Y-%m-%d")
