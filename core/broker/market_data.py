"""
Live market data from IBKR.

Provides:
- Historical OHLCV bars (1H, 15m, 1D) via reqHistoricalDataAsync
- Real-time streaming quotes (bid/ask/last) via reqMktData
- Thin wrappers that return pandas DataFrames matching the backtesting format

All functions accept an IBKRClient and use its .ib attribute.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
from ib_insync import Contract, IB, Stock, Crypto, Forex
from loguru import logger

from core.config import cfg


# ── Contract helpers ──────────────────────────────────────────────────────────


def build_contract(
    symbol: str | None = None,
    asset_type: str | None = None,
    currency: str | None = None,
    exchange: str | None = None,
) -> Contract:
    """
    Build an ib_insync Contract from config or explicit arguments.

    Supports CRYPTO (e.g. BTC/ETH on PAXOS), STK (e.g. AAPL on SMART),
    and FOREX (e.g. EURUSD on IDEALPRO).
    """
    sym = (symbol or cfg.asset.symbol).upper()
    atype = (asset_type or cfg.asset.type).upper()
    cur = (currency or cfg.asset.currency).upper()
    exch = (exchange or cfg.asset.exchange).upper()

    if atype == "CRYPTO":
        return Crypto(sym, exch, cur)
    elif atype == "STK":
        return Stock(sym, exch, cur)
    elif atype == "FOREX":
        # ib_insync Forex takes the 6-char pair e.g. "EURUSD"
        return Forex(sym)
    else:
        raise ValueError(f"Unsupported asset type: {atype}")


# ── Historical OHLCV ──────────────────────────────────────────────────────────


_IB_DURATION = {
    "1 min": "1 D",
    "5 mins": "2 D",
    "15 mins": "5 D",
    "30 mins": "7 D",
    "1 hour": "30 D",
    "4 hours": "60 D",
    "1 day": "365 D",
}

_INTERVAL_TO_IB = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
}


async def fetch_ohlcv(
    ib: IB,
    contract: Contract | None = None,
    interval: str = "1h",
    lookback_days: int | None = None,
    end_datetime: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV bars from IBKR.

    Args:
        ib:            Connected IB instance
        contract:      Contract to fetch; builds from config if None
        interval:      Bar size: "1m", "15m", "1h", "1d"
        lookback_days: How many calendar days to fetch (default varies by interval)
        end_datetime:  End of the bar range (defaults to now)

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: UTC DatetimeIndex (matching backtesting format)
    """
    contract = contract or build_contract()
    ib_bar_size = _INTERVAL_TO_IB.get(interval)
    if ib_bar_size is None:
        raise ValueError(f"Unsupported interval: {interval}. Use: {list(_INTERVAL_TO_IB)}")

    if lookback_days is None:
        # Sensible defaults per interval
        lookback_days = {"1m": 1, "5m": 2, "15m": 5, "30m": 7, "1h": 30, "4h": 60, "1d": 365}[interval]

    duration_str = f"{lookback_days} D"
    end_dt = end_datetime or datetime.now(timezone.utc)
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S UTC")

    logger.debug(
        f"Fetching {contract.symbol} {interval} bars | last {lookback_days}d | end={end_str}"
    )

    bars = await ib.reqHistoricalDataAsync(
        contract=contract,
        endDateTime=end_str,
        durationStr=duration_str,
        barSizeSetting=ib_bar_size,
        whatToShow="MIDPOINT" if cfg.asset.type in ("CRYPTO", "FOREX") else "TRADES",
        useRTH=False,
        formatDate=2,
    )

    if not bars:
        logger.warning(f"No bars returned for {contract.symbol} {interval}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        },
        index=pd.to_datetime([b.date for b in bars], utc=True),
    )
    df.index.name = "datetime"
    df = df.sort_index()
    logger.debug(f"Fetched {len(df)} {interval} bars for {contract.symbol}")
    return df


async def fetch_multi_timeframe(
    ib: IB,
    contract: Contract | None = None,
    lookback_days_1h: int = 30,
    lookback_days_15m: int = 5,
    lookback_days_1d: int = 365,
) -> dict[str, pd.DataFrame]:
    """
    Fetch 1H, 15m, and 1D bars in parallel.

    Returns:
        {"1h": df_1h, "15m": df_15m, "1d": df_1d}
    """
    contract = contract or build_contract()
    df_1h, df_15m, df_1d = await asyncio.gather(
        fetch_ohlcv(ib, contract, "1h", lookback_days_1h),
        fetch_ohlcv(ib, contract, "15m", lookback_days_15m),
        fetch_ohlcv(ib, contract, "1d", lookback_days_1d),
    )
    return {"1h": df_1h, "15m": df_15m, "1d": df_1d}


# ── Real-time quote streaming ─────────────────────────────────────────────────


class LiveQuote:
    """Holds the latest streaming quote for a contract."""

    def __init__(self) -> None:
        self.bid: float | None = None
        self.ask: float | None = None
        self.last: float | None = None
        self.mid: float | None = None

    def update(self, ticker) -> None:
        """Update fields from an ib_insync Ticker object."""
        import math

        def _is_valid(val) -> bool:
            """Check if a ticker value is a real, positive number."""
            if val is None:
                return False
            try:
                return not math.isnan(val) and val > 0
            except TypeError:
                return False

        if _is_valid(ticker.bid):
            self.bid = ticker.bid
        if _is_valid(ticker.ask):
            self.ask = ticker.ask
        if _is_valid(ticker.last):
            self.last = ticker.last
        if self.bid and self.ask:
            self.mid = (self.bid + self.ask) / 2

    @property
    def price(self) -> float | None:
        """Best available price: mid → last → None."""
        return self.mid or self.last

    def __repr__(self) -> str:
        return f"<LiveQuote bid={self.bid} ask={self.ask} last={self.last}>"


def subscribe_quote(ib: IB, contract: Contract | None = None) -> LiveQuote:
    """
    Subscribe to real-time streaming quote for a contract.

    Returns a LiveQuote object that updates automatically as ticks arrive.
    Call ib.cancelMktData(contract) to unsubscribe.

    Args:
        ib:       Connected IB instance
        contract: Contract to stream; builds from config if None

    Returns:
        LiveQuote instance (will be populated as ticks arrive)
    """
    contract = contract or build_contract()
    quote = LiveQuote()

    ticker = ib.reqMktData(contract, "", False, False)

    ticker.updateEvent += lambda t: quote.update(t)

    logger.info(f"Subscribed to live quote for {contract.symbol}")
    return quote


def unsubscribe_quote(ib: IB, contract: Contract | None = None) -> None:
    """Cancel market data subscription for a contract."""
    contract = contract or build_contract()
    ib.cancelMktData(contract)
    logger.debug(f"Unsubscribed from live quote for {contract.symbol}")


# ── asyncio import guard ──────────────────────────────────────────────────────
