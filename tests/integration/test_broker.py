"""
Integration tests for IBKR broker modules.

REQUIRES: IB Gateway paper account running on localhost:7497.
Skip automatically when IB Gateway is not available.

Run explicitly with:
    pytest tests/integration/test_broker.py -v -s
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from core.broker.client import IBKRClient
from core.broker.market_data import build_contract, fetch_ohlcv, subscribe_quote
from core.broker.orders import cancel_bracket, check_slippage, place_bracket_order
from core.broker.portfolio import get_account_summary, get_positions


pytestmark = pytest.mark.integration


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def ibkr():
    """
    Yield a connected IBKRClient for the duration of a test.
    Skips automatically if IB Gateway is unreachable.
    """
    client = IBKRClient(host="127.0.0.1", port=7497, client_id=10)
    try:
        await client.connect_with_retry(max_attempts=1, backoff_base=1)
    except ConnectionError:
        pytest.skip("IB Gateway not available on localhost:7497")

    yield client

    await client.disconnect()


# ── Connection ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_to_paper_account(ibkr: IBKRClient):
    """Connect to IB Gateway paper account successfully."""
    assert ibkr.connected is True
    assert ibkr.ib.isConnected() is True


# ── Market data ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_btc_1h_bars(ibkr: IBKRClient):
    """Fetch BTC 1H bars — expect ≥20 bars for a 1-day lookback."""
    contract = build_contract("BTC", "CRYPTO", "USD", "PAXOS")
    df = await fetch_ohlcv(ibkr.ib, contract, interval="1h", lookback_days=2)

    if df.empty:
        pytest.skip("No BTC market data — subscribe to PAXOS CRYPTO data on the paper account")

    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert df.index.tz is not None, "Index should be timezone-aware"
    assert len(df) >= 20, f"Expected ≥20 bars, got {len(df)}"


@pytest.mark.asyncio
async def test_fetch_btc_15m_bars(ibkr: IBKRClient):
    """Fetch BTC 15m bars — expect ≥ 60 bars for a 1-day lookback."""
    contract = build_contract("BTC", "CRYPTO", "USD", "PAXOS")
    df = await fetch_ohlcv(ibkr.ib, contract, interval="15m", lookback_days=1)

    if df.empty:
        pytest.skip("No BTC market data — subscribe to PAXOS CRYPTO data on the paper account")

    assert len(df) >= 60, f"Expected ≥60 15m bars, got {len(df)}"


@pytest.mark.asyncio
async def test_live_quote_subscribes(ibkr: IBKRClient):
    """Subscribe to live BTC quote and receive a price within 5 seconds."""
    contract = build_contract("BTC", "CRYPTO", "USD", "PAXOS")
    quote = subscribe_quote(ibkr.ib, contract)

    # Wait for tick data to arrive
    for _ in range(10):
        await asyncio.sleep(0.5)
        if quote.price is not None:
            break

    ibkr.ib.cancelMktData(contract)

    if quote.price is None:
        pytest.skip(
            "No live quote received — subscribe to PAXOS CRYPTO market data on the paper account"
        )

    assert quote.price > 1000, f"BTC price looks wrong: {quote.price}"


# ── Account / Portfolio ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_account_summary(ibkr: IBKRClient):
    """Account summary returns valid net liquidation value."""
    summary = await get_account_summary(ibkr.ib)

    assert summary.net_liquidation > 0, (
        f"Net liquidation {summary.net_liquidation} — paper account should have positive NLV"
    )
    # Currency depends on the paper account base currency (USD, SEK, EUR, etc.)
    assert summary.currency != "", "Currency must be set"


@pytest.mark.asyncio
async def test_get_positions_returns_list(ibkr: IBKRClient):
    """get_positions returns a list (may be empty if no open positions)."""
    positions = get_positions(ibkr.ib)
    assert isinstance(positions, list)


# ── Order placement ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_place_and_cancel_bracket_order(ibkr: IBKRClient):
    """
    Place a BTC bracket order far from market (will not fill) and cancel it.

    Uses a very low limit price (~50% below market) to avoid accidental fills.
    """
    contract = build_contract("BTC", "CRYPTO", "USD", "PAXOS")

    # Fetch current price
    df = await fetch_ohlcv(ibkr.ib, contract, interval="1h", lookback_days=1)
    if df.empty:
        pytest.skip("Cannot determine BTC price for safe order test")

    current_price = float(df["close"].iloc[-1])
    # Far-below-market limit so order will not fill
    entry_price = round(current_price * 0.50, 0)
    stop_loss = round(entry_price * 0.95, 0)
    take_profit = round(entry_price * 1.10, 0)

    bracket = await place_bracket_order(
        ib=ibkr.ib,
        contract=contract,
        action="BUY",
        quantity=0.0001,   # minimum BTC size
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    assert bracket.entry_order_id > 0
    assert not bracket.is_filled   # far from market, must not fill

    # Cancel immediately
    await cancel_bracket(ibkr.ib, bracket)

    # Give IB a moment to process the cancellation
    await asyncio.sleep(2)

    # Verify cancelled
    open_orders = ibkr.ib.openOrders()
    open_ids = {o.orderId for o in open_orders}
    assert bracket.entry_order_id not in open_ids, (
        "Order should have been cancelled"
    )
