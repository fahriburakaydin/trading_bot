"""
Unit tests for core/broker modules.

These tests use mocks — no IB Gateway required.
Integration tests (tests/integration/test_broker.py) require a live paper account.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.broker.client import IBKRClient
from core.broker.market_data import LiveQuote, build_contract
from core.broker.orders import check_slippage as orders_check_slippage
from core.broker.portfolio import (
    AccountSummary,
    PositionSummary,
    count_open_positions,
    get_position,
    get_positions,
)


# ── IBKRClient tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_with_retry_success_on_first_attempt():
    """Connect succeeds on the first attempt — no retry."""
    client = IBKRClient(host="127.0.0.1", port=7497, client_id=99)
    client.ib = MagicMock()
    client.ib.connectAsync = AsyncMock()
    client.ib.isConnected = MagicMock(return_value=True)

    with patch.object(client._notifier, "send", new=AsyncMock()):
        await client.connect_with_retry(max_attempts=3, backoff_base=1)

    assert client.connected is True
    client.ib.connectAsync.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_with_retry_succeeds_on_second_attempt():
    """First connection fails; second succeeds."""
    client = IBKRClient(host="127.0.0.1", port=7497, client_id=99)
    call_count = 0

    async def fake_connect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionRefusedError("port closed")

    client.ib = MagicMock()
    client.ib.connectAsync = fake_connect
    client.ib.isConnected = MagicMock(return_value=True)

    with patch.object(client._notifier, "send", new=AsyncMock()):
        with patch("asyncio.sleep", new=AsyncMock()):
            await client.connect_with_retry(max_attempts=3, backoff_base=1)

    assert client.connected is True
    assert call_count == 2


@pytest.mark.asyncio
async def test_connect_with_retry_raises_after_exhausting_attempts():
    """All attempts fail — raises ConnectionError and sends Telegram alert."""
    client = IBKRClient(host="127.0.0.1", port=7497, client_id=99)
    client.ib = MagicMock()
    client.ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("port closed"))
    client.ib.isConnected = MagicMock(return_value=False)

    alert_sent = []
    with patch.object(client._notifier, "send", new=AsyncMock(side_effect=lambda msg: alert_sent.append(msg))):
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ConnectionError):
                await client.connect_with_retry(max_attempts=3, backoff_base=1)

    assert client.connected is False
    assert len(alert_sent) == 1   # one alert for exhausted attempts
    assert "unreachable" in alert_sent[0].lower()


@pytest.mark.asyncio
async def test_disconnect_cancels_heartbeat():
    """Disconnect cancels the heartbeat task."""
    client = IBKRClient()
    client.ib = MagicMock()
    client.ib.isConnected = MagicMock(return_value=False)

    # Create a real task that runs forever so we can check it gets cancelled
    async def forever():
        await asyncio.sleep(9999)

    task = asyncio.create_task(forever())
    client._heartbeat_task = task   # assign before disconnect

    await client.disconnect()

    # disconnect() sets _heartbeat_task to None after cancellation; use saved ref
    assert task.cancelled() or task.done()


# ── LiveQuote tests ───────────────────────────────────────────────────────────


def test_live_quote_update_sets_mid():
    """update() populates bid/ask/last and computes mid."""
    quote = LiveQuote()
    ticker = MagicMock()
    ticker.bid = 95_000.0
    ticker.ask = 95_010.0
    ticker.last = 95_005.0

    quote.update(ticker)

    assert quote.bid == 95_000.0
    assert quote.ask == 95_010.0
    assert quote.last == 95_005.0
    assert quote.mid == pytest.approx(95_005.0, abs=0.01)


def test_live_quote_price_prefers_mid():
    """price returns mid when both bid and ask are set."""
    quote = LiveQuote()
    quote.bid = 100.0
    quote.ask = 102.0
    quote.mid = 101.0
    quote.last = 99.0

    assert quote.price == 101.0


def test_live_quote_price_falls_back_to_last():
    """price returns last when mid is not set."""
    quote = LiveQuote()
    quote.last = 99.5

    assert quote.price == 99.5


def test_live_quote_price_none_when_empty():
    """price returns None when no data received yet."""
    quote = LiveQuote()
    assert quote.price is None


def test_live_quote_ignores_zero_bid():
    """Zero or negative values are ignored (IBKR sends 0 for missing fields)."""
    quote = LiveQuote()
    ticker = MagicMock()
    ticker.bid = 0.0
    ticker.ask = 95_010.0
    ticker.last = None

    quote.update(ticker)

    assert quote.bid is None   # 0 ignored
    assert quote.ask == 95_010.0


# ── Slippage check tests ──────────────────────────────────────────────────────


def test_slippage_acceptable():
    """Slippage within an explicit 0.5% threshold."""
    acceptable, pct = orders_check_slippage(100.0, 100.3, max_slippage_pct=0.005)  # 0.3%
    assert acceptable is True
    assert pct == pytest.approx(0.003, rel=0.01)


def test_slippage_exceeded():
    """Slippage beyond threshold returns False."""
    acceptable, pct = orders_check_slippage(100.0, 101.0, max_slippage_pct=0.005)  # 1.0%
    assert acceptable is False
    assert pct == pytest.approx(0.01, rel=0.01)


def test_slippage_symmetric():
    """Works for both long (higher fill) and short (lower fill)."""
    ok_long, _ = orders_check_slippage(100.0, 100.2, max_slippage_pct=0.005)
    ok_short, _ = orders_check_slippage(100.0, 99.8, max_slippage_pct=0.005)
    assert ok_long is True
    assert ok_short is True


# ── Portfolio helper tests ────────────────────────────────────────────────────


def _make_portfolio_item(symbol: str, qty: float, avg_cost: float, mkt_price: float):
    item = MagicMock()
    item.contract.symbol = symbol
    item.position = qty
    item.averageCost = avg_cost
    item.marketPrice = mkt_price
    item.unrealizedPNL = (mkt_price - avg_cost) * qty
    item.realizedPNL = 0.0
    return item


def test_get_positions_returns_all():
    """get_positions maps portfolio items to PositionSummary objects."""
    ib = MagicMock()
    ib.portfolio.return_value = [
        _make_portfolio_item("BTC", 0.5, 90_000, 95_000),
        _make_portfolio_item("ETH", -1.0, 3_000, 2_900),
    ]

    positions = get_positions(ib)

    assert len(positions) == 2
    assert positions[0].symbol == "BTC"
    assert positions[0].is_long is True
    assert positions[1].symbol == "ETH"
    assert positions[1].is_short is True


def test_get_position_by_symbol():
    """get_position finds a specific symbol case-insensitively."""
    ib = MagicMock()
    ib.portfolio.return_value = [
        _make_portfolio_item("BTC", 0.5, 90_000, 95_000),
    ]

    result = get_position(ib, "btc")
    assert result is not None
    assert result.symbol == "BTC"


def test_get_position_returns_none_for_missing():
    """Returns None when symbol is not in portfolio."""
    ib = MagicMock()
    ib.portfolio.return_value = []

    result = get_position(ib, "BTC")
    assert result is None


def test_count_open_positions():
    """count_open_positions ignores zero-quantity items."""
    ib = MagicMock()
    ib.portfolio.return_value = [
        _make_portfolio_item("BTC", 0.5, 90_000, 95_000),
        _make_portfolio_item("ETH", 0.0, 3_000, 2_900),   # closed position
    ]

    assert count_open_positions(ib) == 1


def test_position_summary_notional():
    """notional = abs(qty) × market_price."""
    pos = PositionSummary("BTC", 0.5, 90_000, 95_000, 2500.0, 0.0)
    assert pos.notional == pytest.approx(47_500.0)


def test_position_summary_short_notional():
    """Short position (negative qty) also returns positive notional."""
    pos = PositionSummary("ETH", -2.0, 3_000, 2_900, 200.0, 0.0)
    assert pos.notional == pytest.approx(5_800.0)
    assert pos.is_short is True
