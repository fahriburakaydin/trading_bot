"""
IBKR order management.

Wraps ib_insync bracket order creation, cancellation, and modification.
All public functions are async and accept a connected IB instance.

Order strategy (per architecture plan):
- Entry: LIMIT order at signal price
- Stop Loss: STOP LIMIT (stop triggers at SL price, limit = SL ± sl_limit_offset_pct)
- Take Profit: LIMIT order at target price
All three are linked as a bracket order so IBKR manages the exit legs automatically.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ib_insync import IB, Contract, LimitOrder, Order, StopLimitOrder, Trade
from loguru import logger

from core.config import cfg


@dataclass
class BracketResult:
    """Result of a bracket order placement."""

    entry_trade: Trade
    sl_trade: Trade
    tp_trade: Trade

    @property
    def entry_order_id(self) -> int:
        return self.entry_trade.order.orderId

    @property
    def is_filled(self) -> bool:
        return self.entry_trade.orderStatus.status == "Filled"


# ── Bracket order placement ───────────────────────────────────────────────────


async def place_bracket_order(
    ib: IB,
    contract: Contract,
    action: str,        # "BUY" or "SELL"
    quantity: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    sl_limit_offset_pct: float | None = None,
) -> BracketResult:
    """
    Place a bracket order (entry limit + SL stop-limit + TP limit).

    Args:
        ib:                   Connected IB instance
        contract:             The asset contract
        action:               "BUY" for LONG, "SELL" for SHORT
        quantity:             Number of units
        entry_price:          Limit price for entry
        stop_loss:            Stop trigger price for the SL leg
        take_profit:          Limit price for the TP leg
        sl_limit_offset_pct:  SL limit leg offset from stop (default from config)

    Returns:
        BracketResult containing all three Trade objects
    """
    offset_pct = sl_limit_offset_pct or cfg.trading.orders.sl_limit_offset_pct
    exit_action = "SELL" if action == "BUY" else "BUY"
    qty = _round_quantity(quantity)

    # SL limit price: slightly beyond stop to reduce non-fill risk
    if action == "BUY":
        # Long: stop below entry; limit is below stop (allows some slippage)
        sl_limit_price = round(stop_loss * (1 - offset_pct), 2)
    else:
        # Short: stop above entry; limit is above stop
        sl_limit_price = round(stop_loss * (1 + offset_pct), 2)

    # Build bracket legs
    entry_order = LimitOrder(action, qty, entry_price, tif="GTC", transmit=False)
    sl_order = StopLimitOrder(
        exit_action, qty, stop_loss, sl_limit_price, tif="GTC", transmit=False
    )
    tp_order = LimitOrder(exit_action, qty, take_profit, tif="GTC", transmit=True)

    # Link parent/child relationships for bracket
    entry_id = ib.client.getReqId()
    entry_order.orderId = entry_id
    sl_order.parentId = entry_id
    tp_order.parentId = entry_id
    sl_order.ocaGroup = f"OCA_{entry_id}"
    tp_order.ocaGroup = f"OCA_{entry_id}"
    sl_order.ocaType = 2  # reduce quantity — cancel remaining when one fills
    tp_order.ocaType = 2

    # Place all three (transmit=True on TP triggers submission of all three)
    entry_trade = ib.placeOrder(contract, entry_order)
    sl_trade = ib.placeOrder(contract, sl_order)
    tp_trade = ib.placeOrder(contract, tp_order)

    await asyncio.sleep(0)  # yield to let ib_insync process

    logger.info(
        f"Bracket order placed: {action} {qty} {contract.symbol} "
        f"@ entry={entry_price} SL={stop_loss} TP={take_profit} "
        f"| orderId={entry_id}"
    )

    return BracketResult(
        entry_trade=entry_trade,
        sl_trade=sl_trade,
        tp_trade=tp_trade,
    )


# ── Order cancellation ────────────────────────────────────────────────────────


async def cancel_order(ib: IB, order: Order) -> None:
    """Cancel a single order."""
    ib.cancelOrder(order)
    await asyncio.sleep(0)
    logger.info(f"Cancelled order {order.orderId}")


async def cancel_bracket(ib: IB, bracket: BracketResult) -> None:
    """Cancel all legs of a bracket order."""
    for trade in (bracket.entry_trade, bracket.sl_trade, bracket.tp_trade):
        if trade.orderStatus.status not in ("Filled", "Cancelled", "Inactive"):
            ib.cancelOrder(trade.order)
    await asyncio.sleep(0)
    logger.info(f"Cancelled bracket {bracket.entry_order_id}")


# ── Market close ──────────────────────────────────────────────────────────────


async def close_at_market(
    ib: IB,
    contract: Contract,
    action: str,    # "SELL" to close LONG, "BUY" to close SHORT
    quantity: float,
) -> Trade:
    """
    Close a position with a market order (emergency use only).

    Used when: SL bracket leg is not filled within timeout, or Trader Agent override.
    """
    qty = _round_quantity(quantity)
    market_order = ib.marketOrder(action, qty)
    trade = ib.placeOrder(contract, market_order)
    await asyncio.sleep(0)
    logger.warning(f"Market close: {action} {qty} {contract.symbol}")
    return trade


# ── Slippage check ────────────────────────────────────────────────────────────


def check_slippage(
    intended_price: float,
    fill_price: float,
    max_slippage_pct: float | None = None,
) -> tuple[bool, float]:
    """
    Check if actual fill price is within the acceptable slippage threshold.

    Args:
        intended_price:   The price the order was intended to fill at
        fill_price:       The actual fill price
        max_slippage_pct: Override threshold; defaults to config value

    Returns:
        (acceptable: bool, slippage_pct: float)
    """
    slippage_pct = abs(fill_price - intended_price) / intended_price
    max_slip = max_slippage_pct if max_slippage_pct is not None else cfg.trading.orders.max_slippage_pct
    acceptable = slippage_pct <= max_slip
    if not acceptable:
        logger.warning(
            f"Slippage exceeded: {slippage_pct:.3%} > {max_slip:.3%} "
            f"(intended={intended_price}, fill={fill_price})"
        )
    return acceptable, round(slippage_pct, 6)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _round_quantity(quantity: float) -> float:
    """Round quantity to asset precision from config."""
    precision = cfg.asset.quantity_precision
    return round(quantity, precision)


