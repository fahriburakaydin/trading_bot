"""
Live portfolio and position tracking.

Queries IBKR for current account values, open positions,
and provides helpers used by the Trader Agent and guardrails.

All functions accept a connected IB instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ib_insync import IB, PortfolioItem
from loguru import logger


@dataclass
class PositionSummary:
    """Simplified position snapshot."""

    symbol: str
    quantity: float         # positive = long, negative = short
    avg_cost: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.market_price


@dataclass
class AccountSummary:
    """Key account metrics from IBKR."""

    net_liquidation: float = 0.0
    total_cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    available_funds: float = 0.0
    buying_power: float = 0.0
    currency: str = "USD"


# ── Position queries ──────────────────────────────────────────────────────────


def get_positions(ib: IB) -> list[PositionSummary]:
    """
    Return all open positions from the IBKR account.

    Uses ib_insync's cached portfolio data (updated on each event loop tick).
    """
    positions = []
    for item in ib.portfolio():
        positions.append(
            PositionSummary(
                symbol=item.contract.symbol,
                quantity=item.position,
                avg_cost=item.averageCost,
                market_price=item.marketPrice,
                unrealized_pnl=item.unrealizedPNL,
                realized_pnl=item.realizedPNL,
            )
        )
    return positions


def get_position(ib: IB, symbol: str) -> PositionSummary | None:
    """Return the position for a specific symbol, or None if not held."""
    for pos in get_positions(ib):
        if pos.symbol.upper() == symbol.upper():
            return pos
    return None


def count_open_positions(ib: IB) -> int:
    """Return the number of currently open positions."""
    return sum(1 for p in get_positions(ib) if p.quantity != 0)


# ── Account queries ───────────────────────────────────────────────────────────


async def get_account_summary(ib: IB) -> AccountSummary:
    """
    Fetch key account values from IBKR.

    Returns an AccountSummary with net liquidation, cash, PnL, and buying power.
    """
    tags = {
        "NetLiquidation",
        "TotalCashValue",
        "UnrealizedPnL",
        "RealizedPnL",
        "AvailableFunds",
        "BuyingPower",
    }
    # reqAccountSummaryAsync() is a void trigger; accountSummaryAsync() returns the data.
    # Take the first value per tag — avoids duplicates in multi-currency accounts.
    all_values = await ib.accountSummaryAsync()
    seen: set[str] = set()
    account_values = []
    for av in all_values:
        if av.tag in tags and av.tag not in seen:
            account_values.append(av)
            seen.add(av.tag)

    summary = AccountSummary()
    for av in account_values:
        try:
            val = float(av.value)
        except (ValueError, TypeError):
            continue

        match av.tag:
            case "NetLiquidation":
                summary.net_liquidation = val
                summary.currency = av.currency
            case "TotalCashValue":
                summary.total_cash = val
            case "UnrealizedPnL":
                summary.unrealized_pnl = val
            case "RealizedPnL":
                summary.realized_pnl = val
            case "AvailableFunds":
                summary.available_funds = val
            case "BuyingPower":
                summary.buying_power = val

    logger.debug(
        f"Account summary: NLV={summary.net_liquidation:,.2f} "
        f"Cash={summary.total_cash:,.2f} "
        f"UPnL={summary.unrealized_pnl:,.2f}"
    )
    return summary


async def get_daily_pnl(ib: IB) -> float:
    """
    Return today's realized + unrealized P&L.

    Note: IBKR resets daily PnL at 17:00 ET (after US equity close).
    """
    summary = await get_account_summary(ib)
    return summary.realized_pnl + summary.unrealized_pnl
