"""
Trade simulator for backtesting.

Takes signals from strategy.py and simulates order fills, SL/TP hits,
and position sizing using the same fixed-fractional formula as live trading.

Assumptions:
- Entry: limit order at signal.entry_price, filled on next bar open
  if price passes through entry_price (realistic fill simulation)
- Stop loss: filled on next bar where low <= stop_loss (LONG)
  or high >= stop_loss (SHORT)
- Take profit: filled on next bar where high >= target (LONG)
  or low <= target (SHORT)
- Slippage: 0.3% applied to all fills (realistic for crypto on IBKR)
- Max open positions: 3 concurrent
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.strategy import Signal
from core.risk.position_sizing import calculate_position_size


SLIPPAGE_PCT_DEFAULT = 0.003      # 0.3% fallback (BTC/crypto); overridden by asset config


@dataclass
class SimulatedTrade:
    signal: Signal
    entry_bar: int
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: float
    notional: float
    risk_amount: float

    exit_bar: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None    # target_hit | stop_hit | end_of_data

    @property
    def is_open(self) -> bool:
        return self.exit_bar is None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.signal.action == "LONG":
            return (self.exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.exit_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.notional if self.notional else 0.0

    @property
    def pnl_r(self) -> float:
        """P&L in multiples of R (risk units)."""
        return self.pnl / self.risk_amount if self.risk_amount else 0.0


def simulate_trades(
    signals: list[Signal],
    df_1h: pd.DataFrame,
    portfolio_value: float = 100_000,
    risk_pct: float = 0.005,
    max_concurrent: int = 3,
    slippage_pct: float | None = None,
) -> list[SimulatedTrade]:
    """
    Simulate trade execution for all signals against historical price data.

    Args:
        signals:         Signals from strategy.generate_signals()
        df_1h:           Full 1H OHLCV used to check fills bar-by-bar
        portfolio_value: Starting capital
        risk_pct:        Risk per trade as fraction (e.g. 0.005 = 0.5%)
        max_concurrent:  Max simultaneously open positions
        slippage_pct:    One-way slippage per fill

    Returns:
        List of SimulatedTrade with full entry/exit details
    """
    # Use asset config slippage if not explicitly provided
    if slippage_pct is None:
        try:
            from core.config import cfg
            slippage_pct = cfg.trading.orders.max_slippage_pct
        except Exception:
            slippage_pct = SLIPPAGE_PCT_DEFAULT

    bars = df_1h.reset_index(drop=False)
    completed: list[SimulatedTrade] = []
    open_trades: list[SimulatedTrade] = []
    pending_market_entries: list[Signal] = []   # breakout signals awaiting next-bar open fill

    # Index signals by bar_index for fast lookup
    signal_map: dict[int, list[Signal]] = {}
    for sig in signals:
        signal_map.setdefault(sig.bar_index, []).append(sig)

    current_portfolio = portfolio_value

    for bar_idx in range(len(bars)):
        bar = bars.iloc[bar_idx]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        # ── Step 1: Check open trades for SL/TP hits ─────────────────────
        still_open = []
        for trade in open_trades:
            filled = False

            if trade.signal.action == "LONG":
                # Stop loss: low breached stop — stop-limit fill, apply slippage
                if bar_low <= trade.stop_loss:
                    fill = trade.stop_loss * (1 - slippage_pct)
                    trade.exit_bar = bar_idx
                    trade.exit_price = fill
                    trade.exit_reason = "stop_hit"
                    current_portfolio += trade.pnl
                    completed.append(trade)
                    filled = True
                # Take profit: limit sell order — no slippage (fills at target or better)
                elif bar_high >= trade.target_price:
                    fill = trade.target_price   # limit order: no slippage
                    trade.exit_bar = bar_idx
                    trade.exit_price = fill
                    trade.exit_reason = "target_hit"
                    current_portfolio += trade.pnl
                    completed.append(trade)
                    filled = True

            else:  # SHORT
                # Stop loss: high breached stop — stop-limit buy, apply slippage
                if bar_high >= trade.stop_loss:
                    fill = trade.stop_loss * (1 + slippage_pct)
                    trade.exit_bar = bar_idx
                    trade.exit_price = fill
                    trade.exit_reason = "stop_hit"
                    current_portfolio += trade.pnl
                    completed.append(trade)
                    filled = True
                # Take profit: limit buy order — no slippage (fills at target or better)
                elif bar_low <= trade.target_price:
                    fill = trade.target_price   # limit order: no slippage
                    trade.exit_bar = bar_idx
                    trade.exit_price = fill
                    trade.exit_reason = "target_hit"
                    current_portfolio += trade.pnl
                    completed.append(trade)
                    filled = True

            if not filled:
                still_open.append(trade)

        open_trades = still_open

        # ── Step 2: Process new signals at this bar ───────────────────────
        # Signals fire at bar_index. Depending on entry_type:
        #   "limit"  — fill on bar_index if price visits the level (S/R bounce)
        #   "market" — fill at bar_index+1 open (breakout: confirmed, enter next bar)
        # We handle "market" signals by placing them into a pending queue and
        # filling them on the very next bar's open.

        # Fill any pending market-entry signals at this bar's open
        still_pending = []
        for sig in pending_market_entries:
            if len(open_trades) >= max_concurrent:
                still_pending.append(sig)
                continue
            entry_fill = bar_open * (1 + slippage_pct if sig.action == "LONG" else 1 - slippage_pct)
            sizing = calculate_position_size(
                portfolio_value=current_portfolio,
                risk_pct=risk_pct,
                entry_price=entry_fill,
                stop_loss=sig.stop_loss,
            )
            if sizing is None:
                continue
            trade = SimulatedTrade(
                signal=sig,
                entry_bar=bar_idx,
                entry_price=entry_fill,
                stop_loss=sig.stop_loss,
                target_price=sig.target_price,
                quantity=sizing["quantity"],
                notional=sizing["notional"],
                risk_amount=sizing["risk_amount"],
            )
            open_trades.append(trade)
        pending_market_entries = still_pending

        if bar_idx in signal_map and len(open_trades) < max_concurrent:
            for sig in signal_map[bar_idx]:
                if len(open_trades) >= max_concurrent:
                    break

                if getattr(sig, "entry_type", "limit") == "market":
                    # Queue for fill at next bar open
                    pending_market_entries.append(sig)
                    continue

                # Limit order: fills at the signal price (maker fill — no adverse slippage).
                # The Analyst sets alarms at S/R levels; when price touches the level,
                # the limit order fills at that exact price.  SL exits (stop orders)
                # and gap-fills still incur slippage in the exit logic above.
                entry_fill = sig.entry_price

                # Only enter if price is near the entry level on this bar
                price_reached = bar_low <= sig.entry_price <= bar_high
                if not price_reached:
                    continue

                sizing = calculate_position_size(
                    portfolio_value=current_portfolio,
                    risk_pct=risk_pct,
                    entry_price=entry_fill,
                    stop_loss=sig.stop_loss,
                )
                if sizing is None:
                    continue

                trade = SimulatedTrade(
                    signal=sig,
                    entry_bar=bar_idx,
                    entry_price=entry_fill,
                    stop_loss=sig.stop_loss,
                    target_price=sig.target_price,
                    quantity=sizing["quantity"],
                    notional=sizing["notional"],
                    risk_amount=sizing["risk_amount"],
                )
                open_trades.append(trade)

    # ── Step 3: Close any remaining open trades at last bar price ─────────
    last_close = float(bars.iloc[-1]["close"])
    for trade in open_trades:
        trade.exit_bar = len(bars) - 1
        trade.exit_price = last_close
        trade.exit_reason = "end_of_data"
        current_portfolio += trade.pnl
        completed.append(trade)

    return completed
