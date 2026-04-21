"""
Backtest performance metrics.

All 7 metrics used by both the backtester and the Evaluator Agent.

Metrics:
    win_rate        — % of trades that are profitable
    avg_win         — Average P&L of winning trades (USD)
    avg_loss        — Average P&L of losing trades (USD, negative)
    expectancy      — (WR × avg_win) + (LR × avg_loss) per trade
    profit_factor   — gross_profit / gross_loss
    sharpe          — annualised Sharpe ratio (simplified, daily returns)
    max_drawdown    — maximum peak-to-trough equity decline (%)
    calmar          — annualised_return / max_drawdown
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    profit_factor: float
    sharpe: float
    max_drawdown: float          # as fraction, e.g. 0.15 = 15%
    calmar: float
    total_return: float          # as fraction
    annualised_return: float     # as fraction
    days_tested: int

    # Phase 0 success gate
    @property
    def passes_gate(self) -> bool:
        return (
            self.expectancy > 0
            and self.win_rate >= 0.40
            and self.sharpe > 0.5
            and self.max_drawdown < 0.20
        )

    def summary(self) -> str:
        gate = "PASS" if self.passes_gate else "FAIL"
        return (
            f"═══════════════════════════════════════\n"
            f"  BACKTEST RESULTS — Phase 0 Gate: {gate}\n"
            f"═══════════════════════════════════════\n"
            f"  Trades:        {self.total_trades} "
            f"({self.winning_trades}W / {self.losing_trades}L)\n"
            f"  Win Rate:      {self.win_rate:.1%}\n"
            f"  Total P&L:     ${self.total_pnl:,.2f}\n"
            f"  Avg Win:       ${self.avg_win:,.2f}\n"
            f"  Avg Loss:      ${self.avg_loss:,.2f}\n"
            f"  Expectancy:    ${self.expectancy:,.2f} / trade\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Sharpe (SQN):  {self.sharpe:.2f}\n"
            f"  Max Drawdown:  {self.max_drawdown:.1%}\n"
            f"  Calmar:        {self.calmar:.2f}\n"
            f"  Total Return:  {self.total_return:.1%} over {self.days_tested}d\n"
            f"═══════════════════════════════════════\n"
            f"  Gate criteria:\n"
            f"    Expectancy > 0:       {'✓' if self.expectancy > 0 else '✗'}\n"
            f"    Win rate ≥ 40%:       {'✓' if self.win_rate >= 0.40 else '✗'}\n"
            f"    Sharpe (SQN) > 0.5:   {'✓' if self.sharpe > 0.5 else '✗'}\n"
            f"    Max drawdown < 20%:   {'✓' if self.max_drawdown < 0.20 else '✗'}\n"
            f"═══════════════════════════════════════"
        )


def calculate_metrics(
    trades: list,
    initial_capital: float = 100_000,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> BacktestMetrics:
    """
    Calculate all backtest metrics from a list of SimulatedTrade objects.

    Args:
        trades:          List of SimulatedTrade from simulator.simulate_trades()
        initial_capital: Starting portfolio value
        start_date:      Backtest start (for annualisation)
        end_date:        Backtest end

    Returns:
        BacktestMetrics dataclass with all metrics and gate result
    """
    if not trades:
        return BacktestMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
            total_pnl=0, win_rate=0, avg_win=0, avg_loss=0, expectancy=0,
            profit_factor=0, sharpe=0, max_drawdown=0, calmar=0,
            total_return=0, annualised_return=0, days_tested=0,
        )

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    breakevens = [p for p in pnls if p == 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    loss_rate = 1 - win_rate
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # ── Equity curve and drawdown ─────────────────────────────────────────
    # Build daily equity curve
    if start_date and end_date:
        days_tested = max(1, (end_date - start_date).days)
    else:
        days_tested = 180  # default assumption

    equity = initial_capital
    equity_curve = [equity]
    for t in sorted(trades, key=lambda x: x.entry_bar):
        equity += t.pnl
        equity_curve.append(equity)

    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) > 1 else 0.0

    # ── Returns ───────────────────────────────────────────────────────────
    total_return = total_pnl / initial_capital
    annualised_return = (1 + total_return) ** (365 / days_tested) - 1 if days_tested > 0 else 0.0

    # ── Sharpe (R-multiple SQN approach) ─────────────────────────────────
    # Per-portfolio-return Sharpe is misleading for low-frequency, small-risk-
    # per-trade strategies: each trade's PnL (~$300) looks tiny vs $100k capital.
    # Standard alternative: System Quality Number using R-multiples (pnl / risk).
    # Each trade's R-multiple: +2R on win, -1R on loss (at 2.0 min R:R).
    # Sharpe = (mean R / std R) × sqrt(trades_per_year).
    if len(pnls) >= 2 and hasattr(trades[0], "pnl_r"):
        r_multiples = np.array([t.pnl_r for t in trades])
        mean_rm = float(np.mean(r_multiples))
        std_rm = float(np.std(r_multiples, ddof=1))
        trades_per_year = max(len(pnls) / (days_tested / 365), 1)
        sharpe = (mean_rm / std_rm * math.sqrt(trades_per_year)) if std_rm > 0 else 0.0
    elif len(pnls) >= 2:
        # Fallback: per-trade portfolio return
        returns = np.array(pnls) / initial_capital
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        trades_per_year = max(len(pnls) / (days_tested / 365), 1)
        sharpe = (mean_r / std_r * math.sqrt(trades_per_year)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    calmar = annualised_return / max_drawdown if max_drawdown > 0 else float("inf")

    return BacktestMetrics(
        total_trades=len(pnls),
        winning_trades=len(wins),
        losing_trades=len(losses),
        breakeven_trades=len(breakevens),
        total_pnl=round(total_pnl, 2),
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        expectancy=round(expectancy, 2),
        profit_factor=round(profit_factor, 3),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_drawdown, 4),
        calmar=round(calmar, 3),
        total_return=round(total_return, 4),
        annualised_return=round(annualised_return, 4),
        days_tested=days_tested,
    )
