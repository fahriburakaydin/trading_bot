#!/usr/bin/env python3
"""
CLI script to run the Phase 0 backtest.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --asset BTC --start 2024-09-01 --end 2025-03-01
    python scripts/run_backtest.py --risk-pct 0.5 --min-confidence 0.65
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from backtesting.data_loader import load_ohlcv, default_backtest_range, default_daily_warmup_start
from backtesting.metrics import calculate_metrics
from backtesting.simulator import simulate_trades
from backtesting.strategy import generate_signals
from core.config import cfg

import pandas as pd


def run_backtest(
    asset: str,
    start: str,
    end: str,
    risk_pct: float,
    min_confidence: float,
    min_rr: float,
) -> bool:
    """
    Run full backtest pipeline and return True if Phase 0 gate passes.
    """
    logger.info(f"Starting backtest: {asset} | {start} → {end}")
    logger.info(f"Settings: risk={risk_pct:.2%} min_conf={min_confidence} min_rr={min_rr}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading historical data...")
    df_1h = load_ohlcv(asset, "1h", start, end)
    # Load daily with extra warmup so EMA-200 is valid from the first trading bar
    daily_start = default_daily_warmup_start(start, warmup_days=250)
    df_daily = load_ohlcv(asset, "1d", daily_start, end)
    logger.info(f"Loaded {len(df_1h)} 1H bars, {len(df_daily)} daily bars (incl. {daily_start} warmup)")

    # Load 15m data if the active strategy uses it (yfinance limits: max 60 days for 15m)
    df_15m = None
    try:
        from strategies import get_strategy
        entry = get_strategy(asset)
        uses_15m = "15m" in entry.config.description.lower() or "15m" in entry.config.strategy_id or "scalp" in entry.config.strategy_id
        if uses_15m:
            logger.info("Strategy uses 15m — loading 15m data (yfinance limit: last 60 days)")
            from datetime import datetime, timedelta
            max_15m_start = (datetime.utcnow() - timedelta(days=59)).strftime("%Y-%m-%d")
            df_15m = load_ohlcv(asset, "15m", max_15m_start, end)
            logger.info(f"Loaded {len(df_15m)} 15m bars")
    except Exception as e:
        logger.warning(f"Could not load 15m data: {e} — falling back to 1H")

    if len(df_1h) < 300:
        logger.error("Insufficient data for backtest (need ≥300 1H bars)")
        return False

    # ── 2. Generate signals ───────────────────────────────────────────────
    logger.info("Generating signals (walk-forward, no look-ahead)...")
    signals = generate_signals(
        df_1h=df_1h,
        df_daily=df_daily,
        df_15m=df_15m,
        min_confidence=min_confidence,
        min_rr=min_rr,
        ema_periods=cfg.indicators.ema_periods,
        rsi_period=cfg.indicators.rsi_period,
        macd_fast=cfg.indicators.macd_fast,
        macd_slow=cfg.indicators.macd_slow,
        macd_signal_period=cfg.indicators.macd_signal,
        atr_period=cfg.indicators.atr_period,
    )
    logger.info(f"Generated {len(signals)} signals")

    if len(signals) == 0:
        logger.warning("No signals generated — check min_confidence threshold")
        return False

    # ── 3. Simulate trades ────────────────────────────────────────────────
    logger.info("Simulating trade execution...")
    # Use 15m bars for simulation if available (signals reference 15m bar indices)
    sim_df = df_15m if df_15m is not None else df_1h
    trades = simulate_trades(
        signals=signals,
        df_1h=sim_df,
        portfolio_value=cfg.trading.capital.paper_account_size,
        risk_pct=risk_pct,
        max_concurrent=cfg.trading.capital.max_concurrent_positions,
    )
    logger.info(f"Simulated {len(trades)} trades")

    if len(trades) == 0:
        logger.warning("No trades executed — signals may not have triggered price levels")
        return False

    # ── 4. Calculate metrics ──────────────────────────────────────────────
    start_ts = sim_df.index[0]
    end_ts = sim_df.index[-1]

    metrics = calculate_metrics(
        trades=trades,
        initial_capital=cfg.trading.capital.paper_account_size,
        start_date=start_ts,
        end_date=end_ts,
    )

    # ── 5. Print report ───────────────────────────────────────────────────
    print("\n" + metrics.summary())

    # Signal breakdown
    long_sigs = sum(1 for s in signals if s.action == "LONG")
    short_sigs = sum(1 for s in signals if s.action == "SHORT")
    print(f"\nSignal breakdown: {long_sigs} LONG, {short_sigs} SHORT")

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    print(f"Exit reasons: {exit_reasons}")

    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    print(f"Avg signal confidence: {avg_confidence:.2%}")

    return metrics.passes_gate


def main():
    parser = argparse.ArgumentParser(description="Trading Bot — Phase 0 Backtest")
    parser.add_argument("--asset", default=cfg.asset.symbol, help=f"Asset symbol (default: {cfg.asset.symbol})")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--risk-pct", type=float, default=cfg.trading.capital.risk_per_trade_pct * 100,
                        help=f"Risk %% per trade (default: {cfg.trading.capital.risk_per_trade_pct*100})")
    parser.add_argument("--min-confidence", type=float, default=cfg.risk.min_confidence,
                        help=f"Min confluence score (default: {cfg.risk.min_confidence})")
    parser.add_argument("--min-rr", type=float, default=cfg.risk.min_risk_reward,
                        help=f"Min risk:reward (default: {cfg.risk.min_risk_reward})")
    args = parser.parse_args()

    default_start, default_end = default_backtest_range()
    start = args.start or default_start
    end = args.end or default_end
    risk_pct = args.risk_pct / 100  # convert from % to fraction

    passed = run_backtest(
        asset=args.asset,
        start=start,
        end=end,
        risk_pct=risk_pct,
        min_confidence=args.min_confidence,
        min_rr=args.min_rr,
    )

    if passed:
        logger.success("Phase 0 gate PASSED — ready to proceed to paper trading!")
        sys.exit(0)
    else:
        logger.error("Phase 0 gate FAILED — review strategy before paper trading")
        sys.exit(1)


if __name__ == "__main__":
    main()
