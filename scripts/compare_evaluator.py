#!/usr/bin/env python3
"""
Compare two EvaluatorAgent runs side-by-side with different LLM models.

Both runs use the same trade data (synthetic by default, real DB trades if
--real-trades is passed). DB writes and Telegram notifications are suppressed.

Usage:
    python scripts/compare_evaluator.py
    python scripts/compare_evaluator.py --model-a qwen3:14b --model-b qwen2.5:32b
    python scripts/compare_evaluator.py --real-trades
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# ── Synthetic EURUSD trade fixtures ───────────────────────────────────────────
# 7 trades covering the past week — realistic 2026-04 price levels (~1.08-1.10).
# Mix: 4 wins / 3 losses, win-rate ≈ 57 %, profit factor ≈ 1.4.
# One outsized winner (+2.8 R), one stopped out at a news spike.

_now = datetime.utcnow()
_d = lambda days_ago, h=10: (_now - timedelta(days=days_ago)).replace(  # noqa: E731
    hour=h, minute=0, second=0, microsecond=0
).isoformat()

SYNTHETIC_TRADES: list[dict] = [
    {
        "id": 9001,
        "direction": "LONG",
        "entry_price": 1.0872,
        "exit_price": 1.0950,
        "quantity": 50_000,
        "notional": 54_360.0,
        "stop_loss": 1.0840,
        "target_price": 1.0950,
        "pnl": 390.0,
        "pnl_pct": 0.0072,
        "pnl_r": 2.44,
        "exit_reason": "target_hit",
        "opened_at": _d(6, 9),
        "closed_at": _d(6, 15),
    },
    {
        "id": 9002,
        "direction": "SHORT",
        "entry_price": 1.0961,
        "exit_price": 1.0925,
        "quantity": 50_000,
        "notional": 54_805.0,
        "stop_loss": 1.0990,
        "target_price": 1.0910,
        "pnl": 180.0,
        "pnl_pct": 0.0033,
        "pnl_r": 1.24,
        "exit_reason": "target_hit",
        "opened_at": _d(5, 11),
        "closed_at": _d(5, 16),
    },
    {
        "id": 9003,
        "direction": "LONG",
        "entry_price": 1.0903,
        "exit_price": 1.0865,
        "quantity": 50_000,
        "notional": 54_515.0,
        "stop_loss": 1.0865,
        "target_price": 1.0960,
        "pnl": -190.0,
        "pnl_pct": -0.0035,
        "pnl_r": -1.0,
        "exit_reason": "stop_hit",
        "opened_at": _d(4, 10),
        "closed_at": _d(4, 13),
    },
    {
        "id": 9004,
        "direction": "LONG",
        "entry_price": 1.0885,
        "exit_price": 1.0980,
        "quantity": 50_000,
        "notional": 54_425.0,
        "stop_loss": 1.0851,
        "target_price": 1.0986,
        "pnl": 475.0,
        "pnl_pct": 0.0087,
        "pnl_r": 2.79,
        "exit_reason": "target_hit",
        "opened_at": _d(3, 9),
        "closed_at": _d(3, 17),
    },
    {
        "id": 9005,
        "direction": "SHORT",
        "entry_price": 1.0975,
        "exit_price": 1.1010,
        "quantity": 50_000,
        "notional": 54_875.0,
        "stop_loss": 1.1005,
        "target_price": 1.0920,
        "pnl": -175.0,
        "pnl_pct": -0.0032,
        "pnl_r": -1.17,
        "exit_reason": "news_spike",
        "opened_at": _d(2, 14),
        "closed_at": _d(2, 14),
    },
    {
        "id": 9006,
        "direction": "LONG",
        "entry_price": 1.0920,
        "exit_price": 1.0945,
        "quantity": 50_000,
        "notional": 54_600.0,
        "stop_loss": 1.0890,
        "target_price": 1.0975,
        "pnl": 125.0,
        "pnl_pct": 0.0023,
        "pnl_r": 0.83,
        "exit_reason": "manual_close",
        "opened_at": _d(1, 10),
        "closed_at": _d(1, 15),
    },
    {
        "id": 9007,
        "direction": "SHORT",
        "entry_price": 1.0940,
        "exit_price": 1.0975,
        "quantity": 50_000,
        "notional": 54_700.0,
        "stop_loss": 1.0975,
        "target_price": 1.0885,
        "pnl": -175.0,
        "pnl_pct": -0.0032,
        "pnl_r": -1.0,
        "exit_reason": "stop_hit",
        "opened_at": _d(0, 9),
        "closed_at": _d(0, 11),
    },
]

# ── Formatting helpers ─────────────────────────────────────────────────────────

_W = 70  # terminal width


def _bar(char: str = "═") -> str:
    return char * _W


def _section(title: str) -> str:
    return f"\n{_bar()}\n{title}\n{_bar()}"


def _divider() -> str:
    return "─" * _W


def _print_model_section(label: str, state: dict) -> None:
    analysis: dict = state.get("analysis") or {}
    knowledge: list[dict] = state.get("new_knowledge") or []

    print(f"\n{label}")
    print(_divider())

    summary = analysis.get("performance_summary", "(none)")
    print("Performance Summary:")
    for line in _wrap(summary, indent=2):
        print(line)

    strengths = analysis.get("strengths") or []
    weaknesses = analysis.get("weaknesses") or []
    patterns = analysis.get("notable_patterns") or []

    print(f"\nStrengths ({len(strengths)}):")
    for s in strengths:
        print(f"  + {s}")

    print(f"\nWeaknesses ({len(weaknesses)}):")
    for w in weaknesses:
        print(f"  - {w}")

    print(f"\nNotable Patterns ({len(patterns)}):")
    for p in patterns:
        print(f"  * {p}")

    print(f"\nKnowledge Base Entries ({len(knowledge)}):")
    for i, k in enumerate(knowledge, 1):
        cat = k.get("category", "?")
        title = k.get("title", "?")
        content = k.get("content", "")
        impact = k.get("performance_impact")
        impact_str = f"  impact={impact:+.1f}" if impact is not None else ""
        print(f"  {i}. [{cat}] {title}{impact_str}")
        for line in _wrap(content, indent=6, width=_W - 6):
            print(line)

    print(_divider())


def _print_diff(label_a: str, state_a: dict, label_b: str, state_b: dict) -> None:
    print(_section("DIFF ANALYSIS"))

    ana_a = state_a.get("analysis") or {}
    ana_b = state_b.get("analysis") or {}
    kb_a: list[dict] = state_a.get("new_knowledge") or []
    kb_b: list[dict] = state_b.get("new_knowledge") or []

    def _count(d: dict, key: str) -> int:
        return len(d.get(key) or [])

    def _summary_len(d: dict) -> int:
        return len(d.get("performance_summary") or "")

    print("\nAnalysis depth:")
    print(f"  {'Metric':<28} {'A':>6}   {'B':>6}")
    print(f"  {'─'*28} {'─'*6}   {'─'*6}")
    rows = [
        ("Summary length (chars)", _summary_len(ana_a), _summary_len(ana_b)),
        ("Strengths identified", _count(ana_a, "strengths"), _count(ana_b, "strengths")),
        ("Weaknesses identified", _count(ana_a, "weaknesses"), _count(ana_b, "weaknesses")),
        ("Patterns identified", _count(ana_a, "notable_patterns"), _count(ana_b, "notable_patterns")),
        ("KB entries extracted", len(kb_a), len(kb_b)),
    ]
    for name, va, vb in rows:
        print(f"  {name:<28} {va:>6}   {vb:>6}")

    titles_a = {k.get("title", "") for k in kb_a}
    titles_b = {k.get("title", "") for k in kb_b}
    only_a = titles_a - titles_b
    only_b = titles_b - titles_a
    shared = titles_a & titles_b

    print(f"\nKnowledge unique to {label_a}  ({len(only_a)}):")
    for t in sorted(only_a):
        print(f"  • {t}")
    if not only_a:
        print("  (none)")

    print(f"\nKnowledge unique to {label_b}  ({len(only_b)}):")
    for t in sorted(only_b):
        print(f"  • {t}")
    if not only_b:
        print("  (none)")

    print(f"\nShared themes ({len(shared)}):")
    for t in sorted(shared):
        print(f"  • {t}")
    if not shared:
        print("  (none)")

    # Auto-generated verdict
    score_a = (
        _summary_len(ana_a)
        + _count(ana_a, "strengths") * 20
        + _count(ana_a, "weaknesses") * 20
        + _count(ana_a, "notable_patterns") * 15
        + len(kb_a) * 25
    )
    score_b = (
        _summary_len(ana_b)
        + _count(ana_b, "strengths") * 20
        + _count(ana_b, "weaknesses") * 20
        + _count(ana_b, "notable_patterns") * 15
        + len(kb_b) * 25
    )

    print("\nVerdict:")
    if score_a == 0 and score_b == 0:
        verdict = "Both models produced no analysis output — likely no LLM calls were made."
    elif abs(score_a - score_b) < 30:
        winner = label_a if score_a >= score_b else label_b
        verdict = (
            f"Models are broadly comparable in output depth. "
            f"{winner} has a slight edge on combined richness score "
            f"(A={score_a}, B={score_b})."
        )
    elif score_a > score_b:
        verdict = (
            f"{label_a} produced richer analysis (score {score_a} vs {score_b}): "
            f"more detailed narrative and/or more KB entries extracted."
        )
    else:
        verdict = (
            f"{label_b} produced richer analysis (score {score_b} vs {score_a}): "
            f"more detailed narrative and/or more KB entries extracted."
        )

    for line in _wrap(verdict, indent=2):
        print(line)
    print()


def _wrap(text: str, indent: int = 0, width: int = _W) -> list[str]:
    """Very simple word-wrap that respects terminal width."""
    import textwrap
    return textwrap.wrap(text, width=width - indent, initial_indent=" " * indent, subsequent_indent=" " * indent)


# ── DB helpers ────────────────────────────────────────────────────────────────


def _load_real_trades() -> list[dict]:
    """Query all closed trades from the DB, regardless of date."""
    from core.config import cfg
    from core.memory.database import get_session
    from core.memory.models import Trade

    try:
        with get_session() as session:
            db_trades = (
                session.query(Trade)
                .filter(
                    Trade.asset == cfg.asset.symbol,
                    Trade.status == "closed",
                )
                .order_by(Trade.closed_at)
                .all()
            )
            trades = []
            for t in db_trades:
                trades.append({
                    "id": t.id,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "notional": t.notional,
                    "stop_loss": t.stop_loss,
                    "target_price": t.target_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "pnl_r": t.pnl_r,
                    "exit_reason": t.exit_reason,
                    "opened_at": t.opened_at.isoformat() if t.opened_at else None,
                    "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                })
        return trades
    except Exception as exc:
        logger.warning(f"Could not query DB trades: {exc}")
        return []


# ── Core comparison ───────────────────────────────────────────────────────────


def run_comparison(model_a: str, model_b: str, use_real_trades: bool) -> None:
    from agents.evaluator.agent import EvaluatorAgent
    from core.config import cfg

    # Determine trade source
    trades: list[dict] = []
    trade_source = "synthetic"
    if use_real_trades:
        trades = _load_real_trades()
        if trades:
            trade_source = f"{len(trades)} real DB trades"
        else:
            logger.warning("No real trades found in DB — falling back to synthetic data")

    if not trades:
        trades = SYNTHETIC_TRADES
        trade_source = f"{len(SYNTHETIC_TRADES)} synthetic trades"

    asset = cfg.asset.symbol.upper()
    today = date.today().isoformat()

    print(_section(f"EVALUATOR MODEL COMPARISON\n{today}  |  {asset}  |  {trade_source}"))

    # ── Run model A ───────────────────────────────────────────────────────────
    print(f"\nRunning {model_a}…  (this may take a minute or two)")
    agent_a = EvaluatorAgent(model=model_a)
    state_a = agent_a.run(dry_run=True, trades_override=trades)

    # ── Run model B ───────────────────────────────────────────────────────────
    print(f"\nRunning {model_b}…  (this may take a minute or two)")
    agent_b = EvaluatorAgent(model=model_b)
    state_b = agent_b.run(dry_run=True, trades_override=trades)

    # ── Print results ─────────────────────────────────────────────────────────
    print(_section(f"RESULTS"))
    _print_model_section(f"MODEL A: {model_a}", state_a)
    _print_model_section(f"MODEL B: {model_b}", state_b)
    _print_diff(model_a, state_a, model_b, state_b)


# ── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two EvaluatorAgent models side by side."
    )
    parser.add_argument(
        "--model-a",
        default="qwen3:14b",
        help="First model tag (default: qwen3:14b)",
    )
    parser.add_argument(
        "--model-b",
        default="qwen2.5:32b",
        help="Second model tag (default: qwen2.5:32b)",
    )
    parser.add_argument(
        "--real-trades",
        action="store_true",
        help="Use real closed trades from DB (falls back to synthetic if none found)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_comparison(
        model_a=args.model_a,
        model_b=args.model_b,
        use_real_trades=args.real_trades,
    )


if __name__ == "__main__":
    main()
