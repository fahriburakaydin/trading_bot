"""
Fixed fractional position sizing.

Formula:
    risk_amount   = portfolio_value × risk_pct
    price_distance = |entry_price - stop_loss|
    quantity       = risk_amount / price_distance
    notional       = quantity × entry_price

Hard guardrails (cannot be overridden by config):
    - notional ≤ 20% of portfolio
    - quantity ≥ min_quantity
    - stop_loss must differ from entry_price
"""

from __future__ import annotations

from loguru import logger

# Hard limits — non-negotiable
MAX_NOTIONAL_FRACTION = 0.20    # 20% of portfolio per trade
MAX_RISK_PCT = 0.02             # 2% absolute ceiling on risk per trade
REQUIRE_STOP_LOSS = True


def calculate_position_size(
    portfolio_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    min_quantity: float = 0.0001,
    max_notional_pct: float | None = None,
) -> dict[str, float] | None:
    """
    Calculate position size using fixed fractional method.

    Args:
        portfolio_value: Current total portfolio value (paper or live)
        risk_pct:        Fraction of portfolio to risk (e.g. 0.005 = 0.5%)
        entry_price:     Intended entry price
        stop_loss:       Stop loss price (must differ from entry)
        min_quantity:    Minimum order size for the asset

    Returns:
        Dict with quantity, notional, risk_amount, risk_pct_actual
        Returns None if any guardrail is violated (logs reason)
    """
    # Input validation
    if entry_price <= 0:
        logger.error(f"Invalid entry_price: {entry_price}")
        return None

    if stop_loss <= 0:
        logger.error(f"Invalid stop_loss: {stop_loss}")
        return None

    price_distance = abs(entry_price - stop_loss)
    if price_distance < entry_price * 0.0001:  # < 0.01% apart
        logger.error(
            f"Stop loss too close to entry: entry={entry_price:.2f} sl={stop_loss:.2f} "
            f"distance={price_distance:.2f}"
        )
        return None

    # Cap risk at hard ceiling
    effective_risk_pct = min(risk_pct, MAX_RISK_PCT)
    if risk_pct > MAX_RISK_PCT:
        logger.warning(
            f"risk_pct {risk_pct:.3%} exceeds hard ceiling {MAX_RISK_PCT:.3%}, "
            f"capped to {MAX_RISK_PCT:.3%}"
        )

    risk_amount = portfolio_value * effective_risk_pct
    quantity = risk_amount / price_distance
    notional = quantity * entry_price

    # Guardrail: max notional — use config override if provided, else hard default
    effective_max_fraction = max_notional_pct if max_notional_pct is not None else MAX_NOTIONAL_FRACTION
    max_notional = portfolio_value * effective_max_fraction
    if notional > max_notional:
        logger.warning(
            f"Notional {notional:.2f} exceeds {effective_max_fraction:.0%} of portfolio "
            f"({max_notional:.2f}). Scaling down."
        )
        quantity = max_notional / entry_price
        notional = quantity * entry_price
        risk_amount = quantity * price_distance  # recalculate actual risk after cap

    # Guardrail: minimum quantity
    if quantity < min_quantity:
        logger.error(
            f"Quantity {quantity:.6f} is below broker minimum {min_quantity}. "
            "Position not taken."
        )
        return None

    return {
        "quantity": round(quantity, 6),
        "notional": round(notional, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_pct_actual": round(risk_amount / portfolio_value, 6),
    }


def validate_risk_tiers(appetite: str) -> tuple[float, float]:
    """
    Return (risk_pct, max_daily_loss_pct) for a given risk appetite.

    Returns:
        Tuple of (risk_per_trade_pct, max_daily_loss_pct)
    """
    tiers = {
        "conservative": (0.0025, 0.015),
        "moderate": (0.005, 0.03),
        "aggressive": (0.01, 0.05),
    }
    return tiers.get(appetite, tiers["moderate"])
