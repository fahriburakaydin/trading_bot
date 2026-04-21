"""
Prompts and Telegram templates for the Trader Agent.
"""

from __future__ import annotations

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Trader Agent for an automated trading system.
Your role is to execute a single trade from a triggered alarm with precision and discipline.

Asset: {symbol} ({asset_class})
Current price: {current_price}
Account value: {account_value}

{context}

Rules you must follow:
- Never trade when risk_level is RED.
- Never exceed max_concurrent_positions.
- Never exceed max_daily_loss.
- Always require a valid stop_loss and target_price.
- Temperature is 0.0 — be deterministic and precise.
"""

# ── Pre-trade risk check prompt ───────────────────────────────────────────────

RISK_CHECK_PROMPT = """\
Evaluate whether it is safe to proceed with this trade.

Alarm details:
{alarm_json}

Current account state:
- Net liquidation value: {net_liq}
- Daily P&L so far: {daily_pnl}
- Max daily loss allowed: {max_daily_loss}
- Open positions: {open_positions}
- Max concurrent positions: {max_concurrent}
- Risk level (from latest research): {risk_level}

Position sizing result:
{sizing_json}

Return a JSON object with:
{{
  "approved": true | false,
  "reason": "<one sentence>",
  "slippage_ok": true | false
}}

Only approve if ALL of the following are true:
1. risk_level is not RED
2. daily_pnl > -max_daily_loss (we haven't blown the daily limit)
3. open_positions < max_concurrent
4. sizing is not null
5. The alarm confidence meets the minimum threshold
"""

# ── Telegram templates ────────────────────────────────────────────────────────

TRADE_OPENED_MSG = """\
*Trade Opened* ✅

*{symbol}* — {direction}
Entry: `${entry_price:,.2f}`
Stop Loss: `${stop_loss:,.2f}`
Target: `${target_price:,.2f}`
Qty: `{quantity}` (notional `${notional:,.2f}`)
R:R: `{risk_reward:.1f}` | Conf: `{confidence:.0%}`
IBKR Order: `{order_id}`
"""

TRADE_REJECTED_MSG = """\
*Trade Rejected* ⛔

*{symbol}* — {direction} @ `${trigger_price:,.2f}`
Reason: {reason}
"""

TRADE_ERROR_MSG = """\
*Trade Error* 🚨

*{symbol}* — {direction} @ `${trigger_price:,.2f}`
Error: {error}
Manual intervention may be required.
"""
