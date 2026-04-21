"""
Jinja2-based Telegram message templates.

Centralises all message formatting for Research Agent, Analyst Agent,
Trader Agent, and system alerts.  Use render() to produce a final string.

Usage:
    from notifications.templates import render, RESEARCH_REPORT
    msg = render(RESEARCH_REPORT, date="2026-04-16", sentiment="Bullish", ...)
"""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

# Jinja2 environment — StrictUndefined raises on missing variables
_env = Environment(undefined=StrictUndefined, autoescape=False)


def render(template_str: str, **kwargs: object) -> str:
    """Render a Jinja2 template string with the provided keyword arguments."""
    tmpl = _env.from_string(template_str)
    return tmpl.render(**kwargs)


# ── Research Agent ─────────────────────────────────────────────────────────────

RESEARCH_REPORT = """\
📊 *Daily Research Report* — {{ date }}

*Sentiment:* {{ sentiment_label }} ({{ "{:+.2f}".format(sentiment_score) }})
*Risk Level:* {{ risk_emoji }} {{ risk_level }}

*Key Macro Points:*
{% for bullet in macro_bullets %}\
• {{ bullet }}
{% endfor %}
*Trading Implication:*
{{ trading_implication }}
"""

# ── Analyst Agent ──────────────────────────────────────────────────────────────

ANALYST_ALARMS = """\
🎯 *Analyst Report* — {{ date }}
*{{ symbol }}* @ ${{ "{:,.2f}".format(current_price) }}

*Alarms Set:* {{ alarm_count }}
{% for alarm in alarms %}\
• {{ alarm.action }} @ ${{ "{:,.2f}".format(alarm.trigger_price) }} \
(conf={{ "{:.0%}".format(alarm.confidence) }}, R:R={{ "{:.1f}".format(alarm.risk_reward) }})
{% endfor %}
*Market Structure:*
• Trend (1D): {{ trend_1d }}
• RSI (1H): {{ "{:.1f}".format(rsi_1h) }} ({{ rsi_signal }})
• MACD (1H): {{ macd_direction }}
"""

ANALYST_NO_ALARMS = """\
📊 *Analyst Report* — {{ date }}
*{{ symbol }}* — No actionable setups today.

Risk Level: {{ risk_emoji }} {{ risk_level }}
Reason: {{ reason }}
"""

# ── Trader Agent ───────────────────────────────────────────────────────────────

TRADE_OPENED = """\
🟢 *Trade Opened* — {{ symbol }}

Direction: *{{ direction }}*
Entry: ${{ "{:,.2f}".format(entry_price) }}
Stop Loss: ${{ "{:,.2f}".format(stop_loss) }}
Target: ${{ "{:,.2f}".format(target_price) }}
Quantity: {{ quantity }} {{ symbol }}
Notional: ${{ "{:,.2f}".format(notional) }}
R:R: {{ "{:.1f}".format(risk_reward) }}x
Confidence: {{ "{:.0%}".format(confidence) }}
"""

TRADE_CLOSED_PROFIT = """\
✅ *Trade Closed — Profit* — {{ symbol }}

Direction: {{ direction }}
Entry: ${{ "{:,.2f}".format(entry_price) }}
Exit: ${{ "{:,.2f}".format(exit_price) }}
P&L: +${{ "{:,.2f}".format(pnl) }} (+{{ "{:.2%}".format(pnl_pct) }}) \
[{{ "{:+.2f}".format(pnl_r) }}R]
Reason: {{ exit_reason }}
"""

TRADE_CLOSED_LOSS = """\
🔴 *Trade Closed — Loss* — {{ symbol }}

Direction: {{ direction }}
Entry: ${{ "{:,.2f}".format(entry_price) }}
Exit: ${{ "{:,.2f}".format(exit_price) }}
P&L: -${{ "{:,.2f}".format(pnl|abs) }} ({{ "{:.2%}".format(pnl_pct) }}) \
[{{ "{:+.2f}".format(pnl_r) }}R]
Reason: {{ exit_reason }}
"""

ALARM_TRIGGERED = """\
🔔 *Alarm Triggered* — {{ symbol }}

{{ action }} alarm hit at ${{ "{:,.2f}".format(trigger_price) }}
Evaluating entry conditions...
"""

SLIPPAGE_ABORTED = """\
⚠️ *Trade Aborted — Slippage*

{{ symbol }} {{ action }} @ ${{ "{:,.2f}".format(trigger_price) }}
Expected: ${{ "{:,.2f}".format(expected_price) }}
Got: ${{ "{:,.2f}".format(actual_price) }}
Slippage: {{ "{:.2%}".format(slippage_pct) }} (limit {{ "{:.2%}".format(max_slippage_pct) }})
"""

# ── Price Monitor ─────────────────────────────────────────────────────────────

ALARM_EXPIRED = """\
⏰ *Alarm Expired* — {{ symbol }}

{{ action }} alarm @ ${{ "{:,.2f}".format(trigger_price) }} expired after {{ hours }}h.
"""

# ── System / Error alerts ─────────────────────────────────────────────────────

IBKR_DISCONNECTED = """\
🔌 *IBKR Disconnected*

Attempting reconnect (attempt {{ attempt }}/{{ max_attempts }})...
"""

IBKR_RECONNECTED = """\
✅ *IBKR Reconnected*

Connection restored after {{ attempts }} attempt(s).
"""

DAILY_LIMIT_HIT = """\
🛑 *Daily Loss Limit Reached*

Total daily P&L: -${{ "{:,.2f}".format(loss) }} ({{ "{:.2%}".format(loss_pct) }})
Trading paused until tomorrow.
"""
