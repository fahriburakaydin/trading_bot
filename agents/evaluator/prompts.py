"""
Prompts and Telegram templates for the Evaluator Agent.
"""

from __future__ import annotations

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Evaluator Agent for an automated trading system.
Your role is to review the past week's trading performance, identify patterns,
and extract concrete lessons to improve future decision-making.

Asset: {symbol} ({asset_class})
Week: {week_start} to {week_end}

{context}

Be objective, data-driven, and specific. Avoid platitudes.
"""

# ── Performance analysis prompt ───────────────────────────────────────────────

ANALYSE_PERFORMANCE_PROMPT = """\
Analyse the following completed trades from the past week.

Trades:
{trades_json}

Summary statistics:
- Total trades: {total_trades}
- Win rate: {win_rate:.1%}
- Average win: ${avg_win:.2f}
- Average loss: ${avg_loss:.2f}
- Total P&L: ${total_pnl:.2f}
- Profit factor: {profit_factor:.2f}
- Max drawdown: {max_drawdown:.2%}

Return a JSON object with:
{{
  "performance_summary": "<2-3 sentence narrative>",
  "strengths": ["<specific strength>", ...],
  "weaknesses": ["<specific weakness>", ...],
  "notable_patterns": ["<pattern observed>", ...]
}}
"""

# ── Knowledge extraction prompt ───────────────────────────────────────────────

EXTRACT_KNOWLEDGE_PROMPT = """\
Based on this week's performance analysis:

{analysis_json}

Extract 2–5 actionable rules or insights to add to the knowledge base.
Each rule should be concrete enough to change future agent behaviour.

Return a JSON array:
[
  {{
    "category": "rule" | "pattern" | "insight",
    "applies_to": "research" | "analyst" | "trader" | "all",
    "title": "<short title>",
    "content": "<actionable description>",
    "performance_impact": <float -1.0 to 1.0>
  }},
  ...
]

Only include rules with clear evidence from this week's data.
"""

# ── Telegram templates ────────────────────────────────────────────────────────

WEEKLY_REPORT_MSG = """\
*Weekly Performance Report* 📊
{week_start} → {week_end}

*Trades:* {total_trades} ({winning_trades}W / {losing_trades}L)
*Win Rate:* `{win_rate:.1%}`
*Total P&L:* `${total_pnl:+,.2f}`
*Profit Factor:* `{profit_factor:.2f}`
*Max Drawdown:* `{max_drawdown:.1%}`
*Expectancy:* `${expectancy:+.2f}` per trade

*Key Insights:*
{insight_bullets}

*Knowledge Base:* {kb_entries_added} new rules added.
"""

NO_TRADES_MSG = """\
*Weekly Report* 📊
{week_start} → {week_end}

No completed trades this week.
{reason}
"""
