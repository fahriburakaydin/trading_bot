"""
Prompt templates for the Analyst Agent.

The Analyst Agent runs each morning after the Research Agent to produce
price alarms (entry levels) for the Trader Agent to monitor.
"""

SYSTEM_PROMPT = """\
You are a professional technical analyst specialising in {asset_class} markets.
You use multi-timeframe analysis to identify high-probability trade setups.

Your analysis must be:
- Data-driven: base every conclusion on the indicator values provided
- Multi-timeframe confluent: only flag setups confirmed on 2+ timeframes
- Risk-first: always define stop-loss before entry
- Concise: one clear bullet point per finding

Current asset: {symbol} ({asset_class})
Current price: {current_price}

Entry type rules (CRITICAL — read carefully):
- LONG pullback: trigger_price BELOW current price. Price must fall to support before buying.
- LONG breakout: trigger_price ABOVE current price. Price must break resistance before buying.
- SHORT pullback: trigger_price ABOVE current price. Price must rally to resistance before selling.
- SHORT breakout: trigger_price BELOW current price. Price must break support before selling.

Only set breakout entries when price is coiling near a key level with strong momentum.
Prefer pullback entries at confirmed support/resistance levels for better risk/reward.

{context}
"""

FETCH_DATA_PROMPT = """\
Fetch the OHLCV market data needed for multi-timeframe technical analysis.

Use the fetch_multi_timeframe tool to retrieve data for these timeframes:
["1 hour", "4 hours", "1 day"]

Duration: "30 D" (30 days of history for reliable indicator calculation)
"""

CALCULATE_INDICATORS_PROMPT = """\
Calculate technical indicators for each timeframe using the OHLCV data provided.

For each timeframe, call these tools in order:
1. calculate_rsi — with the timeframe's OHLCV JSON
2. calculate_macd — with the timeframe's OHLCV JSON
3. calculate_ema — with the timeframe's OHLCV JSON
4. calculate_atr — with the timeframe's OHLCV JSON
5. calculate_vwap — with the 1h OHLCV JSON only (intraday VWAP)

OHLCV data by timeframe:
{ohlcv_data}

Return a summary JSON with structure:
{{
  "1h": {{"rsi": ..., "macd_direction": ..., "trend": ..., "atr": ...}},
  "4h": {{"rsi": ..., "macd_direction": ..., "trend": ..., "atr": ...}},
  "1d": {{"rsi": ..., "macd_direction": ..., "trend": ..., "atr": ...}},
  "vwap": {{"value": ..., "price_vs_vwap": ...}}
}}
"""

DETECT_LEVELS_PROMPT = """\
Detect and score key support/resistance levels using multi-timeframe confluence analysis.

Call the score_confluence_levels tool with:
- ohlcv_json: the 1h OHLCV data (provides granular level detection)
- current_price: {current_price}

After getting results, also call detect_support_resistance with the 4h OHLCV data
to capture higher-timeframe levels.

Return a combined list of the most significant levels, de-duplicated.
"""

FILTER_SETUPS_PROMPT = """\
Review the scored levels and indicator data, then select the best trade setups to alarm.

Selection criteria:
1. Confidence score >= {min_confidence} (from score_confluence_levels output)
2. Risk/reward ratio >= {min_risk_reward}
3. Confirmed by indicators on at least 2 of 3 timeframes
4. Consistent with the current risk environment: {risk_level}

Indicator summary:
{indicators_summary}

Scored levels:
{scored_levels}

Research context (from today's report):
{research_context}

Rules:
- RED risk environment → skip all new setups
- YELLOW → only the highest-confidence setup (if any)
- GREEN → up to {max_alarms} setups

Return a JSON list of approved setups:
[{{
  "trigger_price": float,
  "entry_type": "pullback|breakout",
  "direction": "above|below",
  "action": "LONG|SHORT",
  "confidence": float,
  "stop_loss": float,
  "target_price": float,
  "risk_reward": float,
  "timeframe": "1h|4h|1d",
  "confluence_factors": [str, ...],
  "reasoning": str
}}, ...]

Validate before returning:
- LONG pullback: trigger_price < current_price ✓
- LONG breakout: trigger_price > current_price ✓
- SHORT pullback: trigger_price > current_price ✓
- SHORT breakout: trigger_price < current_price ✓
If a setup fails this check, drop it.

If no setups meet criteria, return [].
"""

WRITE_SUMMARY_PROMPT = """\
Write a brief Analyst Agent summary for logging and notification.

Approved alarms:
{approved_alarms}

Indicator snapshot:
{indicators_summary}

Format:
---
ANALYST SUMMARY — {date}
Asset: {symbol} @ {current_price}

Alarms Set: {alarm_count}
{alarm_bullets}

Market Structure:
- Trend (1D): <direction>
- Trend (4H): <direction>
- RSI (1H): <value> (<signal>)
- MACD (1H): <direction>
- VWAP: <price_vs_vwap>
---

Keep under 200 words.
"""

TELEGRAM_ALARM_TEMPLATE = """\
🎯 <b>Analyst Report</b> — {date}
<b>{symbol}</b> @ {current_price:,.5f}

<b>Alarms Set:</b> {alarm_count}
{alarm_lines}

<b>Market Structure:</b>
• Trend (1D): {trend_1d}
• RSI (1H): {rsi_1h:.1f} ({rsi_signal})
• MACD (1H): {macd_direction}
"""

TELEGRAM_NO_ALARMS = """\
📊 <b>Analyst Report</b> — {date}
<b>{symbol}</b> — No actionable setups today.

Risk Level: {risk_level_emoji} {risk_level}
Reason: {reason}
"""
