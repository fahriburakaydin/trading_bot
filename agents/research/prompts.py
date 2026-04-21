"""
Prompt templates for the Research Agent.

The Research Agent runs each morning to produce a macro/sentiment report that
the Analyst Agent and Trader Agent consume as context.

All prompts receive {symbol} and {asset_class} at render time from the agent,
so they work correctly regardless of the configured asset.
"""

SYSTEM_PROMPT = """\
You are a professional macro-economic and financial news analyst. Your job is to \
gather and synthesise information from multiple sources into a structured, actionable \
intelligence report.

Your reports must be:
- Factual and source-attributed — never invent data
- Concise — one clear sentence per finding where possible
- Actionable — end every section with a brief "implication" for a {asset_class} trader trading {symbol}
- Risk-aware — explicitly flag elevated-risk conditions

{context}
"""

GATHER_NEWS_PROMPT = """\
Search for the latest financial news published in the past 24 hours relevant to {symbol}.

Focus on:
1. {symbol}-specific price-moving events
2. US macro data releases (CPI, NFP, FOMC statements, Fed speeches)
3. Global macro risk factors (DXY, US10Y yield, equity indices S&P500 / Nasdaq)
4. Any major geopolitical events that affect risk appetite

Use the web_search tool with these queries (run all in sequence):
- "{symbol} market news today"
- "US macro economic data release today {date}"
- "Federal Reserve interest rate news {date}"
- "DXY dollar index {asset_class} correlation {date}"

Return a JSON list of findings, each with keys: source_title, url, summary, relevance_score (0-1).
"""

ECONOMIC_CALENDAR_PROMPT = """\
Search for today's economic calendar events that could move markets for {symbol}.
Use the web_search tool with query: "economic calendar high impact events {date}"

Look specifically for:
- FOMC meetings or Fed speeches
- US CPI / PPI / PCE releases
- US Non-Farm Payrolls
- US GDP data
- Any surprise announcements

Return a JSON list with keys: event_name, time_utc, expected_impact (high/medium/low), \
previous_value, forecast_value.
If no data found, return an empty list [].
"""

SENTIMENT_PROMPT = """\
Based on the news findings below, assess overall market sentiment for {symbol} ({asset_class}).

News findings:
{news_findings}

Rate each dimension on a scale of -1.0 (strongly bearish) to +1.0 (strongly bullish):
- macro_sentiment: driven by Fed policy, DXY, yields
- risk_appetite: driven by equity markets, VIX, overall risk-on/off
- asset_specific: {symbol}-specific news and catalysts
- overall_sentiment: weighted average of the above

Return a JSON object with those four keys plus a "reasoning" string (2-3 sentences).
"""

RISK_ENVIRONMENT_PROMPT = """\
Based on the collected data, assess the current risk environment for a {symbol} trader.

Market context:
{market_context}

Classify the risk environment as one of:
- GREEN: Normal conditions, standard position sizing appropriate
- YELLOW: Elevated uncertainty, reduce position sizes by 50%
- RED: High risk / avoid new positions entirely

Provide:
{{
  "risk_level": "GREEN|YELLOW|RED",
  "primary_risk_factor": "<single sentence>",
  "secondary_risk_factors": ["<factor 1>", "<factor 2>"],
  "trading_implication": "<1-2 sentence actionable guidance>"
}}
"""

WRITE_REPORT_PROMPT = """\
Compile the research findings into a final structured intelligence report.

Data to include:
- News findings: {news_findings}
- Economic calendar: {economic_calendar}
- Sentiment analysis: {sentiment}
- Risk environment: {risk_environment}

Format the report as:
---
DAILY INTELLIGENCE REPORT — {date}
Asset: {symbol}

## MACRO ENVIRONMENT
<2-3 bullet points>

## SENTIMENT
Overall: <score> (<label>)
<2-3 bullet points>

## KEY RISKS
Risk Level: <GREEN|YELLOW|RED>
<2-3 bullet points>

## ECONOMIC CALENDAR
<table or bullet list of high-impact events today>

## TRADING IMPLICATION
<1-2 sentences: what does this mean for {symbol} traders today?>
---

Keep the report under 400 words. Be direct. No filler text.
"""

TELEGRAM_SUMMARY_TEMPLATE = """\
📊 *Daily Research Report* — {date}

*Sentiment:* {overall_sentiment_label} ({overall_sentiment_score:+.2f})
*Risk Level:* {risk_level_emoji} {risk_level}

*Key Macro Points:*
{macro_bullets}

*Trading Implication:*
{trading_implication}
"""

# Emoji mapping for risk levels
RISK_LEVEL_EMOJI = {
    "GREEN": "🟢",
    "YELLOW": "🟡",
    "RED": "🔴",
}

SENTIMENT_LABELS = {
    (-1.0, -0.5): "Strongly Bearish",
    (-0.5, -0.1): "Bearish",
    (-0.1, 0.1): "Neutral",
    (0.1, 0.5): "Bullish",
    (0.5, 1.01): "Strongly Bullish",
}


def sentiment_label(score: float) -> str:
    """Convert a numeric sentiment score to a human-readable label."""
    for (low, high), label in SENTIMENT_LABELS.items():
        if low <= score < high:
            return label
    return "Neutral"
