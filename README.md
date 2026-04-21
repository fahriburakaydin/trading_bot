# LLM Trading Bot

An autonomous trading bot that uses local LLM agents (running on Ollama) to analyse markets, set price alarms, execute trades through Interactive Brokers, and continuously improve its own strategy through weekly self-reflection.

Currently configured for **EURUSD** on a **paper trading** account.

---

## How It Works

The bot runs four AI agents that hand off work to each other through a shared SQLite database. Each agent has a specific job and runs on a schedule:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DAILY CYCLE                                 │
│                                                                     │
│  10:00 IST                    10:30 IST                             │
│  ┌──────────────┐             ┌──────────────┐                      │
│  │   Research   │────────────▶│   Analyst    │                      │
│  │   Agent      │  report     │   Agent      │                      │
│  │              │  saved      │              │                      │
│  │ Searches web │  to DB      │ Reads report │                      │
│  │ for news,    │             │ + calculates │                      │
│  │ macro events │             │ indicators   │                      │
│  │ & sentiment  │             │ → sets price │                      │
│  └──────────────┘             │   ALARMS     │                      │
│                               └──────┬───────┘                      │
│                                      │ alarms saved to DB           │
│                                      ▼                               │
│                          ┌─────────────────────┐                    │
│  ALWAYS RUNNING          │    Price Monitor     │                    │
│  (every 30 seconds)      │                      │                    │
│                          │  Polls live price    │                    │
│                          │  from IB Gateway     │                    │
│                          │  → fires Trader      │                    │
│                          │    when price hits   │                    │
│                          │    an alarm level    │                    │
│                          └──────────┬──────────┘                    │
│                                     │ alarm triggered               │
│                                     ▼                               │
│                          ┌─────────────────────┐                    │
│  EVENT-DRIVEN            │    Trader Agent      │                    │
│                          │                      │                    │
│                          │  Final risk check    │                    │
│                          │  → places order      │                    │
│                          │    via IBKR API      │                    │
│                          │  → writes Trade      │                    │
│                          │    record to DB      │                    │
│                          └─────────────────────┘                    │
│                                                                     │
│  WEEKLY CYCLE (Sunday 20:00 IST)                                    │
│  ┌──────────────────────────────────────────────────────┐           │
│  │   Evaluator Agent                                     │           │
│  │                                                       │           │
│  │  Reads all closed trades → identifies patterns       │           │
│  │  → writes strategy rules to Knowledge Base           │           │
│  │  → those rules are injected into ALL future prompts  │           │
│  └──────────────────────────────────────────────────────┘           │
│            │                                                        │
│            └─────────────────────────────────────────────────────▶  │
│              Knowledge Base rules feed back into Research,          │
│              Analyst, and Trader prompts on next cycle              │
└─────────────────────────────────────────────────────────────────────┘
```

**The self-improvement loop:** Every week, the Evaluator reads what happened (wins and losses), extracts lessons, and writes rules into a Knowledge Base. Those rules are automatically injected into every agent's prompt on the next run — so the bot literally learns from its own mistakes.

---

## Components

| Component | Location | What It Does | When It Runs |
|-----------|----------|--------------|--------------|
| **Research Agent** | `agents/research/` | Searches news and macroeconomic events. Summarises market context into a structured report. | Daily at 10:00 IST |
| **Analyst Agent** | `agents/analyst/` | Reads the Research report + calculates RSI, MACD, EMA, ATR, VWAP across 1h/4h/1d timeframes. Scores support/resistance levels. Sets price alarms. | Daily at 10:30 IST |
| **Price Monitor** | `core/monitor/` | Polls the live IBKR price feed every 30 seconds. Fires the Trader Agent when price reaches an alarm level. | Always running |
| **Trader Agent** | `agents/trader/` | Performs a final risk check (position size, daily loss limit, concurrent positions). If approved, places a limit order via IBKR with stop-loss and take-profit. | When alarm is triggered |
| **Evaluator Agent** | `agents/evaluator/` | Reviews closed trades for the past week. Identifies what worked, what didn't. Writes actionable rules to the Knowledge Base. | Every Sunday at 20:00 IST |
| **Knowledge Base** | `core/memory/models.py` | SQLite table of strategy rules written by the Evaluator. Loaded into every agent's prompt automatically via `BaseAgent.load_context()`. | Persistent |
| **Telegram Notifier** | `notifications/` | Sends trade alerts, analyst reports, startup status, and error warnings. Accepts `/improve`, `/show_alarms`, `/set_alarms` commands. | On events |
| **IB Gateway Client** | `core/broker/` | Manages connection to Interactive Brokers Gateway. Handles reconnects with exponential backoff. | Always running |
| **Scheduler** | `core/scheduler/` | APScheduler that fires the Research, Analyst, and Evaluator agents on their cron schedules. | Always running |

---

## Tech Stack

| Purpose | Library |
|---------|---------|
| LLM agents & graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) + [LangChain](https://github.com/langchain-ai/langchain) |
| Local LLM inference | [Ollama](https://ollama.ai) (`qwen2.5:32b` for trading, `qwen3:14b` for research) |
| Cloud LLM fallback | Anthropic Claude, OpenAI GPT (optional) |
| Broker connection | [ib_insync](https://github.com/erdewit/ib_insync) → Interactive Brokers Gateway |
| Technical indicators | [pandas-ta](https://github.com/twopirllc/pandas-ta) |
| Market data | [yfinance](https://github.com/ranaroussi/yfinance) (backtesting) + live IBKR feed |
| Database | SQLite via [SQLAlchemy](https://www.sqlalchemy.org) |
| Configuration | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) + YAML |
| Scheduling | [APScheduler](https://apscheduler.readthedocs.io) |
| Notifications | [python-telegram-bot](https://python-telegram-bot.org) |
| News search | [Brave Search API](https://brave.com/search/api/) (falls back to DuckDuckGo) |
| Dashboard | [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com/python/) |
| Testing | [pytest](https://pytest.org) + pytest-asyncio + pytest-cov |
| Logging | [Loguru](https://github.com/Delgan/loguru) |

---

## Setup

### Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.ai)** installed and running locally
- **Interactive Brokers** account with [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway.php) or TWS running on paper mode (port 7497)
- A **Telegram bot** token (create one via [@BotFather](https://t.me/botfather))
- (Optional) A **Brave Search API** key for better news results

### Install

```bash
git clone <repo-url>
cd trading-bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Pull the LLM models

```bash
ollama pull qwen2.5:32b    # analyst, trader, evaluator
ollama pull qwen3:14b      # research agent
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
BRAVE_API_KEY=your_brave_api_key_here   # optional
```

Review `config/config.yaml` for trading parameters (capital, risk per trade, schedules, etc.).

### Start IB Gateway

1. Open IB Gateway or TWS
2. Set it to **paper trading** mode
3. Enable the API on port **7497**
4. Enable "ActiveX and Socket Clients" in API settings

### Run

```bash
# Full run (requires IB Gateway)
python main.py

# Dry run — scheduler + Telegram only, no IBKR connection
python main.py --dry-run
```

The bot will send a Telegram message when it starts and run the Analyst Agent automatically if no active alarms exist.

---

## Telegram Commands

Send these commands to your bot in Telegram:

| Command | What It Does |
|---------|--------------|
| `/show_alarms` | Lists all active price alarms with entry, stop-loss, and take-profit levels |
| `/set_alarms` | Triggers the Analyst Agent immediately to run a fresh analysis and set new alarms |
| `/improve` | Triggers the Evaluator Agent to review recent trades and update the Knowledge Base |

---

## Configuration Reference

Key settings in `config/config.yaml`:

```yaml
active_asset: "eurusd"          # which asset to trade

trading:
  capital:
    paper_account_size: 100000  # paper account size in USD
    risk_per_trade_pct: 0.005   # risk 0.5% of capital per trade
    max_concurrent_positions: 3
    max_daily_loss_pct: 0.03    # stop trading after 3% daily loss
  price_monitor:
    poll_interval_seconds: 30   # how often to check price
    alarm_expiry_hours: 48      # alarms expire after 48 hours

llm:
  provider: "ollama"            # ollama | anthropic | openai
  model: "qwen3:14b"            # global default model

agents:
  analyst:
    model: "qwen2.5:32b"        # override per agent
```

To switch to cloud LLMs, change `provider` to `anthropic` or `openai` and add the API key to `.env`.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov-report=html

# Run a specific test file
pytest tests/unit/test_indicators.py

# Run a specific test
pytest -k test_broker_connect
```

---

## Backtesting

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --asset EURUSD --start 2025-01-01 --end 2025-04-01
```

---

## Database

SQLite at `trading_bot.db`. Key tables:

| Table | Purpose |
|-------|---------|
| `alarms` | Price levels set by the Analyst. Status: `active` → `triggered` / `expired` / `cancelled` |
| `trades` | Full trade lifecycle. P&L calculated automatically on close. |
| `knowledge_base` | Strategy rules written by the Evaluator. Injected into all agent prompts. |
| `research_reports` | Daily research summaries from the Research Agent |
| `evaluator_reports` | Weekly performance snapshots |
| `agent_logs` | Per-run log of every agent execution with timing and outputs |

---

## Project Structure

```
trading-bot/
├── agents/
│   ├── base.py              # Shared LLM init, KB loading, logging
│   ├── analyst/             # Technical analysis + alarm setting
│   ├── research/            # News search + market context
│   ├── trader/              # Order execution + risk check
│   └── evaluator/           # Weekly strategy reflection
├── core/
│   ├── broker/              # IB Gateway connection + order management
│   ├── config.py            # Config loading (YAML + .env → Pydantic models)
│   ├── indicators/          # RSI, MACD, EMA, ATR, VWAP, confluence scoring
│   ├── memory/              # SQLAlchemy models + database session management
│   ├── monitor/             # Price polling loop + alarm trigger logic
│   └── scheduler/           # APScheduler job registration
├── notifications/
│   ├── telegram.py          # Outbound Telegram messages
│   ├── telegram_commands.py # Inbound Telegram slash commands
│   └── templates.py         # Jinja2 message templates
├── strategies/              # Asset-specific confluence weights
├── tools/                   # LangChain tools: market data, search, indicators
├── tests/                   # pytest test suite
├── config/
│   ├── config.yaml          # Main configuration
│   └── assets/eurusd.yaml   # Asset-specific overrides
├── scripts/
│   ├── run_backtest.py      # Backtest runner
│   └── compare_evaluator.py # A/B test two LLM models for the Evaluator
└── main.py                  # Entry point
```

---

## Potential Next Steps

**Near term**
- [ ] Switch to live trading — change IBKR port to `7496` in `config.yaml`
- [ ] Add more assets — create `config/assets/btcusd.yaml` and set `active_asset`
- [ ] Web dashboard — `streamlit run dashboard/app.py` (foundation already present)
- [ ] Trade journal export — CSV/PDF report generation from closed trades

**Medium term**
- [ ] Multi-asset parallel trading — run separate bot instances per asset
- [ ] Options strategies — extend the Trader Agent to handle options chains
- [ ] Backtesting integration with live strategy weights — close the feedback loop
- [ ] Cloud deployment on a VPS — always-on without a desktop IB Gateway

**Longer term**
- [ ] Portfolio-level risk management — aggregate exposure across all assets
- [ ] Reinforcement learning layer — let the Evaluator tune indicator weights, not just write rules
- [ ] Multi-broker support — abstract the IBKR client behind an interface
- [ ] Real-time dashboard with WebSocket price feed

---

## Security Notes

- Never commit your `.env` file — it is excluded by `.gitignore`
- The `.env.example` file shows the required variables with placeholder values
- IBKR paper trading port `7497` is safe for development; live trading uses port `7496`
- All secrets are loaded from environment variables at startup — nothing is hardcoded in code
