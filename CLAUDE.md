# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to Operate

**1. Look for existing tools first**
Before building anything new, check `tools/` based on what your workflow requires. Only create new scripts when nothing exists for that task.

**2. Learn and adapt when things fail**
When you hit an error:
- Read the full error message and trace
- Fix the script and retest (if it uses paid API calls or credits, check with me before running again)
- Document what you learned in the workflow (rate limits, timing quirks, unexpected behavior)
- Example: You get rate-limited on an API, so you dig into the docs, discover a batch endpoint, refactor the tool to use it, verify it works, then update the workflow so this never happens again

**3. Keep workflows current**
Workflows should evolve as you learn. When you find better methods, discover constraints, or encounter recurring issues, update the workflow. That said, don't create or overwrite workflows without asking unless I explicitly tell you to. These are your instructions and need to be preserved and refined, not tossed after one use.

## The Self-Improvement Loop

Every failure is a chance to make the system stronger:
1. Identify what broke
2. Fix the tool
3. Verify the fix works
4. Update the workflow with the new approach
5. Move on with a more robust system

This loop is how the framework improves over time.


## Commands

```bash
# Run the bot (requires IB Gateway on port 7497)
python main.py

# Run without IBKR connection (scheduler + Telegram only)
python main.py --dry-run

# Run all tests
pytest

# Run a single test file
pytest tests/unit/test_indicators.py

# Run a single test
pytest -k test_broker_connect

# Run with coverage
pytest --cov=core --cov-report=html

# Backtest the active strategy
python scripts/run_backtest.py
python scripts/run_backtest.py --asset EURUSD --start 2025-01-01 --end 2025-04-01

# Compare two LLM models for the Evaluator Agent
python scripts/compare_evaluator.py --model-a qwen3:14b --model-b qwen2.5:32b
```

No linter or formatter is configured. No Makefile.

## Architecture

This is an **LLM-agent trading bot** that trades EURUSD (configurable) via Interactive Brokers. Four LangGraph agents run on a scheduler and hand off through a shared SQLite database:

```
Research Agent (10:00 daily)
  → ResearchReport saved to DB

Analyst Agent (10:30 daily)
  → Reads ResearchReport + TA indicators → sets Alarms in DB

PriceMonitor (always on, polls every 30s)
  → When price hits an Alarm → fires Trader Agent

Trader Agent (event-driven)
  → Places order via IBKR → writes Trade to DB

Evaluator Agent (Sunday 20:00)
  → Reads closed Trades → extracts KnowledgeEntry rows → writes EvaluatorReport
```

### Agent pattern

All agents in `agents/` subclass `BaseAgent` (`agents/base.py`), which handles:
- LLM instantiation with 3-level model resolution: runtime arg > per-agent YAML config > global default
- Knowledge-base context loading (`load_context()` injects active `KnowledgeEntry` rows into prompts)
- Structured logging

Each agent builds a LangGraph `StateGraph`, compiles it, and exposes a single `run()` method that invokes the graph and returns the final state dict.

`run(dry_run=True, trades_override=[...])` is supported on `EvaluatorAgent` to suppress DB writes and Telegram notifications (used by `scripts/compare_evaluator.py`).

### Configuration

`config/config.yaml` is the primary config. Asset-specific overrides live in `config/assets/{asset}.yaml`. The active asset is set by `active_asset: "eurusd"` at the top of `config.yaml`.

Per-agent model overrides are under `agents:` in `config.yaml`. To switch a model, edit that section — do not hardcode model names in agent code.

LLM provider routing (`ollama` / `anthropic` / `openai`) is handled in `agents/base.py:_build_llm()`.

### Database

SQLite at `trading_bot.db`. Seven tables defined in `core/memory/models.py`:

| Table | Purpose |
|-------|---------|
| `alarms` | Price levels set by Analyst; `status`: active → triggered/expired/cancelled |
| `trades` | Full trade lifecycle; `close()` method calculates P&L automatically |
| `knowledge_base` | Rules/insights written by Evaluator; loaded into all agent prompts via `load_context()` |
| `evaluator_reports` | Weekly metrics snapshots |
| `research_reports` | Daily research summaries |
| `analysis_runs` | Analyst TA run metadata |
| `agent_logs` | Per-agent action log with input/output JSON |

Session pattern everywhere:
```python
from core.memory.database import get_session
with get_session() as session:
    session.query(Model).filter(...).all()
    session.add(new_record)
    # auto-commits on exit, rolls back on exception
```

### Strategies

`strategies/` holds `StrategyConfig` dataclasses per asset, defining confluence weights and signal parameters. `strategies/__init__.py` exposes `get_strategy(symbol)` as a registry lookup. The Analyst Agent calls `get_strategy()` to get asset-specific scoring weights when evaluating indicator confluence.

### Technical indicators

`core/indicators/` has seven modules (trend, momentum, volume, levels, pivots, candles, confluence). The `confluence.py` module aggregates scores from the others into a single signal, weighted by the active strategy config.

### Notifications

`notifications/telegram.py` — async `TelegramNotifier.send()` for outbound messages.
`notifications/telegram_commands.py` — `TelegramCommandHandler` handles `/improve`, `/show_alarms`, `/set_alarms` slash commands via polling.
`notifications/templates.py` — all Telegram message templates as Jinja2 strings.

`send_sync()` is safe to call from synchronous/threaded contexts (executor threads, scheduler jobs) — it detects whether a loop is running and uses `asyncio.run()` when not.

## Environment

Copy `.env.example` to `.env`. Required:
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — notifications
- `BRAVE_API_KEY` — news search (falls back to DuckDuckGo if missing)

Optional (only if switching from Ollama):
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`

IBKR connects to `127.0.0.1:7497` (paper) by default. Live trading uses port 7496.

## Key gotchas

- **qwen3 thinking mode**: qwen3 models in Ollama run with extended thinking enabled by default, which can consume the entire context window before producing output. Use `qwen2.5:32b` for the evaluator and trader agents — it produces structured JSON reliably and is already configured for the trader.
- **Async in threads**: Agents scheduled via APScheduler run in executor threads with no event loop. Always use `asyncio.run()` for one-shot async calls in synchronous agent nodes, not `get_event_loop().run_until_complete()`.
- **LLM JSON parsing**: All agents use `_extract_json()` in `agents/evaluator/agent.py` (or equivalent per-agent helpers) to strip markdown fences before `json.loads()`. If an LLM returns a list where a dict is expected, the caller must handle it — the parser won't.
