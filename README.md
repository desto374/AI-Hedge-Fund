# AI Hedge Fund

A CrewAI-based multi-agent trading prototype where specialist agents analyze a ticker, report to a portfolio manager, and the manager makes the final trading decision.

This repo is built for research and paper-trading workflows first. It is not production-ready autonomous trading software.

## What It Does

- Pulls live market data with `Alpaca`, with automatic fallback to `yfinance`
- Summarizes news sentiment for a ticker
- Reviews technical context and options-chain context
- Checks account and position state from Alpaca
- Runs a simple historical backtest for added manager context
- Lets the portfolio manager choose `buy`, `sell`, `reduce`, or `hold`
- Keeps execution separate so order submission can stay disabled until you are ready

## Agent Structure

- `market_research_analyst`: catalyst, macro, and price-action context
- `technical_analyst`: RSI, MACD direction, moving averages, Bollinger position
- `sentiment_analyst`: news and sentiment summary
- `options_strategist`: options-chain context and simple expression ideas
- `risk_manager`: position sizing and portfolio limits
- `strategy_backtester`: simple historical strategy review
- `portfolio_manager`: final decision-maker
- `execution_operator`: manual or paper-execution handoff

## Stack

- Python `3.10+`
- `CrewAI`
- `OpenAI` for agent reasoning
- `Alpaca` for paper trading, account state, and market data
- `yfinance` as read-only fallback market data

## Project Layout

```text
.
├── .env.example
├── pyproject.toml
├── README.md
├── src/ai_hedge_fund/
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── tools/
│   │   ├── __init__.py
│   │   └── custom_tools.py
│   ├── crew.py
│   └── main.py
└── tests/
    └── test_custom_tools.py
```

## Environment

Create a local `.env` from `.env.example`.

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
MODEL=gpt-4.1-mini
CREWAI_TRACING_ENABLED=false

MARKET_DATA_PROVIDER=alpaca

ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
ALPACA_DATA_BASE_URL=https://data.alpaca.markets/v2
ALPACA_STOCK_FEED=iex
ALPACA_OPTIONS_FEED=indicative
ALPACA_ENABLE_SUBMISSION=false
ALPACA_ALLOW_EXTENDED_HOURS=false
ALPACA_KILL_SWITCH=false
```

Important:

- Keep `.env` local and never commit it
- Leave `ALPACA_ENABLE_SUBMISSION=false` until you explicitly want paper orders submitted
- Rotate any key that has ever been pasted into chat, a screenshot, or a public file

## Installation

Standard install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

If you are using the validated local setup from this project:

```bash
./.venv311/bin/pip install -e .[dev]
```

## Running The Crew

Safe manual-mode run:

```bash
HOME='/Volumes/new life /Websites/AI Hedge Fund' \
CREWAI_STORAGE_DIR='/Volumes/new life /Websites/AI Hedge Fund/.crewai_storage' \
CREWAI_TRACING_ENABLED=false \
PYTHONPATH=src \
./.venv311/bin/python -m ai_hedge_fund.main --ticker AAPL --execution-mode manual
```

Example paper-mode run with execution submission still disabled:

```bash
HOME='/Volumes/new life /Websites/AI Hedge Fund' \
CREWAI_STORAGE_DIR='/Volumes/new life /Websites/AI Hedge Fund/.crewai_storage' \
CREWAI_TRACING_ENABLED=false \
PYTHONPATH=src \
./.venv311/bin/python -m ai_hedge_fund.main \
  --ticker NVDA \
  --thesis "AI demand remains strong into the next earnings print" \
  --macro-view "Growth remains favored while rates stabilize" \
  --upcoming-event "Quarterly earnings" \
  --execution-mode paper \
  --order-type market \
  --time-in-force day
```

## Data Behavior

- `MARKET_DATA_PROVIDER=alpaca`: uses Alpaca first, then falls back to `yfinance`
- `MARKET_DATA_PROVIDER=yfinance`: forces `yfinance` only

Current live-data coverage includes:

- market snapshot and recent bars
- derived technical context
- news sentiment summary
- options-chain summary
- Alpaca account and positions state
- simple historical backtest context

## Safety Controls

- `ALPACA_ENABLE_SUBMISSION=false` prevents paper-order submission
- `ALPACA_KILL_SWITCH=true` blocks all submissions
- `ALPACA_ALLOW_EXTENDED_HOURS=false` blocks submissions outside regular market hours
- duplicate open orders with the same ticker, side, and quantity are blocked

## Testing

Run tests with:

```bash
HOME='/Volumes/new life /Websites/AI Hedge Fund' \
CREWAI_STORAGE_DIR='/Volumes/new life /Websites/AI Hedge Fund/.crewai_storage' \
PYTHONPATH=src \
./.venv311/bin/pytest -q
```

Current status:

- tool-layer tests pass
- CrewAI boot path is verified
- full live execution depends on working outbound API access from your machine

## Current Limits

- sentiment is still heuristic, not a dedicated NLP model
- backtesting is intentionally simple
- options analysis is summary-level, not advanced strategy modeling
- this should be treated as a prototype for research and paper testing, not unattended live trading

## GitHub Push Checklist

- commit `.env.example`, not `.env`
- confirm `.env`, `.venv311`, `.crewai_storage`, `Library`, and `.config` are ignored
- rotate any key that has been exposed
- push only after checking `git status`

## License

Add a license before public release if you plan to open-source the repo.
