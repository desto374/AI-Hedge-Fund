from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Type
from urllib import error, request
from urllib.parse import urlencode

try:
    from crewai.tools import BaseTool
except ModuleNotFoundError:
    class BaseTool:  # type: ignore[no-redef]
        """Minimal fallback so local tool tests can run without CrewAI installed."""

        name: str = ""
        description: str = ""
from pydantic import BaseModel, Field

VALID_ACTIONS = {"buy", "sell", "hold", "reduce"}
VALID_EXECUTION_MODES = {"manual", "paper", "live"}
VALID_ORDER_TYPES = {"market", "limit"}
VALID_MARKET_DATA_PROVIDERS = {"alpaca", "yfinance"}
DEFAULT_ALPACA_DATA_BASE_URL = "https://data.alpaca.markets/v2"


class MarketResearchInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    thesis: str = Field(..., description="Initial thesis or catalyst to evaluate.")
    macro_view: str = Field(..., description="Macro context for the trade idea.")
    upcoming_event: str = Field(..., description="Closest relevant company or macro event.")


class CandidateDiscoveryInput(BaseModel):
    ticker: str = Field(default="", description="Optional manually provided ticker symbol.")
    auto_discover: bool = Field(
        default=False, description="Whether the system should discover a ticker automatically."
    )
    discovery_window_days: int = Field(
        default=7, description="Maximum number of days until earnings for eligible candidates."
    )
    discovery_max_symbols: int = Field(
        default=75, description="Maximum number of symbols to scan during discovery."
    )
    discovery_min_price: float = Field(
        default=10.0, description="Minimum share price for eligible candidates."
    )
    discovery_universe_file: str = Field(
        default="", description="Optional newline or CSV file with symbols to scan."
    )
    discovery_min_score: float = Field(
        default=2.5, description="Minimum discovery score required for an auto-discovered candidate."
    )
    discovery_retry_attempts: int = Field(
        default=2, description="How many automatic discovery retries to attempt after rejection."
    )
    thesis: str = Field(
        default="", description="Current thesis text, used when manual ticker mode is active."
    )
    upcoming_event: str = Field(
        default="", description="Current event text, used when manual ticker mode is active."
    )


class CandidateDiscoveryTool(BaseTool):
    name: str = "candidate_discovery_tool"
    description: str = "Select a candidate ticker ahead of earnings or confirm the manually supplied ticker."
    args_schema: Type[BaseModel] = CandidateDiscoveryInput

    def _run(
        self,
        ticker: str = "",
        auto_discover: bool = False,
        discovery_window_days: int = 7,
        discovery_max_symbols: int = 75,
        discovery_min_price: float = 10.0,
        discovery_universe_file: str = "",
        discovery_min_score: float = 2.5,
        discovery_retry_attempts: int = 2,
        thesis: str = "",
        upcoming_event: str = "",
    ) -> str:
        normalized_ticker = ticker.strip().upper()
        force_auto_discover = os.getenv("AI_HEDGE_FUND_FORCE_AUTO_DISCOVER", "false").lower() == "true"
        if force_auto_discover and not auto_discover:
            return (
                "Discovery mode: auto\n"
                "Discovery status: rejected\n"
                "Rejection reason: runtime is locked to auto-discover, so manual override is not allowed.\n"
                "Instruction: Re-run candidate_discovery_tool with auto_discover=true."
            )
        if not auto_discover:
            payload = {
                "discovery_mode": "manual",
                "discovery_status": "manual",
                "selected_ticker": normalized_ticker,
                "company_name": normalized_ticker,
                "earnings_date": "",
                "days_until_earnings": 0,
                "discovery_score": 0.0,
                "discovery_attempts_used": 1,
                "price": 0.0,
                "momentum_20d_pct": 0.0,
                "momentum_60d_pct": 0.0,
                "news_score": 0.0,
                "upcoming_event": upcoming_event.strip(),
                "thesis": thesis.strip(),
                "instruction": "Use the selected ticker for all downstream analysis.",
            }
            _write_structured_output("discovery_selection.json", payload)
            return (
                "Discovery mode: manual\n"
                f"Selected ticker: {normalized_ticker}\n"
                f"Upcoming event: {upcoming_event.strip()}\n"
                f"Thesis: {thesis.strip()}\n"
                "Instruction: Use the selected ticker for all downstream analysis.\n"
                f"Structured payload: {json.dumps(payload, sort_keys=True)}"
            )

        candidate, attempts_used = _discover_with_retries(
            discovery_window_days=discovery_window_days,
            discovery_max_symbols=discovery_max_symbols,
            discovery_min_price=discovery_min_price,
            discovery_universe_file=discovery_universe_file,
            discovery_min_score=discovery_min_score,
            discovery_retry_attempts=discovery_retry_attempts,
        )
        if candidate.score < discovery_min_score:
            payload = {
                "discovery_mode": "auto",
                "discovery_status": "rejected",
                "selected_ticker": candidate.ticker,
                "company_name": candidate.company_name,
                "earnings_date": candidate.earnings_date.isoformat(),
                "days_until_earnings": candidate.days_until_earnings,
                "discovery_score": candidate.score,
                "discovery_attempts_used": attempts_used,
                "price": candidate.price,
                "momentum_20d_pct": candidate.momentum_20d_pct,
                "momentum_60d_pct": candidate.momentum_60d_pct,
                "news_score": candidate.news_score,
                "upcoming_event": candidate.upcoming_event,
                "thesis": candidate.thesis,
                "instruction": "Do not continue with trade analysis. Hold and wait for a stronger setup.",
            }
            _write_structured_output("discovery_selection.json", payload)
            return (
                "Discovery mode: auto\n"
                "Discovery status: rejected\n"
                f"Discovery attempts used: {attempts_used}\n"
                f"Rejection reason: best candidate score {candidate.score:.2f} is below "
                f"minimum threshold {discovery_min_score:.2f}\n"
                f"Selected ticker: {candidate.ticker}\n"
                f"Company: {candidate.company_name}\n"
                f"Earnings date: {candidate.earnings_date.isoformat()}\n"
                "Instruction: Do not continue with trade analysis. Hold and wait for a stronger setup.\n"
                f"Structured payload: {json.dumps(payload, sort_keys=True)}"
            )
        payload = {
            "discovery_mode": "auto",
            "discovery_status": "accepted",
            "selected_ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "earnings_date": candidate.earnings_date.isoformat(),
            "days_until_earnings": candidate.days_until_earnings,
            "discovery_score": candidate.score,
            "discovery_attempts_used": attempts_used,
            "price": candidate.price,
            "momentum_20d_pct": candidate.momentum_20d_pct,
            "momentum_60d_pct": candidate.momentum_60d_pct,
            "news_score": candidate.news_score,
            "upcoming_event": candidate.upcoming_event,
            "thesis": candidate.thesis,
            "instruction": "Use the selected ticker for all downstream analysis.",
        }
        _write_structured_output("discovery_selection.json", payload)
        return (
            "Discovery mode: auto\n"
            "Discovery status: accepted\n"
            f"Discovery attempts used: {attempts_used}\n"
            f"Selected ticker: {candidate.ticker}\n"
            f"Company: {candidate.company_name}\n"
            f"Earnings date: {candidate.earnings_date.isoformat()}\n"
            f"Days until earnings: {candidate.days_until_earnings}\n"
            f"Discovery score: {candidate.score:.2f}\n"
            f"Price: {candidate.price:.2f}\n"
            f"20d momentum: {candidate.momentum_20d_pct:.2f}%\n"
            f"60d momentum: {candidate.momentum_60d_pct:.2f}%\n"
            f"News score: {candidate.news_score:.2f}\n"
            f"Upcoming event: {candidate.upcoming_event}\n"
            f"Thesis: {candidate.thesis}\n"
            "Instruction: Use the selected ticker for all downstream analysis.\n"
            f"Structured payload: {json.dumps(payload, sort_keys=True)}"
        )


class TradeContextInput(BaseModel):
    include_decision: bool = Field(
        default=False, description="Whether to include the portfolio decision JSON if available."
    )


class TradeContextTool(BaseTool):
    name: str = "trade_context_tool"
    description: str = "Load persisted structured discovery and decision context for downstream tasks."
    args_schema: Type[BaseModel] = TradeContextInput

    def _run(self, include_decision: bool = False) -> str:
        discovery = _read_structured_output("discovery_selection.json")
        if discovery is None:
            return "Trade context unavailable. discovery_selection.json has not been created yet."

        lines = [
            "Structured trade context",
            f"Discovery status: {discovery.get('discovery_status', 'unknown')}",
            f"Selected ticker: {discovery.get('selected_ticker', '')}",
            f"Upcoming event: {discovery.get('upcoming_event', '')}",
            f"Thesis: {discovery.get('thesis', '')}",
        ]
        if include_decision:
            decision = _read_structured_output("portfolio_decision.json")
            if decision is None:
                lines.append("Portfolio decision: unavailable")
            else:
                lines.extend(
                    [
                        f"Decision ticker: {decision.get('ticker', '')}",
                        f"Final action: {decision.get('final_action', '')}",
                        f"Decision confidence: {decision.get('confidence', '')}",
                    ]
                )
        return "\n".join(lines)

class MarketResearchTool(BaseTool):
    name: str = "market_research_tool"
    description: str = "Create a concise market research brief from supplied catalysts."
    args_schema: Type[BaseModel] = MarketResearchInput

    def _run(
        self, ticker: str, thesis: str, macro_view: str, upcoming_event: str
    ) -> str:
        normalized_ticker = ticker.strip().upper()
        return (
            f"Ticker: {normalized_ticker}\n"
            f"Core thesis: {thesis.strip()}\n"
            f"Macro backdrop: {macro_view.strip()}\n"
            f"Upcoming event: {upcoming_event.strip()}\n"
            "Research stance: Validate whether the thesis still holds after the next "
            "event and whether the macro backdrop supports risk-taking."
        )


class NewsSentimentInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    limit: int = Field(default=5, description="Maximum number of articles to inspect.")


class NewsSentimentTool(BaseTool):
    name: str = "news_sentiment_tool"
    description: str = "Fetch recent news and summarize headline sentiment for a ticker."
    args_schema: Type[BaseModel] = NewsSentimentInput

    def _run(self, ticker: str, limit: int = 5) -> str:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return "News sentiment unavailable. Ticker cannot be empty."
        if limit <= 0:
            return "News sentiment unavailable. limit must be greater than zero."

        provider = os.getenv("MARKET_DATA_PROVIDER", "alpaca").strip().lower()
        if provider == "yfinance":
            return self._run_yfinance(normalized_ticker, limit)

        alpaca_result = self._run_alpaca(normalized_ticker, limit)
        if not alpaca_result.startswith("News sentiment unavailable."):
            return alpaca_result

        yfinance_result = self._run_yfinance(normalized_ticker, limit)
        if not yfinance_result.startswith("News sentiment unavailable."):
            return f"{yfinance_result}\nFallback note: Alpaca news failed, so yfinance news was used instead."
        return f"{alpaca_result}\nFallback result: {yfinance_result}"

    def _run_alpaca(self, ticker: str, limit: int) -> str:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return "News sentiment unavailable. Missing ALPACA_API_KEY or ALPACA_SECRET_KEY."

        query = urlencode({"symbols": ticker, "limit": limit, "include_content": "false"})
        result = _fetch_json_with_headers(
            f"{os.getenv('ALPACA_DATA_BASE_URL', DEFAULT_ALPACA_DATA_BASE_URL).rstrip('/')}/news?{query}",
            {
                "accept": "application/json",
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
        )
        if isinstance(result, str):
            return f"News sentiment unavailable. {result}"

        items = result.get("news", [])
        if not items:
            return f"News sentiment unavailable. No Alpaca news returned for {ticker}."
        return _format_news_summary("alpaca", ticker, items[:limit])

    def _run_yfinance(self, ticker: str, limit: int) -> str:
        if find_spec("yfinance") is None:
            return "News sentiment unavailable. Install yfinance to use the fallback news provider."

        import yfinance as yf

        items = getattr(yf.Ticker(ticker), "news", []) or []
        if not items:
            return f"News sentiment unavailable. No yfinance news returned for {ticker}."
        normalized_items = []
        for item in items[:limit]:
            content = item.get("content") or {}
            normalized_items.append(
                {
                    "headline": content.get("title") or item.get("title", ""),
                    "summary": content.get("summary") or item.get("summary", ""),
                    "source": content.get("provider", {}).get("displayName") or item.get("publisher", "unknown"),
                    "created_at": content.get("pubDate") or item.get("providerPublishTime", ""),
                    "url": content.get("canonicalUrl", {}).get("url") or item.get("link", ""),
                }
            )
        return _format_news_summary("yfinance", ticker, normalized_items)


class LiveMarketDataInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    bar_limit: int = Field(default=30, description="Number of recent minute bars to fetch.")


class LiveMarketDataTool(BaseTool):
    name: str = "live_market_data_tool"
    description: str = "Fetch live stock data from the configured market data provider."
    args_schema: Type[BaseModel] = LiveMarketDataInput

    def _run(self, ticker: str, bar_limit: int = 30) -> str:
        normalized_ticker = ticker.strip().upper()
        validation_error = self._validate_request(normalized_ticker, bar_limit)
        if validation_error is not None:
            return validation_error

        provider = os.getenv("MARKET_DATA_PROVIDER", "alpaca").strip().lower()
        if provider not in VALID_MARKET_DATA_PROVIDERS:
            return "Live market data unavailable. MARKET_DATA_PROVIDER must be alpaca or yfinance."
        if provider == "yfinance":
            return self._run_yfinance(normalized_ticker, bar_limit)
        alpaca_result = self._run_alpaca(normalized_ticker, bar_limit)
        if not alpaca_result.startswith("Live market data unavailable.") and not alpaca_result.startswith(
            "Live market data request failed"
        ):
            return alpaca_result

        yfinance_result = self._run_yfinance(normalized_ticker, bar_limit)
        if not yfinance_result.startswith("Live market data unavailable."):
            return f"{yfinance_result}\nFallback note: Alpaca failed, so yfinance was used instead."
        return f"{alpaca_result}\nFallback result: {yfinance_result}"

    def _run_alpaca(self, normalized_ticker: str, bar_limit: int) -> str:

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return "Live market data unavailable. Missing ALPACA_API_KEY or ALPACA_SECRET_KEY."

        base_url = os.getenv("ALPACA_DATA_BASE_URL", DEFAULT_ALPACA_DATA_BASE_URL).rstrip("/")
        snapshot_result = self._fetch_json(
            f"{base_url}/stocks/{normalized_ticker}/snapshot",
            api_key,
            secret_key,
        )
        if isinstance(snapshot_result, str):
            return snapshot_result

        bars_query = urlencode(
            {
                "symbols": normalized_ticker,
                "timeframe": "1Min",
                "limit": bar_limit,
                "feed": os.getenv("ALPACA_STOCK_FEED", "iex"),
            }
        )
        bars_result = self._fetch_json(
            f"{base_url}/stocks/bars?{bars_query}",
            api_key,
            secret_key,
        )
        if isinstance(bars_result, str):
            return bars_result

        bars = bars_result.get("bars", {}).get(normalized_ticker, [])
        if not bars:
            return f"Live market data unavailable. No recent bars returned for {normalized_ticker}."

        minute_bar = snapshot_result.get("minuteBar") or {}
        daily_bar = snapshot_result.get("dailyBar") or {}
        previous_daily_bar = snapshot_result.get("prevDailyBar") or {}
        latest_trade = snapshot_result.get("latestTrade") or {}
        latest_quote = snapshot_result.get("latestQuote") or {}

        closes = [float(bar["c"]) for bar in bars if "c" in bar]
        return _format_market_data_summary(
            provider="alpaca",
            ticker=normalized_ticker,
            closes=closes,
            last_trade_price=float(latest_trade.get("p", 0.0)),
            bid=float(latest_quote.get("bp", 0.0)),
            ask=float(latest_quote.get("ap", 0.0)),
            minute_close=float(minute_bar.get("c", 0.0)),
            daily_open=float(daily_bar.get("o", 0.0)),
            daily_high=float(daily_bar.get("h", 0.0)),
            daily_low=float(daily_bar.get("l", 0.0)),
            previous_close=float(previous_daily_bar.get("c", 0.0)),
        )

    def _run_yfinance(self, normalized_ticker: str, bar_limit: int) -> str:
        if find_spec("yfinance") is None:
            return "Live market data unavailable. Install yfinance or switch MARKET_DATA_PROVIDER=alpaca."

        import yfinance as yf

        history = yf.Ticker(normalized_ticker).history(period="5d", interval="1m", auto_adjust=False)
        if history.empty:
            return f"Live market data unavailable. No recent yfinance bars returned for {normalized_ticker}."

        closes = [float(value) for value in history["Close"].dropna().tail(bar_limit).tolist()]
        opens = history["Open"].dropna()
        highs = history["High"].dropna()
        lows = history["Low"].dropna()
        if not closes:
            return f"Live market data unavailable. No recent yfinance closes returned for {normalized_ticker}."

        last_trade_price = closes[-1]
        previous_close = float(closes[-2]) if len(closes) > 1 else last_trade_price
        return _format_market_data_summary(
            provider="yfinance",
            ticker=normalized_ticker,
            closes=closes,
            last_trade_price=last_trade_price,
            bid=0.0,
            ask=0.0,
            minute_close=last_trade_price,
            daily_open=float(opens.iloc[-1]) if not opens.empty else last_trade_price,
            daily_high=float(highs.iloc[-1]) if not highs.empty else last_trade_price,
            daily_low=float(lows.iloc[-1]) if not lows.empty else last_trade_price,
            previous_close=previous_close,
        )

    def _validate_request(self, ticker: str, bar_limit: int) -> str | None:
        if not ticker:
            return "Live market data unavailable. Ticker cannot be empty."
        if bar_limit <= 1:
            return "Live market data unavailable. bar_limit must be greater than 1."
        return None

    def _fetch_json(self, url: str, api_key: str, secret_key: str) -> dict[str, Any] | str:
        return _fetch_json_with_headers(
            url,
            {
                "accept": "application/json",
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
        )


class LiveOptionsChainInput(BaseModel):
    ticker: str = Field(..., description="Underlying ticker symbol.")
    expiration_limit: int = Field(default=1, description="How many expirations to inspect.")


class LiveOptionsChainTool(BaseTool):
    name: str = "live_options_chain_tool"
    description: str = "Fetch a live options chain summary from Alpaca or yfinance."
    args_schema: Type[BaseModel] = LiveOptionsChainInput

    def _run(self, ticker: str, expiration_limit: int = 1) -> str:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return "Options chain unavailable. Ticker cannot be empty."
        provider = os.getenv("MARKET_DATA_PROVIDER", "alpaca").strip().lower()
        if provider == "yfinance":
            return self._run_yfinance(normalized_ticker, expiration_limit)

        alpaca_result = self._run_alpaca(normalized_ticker)
        if not alpaca_result.startswith("Options chain unavailable."):
            return alpaca_result

        yfinance_result = self._run_yfinance(normalized_ticker, expiration_limit)
        if not yfinance_result.startswith("Options chain unavailable."):
            return f"{yfinance_result}\nFallback note: Alpaca options failed, so yfinance options were used instead."
        return f"{alpaca_result}\nFallback result: {yfinance_result}"

    def _run_alpaca(self, ticker: str) -> str:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return "Options chain unavailable. Missing ALPACA_API_KEY or ALPACA_SECRET_KEY."

        feed = os.getenv("ALPACA_OPTIONS_FEED", "indicative")
        result = _fetch_json_with_headers(
            f"{os.getenv('ALPACA_DATA_BASE_URL', DEFAULT_ALPACA_DATA_BASE_URL).rstrip('/')}/options/snapshots/{ticker}?feed={feed}",
            {
                "accept": "application/json",
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
        )
        if isinstance(result, str):
            return f"Options chain unavailable. {result}"

        snapshots = result.get("snapshots", {})
        if not snapshots:
            return f"Options chain unavailable. No Alpaca option snapshots returned for {ticker}."

        parsed = []
        for symbol, snapshot in list(snapshots.items())[:10]:
            latest_quote = snapshot.get("latestQuote") or {}
            greeks = snapshot.get("greeks") or {}
            parsed.append(
                {
                    "symbol": symbol,
                    "mid": _mid_price(float(latest_quote.get("bp", 0.0)), float(latest_quote.get("ap", 0.0))),
                    "iv": float(snapshot.get("impliedVolatility", 0.0) or 0.0),
                    "delta": float(greeks.get("delta", 0.0) or 0.0),
                }
            )
        return _format_options_summary("alpaca", ticker, parsed)

    def _run_yfinance(self, ticker: str, expiration_limit: int) -> str:
        if find_spec("yfinance") is None:
            return "Options chain unavailable. Install yfinance to use the fallback options provider."

        import yfinance as yf

        yf_ticker = yf.Ticker(ticker)
        expirations = list(getattr(yf_ticker, "options", [])[: max(expiration_limit, 1)])
        if not expirations:
            return f"Options chain unavailable. No yfinance expirations returned for {ticker}."

        chain = yf_ticker.option_chain(expirations[0])
        parsed = []
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            if getattr(frame, "empty", True):
                continue
            top = frame.sort_values("openInterest", ascending=False).head(2)
            for _, row in top.iterrows():
                parsed.append(
                    {
                        "symbol": str(row.get("contractSymbol", "")),
                        "side": side,
                        "mid": _mid_price(float(row.get("bid", 0.0)), float(row.get("ask", 0.0))),
                        "iv": float(row.get("impliedVolatility", 0.0) or 0.0),
                        "open_interest": int(row.get("openInterest", 0) or 0),
                    }
                )
        if not parsed:
            return f"Options chain unavailable. No populated yfinance chain rows returned for {ticker}."
        return _format_options_summary("yfinance", ticker, parsed, expiration=expirations[0])


class PortfolioStateInput(BaseModel):
    ticker: str = Field(default="", description="Optional ticker to highlight if already held.")


class PortfolioStateTool(BaseTool):
    name: str = "portfolio_state_tool"
    description: str = "Fetch Alpaca account and positions state for the risk manager."
    args_schema: Type[BaseModel] = PortfolioStateInput

    def _run(self, ticker: str = "") -> str:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return "Portfolio state unavailable. Missing ALPACA_API_KEY or ALPACA_SECRET_KEY."

        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2").rstrip("/")
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        account = _fetch_json_with_headers(f"{base_url}/account", headers)
        if isinstance(account, str):
            return f"Portfolio state unavailable. {account}"
        positions = _fetch_json_with_headers(f"{base_url}/positions", headers)
        if isinstance(positions, str):
            return f"Portfolio state unavailable. {positions}"

        focus_ticker = ticker.strip().upper()
        focus_position = next((p for p in positions if p.get("symbol") == focus_ticker), None)
        positions_preview = ", ".join(
            f"{position.get('symbol')}:{position.get('qty')}" for position in positions[:5]
        ) or "no open positions"
        focus_line = (
            f"\nFocus position: {focus_position.get('symbol')} qty {focus_position.get('qty')} market value {focus_position.get('market_value')}"
            if focus_position
            else ""
        )
        return (
            f"Account status: {account.get('status', 'unknown')}\n"
            f"Equity: {float(account.get('equity', 0.0)):.2f}\n"
            f"Cash: {float(account.get('cash', 0.0)):.2f}\n"
            f"Buying power: {float(account.get('buying_power', 0.0)):.2f}\n"
            f"Day trade count: {account.get('daytrade_count', 0)}\n"
            f"Open positions: {len(positions)}\n"
            f"Positions preview: {positions_preview}"
            f"{focus_line}"
        )


class BacktestInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol to backtest.")
    period: str = Field(default="1y", description="History period such as 6mo, 1y, or 2y.")


class BacktestTool(BaseTool):
    name: str = "backtest_tool"
    description: str = "Run a simple moving-average crossover backtest on historical data."
    args_schema: Type[BaseModel] = BacktestInput

    def _run(self, ticker: str, period: str = "1y") -> str:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return "Backtest unavailable. Ticker cannot be empty."
        closes = self._fetch_history(normalized_ticker, period)
        if isinstance(closes, str):
            return closes
        if len(closes) < 60:
            return f"Backtest unavailable. Not enough history returned for {normalized_ticker}."

        cash = 1.0
        position = 0.0
        entry_price = 0.0
        trades = 0
        wins = 0
        equity_curve: list[float] = []

        for index in range(len(closes)):
            price = closes[index]
            short_ma = _simple_moving_average(closes[max(0, index - 19) : index + 1])
            long_ma = _simple_moving_average(closes[max(0, index - 49) : index + 1])
            if index >= 49 and position == 0.0 and short_ma > long_ma:
                position = cash / price
                cash = 0.0
                entry_price = price
                trades += 1
            elif index >= 49 and position > 0.0 and short_ma < long_ma:
                cash = position * price
                if price > entry_price:
                    wins += 1
                position = 0.0
            equity_curve.append(cash if position == 0.0 else position * price)

        final_equity = cash if position == 0.0 else position * closes[-1]
        benchmark_return = ((closes[-1] / closes[0]) - 1) * 100
        strategy_return = (final_equity - 1.0) * 100
        max_drawdown = _max_drawdown(equity_curve) * 100
        win_rate = ((wins / trades) * 100) if trades else 0.0
        return (
            f"Ticker: {normalized_ticker}\n"
            f"Backtest period: {period}\n"
            f"Strategy: 20/50 moving-average crossover\n"
            f"Strategy return: {strategy_return:.2f}%\n"
            f"Benchmark return: {benchmark_return:.2f}%\n"
            f"Trades: {trades}\n"
            f"Win rate: {win_rate:.2f}%\n"
            f"Max drawdown: {max_drawdown:.2f}%"
        )

    def _fetch_history(self, ticker: str, period: str) -> list[float] | str:
        if find_spec("yfinance") is None:
            return "Backtest unavailable. Install yfinance to run historical backtests."

        import yfinance as yf

        history = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        if history.empty:
            return f"Backtest unavailable. No historical bars returned for {ticker}."
        return [float(value) for value in history["Close"].dropna().tolist()]


class TechnicalIndicatorInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    price: float = Field(..., description="Current share price.")
    rsi: float = Field(..., description="Relative Strength Index value.")
    macd_signal: str = Field(..., description="MACD direction: bullish, bearish, or flat.")
    moving_average_gap_pct: float = Field(
        ..., description="Percent above or below the key moving average."
    )
    bollinger_position: str = Field(
        ..., description="Bollinger location: upper, middle, or lower."
    )


class TechnicalIndicatorTool(BaseTool):
    name: str = "technical_indicator_tool"
    description: str = "Score a trade setup from common technical indicators."
    args_schema: Type[BaseModel] = TechnicalIndicatorInput

    def _run(
        self,
        ticker: str,
        price: float,
        rsi: float,
        macd_signal: str,
        moving_average_gap_pct: float,
        bollinger_position: str,
    ) -> str:
        score = 0
        normalized_macd = macd_signal.strip().lower()
        normalized_band = bollinger_position.strip().lower()

        if 45 <= rsi <= 65:
            score += 1
        elif rsi > 70 or rsi < 30:
            score -= 1

        if normalized_macd == "bullish":
            score += 1
        elif normalized_macd == "bearish":
            score -= 1

        if moving_average_gap_pct > 0:
            score += 1
        elif moving_average_gap_pct < 0:
            score -= 1

        if normalized_band == "upper":
            score += 1
        elif normalized_band == "lower":
            score -= 1

        if score >= 2:
            setup = "bullish"
        elif score <= -2:
            setup = "bearish"
        else:
            setup = "mixed"

        return (
            f"Ticker: {ticker.strip().upper()}\n"
            f"Price: {price:.2f}\n"
            f"Technical score: {score}\n"
            f"Setup: {setup}\n"
            f"RSI: {rsi:.1f}\n"
            f"MACD: {normalized_macd}\n"
            f"Moving average gap: {moving_average_gap_pct:.2f}%\n"
            f"Bollinger position: {normalized_band}"
        )


class SentimentSnapshotInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    news_sentiment: float = Field(..., description="News tone from -1 to 1.")
    social_sentiment: float = Field(..., description="Social tone from -1 to 1.")
    analyst_revision_trend: str = Field(
        ..., description="Analyst revisions: up, down, or flat."
    )


class SentimentSnapshotTool(BaseTool):
    name: str = "sentiment_snapshot_tool"
    description: str = "Summarize broad sentiment from normalized sentiment inputs."
    args_schema: Type[BaseModel] = SentimentSnapshotInput

    def _run(
        self,
        ticker: str,
        news_sentiment: float,
        social_sentiment: float,
        analyst_revision_trend: str,
    ) -> str:
        trend = analyst_revision_trend.strip().lower()
        score = (news_sentiment + social_sentiment) / 2
        if trend == "up":
            score += 0.2
        elif trend == "down":
            score -= 0.2

        if score >= 0.25:
            stance = "positive"
        elif score <= -0.25:
            stance = "negative"
        else:
            stance = "balanced"

        return (
            f"Ticker: {ticker.strip().upper()}\n"
            f"Sentiment score: {score:.2f}\n"
            f"Stance: {stance}\n"
            f"News sentiment: {news_sentiment:.2f}\n"
            f"Social sentiment: {social_sentiment:.2f}\n"
            f"Analyst revisions: {trend}"
        )


class OptionsStrategyInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    directional_bias: str = Field(..., description="Bullish, bearish, or neutral.")
    implied_volatility_pct: float = Field(..., description="Implied volatility percent.")
    days_to_expiry: int = Field(..., description="Target days to expiry.")


class OptionsStrategyTool(BaseTool):
    name: str = "options_strategy_tool"
    description: str = "Suggest a simple options structure from direction and volatility."
    args_schema: Type[BaseModel] = OptionsStrategyInput

    def _run(
        self,
        ticker: str,
        directional_bias: str,
        implied_volatility_pct: float,
        days_to_expiry: int,
    ) -> str:
        bias = directional_bias.strip().lower()

        if bias == "bullish" and implied_volatility_pct < 40:
            strategy = "long call"
        elif bias == "bullish":
            strategy = "bull call spread"
        elif bias == "bearish" and implied_volatility_pct < 40:
            strategy = "long put"
        elif bias == "bearish":
            strategy = "bear put spread"
        else:
            strategy = "cash-secured put or no options trade"

        return (
            f"Ticker: {ticker.strip().upper()}\n"
            f"Directional bias: {bias}\n"
            f"Implied volatility: {implied_volatility_pct:.1f}%\n"
            f"Days to expiry: {days_to_expiry}\n"
            f"Suggested options structure: {strategy}"
        )


class RiskBudgetInput(BaseModel):
    portfolio_value: float = Field(..., description="Portfolio account value.")
    max_position_pct: float = Field(..., description="Maximum position percent of portfolio.")
    stop_loss_pct: float = Field(..., description="Planned stop loss percent.")
    conviction: float = Field(..., description="Conviction score from 0 to 1.")
    price: float = Field(..., description="Current asset price.")


class RiskBudgetTool(BaseTool):
    name: str = "risk_budget_tool"
    description: str = "Calculate a practical position size from risk budget inputs."
    args_schema: Type[BaseModel] = RiskBudgetInput

    def _run(
        self,
        portfolio_value: float,
        max_position_pct: float,
        stop_loss_pct: float,
        conviction: float,
        price: float,
    ) -> str:
        max_notional = portfolio_value * (max_position_pct / 100)
        adjusted_notional = max_notional * max(0.1, min(conviction, 1.0))
        risk_per_share = max(price * (stop_loss_pct / 100), 0.01)
        share_limit = int(adjusted_notional // price) if price > 0 else 0
        loss_limit = int((portfolio_value * 0.01) // risk_per_share)
        shares = max(0, min(share_limit, loss_limit))

        return (
            f"Portfolio value: {portfolio_value:.2f}\n"
            f"Max position: {max_position_pct:.2f}%\n"
            f"Stop loss: {stop_loss_pct:.2f}%\n"
            f"Conviction: {conviction:.2f}\n"
            f"Recommended shares: {shares}\n"
            f"Estimated notional: {shares * price:.2f}\n"
            f"Estimated max loss at stop: {shares * risk_per_share:.2f}"
        )


class ExecutionPlanInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol under review.")
    action: str = Field(..., description="Buy, sell, hold, or reduce.")
    shares: int = Field(..., description="Recommended share quantity.")
    mode: str = Field(..., description="Execution mode: manual, paper, or live.")
    broker: str = Field(..., description="Broker or execution venue.")
    discovery_status: str = Field(
        default="accepted", description="Discovery status: accepted, rejected, or manual."
    )
    order_type: str = Field(
        default="market", description="Order type such as market or limit."
    )
    time_in_force: str = Field(default="day", description="Time in force such as day or gtc.")
    limit_price: Optional[float] = Field(
        default=None, description="Optional limit price when using limit orders."
    )


class ExecutionPlanTool(BaseTool):
    name: str = "execution_plan_tool"
    description: str = "Generate an execution handoff without making the trade decision."
    args_schema: Type[BaseModel] = ExecutionPlanInput

    def _run(
        self,
        ticker: str,
        action: str,
        shares: int,
        mode: str,
        broker: str,
        discovery_status: str = "accepted",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> str:
        normalized_mode = mode.strip().lower()
        normalized_action = action.strip().lower()
        normalized_broker = broker.strip()
        normalized_discovery_status = discovery_status.strip().lower()
        normalized_order_type = order_type.strip().lower()
        normalized_tif = time_in_force.strip().lower()
        validation_error = self._validate_execution_request(
            ticker=ticker.strip(),
            action=normalized_action,
            shares=shares,
            mode=normalized_mode,
            discovery_status=normalized_discovery_status,
            order_type=normalized_order_type,
            limit_price=limit_price,
        )
        broker_line = normalized_broker if normalized_broker else "No broker connected"
        execution_note = (
            "No API integration required."
            if normalized_mode == "manual"
            else "Connect a brokerage API such as Alpaca for paper or live execution."
        )
        submission_status = validation_error or "Not submitted."

        if validation_error is None and normalized_mode in {"paper", "live"} and normalized_action in {
            "buy",
            "sell",
        }:
            submission_status = self._maybe_submit_alpaca_order(
                ticker=ticker.strip().upper(),
                action=normalized_action,
                shares=shares,
                mode=normalized_mode,
                broker=normalized_broker,
                order_type=normalized_order_type,
                time_in_force=normalized_tif,
                limit_price=limit_price,
            )
        elif validation_error is None and normalized_action in {"hold", "reduce"}:
            submission_status = (
                "Not submitted. Manual review required for hold/reduce decisions."
            )

        limit_line = (
            f"\nLimit price: {limit_price:.2f}"
            if limit_price is not None and normalized_order_type == "limit"
            else ""
        )
        return (
            f"Ticker: {ticker.strip().upper()}\n"
            f"Action: {normalized_action}\n"
            f"Shares: {shares}\n"
            f"Mode: {normalized_mode}\n"
            f"Broker: {broker_line}\n"
            f"Discovery status: {normalized_discovery_status}\n"
            f"Order type: {normalized_order_type}\n"
            f"Time in force: {normalized_tif}"
            f"{limit_line}\n"
            f"Execution note: {execution_note}\n"
            f"Submission status: {submission_status}"
        )

    def _validate_execution_request(
        self,
        ticker: str,
        action: str,
        shares: int,
        mode: str,
        discovery_status: str,
        order_type: str,
        limit_price: Optional[float],
    ) -> str | None:
        if not ticker:
            return "Not submitted. Ticker cannot be empty."
        if action not in VALID_ACTIONS:
            return "Not submitted. Action must be one of: buy, sell, hold, reduce."
        if mode not in VALID_EXECUTION_MODES:
            return "Not submitted. Mode must be one of: manual, paper, live."
        if discovery_status not in {"accepted", "rejected", "manual"}:
            return "Not submitted. Discovery status must be one of: accepted, rejected, manual."
        if discovery_status == "rejected" and action != "hold":
            return "Not submitted. Discovery rejected the setup, so the action must remain hold."
        if order_type not in VALID_ORDER_TYPES:
            return "Not submitted. Order type must be one of: market, limit."
        if shares < 0:
            return "Not submitted. Share quantity cannot be negative."
        if order_type == "limit" and limit_price is None:
            return "Not submitted. Limit orders require a limit_price."
        if limit_price is not None and limit_price <= 0:
            return "Not submitted. Limit price must be greater than zero."
        return None

    def _maybe_submit_alpaca_order(
        self,
        ticker: str,
        action: str,
        shares: int,
        mode: str,
        broker: str,
        order_type: str,
        time_in_force: str,
        limit_price: Optional[float],
    ) -> str:
        if broker and broker.lower() != "alpaca":
            return f"Not submitted. Broker '{broker}' is not supported by this tool yet."

        if os.getenv("ALPACA_KILL_SWITCH", "false").lower() == "true":
            return "Not submitted. ALPACA_KILL_SWITCH is enabled."

        if os.getenv("ALPACA_ENABLE_SUBMISSION", "false").lower() != "true":
            return "Not submitted. Set ALPACA_ENABLE_SUBMISSION=true to allow order submission."

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL")

        if not api_key or not secret_key or not base_url:
            return (
                "Not submitted. Missing ALPACA_API_KEY, ALPACA_SECRET_KEY, or "
                "ALPACA_BASE_URL."
            )

        if mode == "paper" and "paper-api.alpaca.markets" not in base_url:
            return "Not submitted. Paper mode requires the Alpaca paper API base URL."

        if shares <= 0:
            return "Not submitted. Share quantity must be greater than zero."

        safety_error = self._validate_alpaca_execution_safety(
            ticker=ticker,
            action=action,
            shares=shares,
            base_url=base_url,
            api_key=api_key,
            secret_key=secret_key,
        )
        if safety_error is not None:
            return safety_error

        payload: dict[str, Any] = {
            "symbol": ticker,
            "qty": shares,
            "side": action,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if order_type == "limit":
            if limit_price is None:
                return "Not submitted. Limit orders require a limit_price."
            payload["limit_price"] = limit_price

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{base_url.rstrip('/')}/orders",
            data=body,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            return f"Submission failed with HTTP {exc.code}: {details}"
        except error.URLError as exc:
            return f"Submission failed: {exc.reason}"

        order_id = response_body.get("id", "unknown")
        order_status = response_body.get("status", "received")
        return f"Submitted to Alpaca. Order id: {order_id}. Status: {order_status}."

    def _validate_alpaca_execution_safety(
        self,
        ticker: str,
        action: str,
        shares: int,
        base_url: str,
        api_key: str,
        secret_key: str,
    ) -> str | None:
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        allow_extended_hours = os.getenv("ALPACA_ALLOW_EXTENDED_HOURS", "false").lower() == "true"
        if not allow_extended_hours:
            clock = _fetch_json_with_headers(f"{base_url.rstrip('/')}/clock", headers)
            if isinstance(clock, str):
                return f"Not submitted. Market hours check failed: {clock}"
            if not clock.get("is_open", False):
                return "Not submitted. Market is closed and ALPACA_ALLOW_EXTENDED_HOURS is not enabled."

        open_orders = _fetch_json_with_headers(
            f"{base_url.rstrip('/')}/orders?status=open&limit=100&direction=desc",
            headers,
        )
        if isinstance(open_orders, str):
            return f"Not submitted. Open-order safety check failed: {open_orders}"

        for order in open_orders:
            if (
                order.get("symbol") == ticker
                and str(order.get("side", "")).lower() == action
                and int(float(order.get("qty", 0) or 0)) == shares
            ):
                return (
                    "Not submitted. Duplicate open order detected for the same ticker, side, "
                    "and share quantity."
                )
        return None


def _simple_moving_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _calculate_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0

    gains: list[float] = []
    losses: list[float] = []
    for current, previous in zip(closes[1:], closes[:-1]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))


def _infer_macd_signal(closes: list[float]) -> str:
    if len(closes) < 26:
        return "flat"

    short_sma = _simple_moving_average(closes[-12:])
    long_sma = _simple_moving_average(closes[-26:])
    if short_sma > long_sma:
        return "bullish"
    if short_sma < long_sma:
        return "bearish"
    return "flat"


def _infer_bollinger_position(closes: list[float], period: int = 20) -> str:
    if len(closes) < period:
        return "middle"

    window = closes[-period:]
    mid = _simple_moving_average(window)
    variance = sum((value - mid) ** 2 for value in window) / period
    std_dev = variance ** 0.5
    upper = mid + (2 * std_dev)
    lower = mid - (2 * std_dev)
    latest = window[-1]

    if latest >= upper:
        return "upper"
    if latest <= lower:
        return "lower"
    return "middle"


def _format_market_data_summary(
    provider: str,
    ticker: str,
    closes: list[float],
    last_trade_price: float,
    bid: float,
    ask: float,
    minute_close: float,
    daily_open: float,
    daily_high: float,
    daily_low: float,
    previous_close: float,
) -> str:
    rsi = _calculate_rsi(closes)
    short_sma = _simple_moving_average(closes[-5:])
    long_sma = _simple_moving_average(closes[-20:])
    moving_average_gap_pct = ((short_sma - long_sma) / long_sma) * 100 if long_sma else 0.0
    macd_signal = _infer_macd_signal(closes)
    bollinger_position = _infer_bollinger_position(closes)

    return (
        f"Provider: {provider}\n"
        f"Ticker: {ticker}\n"
        f"Last trade price: {last_trade_price:.2f}\n"
        f"Bid: {bid:.2f}\n"
        f"Ask: {ask:.2f}\n"
        f"Minute close: {minute_close:.2f}\n"
        f"Daily open: {daily_open:.2f}\n"
        f"Daily high: {daily_high:.2f}\n"
        f"Daily low: {daily_low:.2f}\n"
        f"Previous close: {previous_close:.2f}\n"
        f"Recent bars fetched: {len(closes)}\n"
        f"Derived RSI(14): {rsi:.2f}\n"
        f"Derived MACD signal: {macd_signal}\n"
        f"Derived moving average gap: {moving_average_gap_pct:.2f}%\n"
        f"Derived Bollinger position: {bollinger_position}"
    )


def _fetch_json_with_headers(url: str, headers: dict[str, str]) -> dict[str, Any] | str:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        return f"request failed with HTTP {exc.code}: {details}"
    except error.URLError as exc:
        return f"request failed: {exc.reason}"


def _headline_sentiment_score(headline: str, summary: str) -> float:
    text = f"{headline} {summary}".lower()
    weighted_terms = {
        "beat": 1.2,
        "beats": 1.2,
        "surge": 1.0,
        "gain": 0.8,
        "strong": 0.8,
        "upgrade": 1.0,
        "growth": 0.7,
        "profit": 0.7,
        "record": 0.8,
        "guidance raised": 1.2,
        "miss": -1.2,
        "misses": -1.2,
        "fall": -0.8,
        "drop": -0.8,
        "weak": -0.8,
        "downgrade": -1.0,
        "loss": -0.7,
        "lawsuit": -1.0,
        "probe": -0.8,
        "guidance cut": -1.2,
    }
    score = 0.0
    for token, weight in weighted_terms.items():
        if token in text:
            score += weight
    return score


def _format_news_summary(provider: str, ticker: str, items: list[dict[str, Any]]) -> str:
    scored = []
    for item in items:
        headline = str(item.get("headline") or item.get("title") or "")
        summary = str(item.get("summary") or "")
        source = str(item.get("source", "unknown"))
        created_at = str(item.get("created_at") or item.get("updated_at") or "")
        adjusted_score = (
            _headline_sentiment_score(headline, summary) * _recency_weight(created_at)
            + _source_weight(source)
        )
        scored.append(
            {
                "headline": headline,
                "summary": summary,
                "score": adjusted_score,
                "source": source,
                "created_at": created_at,
            }
        )
    avg_score = sum(entry["score"] for entry in scored) / len(scored)
    stance = "positive" if avg_score > 0.25 else "negative" if avg_score < -0.25 else "balanced"
    headlines = "\n".join(
        f"- {entry['headline']} ({entry['source']}, score={entry['score']:.2f})"
        for entry in scored[:3]
        if entry["headline"]
    ) or "- No headlines parsed"
    return (
        f"Provider: {provider}\n"
        f"Ticker: {ticker}\n"
        f"Articles analyzed: {len(scored)}\n"
        f"Sentiment score: {avg_score:.2f}\n"
        f"Stance: {stance}\n"
        f"Top headlines:\n{headlines}"
    )


def _mid_price(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return max(bid, ask, 0.0)


def _structured_output_path(filename: str) -> Path:
    return Path("output") / filename


def _write_structured_output(filename: str, payload: dict[str, Any]) -> None:
    path = _structured_output_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_structured_output(filename: str) -> dict[str, Any] | None:
    path = _structured_output_path(filename)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _discover_with_retries(
    discovery_window_days: int,
    discovery_max_symbols: int,
    discovery_min_price: float,
    discovery_universe_file: str,
    discovery_min_score: float,
    discovery_retry_attempts: int,
) -> tuple[Any, int]:
    from ai_hedge_fund.discovery import discover_candidate

    attempts_used = 0
    scan_size = max(1, discovery_max_symbols)
    candidate = None

    for attempt in range(discovery_retry_attempts + 1):
        attempts_used = attempt + 1
        candidate = discover_candidate(
            earnings_window_days=discovery_window_days,
            max_symbols=scan_size,
            min_price=discovery_min_price,
            universe_file=discovery_universe_file,
        )
        if candidate.score >= discovery_min_score:
            return candidate, attempts_used
        scan_size = max(scan_size + discovery_max_symbols, scan_size * 2)

    return candidate, attempts_used


def _format_options_summary(
    provider: str, ticker: str, contracts: list[dict[str, Any]], expiration: str = ""
) -> str:
    top_contracts = sorted(contracts, key=lambda item: item.get("iv", 0.0), reverse=True)[:4]
    lines = []
    for contract in top_contracts:
        extra = []
        if "side" in contract:
            extra.append(str(contract["side"]))
        if "open_interest" in contract:
            extra.append(f"oi={contract['open_interest']}")
        if "delta" in contract:
            extra.append(f"delta={contract['delta']:.2f}")
        lines.append(
            f"- {contract.get('symbol', 'unknown')} mid={float(contract.get('mid', 0.0)):.2f} "
            f"iv={float(contract.get('iv', 0.0)):.2f} {' '.join(extra)}".rstrip()
        )
    expiration_line = f"\nExpiration inspected: {expiration}" if expiration else ""
    return (
        f"Provider: {provider}\n"
        f"Ticker: {ticker}"
        f"{expiration_line}\n"
        f"Contracts reviewed: {len(contracts)}\n"
        f"Highlights:\n" + ("\n".join(lines) or "- No contracts parsed")
    )


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    return max_drawdown


def _source_weight(source: str) -> float:
    normalized = source.lower()
    trusted = ["reuters", "bloomberg", "associated press", "dow jones", "wsj", "financial times"]
    if any(token in normalized for token in trusted):
        return 0.15
    return 0.0


def _recency_weight(created_at: str) -> float:
    published = _parse_datetime(created_at)
    if published is None:
        return 1.0
    age_hours = max((datetime.now(timezone.utc) - published).total_seconds() / 3600, 0.0)
    if age_hours <= 24:
        return 1.0
    if age_hours <= 72:
        return 0.75
    return 0.5


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if value.isdigit():
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
