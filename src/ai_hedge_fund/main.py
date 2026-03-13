from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


VALID_SIGNAL_VALUES = {"bullish", "bearish", "flat"}
VALID_BOLLINGER_VALUES = {"upper", "middle", "lower"}
VALID_REVISION_VALUES = {"up", "down", "flat"}
VALID_EXECUTION_MODES = {"manual", "paper", "live"}
VALID_ORDER_TYPES = {"market", "limit"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AI hedge fund multi-agent crew."
    )
    parser.add_argument(
        "--ticker",
        default=os.getenv("DEFAULT_TICKER", "AAPL"),
        help="Ticker symbol to evaluate.",
    )
    parser.add_argument(
        "--thesis",
        default="Momentum remains intact into the next catalyst.",
        help="Initial trade thesis.",
    )
    parser.add_argument(
        "--macro-view",
        default="Rates are stable and equity risk appetite is neutral to positive.",
        help="Macro backdrop for the trade.",
    )
    parser.add_argument(
        "--upcoming-event",
        default="Next earnings report",
        help="Next company or macro catalyst.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=185.0,
        help="Current share price.",
    )
    parser.add_argument(
        "--rsi",
        type=float,
        default=58.0,
        help="RSI value.",
    )
    parser.add_argument(
        "--macd-signal",
        default="bullish",
        help="MACD direction: bullish, bearish, or flat.",
    )
    parser.add_argument(
        "--moving-average-gap-pct",
        type=float,
        default=3.5,
        help="Percent above or below the key moving average.",
    )
    parser.add_argument(
        "--bollinger-position",
        default="middle",
        help="Bollinger position: upper, middle, or lower.",
    )
    parser.add_argument(
        "--news-sentiment",
        type=float,
        default=0.25,
        help="News sentiment from -1 to 1.",
    )
    parser.add_argument(
        "--social-sentiment",
        type=float,
        default=0.10,
        help="Social sentiment from -1 to 1.",
    )
    parser.add_argument(
        "--analyst-revision-trend",
        default="up",
        help="Analyst revision trend: up, down, or flat.",
    )
    parser.add_argument(
        "--implied-volatility-pct",
        type=float,
        default=32.0,
        help="Implied volatility percent.",
    )
    parser.add_argument(
        "--days-to-expiry",
        type=int,
        default=30,
        help="Target days to expiry for options ideas.",
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100000.0,
        help="Portfolio account value.",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=5.0,
        help="Max position size as percent of portfolio.",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=4.0,
        help="Stop loss percent.",
    )
    parser.add_argument(
        "--conviction",
        type=float,
        default=0.65,
        help="Conviction from 0 to 1.",
    )
    parser.add_argument(
        "--execution-mode",
        default="manual",
        help="Execution mode: manual, paper, or live.",
    )
    parser.add_argument(
        "--broker",
        default="Alpaca",
        help="Broker or API venue. Example: Alpaca.",
    )
    parser.add_argument(
        "--order-type",
        default="market",
        help="Order type for execution: market or limit.",
    )
    parser.add_argument(
        "--time-in-force",
        default="day",
        help="Time in force for execution: day, gtc, etc.",
    )
    parser.add_argument(
        "--limit-price",
        type=float,
        default=None,
        help="Optional limit price for limit orders.",
    )
    return parser


def _normalize_choice(value: str) -> str:
    return value.strip().lower()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.ticker.strip():
        raise SystemExit("Ticker cannot be empty.")
    if args.price <= 0:
        raise SystemExit("Price must be greater than zero.")
    if not 0 <= args.rsi <= 100:
        raise SystemExit("RSI must be between 0 and 100.")
    if not -1 <= args.news_sentiment <= 1:
        raise SystemExit("News sentiment must be between -1 and 1.")
    if not -1 <= args.social_sentiment <= 1:
        raise SystemExit("Social sentiment must be between -1 and 1.")
    if args.implied_volatility_pct < 0:
        raise SystemExit("Implied volatility must be zero or greater.")
    if args.days_to_expiry <= 0:
        raise SystemExit("Days to expiry must be greater than zero.")
    if args.portfolio_value <= 0:
        raise SystemExit("Portfolio value must be greater than zero.")
    if not 0 < args.max_position_pct <= 100:
        raise SystemExit("Max position percent must be between 0 and 100.")
    if not 0 < args.stop_loss_pct <= 100:
        raise SystemExit("Stop loss percent must be between 0 and 100.")
    if not 0 <= args.conviction <= 1:
        raise SystemExit("Conviction must be between 0 and 1.")

    macd_signal = _normalize_choice(args.macd_signal)
    if macd_signal not in VALID_SIGNAL_VALUES:
        raise SystemExit("MACD signal must be one of: bullish, bearish, flat.")

    bollinger_position = _normalize_choice(args.bollinger_position)
    if bollinger_position not in VALID_BOLLINGER_VALUES:
        raise SystemExit("Bollinger position must be one of: upper, middle, lower.")

    analyst_revision_trend = _normalize_choice(args.analyst_revision_trend)
    if analyst_revision_trend not in VALID_REVISION_VALUES:
        raise SystemExit("Analyst revision trend must be one of: up, down, flat.")

    execution_mode = _normalize_choice(args.execution_mode)
    if execution_mode not in VALID_EXECUTION_MODES:
        raise SystemExit("Execution mode must be one of: manual, paper, live.")

    order_type = _normalize_choice(args.order_type)
    if order_type not in VALID_ORDER_TYPES:
        raise SystemExit("Order type must be one of: market, limit.")
    if order_type == "limit" and args.limit_price is None:
        raise SystemExit("Limit orders require --limit-price.")
    if args.limit_price is not None and args.limit_price <= 0:
        raise SystemExit("Limit price must be greater than zero.")


def _build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ticker": args.ticker.strip().upper(),
        "thesis": args.thesis.strip(),
        "macro_view": args.macro_view.strip(),
        "upcoming_event": args.upcoming_event.strip(),
        "price": args.price,
        "rsi": args.rsi,
        "macd_signal": _normalize_choice(args.macd_signal),
        "moving_average_gap_pct": args.moving_average_gap_pct,
        "bollinger_position": _normalize_choice(args.bollinger_position),
        "news_sentiment": args.news_sentiment,
        "social_sentiment": args.social_sentiment,
        "analyst_revision_trend": _normalize_choice(args.analyst_revision_trend),
        "implied_volatility_pct": args.implied_volatility_pct,
        "days_to_expiry": args.days_to_expiry,
        "portfolio_value": args.portfolio_value,
        "max_position_pct": args.max_position_pct,
        "stop_loss_pct": args.stop_loss_pct,
        "conviction": args.conviction,
        "execution_mode": _normalize_choice(args.execution_mode),
        "broker": args.broker.strip(),
        "order_type": _normalize_choice(args.order_type),
        "time_in_force": _normalize_choice(args.time_in_force),
        "limit_price": args.limit_price,
    }


def run() -> None:
    load_dotenv()
    try:
        from ai_hedge_fund.crew import AIHedgeFundCrew
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("crewai"):
            raise SystemExit(
                "CrewAI is not installed in this environment. Install project dependencies with "
                "`pip install '.[dev]'` in a Python 3.10+ environment."
            ) from exc
        raise
    args = build_parser().parse_args()
    _validate_args(args)
    Path("output").mkdir(exist_ok=True)
    AIHedgeFundCrew().crew().kickoff(inputs=_build_inputs(args))


if __name__ == "__main__":
    run()
