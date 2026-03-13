from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from ai_hedge_fund.run_batch import run_for_tickers


def _enforce_safe_env() -> None:
    os.environ["ALPACA_ENABLE_SUBMISSION"] = "false"
    os.environ["ALPACA_KILL_SWITCH"] = "true"


def _validate_safe_mode(execution_mode: str) -> str:
    normalized = execution_mode.strip().lower()
    if normalized == "live":
        raise SystemExit("Safe scan does not allow --execution-mode live.")
    return normalized or "manual"


def _build_simple_parser():
    from ai_hedge_fund.automation import build_parser

    parser = build_parser()
    visible = {
        "help",
        "ticker",
        "companies",
        "companies_file",
        "auto_discover",
        "top_percent",
        "notify_on",
        "alert_prefix",
    }
    for action in parser._actions:
        if action.dest not in visible:
            action.help = argparse.SUPPRESS
    parser.set_defaults(
        execution_mode="manual",
        discovery_window_days=7,
        discovery_max_symbols=75,
        discovery_min_price=10.0,
        discovery_min_score=2.5,
        discovery_retry_attempts=2,
        screen_cache_ttl_hours=6,
    )
    return parser


def run() -> None:
    load_dotenv()

    from ai_hedge_fund.automation import (
        _build_alert_summary,
        _maybe_send_alerts,
    )
    from ai_hedge_fund.crew import AIHedgeFundCrew
    from ai_hedge_fund.main import _build_inputs, _validate_args, configure_runtime_flags

    parser = _build_simple_parser()
    parser.description = "Run a safe local research scan with alerts and no order submission."
    args = parser.parse_args()
    args.execution_mode = _validate_safe_mode(args.execution_mode)
    _validate_args(args)
    _enforce_safe_env()
    configure_runtime_flags(args)

    Path("output").mkdir(exist_ok=True)
    print(
        "Safe scan mode enabled: execution_mode="
        f"{args.execution_mode}, ALPACA_ENABLE_SUBMISSION=false, ALPACA_KILL_SWITCH=true"
    )
    def _runner(run_args) -> None:
        AIHedgeFundCrew().crew().kickoff(inputs=_build_inputs(run_args))

    run_for_tickers(args, _runner)

    summary = _build_alert_summary()
    _maybe_send_alerts(
        summary=summary,
        notify_on=args.notify_on.strip().lower(),
        prefix=args.alert_prefix.strip(),
    )


if __name__ == "__main__":
    run()
