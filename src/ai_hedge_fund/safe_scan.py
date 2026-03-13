from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _enforce_safe_env() -> None:
    os.environ["ALPACA_ENABLE_SUBMISSION"] = "false"
    os.environ["ALPACA_KILL_SWITCH"] = "true"


def _validate_safe_mode(execution_mode: str) -> str:
    normalized = execution_mode.strip().lower()
    if normalized == "live":
        raise SystemExit("Safe scan does not allow --execution-mode live.")
    return normalized or "manual"


def run() -> None:
    load_dotenv()

    from ai_hedge_fund.automation import (
        _build_alert_summary,
        _maybe_send_alerts,
        build_parser,
    )
    from ai_hedge_fund.crew import AIHedgeFundCrew
    from ai_hedge_fund.main import _build_inputs, _validate_args, configure_runtime_flags

    parser = build_parser()
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
    AIHedgeFundCrew().crew().kickoff(inputs=_build_inputs(args))

    summary = _build_alert_summary()
    _maybe_send_alerts(
        summary=summary,
        notify_on=args.notify_on.strip().lower(),
        prefix=args.alert_prefix.strip(),
    )


if __name__ == "__main__":
    run()
