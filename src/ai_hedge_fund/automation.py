from __future__ import annotations

import argparse
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib import error, request

from dotenv import load_dotenv
from ai_hedge_fund.run_batch import run_for_tickers


def build_parser() -> argparse.ArgumentParser:
    from ai_hedge_fund.main import build_parser as build_main_parser

    parser = build_main_parser()
    parser.description = "Run the AI hedge fund scan once and optionally send alerts."
    parser.add_argument(
        "--notify-on",
        default=os.getenv("ALERT_NOTIFY_ON", "accepted"),
        help="When to send alerts: accepted, rejected, all, or none.",
    )
    parser.add_argument(
        "--alert-prefix",
        default=os.getenv("ALERT_PREFIX", "[AI Hedge Fund]"),
        help="Prefix used in alert subject lines and messages.",
    )
    return parser


def run() -> None:
    load_dotenv()
    from ai_hedge_fund.crew import AIHedgeFundCrew
    from ai_hedge_fund.main import _build_inputs, _validate_args, configure_runtime_flags

    args = build_parser().parse_args()
    _validate_args(args)
    configure_runtime_flags(args)
    Path("output").mkdir(exist_ok=True)
    def _runner(run_args: argparse.Namespace) -> None:
        AIHedgeFundCrew().crew().kickoff(inputs=_build_inputs(run_args))

    run_for_tickers(args, _runner)

    summary = _build_alert_summary()
    _maybe_send_alerts(
        summary=summary,
        notify_on=args.notify_on.strip().lower(),
        prefix=args.alert_prefix.strip(),
    )


def _build_alert_summary() -> dict[str, Any]:
    discovery = _read_json(Path("output/discovery_selection.json"))
    decision = _read_json(Path("output/portfolio_decision.json"))
    execution = _safe_read_text(Path("output/trade_decision.md"))

    return {
        "discovery": discovery,
        "decision": decision,
        "execution": execution,
    }


def _maybe_send_alerts(summary: dict[str, Any], notify_on: str, prefix: str) -> None:
    discovery = summary.get("discovery") or {}
    decision = summary.get("decision") or {}
    status = str(discovery.get("discovery_status", "")).lower()
    action = str(decision.get("final_action", "")).lower()

    if notify_on not in {"accepted", "rejected", "all", "none"}:
        raise SystemExit("notify-on must be one of: accepted, rejected, all, none.")
    if notify_on == "none":
        return
    if notify_on == "accepted" and status != "accepted":
        return
    if notify_on == "rejected" and status != "rejected":
        return

    message = _format_alert_message(summary=summary, prefix=prefix)
    subject = _format_alert_subject(
        prefix=prefix,
        ticker=str(discovery.get("selected_ticker", "")),
        status=status or "unknown",
        action=action or "unknown",
    )

    sent_any = False
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    generic_webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if slack_webhook:
        _post_json(slack_webhook, {"text": f"{subject}\n{message}"})
        sent_any = True
    if generic_webhook:
        _post_json(generic_webhook, {"subject": subject, "message": message, "summary": summary})
        sent_any = True
    if _email_config_present():
        _send_email(subject=subject, message=message)
        sent_any = True

    if sent_any:
        print(f"Alerts sent for {subject}")
    else:
        print(message)


def _format_alert_subject(prefix: str, ticker: str, status: str, action: str) -> str:
    normalized_ticker = ticker or "NO-TICKER"
    return f"{prefix} {normalized_ticker} discovery={status} action={action}"


def _format_alert_message(summary: dict[str, Any], prefix: str) -> str:
    discovery = summary.get("discovery") or {}
    decision = summary.get("decision") or {}
    execution = (summary.get("execution") or "").strip()
    lines = [
        prefix,
        f"Ticker: {discovery.get('selected_ticker', '')}",
        f"Discovery status: {discovery.get('discovery_status', '')}",
        f"Discovery score: {discovery.get('discovery_score', '')}",
        f"Earnings date: {discovery.get('earnings_date', '')}",
        f"Upcoming event: {discovery.get('upcoming_event', '')}",
        f"Final action: {decision.get('final_action', '')}",
        f"Confidence: {decision.get('confidence', '')}",
        f"Rationale: {decision.get('rationale', '')}",
    ]
    if execution:
        lines.append("")
        lines.append("Execution handoff:")
        lines.append(execution)
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _post_json(url: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=20):
            return
    except error.URLError as exc:
        raise SystemExit(f"Alert webhook failed: {exc.reason}") from exc


def _email_config_present() -> bool:
    required = ["ALERT_EMAIL_TO", "ALERT_EMAIL_FROM", "SMTP_HOST"]
    return all(os.getenv(name, "").strip() for name in required)


def _send_email(subject: str, message: str) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("ALERT_EMAIL_FROM", "").strip()
    recipient = os.getenv("ALERT_EMAIL_TO", "").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() == "true"

    email = EmailMessage()
    email["Subject"] = subject
    email["From"] = sender
    email["To"] = recipient
    email.set_content(message)

    with smtplib.SMTP(host, port, timeout=20) as smtp:
        if use_tls:
            smtp.starttls()
        if username:
            smtp.login(username, password)
        smtp.send_message(email)


if __name__ == "__main__":
    run()
