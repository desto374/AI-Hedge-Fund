from __future__ import annotations

import json

from ai_hedge_fund.automation import (
    _build_alert_summary,
    _format_alert_message,
    _format_alert_subject,
    _maybe_send_alerts,
)


def test_build_alert_summary_reads_output_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "discovery_selection.json").write_text(
        json.dumps({"selected_ticker": "NVDA", "discovery_status": "accepted"}),
        encoding="utf-8",
    )
    (output_dir / "portfolio_decision.json").write_text(
        json.dumps({"final_action": "buy", "confidence": "high"}),
        encoding="utf-8",
    )
    (output_dir / "trade_decision.md").write_text("Execution note", encoding="utf-8")

    summary = _build_alert_summary()

    assert summary["discovery"]["selected_ticker"] == "NVDA"
    assert summary["decision"]["final_action"] == "buy"
    assert "Execution note" in summary["execution"]


def test_format_alert_message_includes_key_fields() -> None:
    summary = {
        "discovery": {
            "selected_ticker": "NVDA",
            "discovery_status": "accepted",
            "discovery_score": 4.2,
            "earnings_date": "2026-03-18",
            "upcoming_event": "Earnings call on 2026-03-18",
        },
        "decision": {
            "final_action": "buy",
            "confidence": "high",
            "rationale": "Momentum and sentiment align.",
        },
        "execution": "Submission status: Not submitted.",
    }

    message = _format_alert_message(summary=summary, prefix="[AI Hedge Fund]")

    assert "Ticker: NVDA" in message
    assert "Final action: buy" in message
    assert "Execution handoff:" in message


def test_maybe_send_alerts_skips_non_matching_status(monkeypatch) -> None:
    posted = []
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test")
    monkeypatch.setattr("ai_hedge_fund.automation._post_json", lambda url, payload: posted.append((url, payload)))

    _maybe_send_alerts(
        summary={"discovery": {"selected_ticker": "NVDA", "discovery_status": "rejected"}, "decision": {}},
        notify_on="accepted",
        prefix="[AI Hedge Fund]",
    )

    assert posted == []


def test_maybe_send_alerts_posts_when_status_matches(monkeypatch) -> None:
    posted = []
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test")
    monkeypatch.setattr("ai_hedge_fund.automation._post_json", lambda url, payload: posted.append((url, payload)))

    _maybe_send_alerts(
        summary={
            "discovery": {"selected_ticker": "NVDA", "discovery_status": "accepted"},
            "decision": {"final_action": "buy", "rationale": "Test"},
            "execution": "",
        },
        notify_on="accepted",
        prefix="[AI Hedge Fund]",
    )

    assert posted
    assert posted[0][0] == "https://hooks.slack.test"
    assert "NVDA" in posted[0][1]["text"]


def test_format_alert_subject_uses_status_and_action() -> None:
    subject = _format_alert_subject(
        prefix="[AI Hedge Fund]",
        ticker="NVDA",
        status="accepted",
        action="buy",
    )

    assert subject == "[AI Hedge Fund] NVDA discovery=accepted action=buy"
