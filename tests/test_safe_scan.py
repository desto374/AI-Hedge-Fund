import os

import pytest

from ai_hedge_fund.safe_scan import _build_simple_parser, _enforce_safe_env, _validate_safe_mode


def test_enforce_safe_env_forces_non_submitting_mode(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_ENABLE_SUBMISSION", "true")
    monkeypatch.setenv("ALPACA_KILL_SWITCH", "false")

    _enforce_safe_env()

    assert os.environ["ALPACA_ENABLE_SUBMISSION"] == "false"
    assert os.environ["ALPACA_KILL_SWITCH"] == "true"


def test_validate_safe_mode_rejects_live() -> None:
    with pytest.raises(SystemExit, match="does not allow --execution-mode live"):
        _validate_safe_mode("live")


def test_validate_safe_mode_allows_manual_and_paper() -> None:
    assert _validate_safe_mode("manual") == "manual"
    assert _validate_safe_mode("paper") == "paper"


def test_simple_parser_accepts_single_visible_companies_input() -> None:
    parser = _build_simple_parser()
    args = parser.parse_args(["NVDA,MSFT,AMD"])

    assert args.companies_input == "NVDA,MSFT,AMD"
