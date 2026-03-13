from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai_hedge_fund.discovery import discover_candidate
from ai_hedge_fund.main import build_parser, _build_inputs
from ai_hedge_fund.tools.custom_tools import CandidateDiscoveryTool


class FakeSeries:
    def __init__(self, values):
        self._values = list(values)

    def dropna(self):
        return self

    def tolist(self):
        return list(self._values)


class FakeHistory:
    empty = False

    def __init__(self, closes):
        self._closes = closes

    def __getitem__(self, key: str):
        if key != "Close":
            raise KeyError(key)
        return FakeSeries(self._closes)


class FakeTicker:
    def __init__(self, symbol: str, dataset: dict[str, dict]):
        payload = dataset[symbol]
        self._payload = payload
        self.calendar = payload["calendar"]
        self.news = payload.get("news", [])
        self.fast_info = payload.get("fast_info", {})
        self.info = payload.get("info", {})

    def history(self, period: str, interval: str, auto_adjust: bool):
        return FakeHistory(self._payload["closes"])


def test_discover_candidate_picks_best_earnings_setup(monkeypatch, tmp_path) -> None:
    today = datetime.now(timezone.utc).date()
    dataset = {
        "AAA": {
            "calendar": {"Earnings Date": [today + timedelta(days=3)]},
            "closes": [100 + i for i in range(61)],
            "news": [
                {
                    "content": {
                        "title": "Company beats estimates",
                        "summary": "Strong growth continues",
                        "provider": {"displayName": "Reuters"},
                        "pubDate": datetime.now(timezone.utc).isoformat(),
                    }
                }
            ],
            "fast_info": {"shortName": "Alpha Co"},
        },
        "BBB": {
            "calendar": {"Earnings Date": [today + timedelta(days=4)]},
            "closes": [80.0 for _ in range(61)],
            "news": [
                {
                    "content": {
                        "title": "Guidance cut after weak quarter",
                        "summary": "Demand slows",
                        "provider": {"displayName": "Wire"},
                        "pubDate": datetime.now(timezone.utc).isoformat(),
                    }
                }
            ],
            "fast_info": {"shortName": "Beta Co"},
        },
    }

    class FakeYF:
        @staticmethod
        def Ticker(symbol: str):
            return FakeTicker(symbol, dataset)

    universe_file = tmp_path / "symbols.txt"
    universe_file.write_text("AAA\nBBB\n", encoding="utf-8")

    monkeypatch.setattr("ai_hedge_fund.discovery.find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF())

    candidate = discover_candidate(
        earnings_window_days=7,
        max_symbols=10,
        min_price=10.0,
        universe_file=str(universe_file),
    )

    assert candidate.ticker == "AAA"
    assert candidate.company_name == "Alpha Co"
    assert candidate.days_until_earnings == 3
    assert candidate.score > 0


def test_build_inputs_preserves_discovery_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--auto-discover",
            "--discovery-window-days",
            "5",
            "--discovery-max-symbols",
            "20",
            "--discovery-min-price",
            "15",
            "--discovery-min-score",
            "3.25",
            "--discovery-retry-attempts",
            "4",
        ]
    )

    inputs = _build_inputs(args)

    assert inputs["auto_discover"] is True
    assert inputs["discovery_window_days"] == 5
    assert inputs["discovery_max_symbols"] == 20
    assert inputs["discovery_min_price"] == 15.0
    assert inputs["discovery_min_score"] == 3.25
    assert inputs["discovery_retry_attempts"] == 4


def test_candidate_discovery_tool_formats_auto_discovery(monkeypatch) -> None:
    tool = CandidateDiscoveryTool()

    class Candidate:
        ticker = "NVDA"
        company_name = "NVIDIA"
        earnings_date = datetime(2026, 3, 18, tzinfo=timezone.utc).date()
        days_until_earnings = 5
        score = 4.25
        price = 120.50
        momentum_20d_pct = 8.5
        momentum_60d_pct = 22.0
        news_score = 0.8
        upcoming_event = "Earnings call on 2026-03-18"
        thesis = "Auto-discovered ahead of earnings."

    monkeypatch.setattr("ai_hedge_fund.discovery.discover_candidate", lambda **kwargs: Candidate())

    result = tool._run(auto_discover=True)

    assert "Discovery mode: auto" in result
    assert "Discovery status: accepted" in result
    assert "Selected ticker: NVDA" in result
    assert "Earnings date: 2026-03-18" in result
    assert '"selected_ticker": "NVDA"' in result


def test_candidate_discovery_tool_rejects_low_scoring_candidate(monkeypatch) -> None:
    tool = CandidateDiscoveryTool()

    class Candidate:
        ticker = "SNOW"
        company_name = "Snowflake"
        earnings_date = datetime(2026, 3, 19, tzinfo=timezone.utc).date()
        days_until_earnings = 6
        score = 1.4
        price = 155.0
        momentum_20d_pct = 1.0
        momentum_60d_pct = 2.5
        news_score = 0.1
        upcoming_event = "Earnings call on 2026-03-19"
        thesis = "Auto-discovered ahead of earnings."

    monkeypatch.setattr("ai_hedge_fund.discovery.discover_candidate", lambda **kwargs: Candidate())

    result = tool._run(auto_discover=True, discovery_min_score=2.5)

    assert "Discovery status: rejected" in result
    assert "below minimum threshold 2.50" in result
    assert "Do not continue with trade analysis" in result
    assert '"discovery_status": "rejected"' in result


def test_candidate_discovery_tool_retries_until_candidate_is_accepted(monkeypatch) -> None:
    tool = CandidateDiscoveryTool()
    calls = []

    class RejectedCandidate:
        ticker = "SNOW"
        company_name = "Snowflake"
        earnings_date = datetime(2026, 3, 19, tzinfo=timezone.utc).date()
        days_until_earnings = 6
        score = 1.4
        price = 155.0
        momentum_20d_pct = 1.0
        momentum_60d_pct = 2.5
        news_score = 0.1
        upcoming_event = "Earnings call on 2026-03-19"
        thesis = "Auto-discovered ahead of earnings."

    class AcceptedCandidate:
        ticker = "NVDA"
        company_name = "NVIDIA"
        earnings_date = datetime(2026, 3, 18, tzinfo=timezone.utc).date()
        days_until_earnings = 5
        score = 4.1
        price = 121.0
        momentum_20d_pct = 8.2
        momentum_60d_pct = 20.0
        news_score = 0.7
        upcoming_event = "Earnings call on 2026-03-18"
        thesis = "Auto-discovered ahead of earnings."

    def fake_discover_candidate(**kwargs):
        calls.append(kwargs["max_symbols"])
        if len(calls) == 1:
            return RejectedCandidate()
        return AcceptedCandidate()

    monkeypatch.setattr("ai_hedge_fund.discovery.discover_candidate", fake_discover_candidate)

    result = tool._run(
        auto_discover=True,
        discovery_max_symbols=20,
        discovery_min_score=2.5,
        discovery_retry_attempts=2,
    )

    assert calls == [20, 40]
    assert "Discovery status: accepted" in result
    assert "Discovery attempts used: 2" in result
    assert "Selected ticker: NVDA" in result
    assert '"discovery_attempts_used": 2' in result


def test_candidate_discovery_tool_blocks_manual_override_when_auto_locked(monkeypatch) -> None:
    tool = CandidateDiscoveryTool()
    monkeypatch.setenv("AI_HEDGE_FUND_FORCE_AUTO_DISCOVER", "true")

    result = tool._run(ticker="XFLT", auto_discover=False)

    assert "runtime is locked to auto-discover" in result
    assert "auto_discover=true" in result
