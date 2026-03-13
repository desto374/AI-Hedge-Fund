from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from ai_hedge_fund.run_batch import parse_tickers, run_for_tickers


@dataclass
class FakeScreenResult:
    ticker: str
    company_name: str
    score: float
    price: float = 100.0
    momentum_20d_pct: float = 5.0
    momentum_60d_pct: float = 10.0
    news_score: float = 0.5
    earnings_date: str = ""


def test_parse_tickers_supports_csv_and_dedupes() -> None:
    tickers = parse_tickers(ticker="", tickers="nvda,msft,NVDA, amd ", tickers_file="")
    assert tickers == ["NVDA", "MSFT", "AMD"]


def test_parse_tickers_supports_file(tmp_path) -> None:
    path = tmp_path / "tickers.txt"
    path.write_text("nvda\nmsft,amd\n", encoding="utf-8")

    tickers = parse_tickers(ticker="", tickers="", tickers_file=str(path))

    assert tickers == ["NVDA", "MSFT", "AMD"]


def test_parse_tickers_supports_companies_alias() -> None:
    tickers = parse_tickers(
        ticker="",
        tickers="",
        tickers_file="",
        companies="nvda,msft",
        companies_file="",
    )
    assert tickers == ["NVDA", "MSFT"]


def test_run_for_tickers_runs_each_manual_ticker_and_archives_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    calls = []
    monkeypatch.setattr(
        "ai_hedge_fund.run_batch.screen_companies",
        lambda **kwargs: [
            FakeScreenResult(ticker="NVDA", company_name="NVIDIA", score=4.0),
            FakeScreenResult(ticker="MSFT", company_name="Microsoft", score=3.0),
        ],
    )
    monkeypatch.setattr(
        "ai_hedge_fund.run_batch.write_screen_summary",
        lambda all_results, selected_results: tmp_path / "output" / "batch" / "company_screen_summary.json",
    )

    def runner(args):
        calls.append(args.ticker)
        (tmp_path / "output" / "trade_decision.md").write_text(f"Decision for {args.ticker}", encoding="utf-8")
        (tmp_path / "output" / "portfolio_decision.json").write_text("{}", encoding="utf-8")
        (tmp_path / "output" / "discovery_selection.json").write_text("{}", encoding="utf-8")

    args = Namespace(
        ticker="",
        tickers="NVDA,MSFT",
        tickers_file="",
        companies="",
        companies_file="",
        auto_discover=False,
        discovery_min_price=10.0,
        discovery_window_days=7,
        screen_cache_ttl_hours=6,
        top_percent=100.0,
    )

    archived = run_for_tickers(args, runner)

    assert calls == ["NVDA", "MSFT"]
    assert any(path.endswith("nvda_trade_decision.md") for path in archived)
    assert any(path.endswith("msft_trade_decision.md") for path in archived)


def test_run_for_tickers_keeps_single_auto_discover_run() -> None:
    calls = []

    def runner(args):
        calls.append(args.auto_discover)

    args = Namespace(
        ticker="",
        tickers="NVDA,MSFT",
        tickers_file="",
        companies="",
        companies_file="",
        auto_discover=True,
    )

    archived = run_for_tickers(args, runner)

    assert calls == [True]
    assert archived == []


def test_run_for_tickers_selects_top_percent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    calls = []
    monkeypatch.setattr(
        "ai_hedge_fund.run_batch.screen_companies",
        lambda **kwargs: [
            FakeScreenResult(ticker="NVDA", company_name="NVIDIA", score=5.0),
            FakeScreenResult(ticker="MSFT", company_name="Microsoft", score=4.0),
            FakeScreenResult(ticker="AMD", company_name="AMD", score=3.0),
            FakeScreenResult(ticker="AVGO", company_name="Broadcom", score=2.0),
        ],
    )
    monkeypatch.setattr(
        "ai_hedge_fund.run_batch.write_screen_summary",
        lambda all_results, selected_results: tmp_path / "output" / "batch" / "company_screen_summary.json",
    )

    def runner(args):
        calls.append(args.ticker)
        (tmp_path / "output" / "trade_decision.md").write_text("ok", encoding="utf-8")

    args = Namespace(
        ticker="",
        tickers="NVDA,MSFT,AMD,AVGO",
        tickers_file="",
        companies="",
        companies_file="",
        auto_discover=False,
        discovery_min_price=10.0,
        discovery_window_days=7,
        screen_cache_ttl_hours=6,
        top_percent=30.0,
    )

    run_for_tickers(args, runner)

    assert calls == ["NVDA"]
