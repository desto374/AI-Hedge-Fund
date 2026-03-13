from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from ai_hedge_fund.screening import CompanyScreenResult
from ai_hedge_fund.screening import _read_cached_result
from ai_hedge_fund.screening import select_top_company_results


def test_select_top_company_results_keeps_top_percent() -> None:
    results = [
        CompanyScreenResult(ticker="NVDA", company_name="NVIDIA", score=5.0, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="MSFT", company_name="Microsoft", score=4.0, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="AMD", company_name="AMD", score=3.0, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="AVGO", company_name="Broadcom", score=2.0, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="META", company_name="Meta", score=1.0, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="NFLX", company_name="Netflix", score=0.5, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="GOOGL", company_name="Google", score=0.4, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="AMZN", company_name="Amazon", score=0.3, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="AAPL", company_name="Apple", score=0.2, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
        CompanyScreenResult(ticker="CRM", company_name="Salesforce", score=0.1, price=100, momentum_20d_pct=1, momentum_60d_pct=1, news_score=1),
    ]

    selected = select_top_company_results(results, top_percent=30.0)

    assert [item.ticker for item in selected] == ["NVDA", "MSFT", "AMD"]


def test_read_cached_result_respects_ttl() -> None:
    now = datetime.now(timezone.utc)
    fresh = {
        "NVDA": {
            "fetched_at": now.isoformat(),
            "result": asdict(
                CompanyScreenResult(
                    ticker="NVDA",
                    company_name="NVIDIA",
                    score=5.0,
                    price=100.0,
                    momentum_20d_pct=5.0,
                    momentum_60d_pct=10.0,
                    news_score=0.5,
                )
            ),
        }
    }
    stale = {
        "NVDA": {
            "fetched_at": (now - timedelta(hours=10)).isoformat(),
            "result": fresh["NVDA"]["result"],
        }
    }

    assert _read_cached_result(fresh, "NVDA", now, cache_ttl_hours=6) is not None
    assert _read_cached_result(stale, "NVDA", now, cache_ttl_hours=6) is None
