from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_hedge_fund.discovery import (
    _extract_close_values,
    _extract_company_name,
    _extract_earnings_date,
)
from ai_hedge_fund.tools.custom_tools import (
    _headline_sentiment_score,
    _parse_datetime,
    _recency_weight,
    _simple_moving_average,
    _source_weight,
)


DEFAULT_SCREEN_CACHE_PATH = Path("output/cache/company_screen_cache.json")


@dataclass
class CompanyScreenResult:
    ticker: str
    company_name: str
    score: float
    price: float
    momentum_20d_pct: float
    momentum_60d_pct: float
    news_score: float
    earnings_date: str = ""


def screen_companies(
    tickers: list[str],
    min_price: float = 10.0,
    earnings_window_days: int = 7,
    cache_ttl_hours: int = 6,
) -> list[CompanyScreenResult]:
    if not tickers:
        return []

    import yfinance as yf

    cache = _load_screen_cache()
    now = datetime.now(timezone.utc)
    results: list[CompanyScreenResult] = []

    for ticker in tickers:
        cached = _read_cached_result(cache, ticker=ticker, now=now, cache_ttl_hours=cache_ttl_hours)
        if cached is not None:
            results.append(cached)
            continue

        yf_ticker = yf.Ticker(ticker)
        history = yf_ticker.history(period="6mo", interval="1d", auto_adjust=False)
        closes = _extract_close_values(history)
        if len(closes) < 60:
            continue
        price = closes[-1]
        if price < min_price:
            continue

        earnings_date = _extract_earnings_date(yf_ticker)
        score, metrics = _score_screened_company(
            closes=closes,
            news_items=list(getattr(yf_ticker, "news", []) or []),
            earnings_date=earnings_date,
            earnings_window_days=earnings_window_days,
            now=now,
        )
        result = CompanyScreenResult(
            ticker=ticker,
            company_name=_extract_company_name(yf_ticker, fallback=ticker),
            score=score,
            price=price,
            momentum_20d_pct=metrics["momentum_20d_pct"],
            momentum_60d_pct=metrics["momentum_60d_pct"],
            news_score=metrics["news_score"],
            earnings_date=earnings_date.isoformat() if earnings_date is not None else "",
        )
        results.append(result)
        cache[ticker] = {
            "fetched_at": now.isoformat(),
            "result": asdict(result),
        }

    _write_screen_cache(cache)
    return sorted(results, key=lambda item: item.score, reverse=True)


def select_top_company_results(
    results: list[CompanyScreenResult],
    top_percent: float = 30.0,
) -> list[CompanyScreenResult]:
    if not results:
        return []
    bounded_percent = min(max(top_percent, 0.0), 100.0)
    if bounded_percent == 0:
        return []
    count = max(1, round(len(results) * (bounded_percent / 100)))
    return results[:count]


def write_screen_summary(
    all_results: list[CompanyScreenResult],
    selected_results: list[CompanyScreenResult],
) -> Path:
    path = Path("output/batch/company_screen_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "screened_count": len(all_results),
        "selected_count": len(selected_results),
        "all_results": [asdict(result) for result in all_results],
        "selected_results": [asdict(result) for result in selected_results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _score_screened_company(
    closes: list[float],
    news_items: list[dict[str, Any]],
    earnings_date,
    earnings_window_days: int,
    now: datetime,
) -> tuple[float, dict[str, float]]:
    latest_price = closes[-1]
    sma_20 = _simple_moving_average(closes[-20:])
    base_20 = closes[-21] if len(closes) > 20 else closes[0]
    base_60 = closes[-61] if len(closes) > 60 else closes[0]
    momentum_20d_pct = ((latest_price / base_20) - 1) * 100 if base_20 else 0.0
    momentum_60d_pct = ((latest_price / base_60) - 1) * 100 if base_60 else 0.0
    news_score = _average_news_score(news_items[:5])
    score = 0.0
    score += max(min(momentum_20d_pct / 8, 2.0), -2.0)
    score += max(min(momentum_60d_pct / 12, 2.0), -2.0)
    score += 1.0 if latest_price >= sma_20 else -1.0
    score += max(min(news_score, 1.5), -1.5)
    if earnings_date is not None:
        days_until = (earnings_date - now.date()).days
        if 0 <= days_until <= earnings_window_days:
            score += max(0.0, earnings_window_days - days_until + 1) * 0.35
    return score, {
        "momentum_20d_pct": momentum_20d_pct,
        "momentum_60d_pct": momentum_60d_pct,
        "news_score": news_score,
    }


def _average_news_score(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    scores = []
    for item in items:
        content = item.get("content") or {}
        headline = str(content.get("title") or item.get("title") or item.get("headline") or "")
        summary = str(content.get("summary") or item.get("summary") or "")
        source = str(
            content.get("provider", {}).get("displayName")
            or item.get("publisher")
            or item.get("source")
            or "unknown"
        )
        created_at = str(content.get("pubDate") or item.get("providerPublishTime") or item.get("created_at") or "")
        score = (
            _headline_sentiment_score(headline, summary) * _recency_weight(created_at)
            + _source_weight(source)
        )
        scores.append(score)
    return sum(scores) / len(scores)


def _load_screen_cache() -> dict[str, Any]:
    if not DEFAULT_SCREEN_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(DEFAULT_SCREEN_CACHE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _write_screen_cache(cache: dict[str, Any]) -> None:
    DEFAULT_SCREEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SCREEN_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _read_cached_result(
    cache: dict[str, Any],
    ticker: str,
    now: datetime,
    cache_ttl_hours: int,
) -> CompanyScreenResult | None:
    record = cache.get(ticker)
    if not isinstance(record, dict):
        return None
    fetched_at = _parse_datetime(str(record.get("fetched_at", "")))
    if fetched_at is None:
        return None
    age_hours = (now - fetched_at).total_seconds() / 3600
    if age_hours > cache_ttl_hours:
        return None
    result = record.get("result")
    if not isinstance(result, dict):
        return None
    return CompanyScreenResult(**result)
