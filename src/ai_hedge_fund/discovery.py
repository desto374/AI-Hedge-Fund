from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ai_hedge_fund.tools.custom_tools import (
    _fetch_json_with_headers,
    _headline_sentiment_score,
    _parse_datetime,
    _recency_weight,
    _source_weight,
    _simple_moving_average,
)

DEFAULT_DISCOVERY_UNIVERSE = [
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "COST",
    "CRM",
    "CRWD",
    "GOOGL",
    "HOOD",
    "INTU",
    "LLY",
    "META",
    "MSFT",
    "NET",
    "NFLX",
    "NOW",
    "NVDA",
    "ORCL",
    "PANW",
    "PLTR",
    "SHOP",
    "SNOW",
    "TSLA",
    "UBER",
]


@dataclass
class DiscoveryCandidate:
    ticker: str
    company_name: str
    earnings_date: date
    days_until_earnings: int
    score: float
    price: float
    momentum_20d_pct: float
    momentum_60d_pct: float
    news_score: float

    @property
    def upcoming_event(self) -> str:
        return f"Earnings call on {self.earnings_date.isoformat()}"

    @property
    def thesis(self) -> str:
        parts = [
            f"Auto-discovered ahead of earnings in {self.days_until_earnings} day(s)",
            f"20d momentum {self.momentum_20d_pct:.1f}%",
            f"60d momentum {self.momentum_60d_pct:.1f}%",
            f"news score {self.news_score:.2f}",
        ]
        return "; ".join(parts) + "."


def discover_candidate(
    earnings_window_days: int = 7,
    max_symbols: int = 75,
    min_price: float = 10.0,
    universe_file: str = "",
) -> DiscoveryCandidate:
    if earnings_window_days <= 0:
        raise ValueError("earnings_window_days must be greater than zero.")
    if max_symbols <= 0:
        raise ValueError("max_symbols must be greater than zero.")
    if min_price <= 0:
        raise ValueError("min_price must be greater than zero.")
    if find_spec("yfinance") is None:
        raise RuntimeError("Install yfinance to use auto-discovery.")

    import yfinance as yf

    symbols = _load_symbol_universe(max_symbols=max_symbols, universe_file=universe_file)
    today = datetime.now(timezone.utc).date()
    best_candidate: DiscoveryCandidate | None = None
    rejection_reasons: list[str] = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            earnings_date = _extract_earnings_date(ticker)
            if earnings_date is None:
                continue
            days_until = (earnings_date - today).days
            if days_until < 0 or days_until > earnings_window_days:
                continue

            history = ticker.history(period="6mo", interval="1d", auto_adjust=False)
            closes = _extract_close_values(history)
            if len(closes) < 60:
                rejection_reasons.append(f"{symbol}: insufficient price history")
                continue

            price = closes[-1]
            if price < min_price:
                rejection_reasons.append(f"{symbol}: price below min_price")
                continue

            score, details = _score_candidate(
                closes=closes,
                earnings_window_days=earnings_window_days,
                days_until_earnings=days_until,
                news_items=list(getattr(ticker, "news", []) or []),
            )
            company_name = _extract_company_name(ticker, fallback=symbol)
            candidate = DiscoveryCandidate(
                ticker=symbol,
                company_name=company_name,
                earnings_date=earnings_date,
                days_until_earnings=days_until,
                score=score,
                price=price,
                momentum_20d_pct=details["momentum_20d_pct"],
                momentum_60d_pct=details["momentum_60d_pct"],
                news_score=details["news_score"],
            )
            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate
        except Exception as exc:
            rejection_reasons.append(f"{symbol}: {exc}")

    if best_candidate is None:
        reason = "; ".join(rejection_reasons[:5]) or "no symbols matched the earnings window"
        raise RuntimeError(
            "Auto-discovery did not find a candidate. Adjust the universe, earnings window, "
            f"or price filter. Recent issues: {reason}."
        )
    return best_candidate


def _load_symbol_universe(max_symbols: int, universe_file: str) -> list[str]:
    explicit_file = universe_file.strip()
    env_file = os.getenv("DISCOVERY_UNIVERSE_FILE", "").strip()
    if explicit_file:
        symbols = _read_symbols_from_file(Path(explicit_file))
    elif env_file:
        symbols = _read_symbols_from_file(Path(env_file))
    else:
        symbols = _fetch_alpaca_universe()
        if not symbols:
            symbols = list(DEFAULT_DISCOVERY_UNIVERSE)

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        cleaned = str(symbol).strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
        if len(normalized) >= max_symbols:
            break
    return normalized


def _read_symbols_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"Discovery universe file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            return []
        fieldnames = reader.fieldnames or []
        symbol_key = next(
            (name for name in fieldnames if name.lower() in {"symbol", "ticker", "tickers"}),
            fieldnames[0],
        )
        return [str(row.get(symbol_key, "")).strip() for row in rows]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _fetch_alpaca_universe() -> list[str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return []

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2").rstrip("/")
    result = _fetch_json_with_headers(
        f"{base_url}/assets?status=active&asset_class=us_equity",
        headers,
    )
    if isinstance(result, str):
        return []

    filtered = []
    for asset in result:
        if not asset.get("tradable", False):
            continue
        if not asset.get("fractionable", True):
            continue
        exchange = str(asset.get("exchange", "")).upper()
        if exchange not in {"NASDAQ", "NYSE", "ARCA"}:
            continue
        filtered.append(str(asset.get("symbol", "")).strip().upper())
    return filtered


def _extract_earnings_date(ticker: Any) -> date | None:
    calendar = getattr(ticker, "calendar", None)
    parsed = _extract_date_from_calendar(calendar)
    if parsed is not None:
        return parsed

    earnings_dates = getattr(ticker, "earnings_dates", None)
    parsed = _extract_date_from_earnings_dates(earnings_dates)
    if parsed is not None:
        return parsed

    getter = getattr(ticker, "get_earnings_dates", None)
    if callable(getter):
        parsed = _extract_date_from_earnings_dates(getter(limit=4))
        if parsed is not None:
            return parsed
    return None


def _extract_date_from_calendar(calendar: Any) -> date | None:
    if calendar is None:
        return None
    if isinstance(calendar, dict):
        for key, value in calendar.items():
            if "earnings" not in str(key).lower():
                continue
            parsed = _coerce_date(value)
            if parsed is not None:
                return parsed
        return None
    if hasattr(calendar, "to_dict"):
        return _extract_date_from_calendar(calendar.to_dict())
    if hasattr(calendar, "items"):
        try:
            return _extract_date_from_calendar(dict(calendar.items()))
        except Exception:
            return None
    return _coerce_date(calendar)


def _extract_date_from_earnings_dates(earnings_dates: Any) -> date | None:
    if earnings_dates is None:
        return None
    index = getattr(earnings_dates, "index", None)
    if index is not None:
        for value in index:
            parsed = _coerce_date(value)
            if parsed is not None:
                return parsed
    if isinstance(earnings_dates, list):
        for item in earnings_dates:
            parsed = _coerce_date(item)
            if parsed is not None:
                return parsed
    return None


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        parsed_dt = _parse_datetime(value)
        if parsed_dt is not None:
            return parsed_dt.date()
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _coerce_date(item)
            if parsed is not None:
                return parsed
    if hasattr(value, "tolist"):
        try:
            return _coerce_date(value.tolist())
        except Exception:
            return None
    if hasattr(value, "item"):
        try:
            return _coerce_date(value.item())
        except Exception:
            return None
    return None


def _extract_close_values(history: Any) -> list[float]:
    if getattr(history, "empty", False):
        return []
    closes = history["Close"].dropna().tolist()
    return [float(value) for value in closes]


def _score_candidate(
    closes: list[float],
    earnings_window_days: int,
    days_until_earnings: int,
    news_items: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    latest_price = closes[-1]
    sma_20 = _simple_moving_average(closes[-20:])
    base_20 = closes[-21] if len(closes) > 20 else closes[0]
    base_60 = closes[-61] if len(closes) > 60 else closes[0]
    momentum_20d_pct = ((latest_price / base_20) - 1) * 100 if base_20 else 0.0
    momentum_60d_pct = ((latest_price / base_60) - 1) * 100 if base_60 else 0.0
    news_score = _average_news_score(news_items[:5])

    score = 0.0
    score += max(0.0, earnings_window_days - days_until_earnings + 1) * 0.35
    score += max(min(momentum_20d_pct / 8, 2.0), -2.0)
    score += max(min(momentum_60d_pct / 12, 2.0), -2.0)
    score += 1.0 if latest_price >= sma_20 else -1.0
    score += max(min(news_score, 1.5), -1.5)
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


def _extract_company_name(ticker: Any, fallback: str) -> str:
    info = getattr(ticker, "fast_info", None)
    if isinstance(info, dict):
        for key in ("shortName", "longName", "displayName"):
            value = info.get(key)
            if value:
                return str(value)

    info = getattr(ticker, "info", None)
    if isinstance(info, dict):
        for key in ("shortName", "longName", "displayName"):
            value = info.get(key)
            if value:
                return str(value)
    return fallback
