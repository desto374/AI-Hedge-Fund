from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ai_hedge_fund.screening import select_top_company_results
from ai_hedge_fund.screening import screen_companies
from ai_hedge_fund.screening import write_screen_summary


RUN_OUTPUT_FILES = [
    "discovery_selection.json",
    "portfolio_decision.json",
    "trade_decision.md",
]


def parse_tickers(
    ticker: str,
    tickers: str = "",
    tickers_file: str = "",
    companies: str = "",
    companies_file: str = "",
) -> list[str]:
    parsed: list[str] = []
    seen: set[str] = set()

    for raw in _iter_ticker_inputs(
        ticker=ticker,
        tickers=tickers,
        tickers_file=tickers_file,
        companies=companies,
        companies_file=companies_file,
    ):
        normalized = raw.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parsed.append(normalized)
    return parsed


def run_for_tickers(args: Any, runner: Any) -> list[str]:
    tickers = parse_tickers(
        ticker=getattr(args, "ticker", ""),
        tickers=getattr(args, "tickers", ""),
        tickers_file=getattr(args, "tickers_file", ""),
        companies=getattr(args, "companies", ""),
        companies_file=getattr(args, "companies_file", ""),
    )
    if getattr(args, "auto_discover", False):
        runner(args)
        return []

    if not tickers:
        raise SystemExit("Provide --ticker, --tickers, or --tickers-file.")

    screen_results = screen_companies(
        tickers=tickers,
        min_price=getattr(args, "discovery_min_price", 10.0),
        earnings_window_days=getattr(args, "discovery_window_days", 7),
        cache_ttl_hours=getattr(args, "screen_cache_ttl_hours", 6),
    )
    if not screen_results:
        raise SystemExit("No companies passed the screening stage.")
    selected_results = select_top_company_results(
        results=screen_results,
        top_percent=getattr(args, "top_percent", 30.0),
    )
    if not selected_results:
        raise SystemExit("Top percent selection returned no companies to analyze.")
    write_screen_summary(screen_results, selected_results)
    print(
        "Batch screening selected: "
        + ", ".join(result.ticker for result in selected_results)
        + f" out of {len(screen_results)} screened companies"
    )

    archived_paths: list[str] = []
    original_ticker = getattr(args, "ticker", "")
    for result in selected_results:
        args.ticker = result.ticker
        args.auto_discover = False
        os.environ["AI_HEDGE_FUND_FORCED_TICKER"] = result.ticker
        os.environ["AI_HEDGE_FUND_FORCE_MANUAL_TICKER"] = "true"
        print(f"Running crew for ticker={args.ticker} auto_discover={args.auto_discover}")
        runner(args)
        archived_paths.extend(_archive_run_outputs(result.ticker))
    args.ticker = original_ticker
    os.environ.pop("AI_HEDGE_FUND_FORCED_TICKER", None)
    os.environ["AI_HEDGE_FUND_FORCE_MANUAL_TICKER"] = "false"
    return archived_paths


def _iter_ticker_inputs(
    ticker: str,
    tickers: str,
    tickers_file: str,
    companies: str,
    companies_file: str,
):
    if companies_file.strip():
        path = Path(companies_file.strip())
        if not path.exists():
            raise SystemExit(f"Companies file not found: {path}")
        yield from path.read_text(encoding="utf-8").replace(",", "\n").splitlines()
        return
    if companies.strip():
        yield from companies.replace(" ", "").split(",")
        return
    if tickers_file.strip():
        path = Path(tickers_file.strip())
        if not path.exists():
            raise SystemExit(f"Tickers file not found: {path}")
        yield from path.read_text(encoding="utf-8").replace(",", "\n").splitlines()
        return
    if tickers.strip():
        yield from tickers.replace(" ", "").split(",")
        return
    if ticker.strip():
        yield ticker


def _archive_run_outputs(ticker: str) -> list[str]:
    output_dir = Path("output")
    batch_dir = output_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    archived: list[str] = []
    for filename in RUN_OUTPUT_FILES:
        source = output_dir / filename
        if not source.exists():
            continue
        target = batch_dir / f"{ticker.lower()}_{filename}"
        shutil.copy2(source, target)
        archived.append(str(target))
    return archived
