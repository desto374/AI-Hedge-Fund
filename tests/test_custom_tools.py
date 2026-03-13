from ai_hedge_fund.tools.custom_tools import (
    BacktestTool,
    CandidateDiscoveryTool,
    ExecutionPlanTool,
    LiveMarketDataTool,
    LiveOptionsChainTool,
    MarketResearchTool,
    NewsSentimentTool,
    OptionsStrategyTool,
    PortfolioStateTool,
    RiskBudgetTool,
    TechnicalIndicatorTool,
    TradeContextTool,
)


def test_market_research_tool_formats_brief() -> None:
    tool = MarketResearchTool()
    result = tool._run(
        ticker="msft",
        thesis="Cloud demand remains firm",
        macro_view="Enterprise spending is stable",
        upcoming_event="Earnings call",
    )
    assert "Ticker: MSFT" in result
    assert "Core thesis: Cloud demand remains firm" in result


def test_trade_context_tool_reads_structured_discovery_and_decision(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "discovery_selection.json").write_text(
        (
            '{"discovery_status":"accepted","selected_ticker":"NVDA",'
            '"upcoming_event":"Earnings call on 2026-03-18","thesis":"Auto-discovered."}'
        ),
        encoding="utf-8",
    )
    (output_dir / "portfolio_decision.json").write_text(
        '{"ticker":"NVDA","final_action":"buy","confidence":"high"}',
        encoding="utf-8",
    )

    tool = TradeContextTool()
    result = tool._run(include_decision=True)

    assert "Selected ticker: NVDA" in result
    assert "Final action: buy" in result


def test_candidate_discovery_tool_persists_structured_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tool = CandidateDiscoveryTool()
    result = tool._run(
        ticker="AAPL",
        auto_discover=False,
        thesis="Manual thesis",
        upcoming_event="Manual event",
    )
    saved = (tmp_path / "output" / "discovery_selection.json").read_text(encoding="utf-8")

    assert "Structured payload:" in result
    assert '"selected_ticker": "AAPL"' in saved


def test_live_market_data_tool_formats_alpaca_snapshot(monkeypatch) -> None:
    tool = LiveMarketDataTool()
    responses = iter(
        [
            {
                "latestTrade": {"p": 101.25},
                "latestQuote": {"bp": 101.2, "ap": 101.3},
                "minuteBar": {"c": 101.1},
                "dailyBar": {"o": 99.5, "h": 102.0, "l": 99.0},
                "prevDailyBar": {"c": 98.75},
            },
            {
                "bars": {
                    "AAPL": [
                        {"c": 95.0},
                        {"c": 95.5},
                        {"c": 96.0},
                        {"c": 96.5},
                        {"c": 97.0},
                        {"c": 97.5},
                        {"c": 98.0},
                        {"c": 98.5},
                        {"c": 99.0},
                        {"c": 99.5},
                        {"c": 100.0},
                        {"c": 100.5},
                        {"c": 101.0},
                        {"c": 101.5},
                        {"c": 102.0},
                        {"c": 102.5},
                        {"c": 103.0},
                        {"c": 103.5},
                        {"c": 104.0},
                        {"c": 104.5},
                        {"c": 105.0},
                        {"c": 105.5},
                        {"c": 106.0},
                        {"c": 106.5},
                        {"c": 107.0},
                        {"c": 107.5},
                        {"c": 108.0},
                        {"c": 108.5},
                        {"c": 109.0},
                        {"c": 109.5},
                    ]
                }
            },
        ]
    )

    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(tool, "_fetch_json", lambda url, api_key, secret_key: next(responses))

    result = tool._run("aapl", bar_limit=30)
    assert "Provider: alpaca" in result
    assert "Ticker: AAPL" in result
    assert "Last trade price: 101.25" in result
    assert "Derived MACD signal: bullish" in result


def test_live_market_data_tool_rejects_unknown_provider(monkeypatch) -> None:
    tool = LiveMarketDataTool()
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "unknown")
    result = tool._run("aapl", bar_limit=30)
    assert "MARKET_DATA_PROVIDER must be alpaca or yfinance." in result


def test_live_market_data_tool_falls_back_to_yfinance(monkeypatch) -> None:
    tool = LiveMarketDataTool()

    class FakeTicker:
        def history(self, period: str, interval: str, auto_adjust: bool):
            class FakeSeries:
                def __init__(self, values):
                    self._values = values

                def dropna(self):
                    return self

                def tail(self, count: int):
                    return FakeSeries(self._values[-count:])

                def tolist(self):
                    return list(self._values)

                @property
                def empty(self):
                    return len(self._values) == 0

                @property
                def iloc(self):
                    class _ILoc:
                        def __init__(self, values):
                            self._values = values

                        def __getitem__(self, index: int):
                            return self._values[index]

                    return _ILoc(self._values)

            class FakeHistory:
                empty = False

                def __getitem__(self, key: str):
                    values = {
                        "Close": [100.0 + i for i in range(30)],
                        "Open": [99.0 + i for i in range(30)],
                        "High": [101.0 + i for i in range(30)],
                        "Low": [98.0 + i for i in range(30)],
                    }[key]
                    return FakeSeries(values)

            return FakeHistory()

    class FakeYFinanceModule:
        @staticmethod
        def Ticker(symbol: str):
            return FakeTicker()

    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr("ai_hedge_fund.tools.custom_tools.find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYFinanceModule())

    result = tool._run("aapl", bar_limit=30)
    assert "Provider: yfinance" in result
    assert "Fallback note: Alpaca failed" in result


def test_technical_indicator_tool_scores_bullish_setup() -> None:
    tool = TechnicalIndicatorTool()
    result = tool._run(
        ticker="NVDA",
        price=910.0,
        rsi=60.0,
        macd_signal="bullish",
        moving_average_gap_pct=4.0,
        bollinger_position="upper",
    )
    assert "Technical score: 4" in result
    assert "Setup: bullish" in result


def test_news_sentiment_tool_formats_news_summary(monkeypatch) -> None:
    tool = NewsSentimentTool()
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(
        "ai_hedge_fund.tools.custom_tools._fetch_json_with_headers",
        lambda url, headers: {
            "news": [
                {"headline": "Company beats estimates", "summary": "Strong growth continues", "source": "Wire"},
                {"headline": "Analyst upgrades stock", "summary": "Profit outlook improves", "source": "Desk"},
            ]
        },
    )
    result = tool._run("TSLA", limit=2)
    assert "Ticker: TSLA" in result
    assert "Stance: positive" in result
    assert "score=" in result


def test_live_options_chain_tool_formats_yfinance_fallback(monkeypatch) -> None:
    tool = LiveOptionsChainTool()

    class FakeFrame:
        empty = False

        def sort_values(self, by: str, ascending: bool):
            return self

        def head(self, count: int):
            return self

        def iterrows(self):
            yield 0, {
                "contractSymbol": "AAPL240621C00190000",
                "bid": 4.0,
                "ask": 4.2,
                "impliedVolatility": 0.25,
                "openInterest": 1200,
            }

    class FakeChain:
        calls = FakeFrame()
        puts = FakeFrame()

    class FakeTicker:
        options = ["2026-06-19"]

        def option_chain(self, expiration: str):
            return FakeChain()

    class FakeYF:
        @staticmethod
        def Ticker(symbol: str):
            return FakeTicker()

    monkeypatch.setenv("MARKET_DATA_PROVIDER", "yfinance")
    monkeypatch.setattr("ai_hedge_fund.tools.custom_tools.find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF())
    result = tool._run("AAPL")
    assert "Provider: yfinance" in result
    assert "Contracts reviewed:" in result


def test_portfolio_state_tool_formats_account(monkeypatch) -> None:
    tool = PortfolioStateTool()
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    responses = iter(
        [
            {"status": "ACTIVE", "equity": "105000", "cash": "25000", "buying_power": "50000", "daytrade_count": 0},
            [{"symbol": "AAPL", "qty": "10", "market_value": "1800"}],
        ]
    )
    monkeypatch.setattr(
        "ai_hedge_fund.tools.custom_tools._fetch_json_with_headers",
        lambda url, headers: next(responses),
    )
    result = tool._run("AAPL")
    assert "Equity: 105000.00" in result
    assert "Focus position: AAPL qty 10" in result


def test_backtest_tool_formats_strategy_metrics(monkeypatch) -> None:
    tool = BacktestTool()

    class FakeSeries:
        def __init__(self, values):
            self._values = values

        def dropna(self):
            return self

        def tolist(self):
            return list(self._values)

    class FakeHistory:
        empty = False

        def __getitem__(self, key: str):
            return FakeSeries([100.0 + i for i in range(80)])

    class FakeTicker:
        def history(self, period: str, interval: str, auto_adjust: bool):
            return FakeHistory()

    class FakeYF:
        @staticmethod
        def Ticker(symbol: str):
            return FakeTicker()

    monkeypatch.setattr("ai_hedge_fund.tools.custom_tools.find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF())
    result = tool._run("AAPL", period="1y")
    assert "Strategy: 20/50 moving-average crossover" in result
    assert "Benchmark return:" in result


def test_options_strategy_tool_selects_spread_for_high_iv_bullish_case() -> None:
    tool = OptionsStrategyTool()
    result = tool._run(
        ticker="AAPL",
        directional_bias="bullish",
        implied_volatility_pct=45.0,
        days_to_expiry=14,
    )
    assert "Suggested options structure: bull call spread" in result


def test_risk_budget_tool_returns_position_size() -> None:
    tool = RiskBudgetTool()
    result = tool._run(
        portfolio_value=100000.0,
        max_position_pct=5.0,
        stop_loss_pct=4.0,
        conviction=0.5,
        price=100.0,
    )
    assert "Recommended shares: 25" in result
    assert "Estimated notional: 2500.00" in result


def test_execution_plan_tool_flags_alpaca_only_for_api_modes() -> None:
    tool = ExecutionPlanTool()
    manual_result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="manual",
        broker="",
    )
    api_result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="paper",
        broker="Alpaca",
    )
    assert "No API integration required." in manual_result
    assert "Connect a brokerage API such as Alpaca" in api_result
    assert "Not submitted. Set ALPACA_ENABLE_SUBMISSION=true" in api_result


def test_execution_plan_tool_respects_kill_switch(monkeypatch) -> None:
    tool = ExecutionPlanTool()
    monkeypatch.setenv("ALPACA_ENABLE_SUBMISSION", "true")
    monkeypatch.setenv("ALPACA_KILL_SWITCH", "true")
    result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="paper",
        broker="Alpaca",
    )
    assert "ALPACA_KILL_SWITCH is enabled." in result


def test_execution_plan_tool_blocks_when_market_closed(monkeypatch) -> None:
    tool = ExecutionPlanTool()
    monkeypatch.setenv("ALPACA_ENABLE_SUBMISSION", "true")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.delenv("ALPACA_ALLOW_EXTENDED_HOURS", raising=False)
    responses = iter([{"is_open": False}])
    monkeypatch.setattr(
        "ai_hedge_fund.tools.custom_tools._fetch_json_with_headers",
        lambda url, headers: next(responses),
    )
    result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="paper",
        broker="Alpaca",
    )
    assert "Market is closed" in result


def test_execution_plan_tool_blocks_duplicate_open_order(monkeypatch) -> None:
    tool = ExecutionPlanTool()
    monkeypatch.setenv("ALPACA_ENABLE_SUBMISSION", "true")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.setenv("ALPACA_ALLOW_EXTENDED_HOURS", "true")
    responses = iter([[{"symbol": "AAPL", "side": "buy", "qty": "25"}]])
    monkeypatch.setattr(
        "ai_hedge_fund.tools.custom_tools._fetch_json_with_headers",
        lambda url, headers: {"is_open": True} if url.endswith("/clock") else next(responses),
    )
    result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="paper",
        broker="Alpaca",
    )
    assert "Duplicate open order detected" in result


def test_execution_plan_tool_does_not_submit_hold_or_reduce() -> None:
    tool = ExecutionPlanTool()
    result = tool._run(
        ticker="AAPL",
        action="hold",
        shares=25,
        mode="paper",
        broker="Alpaca",
    )
    assert "Manual review required for hold/reduce decisions." in result
    assert "Discovery status: accepted" in result


def test_execution_plan_tool_blocks_non_hold_when_discovery_rejected() -> None:
    tool = ExecutionPlanTool()
    result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="paper",
        broker="Alpaca",
        discovery_status="rejected",
    )
    assert "Discovery rejected the setup" in result


def test_execution_plan_tool_allows_hold_when_discovery_rejected() -> None:
    tool = ExecutionPlanTool()
    result = tool._run(
        ticker="AAPL",
        action="hold",
        shares=0,
        mode="paper",
        broker="Alpaca",
        discovery_status="rejected",
    )
    assert "Discovery status: rejected" in result
    assert "Manual review required for hold/reduce decisions." in result


def test_execution_plan_tool_rejects_invalid_limit_order() -> None:
    tool = ExecutionPlanTool()
    result = tool._run(
        ticker="AAPL",
        action="buy",
        shares=25,
        mode="paper",
        broker="Alpaca",
        order_type="limit",
    )
    assert "Limit orders require a limit_price." in result
