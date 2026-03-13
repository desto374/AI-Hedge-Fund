from __future__ import annotations

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from ai_hedge_fund.models import DiscoverySelection, PortfolioDecision
from ai_hedge_fund.tools.custom_tools import (
    BacktestTool,
    CandidateDiscoveryTool,
    ExecutionPlanTool,
    LiveMarketDataTool,
    LiveOptionsChainTool,
    MarketResearchTool,
    NewsSentimentTool,
    PortfolioStateTool,
    OptionsStrategyTool,
    RiskBudgetTool,
    TechnicalIndicatorTool,
    TradeContextTool,
)


@CrewBase
class AIHedgeFundCrew:
    """Multi-agent hedge fund scaffold with manager-led trade decisions."""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def discovery_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["discovery_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[CandidateDiscoveryTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def market_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_research_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), MarketResearchTool(), LiveMarketDataTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), LiveMarketDataTool(), TechnicalIndicatorTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), NewsSentimentTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def options_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["options_strategist"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), LiveOptionsChainTool(), OptionsStrategyTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def risk_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_manager"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), PortfolioStateTool(), RiskBudgetTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def strategy_backtester(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_backtester"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), BacktestTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["portfolio_manager"],  # type: ignore[index]
            verbose=True,
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def execution_operator(self) -> Agent:
        return Agent(
            config=self.agents_config["execution_operator"],  # type: ignore[index]
            verbose=True,
            tools=[TradeContextTool(), ExecutionPlanTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @task
    def discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config["discovery_task"],  # type: ignore[index]
            output_json=DiscoverySelection,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            context=[self.discovery_task()],
        )

    @task
    def technical_task(self) -> Task:
        return Task(
            config=self.tasks_config["technical_task"],  # type: ignore[index]
            context=[self.discovery_task()],
        )

    @task
    def sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config["sentiment_task"],  # type: ignore[index]
            context=[self.discovery_task()],
        )

    @task
    def options_task(self) -> Task:
        return Task(
            config=self.tasks_config["options_task"],  # type: ignore[index]
            context=[self.discovery_task()],
        )

    @task
    def risk_task(self) -> Task:
        return Task(
            config=self.tasks_config["risk_task"],  # type: ignore[index]
            context=[self.discovery_task()],
        )

    @task
    def backtest_task(self) -> Task:
        return Task(
            config=self.tasks_config["backtest_task"],  # type: ignore[index]
            context=[self.discovery_task()],
        )

    @task
    def decision_task(self) -> Task:
        return Task(
            config=self.tasks_config["decision_task"],  # type: ignore[index]
            output_json=PortfolioDecision,
            output_file="output/portfolio_decision.json",
            context=[
                self.discovery_task(),
                self.research_task(),
                self.technical_task(),
                self.sentiment_task(),
                self.options_task(),
                self.risk_task(),
                self.backtest_task(),
            ],
        )

    @task
    def execution_task(self) -> Task:
        return Task(
            config=self.tasks_config["execution_task"],  # type: ignore[index]
            context=[self.decision_task(), self.risk_task()],
            output_file="output/trade_decision.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
