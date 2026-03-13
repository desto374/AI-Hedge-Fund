from __future__ import annotations

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from ai_hedge_fund.tools.custom_tools import (
    BacktestTool,
    ExecutionPlanTool,
    LiveMarketDataTool,
    LiveOptionsChainTool,
    MarketResearchTool,
    NewsSentimentTool,
    PortfolioStateTool,
    OptionsStrategyTool,
    RiskBudgetTool,
    TechnicalIndicatorTool,
)


@CrewBase
class AIHedgeFundCrew:
    """Multi-agent hedge fund scaffold with manager-led trade decisions."""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def market_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_research_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[MarketResearchTool(), LiveMarketDataTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[LiveMarketDataTool(), TechnicalIndicatorTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyst"],  # type: ignore[index]
            verbose=True,
            tools=[NewsSentimentTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def options_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["options_strategist"],  # type: ignore[index]
            verbose=True,
            tools=[LiveOptionsChainTool(), OptionsStrategyTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def risk_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_manager"],  # type: ignore[index]
            verbose=True,
            tools=[PortfolioStateTool(), RiskBudgetTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @agent
    def strategy_backtester(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_backtester"],  # type: ignore[index]
            verbose=True,
            tools=[BacktestTool()],
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
            tools=[ExecutionPlanTool()],
            max_iter=20,
            respect_context_window=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])  # type: ignore[index]

    @task
    def technical_task(self) -> Task:
        return Task(config=self.tasks_config["technical_task"])  # type: ignore[index]

    @task
    def sentiment_task(self) -> Task:
        return Task(config=self.tasks_config["sentiment_task"])  # type: ignore[index]

    @task
    def options_task(self) -> Task:
        return Task(config=self.tasks_config["options_task"])  # type: ignore[index]

    @task
    def risk_task(self) -> Task:
        return Task(config=self.tasks_config["risk_task"])  # type: ignore[index]

    @task
    def backtest_task(self) -> Task:
        return Task(config=self.tasks_config["backtest_task"])  # type: ignore[index]

    @task
    def decision_task(self) -> Task:
        return Task(
            config=self.tasks_config["decision_task"],  # type: ignore[index]
            context=[
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
