from __future__ import annotations

from pydantic import BaseModel, Field


class DiscoverySelection(BaseModel):
    discovery_mode: str = Field(description="manual or auto")
    discovery_status: str = Field(description="accepted, rejected, or manual")
    selected_ticker: str = Field(description="Ticker selected for downstream analysis.")
    company_name: str = Field(description="Company name tied to the selected ticker.")
    earnings_date: str = Field(description="Earnings date in YYYY-MM-DD format.")
    days_until_earnings: int = Field(description="Days until the earnings event.")
    discovery_score: float = Field(description="Final discovery score for the selected ticker.")
    discovery_attempts_used: int = Field(description="How many discovery attempts were used.")
    price: float = Field(description="Latest stock price used during discovery.")
    momentum_20d_pct: float = Field(description="20-day price momentum in percent.")
    momentum_60d_pct: float = Field(description="60-day price momentum in percent.")
    news_score: float = Field(description="Average recent news score.")
    upcoming_event: str = Field(description="Human-readable event string.")
    thesis: str = Field(description="Short thesis for the selected ticker.")
    instruction: str = Field(description="Instruction for downstream tasks.")


class PortfolioDecision(BaseModel):
    ticker: str = Field(description="Ticker under final review.")
    discovery_status: str = Field(description="accepted, rejected, or manual")
    final_action: str = Field(description="buy, sell, reduce, or hold")
    confidence: str = Field(description="Confidence label or score from the manager.")
    rationale: str = Field(description="Short explanation for the final action.")
    conditions_to_change: str = Field(description="What would invalidate or change the decision.")
