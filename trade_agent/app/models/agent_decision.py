from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator

from app.models.enums import (
    AgentAgreement,
    Bias,
    ContradictionLevel,
    FinalDecision,
    GateDecision,
    ImpactLevel,
    MarketEnvironment,
    SentimentStrength,
    TechnicalDecision,
    TradeCompatibility,
)


class AgentDecisionBase(BaseModel):
    """Common envelope every agent result shares. Confidence here means
    'how reliable is this analysis', never 'probability the trade wins'
    (see spec section 19)."""

    agent: str
    symbol: str
    # Anthropic's strict tool schema rejects minimum/maximum on the wire
    # (see app.providers.llm.anthropic_provider._strip_unsupported_
    # constraints), so the bound is stated here in the description instead --
    # Pydantic still enforces it when the response is parsed back.
    confidence: float = Field(ge=0.0, le=1.0, description="Between 0.0 and 1.0.")
    # Not `Field(...)` (required) -- in practice the model reliably fills the
    # more specific agreement_explanation/independent_finding fields below
    # but sometimes omits this more generic one, which used to hard-fail the
    # whole agent call (and reject the trade) over a redundant paragraph.
    # ChainAwareResult backfills it from those fields when blank.
    reasoning: str = Field(
        default="",
        description="A few sentences explaining how the evidence led to this verdict.",
    )
    data_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    is_degraded: bool = False


class ChainAwareResult(AgentDecisionBase):
    """Adds the fields that make an agent's independence auditable.

    Every agent after the first must state how its own conclusion relates
    to the upstream chain and what it contributes that the chain did not
    already contain. This is how "do not blindly accept the previous
    agent" (section 18) becomes measurable instead of aspirational: a run
    where every agent reports AGREE with an empty independent_finding is a
    red flag visible in the audit trail.
    """

    agreement_with_previous: AgentAgreement = AgentAgreement.NOT_APPLICABLE
    agreement_explanation: str = ""
    independent_finding: str = ""

    @model_validator(mode="after")
    def _backfill_reasoning(self) -> "ChainAwareResult":
        if not self.reasoning:
            self.reasoning = " ".join(
                p for p in (self.agreement_explanation, self.independent_finding) if p
            ) or "(model did not provide a reasoning field for this verdict)"
        # `summary` is declared per-subclass (News/Sentiment/Technical/Final),
        # not here, but every concrete subclass has it by the time this runs.
        if hasattr(self, "summary") and not self.summary:
            self.summary = self.reasoning
        return self


class NewsSourceRef(BaseModel):
    title: str
    source: str
    timestamp: datetime
    impact: ImpactLevel
    bias: Bias
    relevance: str = ""


class NewsAgentResult(ChainAwareResult):
    agent: str = "news_agent"
    signal_side: str
    decision: GateDecision
    news_bias: Bias
    market_environment: MarketEnvironment
    major_events_detected: bool = False
    high_impact_event_within_minutes: bool = False
    risk_of_news_reversal: str = "LOW"  # LOW/MEDIUM/HIGH
    summary: str = Field(default="", description="A few sentences summarizing the news picture and why it supports this decision.")
    sources: list[NewsSourceRef] = Field(default_factory=list)


class SentimentAgentResult(ChainAwareResult):
    agent: str = "sentiment_agent"
    signal_side: str
    decision: GateDecision
    overall_sentiment: Bias
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Between -1.0 and 1.0.")
    sentiment_strength: SentimentStrength
    sentiment_momentum: str = "FLAT"  # RISING/FALLING/FLAT
    contradiction_level: ContradictionLevel
    trade_compatibility: TradeCompatibility
    low_quality_source_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Between 0.0 and 1.0."
    )
    manipulation_suspected: bool = False
    summary: str = Field(default="", description="A few sentences summarizing the sentiment read and why it supports this decision.")


class TechnicalAgentResult(ChainAwareResult):
    agent: str = "technical_agent"
    decision: TechnicalDecision
    higher_tf_trend: str = "RANGE"  # UPTREND/DOWNTREND/RANGE
    aligned_with_higher_tf: bool = True
    entry_valid: bool = True
    is_overextended: bool = False
    breakout_confirmed: bool | None = None
    false_breakout_risk: str = "LOW"
    risk_reward_ratio: float = 0.0
    stop_loss_logical: bool = True
    take_profit_realistic: bool = True
    volatility_acceptable: bool = True
    confluence_factors: list[str] = Field(default_factory=list)
    conflicting_factors: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="A few sentences summarizing the technical read and why it supports this decision.")


class DimensionScores(BaseModel):
    news_score: float = Field(ge=0, le=100, description="0 to 100.")
    sentiment_score: float = Field(ge=0, le=100, description="0 to 100.")
    technical_score: float = Field(ge=0, le=100, description="0 to 100.")
    risk_score: float = Field(ge=0, le=100, description="0 to 100.")
    weighted_total: float = Field(ge=0, le=100, description="0 to 100.")


class FinalDecisionResult(ChainAwareResult):
    agent: str = "final_decision_agent"
    decision: FinalDecision
    scores: DimensionScores
    veto_triggered: bool = False
    veto_reason: str | None = None
    chain_conflicts: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="A few sentences summarizing why this final decision was reached.")


class UnifiedDecisionResult(AgentDecisionBase):
    """The single-agent replacement for the news/sentiment/technical/final
    four-agent chain: one call sees the full multi-timeframe technical
    picture, news and sentiment together and decides once. No
    agreement_with_previous / independent_finding fields -- there is no
    upstream chain to be independent from."""

    agent: str = "unified_agent"
    daily_trend: str = "RANGE"  # UPTREND / DOWNTREND / RANGE, the agent's own D1 read
    trade_vs_daily_trend: str = "NEUTRAL"  # WITH / AGAINST / NEUTRAL
    high_impact_event_within_minutes: bool = False
    summary: str = Field(
        default="",
        description="A few sentences summarizing the full picture and why this decision was reached.",
    )
    # Declared LAST deliberately (2026-09-19, after a real mismatch found in
    # a backtest: `summary` argued "approving per default policy" while
    # `decision` -- generated earlier in the object -- said REJECT). Schema
    # field order is generation order for structured output, so `decision`
    # coming after every analysis field forces it to follow from what was
    # just written instead of being committed to before the reasoning
    # exists. `reasoning` (inherited from AgentDecisionBase) still precedes
    # it too now that `decision` is last, not the other way around.
    decision: FinalDecision

    @model_validator(mode="after")
    def _backfill_summary(self) -> "UnifiedDecisionResult":
        if not self.summary:
            self.summary = self.reasoning or "(model did not provide a summary for this verdict)"
        return self
