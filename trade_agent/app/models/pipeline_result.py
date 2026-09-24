from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.agent_decision import (
    FinalDecisionResult,
    NewsAgentResult,
    SentimentAgentResult,
    TechnicalAgentResult,
    UnifiedDecisionResult,
)
from app.models.enums import ApprovalStatus, FinalDecision, PipelineStage
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal
from app.models.trade import ModifiedTrade


class StageLatency(BaseModel):
    stage: str
    seconds: float


class AgentTrace(BaseModel):
    """The exact input an agent received and what it returned, kept so any
    decision can be reconstructed later (section 17)."""

    agent_name: str
    input_snapshot: dict
    output: dict | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    attempts: int = 1
    model: str | None = None
    model_version: str | None = None
    decision: str | None = None


class DataSources(BaseModel):
    """Which concrete providers and articles backed this decision, so
    source usefulness can be evaluated later (section 16)."""

    llm_provider: str | None = None
    llm_model: str | None = None
    market_data_provider: str | None = None
    news_provider: str | None = None
    sentiment_provider: str | None = None
    news_item_count: int = 0
    news_titles: list[str] = Field(default_factory=list)
    sentiment_item_count: int = 0
    sentiment_sources: list[str] = Field(default_factory=list)
    news_degraded: bool = False
    sentiment_degraded: bool = False
    market_data_stale: bool = False


class PipelineResult(BaseModel):
    """Everything produced while deciding one signal -- the full audit
    record, and the source for the external API response."""

    signal_id: str
    signal: TradeSignal
    decision: FinalDecision
    confidence: float
    reason: str
    modified_trade: ModifiedTrade | None = None

    market_snapshot: MarketSnapshot | None = None
    news_result: NewsAgentResult | None = None
    sentiment_result: SentimentAgentResult | None = None
    technical_result: TechnicalAgentResult | None = None
    final_result: FinalDecisionResult | None = None
    # Populated only by SingleAgentPipeline, in place of the four fields
    # above.
    unified_result: UnifiedDecisionResult | None = None

    agent_traces: list[AgentTrace] = Field(default_factory=list)
    data_sources: DataSources = Field(default_factory=DataSources)

    weighted_score: float = 0.0
    veto_triggered: bool = False
    veto_reasons: list[str] = Field(default_factory=list)
    policy_warnings: list[str] = Field(default_factory=list)
    guard_violations: list[str] = Field(default_factory=list)
    ai_decision: FinalDecision | None = None  # before deterministic overrides

    # True whenever execution must be prevented, whatever the label says.
    # REJECT always sets it; so does a WAIT caused by a safety condition
    # (imminent high-impact event, stale quote) as opposed to one caused by
    # thin evidence. A caller keying only on decision == REJECT silently
    # traded through the safety cases.
    execution_blocked: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    stage_reached: PipelineStage
    short_circuited: bool = False
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)
    latencies: list[StageLatency] = Field(default_factory=list)
    total_latency_seconds: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_api_response(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "execution_blocked": self.execution_blocked,
            "modified_trade": (
                self.modified_trade.model_dump(mode="json")
                if self.modified_trade
                else None
            ),
        }
