from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TradeDecision(Base):
    """One row per signal processed. This is the summary record; the
    full per-agent audit trail lives in AgentAuditLog rows keyed to the
    same decision. Never overwritten (section 17: complete audit trail) --
    the only mutable fields are the human-approval columns, which record a
    later, separate human action rather than revising the AI decision.
    """

    __tablename__ = "trade_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(8))
    original_signal: Mapped[dict] = mapped_column(JSON)
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    news_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sentiment_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    technical_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    decision: Mapped[str] = mapped_column(String(24), index=True)
    ai_decision: Mapped[str | None] = mapped_column(String(24), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    weighted_score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    modified_trade: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    veto_triggered: Mapped[bool] = mapped_column(default=False)
    veto_reasons: Mapped[list] = mapped_column(JSON, default=list)
    policy_warnings: Mapped[list] = mapped_column(JSON, default=list)
    guard_violations: Mapped[list] = mapped_column(JSON, default=list)

    stage_reached: Mapped[str] = mapped_column(String(32))
    short_circuited: Mapped[bool] = mapped_column(default=False)
    degraded: Mapped[bool] = mapped_column(default=False)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    latencies: Mapped[list] = mapped_column(JSON, default=list)
    latency_seconds: Mapped[float] = mapped_column(Float)
    trading_mode: Mapped[str] = mapped_column(String(16))
    data_sources: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    execution_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Human-in-the-loop (MANUAL mode, section 30).
    approval_status: Mapped[str] = mapped_column(String(24), default="NOT_REQUIRED", index=True)
    human_reviewer: Mapped[str | None] = mapped_column(String(120), nullable=True)
    human_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    human_decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    agent_logs: Mapped[list["AgentAuditLog"]] = relationship(
        back_populates="trade_decision", cascade="all, delete-orphan"
    )


class TradeOutcome(Base):
    """What actually happened to a position the bot opened after a prior
    TradeDecision approved it. Reported by the bot itself when MT5 tells it
    the position closed (stop/target/manual/time-stop) -- this service never
    talks to the broker, so it has no other way to know. Append-only, keyed
    by signal_id rather than merged into TradeDecision, for the same reason
    AgentAuditLog is separate (section 17: the original decision record is
    never revised in place)."""

    __tablename__ = "trade_outcomes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    ticket: Mapped[str | None] = mapped_column(String(32), nullable=True)
    profit: Mapped[float] = mapped_column(Float)
    exit_reason: Mapped[str] = mapped_column(String(32))
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class AgentAuditLog(Base):
    """Exact input/output for one agent call on one signal. Append-only --
    every pipeline run creates new rows, nothing is ever updated in place
    (section 17). `input_snapshot` is the actual payload that agent was
    given, not a summary, so a decision can be replayed."""

    __tablename__ = "agent_audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    trade_decision_id: Mapped[str] = mapped_column(ForeignKey("trade_decisions.id"), index=True)
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(64))
    input_snapshot: Mapped[dict] = mapped_column(JSON)
    output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempts: Mapped[int] = mapped_column(default=1)
    latency_seconds: Mapped[float] = mapped_column(Float)
    decision: Mapped[str | None] = mapped_column(String(24), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    trade_decision: Mapped[TradeDecision] = relationship(back_populates="agent_logs")
