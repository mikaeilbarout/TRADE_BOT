from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import AgentAuditLog, TradeDecision, TradeOutcome
from app.models.enums import ApprovalStatus
from app.models.pipeline_result import PipelineResult


class DecisionRepository:
    """Append-only writer/reader for the audit trail. AI decisions are never
    revised in place -- every pipeline run is a new record. The only updates
    are human approval columns, which record a distinct later action
    (section 17/30)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self, result: PipelineResult, execution_response: dict
    ) -> TradeDecision:
        approval_status = execution_response.get(
            "approval_status", result.approval_status.value
        )
        row = TradeDecision(
            signal_id=result.signal_id,
            symbol=result.signal.symbol,
            side=result.signal.side.value,
            original_signal=result.signal.model_dump(mode="json"),
            market_price=result.market_snapshot.last_price if result.market_snapshot else None,
            market_snapshot=(
                result.market_snapshot.model_dump(
                    mode="json", exclude={"timeframes"}
                )
                if result.market_snapshot
                else None
            ),
            news_result=result.news_result.model_dump(mode="json") if result.news_result else None,
            sentiment_result=(
                result.sentiment_result.model_dump(mode="json")
                if result.sentiment_result
                else None
            ),
            technical_result=(
                result.technical_result.model_dump(mode="json")
                if result.technical_result
                else None
            ),
            final_result=(
                result.final_result.model_dump(mode="json")
                if result.final_result
                # SingleAgentPipeline has no separate final agent -- its one
                # decision plays the same role, so it lands in the same
                # column rather than leaving dashboards/queries that read
                # final_result empty for every single-agent decision.
                else (
                    result.unified_result.model_dump(mode="json")
                    if result.unified_result
                    else None
                )
            ),
            decision=execution_response.get("decision", result.decision.value),
            ai_decision=(result.ai_decision.value if result.ai_decision else None),
            confidence=result.confidence,
            weighted_score=result.weighted_score,
            reason=result.reason,
            modified_trade=(
                result.modified_trade.model_dump(mode="json") if result.modified_trade else None
            ),
            veto_triggered=result.veto_triggered,
            veto_reasons=result.veto_reasons,
            policy_warnings=result.policy_warnings,
            guard_violations=result.guard_violations,
            stage_reached=result.stage_reached.value,
            short_circuited=result.short_circuited,
            degraded=result.degraded,
            errors=result.errors,
            latencies=[latency.model_dump(mode="json") for latency in result.latencies],
            latency_seconds=result.total_latency_seconds,
            trading_mode=execution_response.get("trading_mode", "unknown"),
            data_sources=result.data_sources.model_dump(mode="json"),
            execution_result=execution_response,
            approval_status=approval_status,
        )
        self._session.add(row)
        await self._session.flush()

        for trace in result.agent_traces:
            self._session.add(
                AgentAuditLog(
                    trade_decision_id=row.id,
                    signal_id=result.signal_id,
                    agent_name=trace.agent_name,
                    input_snapshot=trace.input_snapshot,
                    output=trace.output,
                    error=trace.error,
                    model=trace.model,
                    model_version=trace.model_version,
                    attempts=trace.attempts,
                    latency_seconds=trace.latency_seconds,
                    decision=trace.decision,
                )
            )

        await self._session.commit()
        return row

    async def get_by_signal_id(self, signal_id: str) -> TradeDecision | None:
        stmt = (
            select(TradeDecision)
            .where(TradeDecision.signal_id == signal_id)
            .order_by(TradeDecision.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def get_agent_logs(self, trade_decision_id: str) -> list[AgentAuditLog]:
        stmt = (
            select(AgentAuditLog)
            .where(AgentAuditLog.trade_decision_id == trade_decision_id)
            .order_by(AgentAuditLog.timestamp)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent(self, limit: int = 50) -> list[TradeDecision]:
        stmt = select(TradeDecision).order_by(TradeDecision.created_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_awaiting_approval(self, limit: int = 50) -> list[TradeDecision]:
        stmt = (
            select(TradeDecision)
            .where(TradeDecision.approval_status == ApprovalStatus.AWAITING_HUMAN.value)
            .order_by(TradeDecision.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def record_outcome(
        self,
        *,
        signal_id: str,
        profit: float,
        exit_reason: str,
        ticket: str | None = None,
        entry_price: float | None = None,
        close_price: float | None = None,
        volume: float | None = None,
        closed_at: datetime | None = None,
    ) -> TradeOutcome:
        row = TradeOutcome(
            signal_id=signal_id,
            ticket=ticket,
            profit=profit,
            exit_reason=exit_reason,
            entry_price=entry_price,
            close_price=close_price,
            volume=volume,
            closed_at=closed_at or datetime.now(timezone.utc),
        )
        self._session.add(row)
        await self._session.commit()
        return row

    async def get_outcomes(self, signal_id: str) -> list[TradeOutcome]:
        stmt = (
            select(TradeOutcome)
            .where(TradeOutcome.signal_id == signal_id)
            .order_by(TradeOutcome.reported_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_outcomes_for(self, signal_ids: list[str]) -> dict[str, list[TradeOutcome]]:
        """Batch lookup so listing endpoints don't issue one query per row."""
        if not signal_ids:
            return {}
        stmt = (
            select(TradeOutcome)
            .where(TradeOutcome.signal_id.in_(signal_ids))
            .order_by(TradeOutcome.reported_at)
        )
        result = await self._session.execute(stmt)
        by_signal: dict[str, list[TradeOutcome]] = {}
        for row in result.scalars().all():
            by_signal.setdefault(row.signal_id, []).append(row)
        return by_signal

    async def set_human_decision(
        self,
        row: TradeDecision,
        status: ApprovalStatus,
        reviewer: str | None,
        note: str | None,
    ) -> TradeDecision:
        row.approval_status = status.value
        row.human_reviewer = reviewer
        row.human_note = note
        row.human_decided_at = datetime.now(timezone.utc)
        await self._session.commit()
        return row
