from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import AppContainer, get_container, get_repository, require_api_key
from app.database.models import TradeDecision, TradeOutcome
from app.database.repository import DecisionRepository
from app.models.enums import ApprovalStatus, FinalDecision
from app.models.signal import TradeSignal
from app.observability.logging import get_logger
from app.services.risk_service import AccountState

router = APIRouter(prefix="/api/v1", tags=["decisions"], dependencies=[Depends(require_api_key)])
logger = get_logger(__name__)


class HumanDecisionRequest(BaseModel):
    reviewer: str | None = None
    note: str | None = None


class TradeOutcomeRequest(BaseModel):
    """What the bot reports once MT5 shows the position closed. This
    service never talks to the broker itself, so this is the only way it
    learns what actually happened after a decision was approved."""

    profit: float
    exit_reason: str  # STOP_LOSS | TAKE_PROFIT | TIME_STOP | manual/expert/unknown
    ticket: str | None = None
    entry_price: float | None = None
    close_price: float | None = None
    volume: float | None = None
    closed_at: datetime | None = None


def _outcome_dict(o: TradeOutcome) -> dict:
    return {
        "profit": o.profit,
        "exit_reason": o.exit_reason,
        "ticket": o.ticket,
        "entry_price": o.entry_price,
        "close_price": o.close_price,
        "volume": o.volume,
        "closed_at": o.closed_at.isoformat(),
        "reported_at": o.reported_at.isoformat(),
    }


def _summary(row: TradeDecision, outcomes: list[TradeOutcome] | None = None) -> dict:
    outcomes = outcomes or []
    return {
        "signal_id": row.signal_id,
        "symbol": row.symbol,
        "side": row.side,
        "decision": row.decision,
        "ai_decision": row.ai_decision,
        "confidence": row.confidence,
        "weighted_score": row.weighted_score,
        "approval_status": row.approval_status,
        "stage_reached": row.stage_reached,
        "veto_triggered": row.veto_triggered,
        "degraded": row.degraded,
        "created_at": row.created_at.isoformat(),
        # last-reported outcome, if the bot has told us what happened yet
        "outcome": _outcome_dict(outcomes[-1]) if outcomes else None,
    }


def _detail(row: TradeDecision, outcomes: list[TradeOutcome] | None = None) -> dict:
    return {
        **_summary(row, outcomes),
        "original_signal": row.original_signal,
        "market_price": row.market_price,
        "market_snapshot": row.market_snapshot,
        "reason": row.reason,
        "modified_trade": row.modified_trade,
        "news_result": row.news_result,
        "sentiment_result": row.sentiment_result,
        "technical_result": row.technical_result,
        "final_result": row.final_result,
        "veto_reasons": row.veto_reasons,
        "policy_warnings": row.policy_warnings,
        "guard_violations": row.guard_violations,
        "short_circuited": row.short_circuited,
        "errors": row.errors,
        "latencies": row.latencies,
        "latency_seconds": row.latency_seconds,
        "trading_mode": row.trading_mode,
        "data_sources": row.data_sources,
        "execution_result": row.execution_result,
        "human_reviewer": row.human_reviewer,
        "human_note": row.human_note,
        "human_decided_at": (
            row.human_decided_at.isoformat() if row.human_decided_at else None
        ),
    }


@router.get("/decisions")
async def list_decisions(
    limit: int = 50, repository: DecisionRepository = Depends(get_repository)
) -> list[dict]:
    rows = await repository.list_recent(limit=limit)
    outcomes_by_signal = await repository.get_outcomes_for([row.signal_id for row in rows])
    return [_summary(row, outcomes_by_signal.get(row.signal_id)) for row in rows]


@router.get("/decisions/pending")
async def list_pending_decisions(
    limit: int = 50, repository: DecisionRepository = Depends(get_repository)
) -> list[dict]:
    """Everything awaiting human review in MANUAL mode. Returns the full
    detail (signal, news, sentiment, technical, final recommendation) so a
    reviewer has the complete picture without a second call (section 30)."""
    rows = await repository.list_awaiting_approval(limit=limit)
    outcomes_by_signal = await repository.get_outcomes_for([row.signal_id for row in rows])
    return [_detail(row, outcomes_by_signal.get(row.signal_id)) for row in rows]


@router.get("/decisions/{signal_id}")
async def get_decision(
    signal_id: str, repository: DecisionRepository = Depends(get_repository)
) -> dict:
    row = await repository.get_by_signal_id(signal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="signal_id not found")
    logs = await repository.get_agent_logs(row.id)
    outcomes = await repository.get_outcomes(signal_id)
    return {
        **_detail(row, outcomes),
        "outcomes": [_outcome_dict(o) for o in outcomes],
        "agent_audit_log": [
            {
                "agent_name": log.agent_name,
                "decision": log.decision,
                "model": log.model,
                "model_version": log.model_version,
                "attempts": log.attempts,
                "latency_seconds": log.latency_seconds,
                "error": log.error,
                "timestamp": log.timestamp.isoformat(),
                "input_snapshot": log.input_snapshot,
                "output": log.output,
            }
            for log in logs
        ],
    }


@router.post("/decisions/{signal_id}/outcome")
async def report_outcome(
    signal_id: str,
    payload: TradeOutcomeRequest,
    repository: DecisionRepository = Depends(get_repository),
) -> dict:
    """The bot calls this once MT5 shows a position it opened has closed.
    Not restricted to APPROVE/MODIFY decisions or any particular
    approval_status -- record whatever the bot reports and let the reader
    (dashboard, analysis) make sense of it; this endpoint's job is only to
    capture the fact, not to judge whether it was expected."""
    row = await repository.get_by_signal_id(signal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="signal_id not found")
    outcome = await repository.record_outcome(
        signal_id=signal_id,
        profit=payload.profit,
        exit_reason=payload.exit_reason,
        ticket=payload.ticket,
        entry_price=payload.entry_price,
        close_price=payload.close_price,
        volume=payload.volume,
        closed_at=payload.closed_at,
    )
    logger.info(
        "trade outcome reported",
        extra={"signal_id": signal_id, "profit": payload.profit, "exit_reason": payload.exit_reason},
    )
    return {"signal_id": signal_id, "recorded": True, "outcome_id": outcome.id}


@router.post("/decisions/{signal_id}/approve")
async def approve_decision(
    signal_id: str,
    payload: HumanDecisionRequest,
    container: AppContainer = Depends(get_container),
    repository: DecisionRepository = Depends(get_repository),
) -> dict:
    """Human approval for a MANUAL-mode decision.

    The deterministic risk rules are re-applied at approval time: a human
    can decline what the AI approved, but neither a human nor the AI can
    approve a trade that violates the hard limits, and time has passed
    since the AI decision (section 32).
    """
    row = await repository.get_by_signal_id(signal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="signal_id not found")
    if row.approval_status != ApprovalStatus.AWAITING_HUMAN.value:
        raise HTTPException(
            status_code=409,
            detail=f"decision is not awaiting approval (status: {row.approval_status})",
        )

    signal = TradeSignal.model_validate(row.original_signal)
    # Always the bot's own signal: MODIFY was removed 2026-09-19, and rows
    # from before that carrying modified_trade are history, not instructions.
    guard = container.risk_service.final_guard(
        signal, AccountState(balance=None), market=None, original_signal=signal
    )
    if not guard.passed:
        logger.warning(
            "human approval blocked by risk guard",
            extra={"signal_id": signal_id, "violations": guard.violations},
        )
        raise HTTPException(
            status_code=409,
            detail={
                "message": "approval blocked by deterministic risk rules",
                "violations": guard.violations,
            },
        )

    updated = await repository.set_human_decision(
        row, ApprovalStatus.HUMAN_APPROVED, payload.reviewer, payload.note
    )
    logger.info(
        "decision approved by human",
        extra={"signal_id": signal_id, "reviewer": payload.reviewer},
    )
    return {
        "signal_id": updated.signal_id,
        "approval_status": updated.approval_status,
        "decision": updated.decision,
        "modified_trade": updated.modified_trade,
        "execute": updated.decision == FinalDecision.APPROVE.value,
    }


@router.post("/decisions/{signal_id}/reject")
async def reject_decision(
    signal_id: str,
    payload: HumanDecisionRequest,
    repository: DecisionRepository = Depends(get_repository),
) -> dict:
    row = await repository.get_by_signal_id(signal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="signal_id not found")
    if row.approval_status != ApprovalStatus.AWAITING_HUMAN.value:
        raise HTTPException(
            status_code=409,
            detail=f"decision is not awaiting approval (status: {row.approval_status})",
        )

    updated = await repository.set_human_decision(
        row, ApprovalStatus.HUMAN_REJECTED, payload.reviewer, payload.note
    )
    logger.info(
        "decision rejected by human",
        extra={"signal_id": signal_id, "reviewer": payload.reviewer},
    )
    return {
        "signal_id": updated.signal_id,
        "approval_status": updated.approval_status,
        "execute": False,
    }
