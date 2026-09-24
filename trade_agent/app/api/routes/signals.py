from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import AppContainer, get_container, get_repository, require_api_key
from app.database.repository import DecisionRepository
from app.models.enums import Side
from app.models.signal import TradeSignal
from app.observability.logging import get_logger
from app.services.risk_service import AccountState

router = APIRouter(prefix="/api/v1", tags=["signals"], dependencies=[Depends(require_api_key)])
logger = get_logger(__name__)


class AccountStatePayload(BaseModel):
    """Deterministic account facts the bot supplies -- never credentials
    (section 31). All optional; omitted fields default conservatively.

    `balance` is required to enforce MAX_RISK_PER_TRADE_PCT; when it is
    omitted and REQUIRE_ACCOUNT_BALANCE is true (the default) the signal is
    rejected rather than having that limit silently skipped.
    """

    balance: float | None = None
    daily_loss_pct: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    exposure_by_asset_pct: dict[str, float] = Field(default_factory=dict)
    current_leverage: float = 0.0
    market_open: bool = True
    recent_loss_streak: int = 0


class TradeSignalRequest(BaseModel):
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: Side
    entry: float
    stop_loss: float
    take_profit: float
    volume: float
    timeframe: str = "M5"
    strategy: str = "unknown"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    account: AccountStatePayload = Field(default_factory=AccountStatePayload)


class TradeSignalResponse(BaseModel):
    """The API contract's core fields, plus mode-specific extras (e.g.
    SHADOW's ai_decision/ai_confidence, MANUAL's requires_human_approval,
    PAPER's hypothetical_execution) that ExecutionService attaches
    depending on TRADING_MODE."""

    model_config = {"extra": "allow"}

    signal_id: str
    decision: str
    confidence: float
    reason: str
    modified_trade: dict | None = None
    trading_mode: str


@router.post("/trade-signal", response_model=TradeSignalResponse)
async def submit_trade_signal(
    payload: TradeSignalRequest,
    container: AppContainer = Depends(get_container),
    repository: DecisionRepository = Depends(get_repository),
) -> dict:
    try:
        signal = TradeSignal(
            signal_id=payload.signal_id,
            symbol=payload.symbol,
            side=payload.side,
            entry=payload.entry,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            volume=payload.volume,
            timeframe=payload.timeframe,
            strategy=payload.strategy,
            timestamp=payload.timestamp,
        )
    except ValueError as exc:
        logger.warning(
            "rejected malformed signal",
            extra={"signal_id": payload.signal_id, "symbol": payload.symbol, "error": str(exc)},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    account = AccountState(
        balance=payload.account.balance,
        daily_loss_pct=payload.account.daily_loss_pct,
        trades_today=payload.account.trades_today,
        open_positions=payload.account.open_positions,
        exposure_by_asset_pct=payload.account.exposure_by_asset_pct,
        current_leverage=payload.account.current_leverage,
        market_open=payload.account.market_open,
        recent_loss_streak=payload.account.recent_loss_streak,
    )

    result = await container.pipeline.run(signal, account)
    response = container.execution_service.finalize(result)
    await repository.record(result, response)
    return response
