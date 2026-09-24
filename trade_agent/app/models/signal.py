from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.enums import Side


class TradeSignal(BaseModel):
    """Normalized representation of a signal emitted by the trading bot.

    This is the contract at the API boundary (POST /api/v1/trade-signal).
    The bot must not need to know anything about agents, prompts, or LLMs.
    """

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: Side
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    volume: float = Field(gt=0)
    timeframe: str = "M5"
    strategy: str = "unknown"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def check_sl_tp_direction(self) -> "TradeSignal":
        if self.side == Side.BUY:
            if not (self.stop_loss < self.entry < self.take_profit):
                raise ValueError(
                    "BUY signal requires stop_loss < entry < take_profit"
                )
        else:
            if not (self.take_profit < self.entry < self.stop_loss):
                raise ValueError(
                    "SELL signal requires take_profit < entry < stop_loss"
                )
        return self

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_distance(self) -> float:
        return abs(self.take_profit - self.entry)

    @property
    def risk_reward_ratio(self) -> float:
        if self.risk_distance == 0:
            return 0.0
        return self.reward_distance / self.risk_distance

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()
