from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.models.enums import Side


class ModifiedTrade(BaseModel):
    """A trade with parameters adjusted by an agent (entry/SL/TP/size).

    Validated exactly as strictly as an incoming signal: an LLM proposing
    a BUY with its stop above entry is a schema violation, not something
    the execution guard should have to catch downstream via absolute
    distances.
    """

    symbol: str
    side: Side
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    volume: float | None = Field(default=None, gt=0)
    order_type: str = "LIMIT"
    delay_seconds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_sl_tp_direction(self) -> "ModifiedTrade":
        if self.side == Side.BUY:
            if not (self.stop_loss < self.entry < self.take_profit):
                raise ValueError(
                    "BUY modification requires stop_loss < entry < take_profit"
                )
        else:
            if not (self.take_profit < self.entry < self.stop_loss):
                raise ValueError(
                    "SELL modification requires take_profit < entry < stop_loss"
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
