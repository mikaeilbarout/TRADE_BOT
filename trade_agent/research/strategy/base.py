from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from app.models.enums import Side


class StrategySignal(BaseModel):
    """One signal, carrying everything the spec asks to be recorded per
    trade -- including WHY it fired and what the market looked like at that
    moment (needed for both the trade log and the AI agents' context)."""

    signal_id: str
    bar_index: int
    signal_time: datetime  # the bar close at which the signal was generated
    symbol: str
    side: Side
    entry: float
    stop_loss: float
    take_profit: float
    entry_reason: str
    market_conditions: dict = Field(default_factory=dict)

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_distance(self) -> float:
        return abs(self.take_profit - self.entry)

    @property
    def risk_reward_ratio(self) -> float:
        return self.reward_distance / self.risk_distance if self.risk_distance else 0.0


class Strategy(ABC):
    """A signal generator over a bar series.

    Contract: `generate` must be causal. A signal emitted at bar i may use
    only bars 0..i, and is executed at bar i+1's open by the engine -- so a
    strategy can never trade on the close of the bar that produced its own
    signal.
    """

    name: str = "base"

    @property
    @abstractmethod
    def params(self) -> dict:
        """The full parameter set, used for sealing and reproducibility."""

    @abstractmethod
    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Attach indicators. Must be causal."""

    @abstractmethod
    def generate(self, prepared: pd.DataFrame) -> list[StrategySignal]:
        """Emit signals over the prepared frame."""
