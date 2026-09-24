from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import Side


class Trade(BaseModel):
    """One executed trade. Carries every field the spec asks to record, plus
    the AI decision chain when the trade came from the AI-filtered run."""

    trade_id: str
    signal_id: str
    symbol: str
    direction: Side

    # Entry
    entry_time: datetime
    entry_price: float
    requested_entry: float
    entry_slippage: float

    # Protection
    stop_loss: float
    take_profit: float

    # Exit
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None  # STOP_LOSS | TAKE_PROFIT | END_OF_DATA

    # Sizing and result
    volume: float = 0.0
    gross_profit: float = 0.0
    commission: float = 0.0
    profit: float = 0.0  # net, after costs
    r_multiple: float = 0.0
    planned_risk_reward: float = 0.0
    risk_amount: float = 0.0
    duration_minutes: float | None = None
    bars_held: int = 0

    # Context
    entry_reason: str = ""
    market_conditions: dict = Field(default_factory=dict)

    # Equity tracking
    balance_before: float = 0.0
    balance_after: float = 0.0

    # Excursion: the best and worst the trade ever looked, in R. Recorded
    # for executed trades as well as counterfactuals, because "the winner
    # that first went 1.5R against us" and "the winner that never traded
    # below entry" are not the same trade.
    max_favorable_excursion_r: float | None = None
    max_adverse_excursion_r: float | None = None

    # AI-run provenance (empty on the baseline run)
    ai_decision: str | None = None
    ai_confidence: float | None = None
    ai_reason: str | None = None
    ai_reason_codes: list[str] = Field(default_factory=list)
    ai_deciding_rule: str | None = None
    ai_weighted_score: float | None = None
    ai_cost_usd: float = 0.0
    was_modified: bool = False
    modified_fill_kind: str | None = None  # NEXT_BAR_OPEN | LIMIT_TOUCH
    original_entry: float | None = None
    original_stop_loss: float | None = None
    original_take_profit: float | None = None
    agent_chain: dict = Field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.profit > 0

    @property
    def is_loss(self) -> bool:
        return self.profit < 0


class SkippedSignal(BaseModel):
    """A signal that never became a trade, and exactly why.

    Kept because the comparison depends on it: to ask "would the trades the
    AI rejected have been profitable?", the rejected signals must be
    recorded alongside their counterfactual outcome.
    """

    signal_id: str
    signal_time: datetime
    symbol: str
    direction: Side
    entry: float
    stop_loss: float
    take_profit: float
    skip_reason: str
    # "risk_engine" | "ai_pipeline" | "engine_constraint" | "execution"
    # | "no_decision"
    skipped_by: str
    ai_decision: str | None = None
    ai_confidence: float | None = None
    ai_reason_codes: list[str] = Field(default_factory=list)
    ai_deciding_rule: str | None = None
    # Which agent (or deterministic component) is answerable for the
    # non-execution. Recorded at decision time so "which agent caused the
    # most false rejections" is a groupby, not text mining.
    blocking_agent: str | None = None
    ai_cost_usd: float = 0.0
    agent_chain: dict = Field(default_factory=dict)

    # --- counterfactual: what this trade WOULD have done if taken ---------
    # Computed on the same engine, with the same fill assumptions, on a fixed
    # notional balance so it can never touch the real equity curve of either
    # experiment. Diagnostic only.
    counterfactual_available: bool = False
    counterfactual_unavailable_reason: str | None = None
    counterfactual_entry_time: datetime | None = None
    counterfactual_entry_price: float | None = None
    counterfactual_exit_time: datetime | None = None
    counterfactual_exit_price: float | None = None
    counterfactual_r: float | None = None
    counterfactual_profit: float | None = None
    counterfactual_exit_reason: str | None = None
    counterfactual_is_win: bool | None = None
    counterfactual_mfe_r: float | None = None
    counterfactual_mae_r: float | None = None
    counterfactual_volume: float | None = None
    counterfactual_bars_held: int | None = None
