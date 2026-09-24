from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class InstrumentSpec(BaseModel):
    """Contract details needed to turn price moves into money. XAUUSD
    defaults match a standard 100oz gold CFD lot."""

    symbol: str = "XAUUSD"
    contract_size: float = 100.0  # units of base asset per 1.0 lot
    price_decimals: int = 3
    min_volume: float = 0.01
    volume_step: float = 0.01
    # Dukascopy stores integer points; XAUUSD ticks are in 1/1000 units.
    point_divisor: float = 1000.0


class CostModel(BaseModel):
    """Transaction costs applied identically to the baseline and the
    AI-filtered run -- a fair comparison requires this to be the same object
    for both (spec: same transaction costs)."""

    # Spread is taken from the tick data itself when available; this is the
    # fallback/minimum used when a bar has no recorded spread.
    fallback_spread_price: float = 0.30
    # Per-side slippage applied to entries and stop exits, in price units.
    slippage_price: float = 0.05
    # Commission per lot per side, in account currency.
    commission_per_lot_per_side: float = 3.5
    # Stops are assumed to fill worse than the trigger; targets at the level.
    stop_slippage_price: float = 0.10


class RiskModel(BaseModel):
    initial_balance: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    max_concurrent_positions: int = 1
    max_trades_per_day: int = 6
    # A trade whose computed size rounds below the instrument minimum is
    # skipped rather than silently up-sized.
    skip_if_below_min_volume: bool = True
    # How long an AI-modified limit entry rests before it is cancelled
    # unfilled. Finite by design: an order that waits indefinitely would let
    # the AI run book entries the live system would have long since dropped.
    modified_limit_expiry_bars: int = 8

    # --- rules the production bot enforces ------------------------------
    # Ported from scalp-sample-v2 so Experiment A reproduces the bot rather
    # than an idealised version of it. All three change WHICH trades exist,
    # so leaving them out would not be a small simplification.
    #
    # Close an open position after this long regardless of stop or target
    # (the bot's `time_stop_minutes`; its M15 profile uses 7 days). 0 disables.
    time_stop_minutes: int = 10080
    # After this many consecutive losses, stop taking new entries for
    # `cooldown_hours` of wall-clock time. The bot's notes record this as the
    # one change that survived walk-forward validation on both halves.
    cooldown_losses_to_trigger: int = 3
    cooldown_hours: float = 2.0
    # Stop trading for the rest of the UTC day once the day's loss reaches
    # this percentage of the day's opening equity. 100 disables it, which is
    # the bot's current live setting.
    max_daily_loss_pct: float = 100.0


class SplitConfig(BaseModel):
    """Chronological 70/30 methodology.

    The boundary is derived from the data's own first/last timestamp, never
    hard-coded, so it stays correct whatever range is actually imported.
    """

    development_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    # Bars of embargo dropped at the boundary so indicators warmed up on
    # development data cannot bleed their state into the first test trades.
    embargo_bars: int = 200


class BacktestConfig(BaseModel):
    symbol: str = "XAUUSD"
    timeframe_minutes: int = 15
    # The span of the supplied XAUUSD M15 export (xauusd_m15_5y.csv, 100,000
    # bars). Set from the data rather than aspirationally, so the readiness
    # report does not show four years of "missing" history that was never
    # available.
    start_date: date = date(2022, 6, 21)
    end_date: date = date(2026, 9, 11)

    instrument: InstrumentSpec = Field(default_factory=InstrumentSpec)
    costs: CostModel = Field(default_factory=CostModel)
    risk: RiskModel = Field(default_factory=RiskModel)
    split: SplitConfig = Field(default_factory=SplitConfig)

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    random_seed: int = 20240101

    @model_validator(mode="after")
    def check_dates(self) -> "BacktestConfig":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    @property
    def tick_dir(self) -> Path:
        return self.data_dir / "ticks" / self.symbol

    @property
    def candle_path(self) -> Path:
        return self.data_dir / "candles" / f"{self.symbol}_M{self.timeframe_minutes}.parquet"

    @property
    def seal_path(self) -> Path:
        return self.results_dir / "strategy_seal.json"
