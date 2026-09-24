from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class Quote(BaseModel):
    """A tick from the feed, carrying the feed's OWN timestamp.

    Staleness must be judged from when the venue produced the price, not
    from how long our fetch took -- a frozen or lagging feed returns
    instantly and would otherwise look perfectly fresh.
    """

    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class TimeframeSeries(BaseModel):
    timeframe: str
    candles: list[Candle]

    @property
    def closes(self) -> list[float]:
        return [c.close for c in self.candles]


class TechnicalIndicators(BaseModel):
    timeframe: str
    ema_50: float | None = None
    ema_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    adx_14: float | None = None  # trend STRENGTH (0-100), independent of direction
    recent_high: float | None = None
    recent_low: float | None = None
    trend: str | None = None  # UPTREND / DOWNTREND / RANGE


class MarketSnapshot(BaseModel):
    """Normalized, shared market view produced by MarketDataService."""

    symbol: str
    bid: float
    ask: float
    last_price: float
    spread: float
    session: str  # ASIA / LONDON / NEW_YORK / OVERLAP / CLOSED
    atr: float | None = None
    volatility_pct: float | None = None
    timeframes: dict[str, TimeframeSeries] = Field(default_factory=dict)
    indicators: dict[str, TechnicalIndicators] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    # When the feed says the quote was produced (not when we fetched it).
    quote_timestamp: datetime | None = None
    is_stale: bool = False
    freshness_seconds: float = 0.0
    fetch_latency_seconds: float = 0.0
    entry_timeframe: str | None = None
