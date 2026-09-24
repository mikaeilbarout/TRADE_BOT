from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.enums import Bias, ImpactLevel


class NewsItem(BaseModel):
    title: str
    source: str
    url: str | None = None
    timestamp: datetime
    summary: str = ""
    impact: ImpactLevel = ImpactLevel.LOW
    bias: Bias = Bias.NEUTRAL
    category: str = "general"
    is_breaking: bool = False

    def age_minutes(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 60.0)


class NewsBundle(BaseModel):
    """Deduplicated, normalized news for one symbol, ready for Agent 1."""

    symbol: str
    items: list[NewsItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lookback_minutes: int = 60
    sources_queried: list[str] = Field(default_factory=list)
    is_degraded: bool = False
    degraded_reason: str | None = None
