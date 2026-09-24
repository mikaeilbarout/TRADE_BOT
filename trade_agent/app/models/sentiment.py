from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceQuality(StrEnum):
    """How much weight the sentiment agent should give a source. The data
    layer assigns this from the source's identity (established financial
    media vs. anonymous social post) -- it is deliberately NOT the agent's
    job to guess provenance, only to weigh it (section 5)."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SentimentKind(StrEnum):
    NEWS_HEADLINE = "news_headline"
    MARKET_COMMENTARY = "market_commentary"
    SOCIAL = "social"
    POSITIONING = "positioning"
    FEAR_GREED = "fear_greed"
    INSTITUTIONAL = "institutional"


class SentimentItem(BaseModel):
    source: str
    kind: SentimentKind
    quality: SourceQuality
    text: str
    timestamp: datetime
    # Optional pre-computed score from the vendor (e.g. a fear/greed index
    # reading). None means "no numeric reading, judge from the text".
    score: float | None = Field(default=None, ge=-1.0, le=1.0)

    def age_minutes(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        ts = self.timestamp if self.timestamp.tzinfo else self.timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 60.0)


class SentimentBundle(BaseModel):
    """Deduplicated, quality-tagged sentiment material for one symbol,
    ready for Agent 2. This is the agent's OWN evidence base -- it exists
    so the sentiment agent analyzes real source material rather than
    paraphrasing the news agent's conclusion."""

    symbol: str
    items: list[SentimentItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lookback_minutes: int = 60
    sources_queried: list[str] = Field(default_factory=list)
    is_degraded: bool = False
    degraded_reason: str | None = None

    @property
    def low_quality_ratio(self) -> float:
        if not self.items:
            return 0.0
        low = sum(1 for i in self.items if i.quality == SourceQuality.LOW)
        return low / len(self.items)
