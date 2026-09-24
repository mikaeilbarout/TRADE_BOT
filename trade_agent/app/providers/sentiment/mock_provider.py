from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.sentiment import SentimentBundle, SentimentItem, SentimentKind, SourceQuality
from app.providers.sentiment.base import SentimentProvider


class MockSentimentProvider(SentimentProvider):
    """Deterministic, network-free sentiment source for tests and local dev.

    Returns a deliberately mixed, mostly-neutral set (including one
    low-quality social item) so that with no real provider configured the
    conservative default holds and the agent has something realistic to
    discount. Replace with a real SentimentProvider for production.
    """

    name = "mock"

    async def fetch(
        self, symbol: str, query_terms: list[str], lookback_minutes: int
    ) -> list[SentimentItem]:
        now = datetime.now(timezone.utc)
        return [
            SentimentItem(
                source="mock-desk-commentary",
                kind=SentimentKind.MARKET_COMMENTARY,
                quality=SourceQuality.HIGH,
                text=f"Desks report balanced two-way flow in {symbol} with no strong directional conviction.",
                timestamp=now - timedelta(minutes=18),
                score=0.05,
            ),
            SentimentItem(
                source="mock-positioning",
                kind=SentimentKind.POSITIONING,
                quality=SourceQuality.MEDIUM,
                text=f"Net positioning in {symbol} little changed week-over-week.",
                timestamp=now - timedelta(minutes=45),
                score=0.0,
            ),
            SentimentItem(
                source="mock-social",
                kind=SentimentKind.SOCIAL,
                quality=SourceQuality.LOW,
                text=f"Retail chatter on {symbol} is loud and directionally inconsistent.",
                timestamp=now - timedelta(minutes=6),
                score=None,
            ),
        ]


def empty_bundle(symbol: str) -> SentimentBundle:
    """Explicitly-empty bundle, used to make 'we have no sentiment data'
    visible to the agent rather than silently looking like neutral data."""
    return SentimentBundle(symbol=symbol, items=[], is_degraded=True,
                           degraded_reason="no sentiment sources returned data")
