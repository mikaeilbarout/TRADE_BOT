from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.enums import Bias, ImpactLevel
from app.models.news import NewsItem
from app.providers.news.base import NewsProvider


class MockNewsProvider(NewsProvider):
    """Deterministic, network-free news source for tests and local dev.

    Returns a small, plausible, NEUTRAL/LOW-impact news set by default so
    that without real news configured, the system's conservative default
    (no strong bias either way) holds. Replace with a real NewsProvider
    (NewsAPIProvider or your own vendor integration) for production use.
    """

    name = "mock"

    async def fetch(
        self, query_terms: list[str], lookback_minutes: int
    ) -> list[NewsItem]:
        now = datetime.now(timezone.utc)
        term = query_terms[0] if query_terms else "market"
        return [
            NewsItem(
                title=f"Markets steady as traders await catalysts related to {term}",
                source="mock-wire",
                timestamp=now - timedelta(minutes=25),
                summary="No major scheduled events reported in the lookback window.",
                impact=ImpactLevel.LOW,
                bias=Bias.NEUTRAL,
                category="general",
                is_breaking=False,
            )
        ]
