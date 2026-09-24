from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.news import NewsItem


class NewsProviderError(Exception):
    """Raised on any upstream failure (network, auth, rate limit, malformed
    payload). NewsService treats this as 'no fresh news available' and the
    pipeline fails closed rather than assuming a quiet news environment."""


class NewsProvider(ABC):
    """Abstraction over wherever raw news actually comes from (a news API,
    an RSS aggregator, a paid data vendor...). Swap the concrete
    implementation without touching NewsService or the news agent.
    """

    name: str = "base"

    @abstractmethod
    async def fetch(
        self, query_terms: list[str], lookback_minutes: int
    ) -> list[NewsItem]:
        raise NotImplementedError
