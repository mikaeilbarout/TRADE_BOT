from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.sentiment import SentimentItem


class SentimentProviderError(Exception):
    """Raised on any upstream failure. SentimentService treats this as 'no
    sentiment data available' and the pipeline fails closed rather than
    assuming neutral sentiment."""


class SentimentProvider(ABC):
    """Abstraction over wherever sentiment material comes from (a social
    listening API, a fear/greed index, positioning data, a commentary
    feed). Implementations are responsible for labelling each item's
    source quality honestly -- that labelling is what lets the agent
    discount noise and possible manipulation."""

    name: str = "base"

    @abstractmethod
    async def fetch(
        self, symbol: str, query_terms: list[str], lookback_minutes: int
    ) -> list[SentimentItem]:
        raise NotImplementedError
