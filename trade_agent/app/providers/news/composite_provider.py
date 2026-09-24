from __future__ import annotations

import asyncio

from app.models.news import NewsItem
from app.observability.logging import get_logger
from app.providers.news.base import NewsProvider, NewsProviderError

logger = get_logger(__name__)


class CompositeNewsProvider(NewsProvider):
    """Merges several NewsProviders into one bundle -- e.g. Finnhub's
    general headline coverage plus FRED's actual economic-release data,
    which cover genuinely different content (chatter vs. hard data prints)
    rather than duplicating each other.

    SOME providers failing does not fail the whole fetch -- fewer real
    sources beats none. But ALL of them failing raises NewsProviderError,
    because an empty list means something completely different downstream:
    NewsService treats a returned list as a successful fetch, so swallowing
    a total outage would build a fresh bundle with is_degraded=False, cache
    it, and tell the news agent "we checked, there is no news" when the
    truth is "we could not reach a single source". That silently disables
    both the degraded-cache fallback and the fail-closed path.
    """

    def __init__(self, providers: list[NewsProvider]) -> None:
        self._providers = providers
        self.name = "+".join(p.name for p in providers)

    async def fetch(self, query_terms: list[str], lookback_minutes: int) -> list[NewsItem]:
        results = await asyncio.gather(
            *(p.fetch(query_terms, lookback_minutes) for p in self._providers),
            return_exceptions=True,
        )
        items: list[NewsItem] = []
        failures: list[str] = []
        for provider, result in zip(self._providers, results):
            if isinstance(result, BaseException):
                failures.append(f"{provider.name}: {result}")
                continue
            items.extend(result)

        if failures and len(failures) == len(self._providers):
            raise NewsProviderError(
                "every news provider failed -- " + "; ".join(failures)
            )
        if failures:
            logger.warning(
                "some news providers failed; continuing with the rest",
                extra={"failed": failures, "surviving_item_count": len(items)},
            )
        return items
