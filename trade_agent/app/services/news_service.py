from __future__ import annotations

from datetime import datetime, timezone

from app.config.assets import get_asset_meta
from app.config.settings import Settings
from app.models.news import NewsBundle, NewsItem
from app.observability.logging import get_logger
from app.providers.news.base import NewsProvider, NewsProviderError
from app.services.cache import BundleCache, InProcessCache
from app.services.clock import Clock, system_clock

logger = get_logger(__name__)


class NewsUnavailableError(Exception):
    """Raised when no valid news bundle could be produced at all (provider
    down and no usable cache). The pipeline fails closed on this."""


def _dedup_key(item: NewsItem) -> str:
    return item.title.strip().lower()[:120]


class NewsService:
    """Normalizes, deduplicates, timestamps, and caches news for Agent 1.

    A short-TTL cache stops repeated signals on one symbol from hammering
    the provider. If the provider fails, the last-known-good bundle can
    bridge the gap -- but only while it is still recent enough to be worth
    deciding on (news_max_fallback_age_seconds); past that the service
    fails closed rather than reasoning about ancient headlines.
    """

    def __init__(
        self,
        provider: NewsProvider,
        settings: Settings,
        cache: BundleCache | None = None,
        clock: Clock = system_clock,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._cache = cache or InProcessCache()
        self._clock = clock

    def _cache_key(self, symbol: str) -> str:
        return f"news:{self._provider.name}:{symbol.upper()}"

    async def get_news(self, symbol: str) -> NewsBundle:
        meta = get_asset_meta(symbol)
        query_terms = [meta.base_asset, symbol, *meta.relevant_factors]
        lookback = self._settings.news_lookback_minutes
        key = self._cache_key(symbol)

        cached = await self._cache.get(key)
        if cached is not None:
            age_seconds, payload = cached
            if age_seconds < self._settings.news_cache_ttl_seconds:
                return NewsBundle.model_validate(payload)

        try:
            raw_items = await self._provider.fetch(query_terms, lookback)
        except NewsProviderError as exc:
            if cached is not None:
                age_seconds, payload = cached
                if age_seconds <= self._settings.news_max_fallback_age_seconds:
                    logger.warning(
                        "news provider failed; serving degraded cached bundle",
                        extra={"symbol": symbol, "cache_age_seconds": age_seconds, "error": str(exc)},
                    )
                    return NewsBundle.model_validate(payload).model_copy(
                        update={
                            "is_degraded": True,
                            "degraded_reason": f"{exc} (serving cache {age_seconds:.0f}s old)",
                        }
                    )
            raise NewsUnavailableError(str(exc)) from exc

        now = self._clock()
        deduped: dict[str, NewsItem] = {}
        for item in raw_items:
            if item.age_minutes(now) > lookback:
                continue
            item.is_breaking = item.is_breaking or item.age_minutes(now) <= 15
            dedup_key = _dedup_key(item)
            existing = deduped.get(dedup_key)
            if existing is None or item.timestamp > existing.timestamp:
                deduped[dedup_key] = item

        bundle = NewsBundle(
            symbol=symbol.upper(),
            items=sorted(deduped.values(), key=lambda i: i.timestamp, reverse=True),
            generated_at=now,
            lookback_minutes=lookback,
            sources_queried=[self._provider.name],
            is_degraded=False,
        )
        await self._cache.set(
            key,
            bundle.model_dump(mode="json"),
            self._settings.news_cache_ttl_seconds,
            # Keep it past its freshness window so the degraded-fallback
            # branch above has something to serve when the provider is down.
            retain_seconds=max(
                self._settings.news_cache_ttl_seconds,
                self._settings.news_max_fallback_age_seconds,
            ),
        )
        return bundle
