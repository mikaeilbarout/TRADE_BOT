from __future__ import annotations

from datetime import datetime, timezone

from app.config.assets import get_asset_meta
from app.config.settings import Settings
from app.models.sentiment import SentimentBundle, SentimentItem
from app.observability.logging import get_logger
from app.providers.sentiment.base import SentimentProvider, SentimentProviderError
from app.services.cache import BundleCache, InProcessCache
from app.services.clock import Clock, system_clock

logger = get_logger(__name__)


class SentimentUnavailableError(Exception):
    """Raised when no sentiment bundle could be produced at all (provider
    down and no usable cache). The pipeline fails closed on this rather
    than treating missing sentiment as neutral sentiment."""


def _dedup_key(item: SentimentItem) -> str:
    return f"{item.source}:{item.text.strip().lower()[:120]}"


class SentimentService:
    """Normalizes, deduplicates, and caches sentiment material for Agent 2.

    Mirrors NewsService's contract deliberately: one shared fetch per
    signal, freshness enforced here, and a short TTL cache. Source-quality
    labelling comes from the provider; this layer filters by freshness and
    removes duplicates so the agent weighs evidence, not noise.
    """

    def __init__(
        self,
        provider: SentimentProvider,
        settings: Settings,
        cache: BundleCache | None = None,
        clock: Clock = system_clock,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._cache = cache or InProcessCache()
        self._clock = clock

    def _cache_key(self, symbol: str) -> str:
        return f"sentiment:{self._provider.name}:{symbol.upper()}"

    async def get_sentiment(self, symbol: str) -> SentimentBundle:
        meta = get_asset_meta(symbol)
        query_terms = [meta.base_asset, symbol, *meta.relevant_factors]
        lookback = self._settings.sentiment_lookback_minutes
        key = self._cache_key(symbol)

        cached = await self._cache.get(key)
        if cached is not None:
            age_seconds, payload = cached
            if age_seconds < self._settings.sentiment_cache_ttl_seconds:
                return SentimentBundle.model_validate(payload)

        try:
            raw_items = await self._provider.fetch(symbol, query_terms, lookback)
        except SentimentProviderError as exc:
            # Age-bounded exactly like NewsService's fallback: without the
            # bound this served any surviving cache entry regardless of
            # age, i.e. it could hand the agent an hours-old sentiment
            # picture and label it merely "degraded".
            if cached is not None:
                age_seconds, payload = cached
                if age_seconds <= self._settings.sentiment_max_fallback_age_seconds:
                    logger.warning(
                        "sentiment provider failed; serving degraded cached bundle",
                        extra={
                            "symbol": symbol,
                            "cache_age_seconds": age_seconds,
                            "error": str(exc),
                        },
                    )
                    return SentimentBundle.model_validate(payload).model_copy(
                        update={
                            "is_degraded": True,
                            "degraded_reason": f"{exc} (serving cache {age_seconds:.0f}s old)",
                        }
                    )
            raise SentimentUnavailableError(str(exc)) from exc

        now = self._clock()
        deduped: dict[str, SentimentItem] = {}
        for item in raw_items:
            if item.age_minutes(now) > lookback:
                continue
            dedup_key = _dedup_key(item)
            existing = deduped.get(dedup_key)
            if existing is None or item.timestamp > existing.timestamp:
                deduped[dedup_key] = item

        items = sorted(deduped.values(), key=lambda i: i.timestamp, reverse=True)
        # A successful fetch that found nothing is NOT degraded -- degraded
        # means "the data you are looking at is unreliable" (a stale
        # fallback bundle, a provider that failed), not "there is nothing
        # to look at". This used to be `is_degraded=not items`, and because
        # gold sentiment is empty most of the time, that flag was nearly
        # always on -> the policy's degraded-inputs veto downgraded every
        # approval to WAIT. Across 200 live decisions that produced ZERO
        # APPROVE and turned all 8 of the final agent's MODIFYs into WAIT.
        # The agent still sees item_count=0 and its prompt already tells it
        # to report low confidence on an empty evidence base, which is the
        # honest way to express this.
        bundle = SentimentBundle(
            symbol=symbol.upper(),
            items=items,
            generated_at=now,
            lookback_minutes=lookback,
            sources_queried=[self._provider.name],
            is_degraded=False,
        )
        await self._cache.set(
            key,
            bundle.model_dump(mode="json"),
            self._settings.sentiment_cache_ttl_seconds,
            retain_seconds=max(
                self._settings.sentiment_cache_ttl_seconds,
                self._settings.sentiment_max_fallback_age_seconds,
            ),
        )
        return bundle
