from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models.market_data import Candle, Quote
from app.models.sentiment import SentimentItem, SentimentKind, SourceQuality
from app.providers.market_data.base import MarketDataProvider
from app.providers.market_data.mock_provider import MockMarketDataProvider
from app.providers.sentiment.base import SentimentProvider, SentimentProviderError
from app.providers.sentiment.mock_provider import MockSentimentProvider
from app.services.cache import InProcessCache, RedisCache
from app.services.market_data import MarketDataService, MarketDataUnavailableError
from app.services.sentiment_service import SentimentService, SentimentUnavailableError
from tests.conftest import make_settings


class _FrozenFeedProvider(MarketDataProvider):
    """Responds instantly but with an old feed timestamp -- the exact case
    that a fetch-duration-based staleness check would miss."""

    name = "frozen"

    def __init__(self, age_seconds: float) -> None:
        self._age = age_seconds

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(
            bid=99.9,
            ask=100.1,
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=self._age),
        )

    async def get_candles(self, symbol: str, timeframe: str, count: int = 200):
        return await MockMarketDataProvider().get_candles(symbol, timeframe, count)


class _EmptyCandlesProvider(MarketDataProvider):
    name = "empty"

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(bid=1.0, ask=1.1, timestamp=datetime.now(timezone.utc))

    async def get_candles(self, symbol: str, timeframe: str, count: int = 200) -> list[Candle]:
        return []


async def test_snapshot_marks_frozen_feed_as_stale():
    service = MarketDataService(
        _FrozenFeedProvider(age_seconds=300), make_settings(market_data_max_staleness_seconds=30)
    )
    snapshot = await service.get_snapshot("XAUUSD", ["H1", "M5"])
    assert snapshot.is_stale is True
    assert snapshot.freshness_seconds > 200
    assert snapshot.fetch_latency_seconds < 1.0  # it answered immediately


async def test_snapshot_is_fresh_for_live_feed():
    service = MarketDataService(MockMarketDataProvider(), make_settings())
    snapshot = await service.get_snapshot("XAUUSD", ["H4", "M15"])
    assert snapshot.is_stale is False
    assert snapshot.entry_timeframe == "M15"
    assert set(snapshot.indicators) == {"H4", "M15"}


async def test_snapshot_fails_closed_on_empty_candles():
    service = MarketDataService(_EmptyCandlesProvider(), make_settings())
    with pytest.raises(MarketDataUnavailableError):
        await service.get_snapshot("XAUUSD", ["H1"])


class _FailingSentimentProvider(SentimentProvider):
    name = "failing"

    async def fetch(self, symbol, query_terms, lookback_minutes):
        raise SentimentProviderError("vendor 503")


async def test_sentiment_service_fails_closed_with_no_cache():
    service = SentimentService(_FailingSentimentProvider(), make_settings())
    with pytest.raises(SentimentUnavailableError):
        await service.get_sentiment("XAUUSD")


async def test_sentiment_service_labels_quality_and_dedups():
    class DuplicatingProvider(SentimentProvider):
        name = "dup"

        async def fetch(self, symbol, query_terms, lookback_minutes):
            now = datetime.now(timezone.utc)
            item = SentimentItem(
                source="desk",
                kind=SentimentKind.MARKET_COMMENTARY,
                quality=SourceQuality.HIGH,
                text="Same note twice",
                timestamp=now - timedelta(minutes=5),
            )
            newer = item.model_copy(update={"timestamp": now})
            stale = SentimentItem(
                source="old",
                kind=SentimentKind.SOCIAL,
                quality=SourceQuality.LOW,
                text="Way outside the lookback window",
                timestamp=now - timedelta(hours=10),
            )
            return [item, newer, stale]

    service = SentimentService(DuplicatingProvider(), make_settings(sentiment_lookback_minutes=60))
    bundle = await service.get_sentiment("XAUUSD")
    assert len(bundle.items) == 1  # duplicate collapsed, stale item dropped
    assert bundle.items[0].quality == SourceQuality.HIGH
    assert bundle.low_quality_ratio == 0.0


async def test_sentiment_service_does_not_mark_an_empty_result_degraded():
    """An empty result is an empty evidence base, not unreliable data.

    This asserted the opposite until 2026-09-18. Because gold sentiment is
    empty most of the time, `is_degraded=not items` meant the flag was
    nearly always on, and the policy's degraded-inputs veto then downgraded
    every approval to WAIT -- across 200 live decisions, zero APPROVE and
    all 8 of the final agent's MODIFYs turned into WAIT. Degraded is
    reserved for genuinely unreliable data (a stale fallback bundle, see
    the test below); an empty bundle is communicated by item_count instead.
    """

    class EmptyProvider(SentimentProvider):
        name = "empty"

        async def fetch(self, symbol, query_terms, lookback_minutes):
            return []

    service = SentimentService(EmptyProvider(), make_settings())
    bundle = await service.get_sentiment("XAUUSD")
    assert bundle.items == []
    assert bundle.is_degraded is False
    assert bundle.degraded_reason is None


async def test_sentiment_service_serves_degraded_cache_on_failure():
    class FlakyProvider(SentimentProvider):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        async def fetch(self, symbol, query_terms, lookback_minutes):
            self.calls += 1
            if self.calls == 1:
                return await MockSentimentProvider().fetch(symbol, query_terms, lookback_minutes)
            raise SentimentProviderError("temporary outage")

    service = SentimentService(FlakyProvider(), make_settings(sentiment_cache_ttl_seconds=0))
    first = await service.get_sentiment("XAUUSD")
    assert not first.is_degraded

    second = await service.get_sentiment("XAUUSD")
    assert second.is_degraded is True
    assert "temporary outage" in second.degraded_reason


async def test_in_process_cache_roundtrip():
    cache = InProcessCache()
    assert await cache.get("missing") is None
    await cache.set("k", {"a": 1}, ttl_seconds=60)
    entry = await cache.get("k")
    assert entry is not None
    age, payload = entry
    assert payload == {"a": 1}
    assert age >= 0


async def test_redis_cache_falls_back_when_redis_unreachable():
    """A Redis outage must degrade to the in-process cache, never surface as
    a decision failure."""

    class BrokenClient:
        async def get(self, key):
            raise ConnectionError("redis down")

        async def set(self, key, value, ex=None):
            raise ConnectionError("redis down")

    cache = RedisCache.__new__(RedisCache)  # bypass __init__'s redis import
    cache._client = BrokenClient()
    cache._fallback = InProcessCache()

    await cache.set("k", {"a": 2}, ttl_seconds=60)
    entry = await cache.get("k")
    assert entry is not None
    assert entry[1] == {"a": 2}


async def test_build_cache_without_redis_url_is_in_process():
    from app.services.cache import build_cache

    assert isinstance(build_cache(None), InProcessCache)
