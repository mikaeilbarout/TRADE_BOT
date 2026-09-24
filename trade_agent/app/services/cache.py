from __future__ import annotations

import json
import time
from typing import Protocol

from app.observability.logging import get_logger

logger = get_logger(__name__)


class BundleCache(Protocol):
    """Minimal cache contract used by NewsService/SentimentService. Values
    are JSON-serializable dicts (already-validated Pydantic dumps).

    Two different horizons, deliberately: `ttl_seconds` is how long an
    entry counts as FRESH (the callers compare it against the returned
    age themselves), while `retain_seconds` is how long the entry is kept
    at all, so a stale-but-recent bundle can still bridge a provider
    outage. Storing at the freshness TTL made the callers'
    `*_max_fallback_age_seconds` settings unreachable -- the entry was
    always gone before it could be used as a fallback.
    """

    async def get(self, key: str) -> tuple[float, dict] | None:
        """Return (age_seconds, payload) or None if absent."""

    async def set(
        self, key: str, payload: dict, ttl_seconds: int, retain_seconds: int | None = None
    ) -> None: ...


class InProcessCache:
    """Default cache: per-worker, no external dependency. Perfectly
    adequate for a single-process VPS deployment."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, dict, float]] = {}

    async def get(self, key: str) -> tuple[float, dict] | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, payload, retain_seconds = entry
        age = time.monotonic() - stored_at
        # Retention used to be ignored entirely here, so entries lived for
        # the life of the process -- a caller whose own fallback rule had
        # no age bound could serve an arbitrarily old bundle.
        if age > retain_seconds:
            self._store.pop(key, None)
            return None
        return age, payload

    async def set(
        self, key: str, payload: dict, ttl_seconds: int, retain_seconds: int | None = None
    ) -> None:
        self._store[key] = (
            time.monotonic(),
            payload,
            float(retain_seconds if retain_seconds is not None else ttl_seconds),
        )


class RedisCache:
    """Shared cache so multiple API workers reuse one news/sentiment fetch.

    Every Redis failure degrades to the in-process fallback instead of
    propagating: a cache outage must never turn into a trading decision
    failure, and the underlying provider fetch still happens either way.
    """

    def __init__(self, redis_url: str, fallback: InProcessCache | None = None) -> None:
        import redis.asyncio as redis_asyncio

        self._client = redis_asyncio.from_url(redis_url, decode_responses=True)
        self._fallback = fallback or InProcessCache()

    async def get(self, key: str) -> tuple[float, dict] | None:
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            logger.warning("redis cache read failed, using in-process cache", extra={"error": str(exc)})
            return await self._fallback.get(key)
        if raw is None:
            return None
        try:
            envelope = json.loads(raw)
            age = max(0.0, time.time() - float(envelope["stored_at"]))
            return age, envelope["payload"]
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("discarding malformed redis cache entry", extra={"error": str(exc)})
            return None

    async def set(
        self, key: str, payload: dict, ttl_seconds: int, retain_seconds: int | None = None
    ) -> None:
        envelope = json.dumps({"stored_at": time.time(), "payload": payload}, default=str)
        # Expire on the RETENTION horizon, not the freshness TTL: the
        # caller decides what is fresh from the returned age, and needs the
        # entry to still exist afterwards to use it as a degraded fallback.
        expiry = max(1, retain_seconds if retain_seconds is not None else ttl_seconds)
        try:
            await self._client.set(key, envelope, ex=expiry)
        except Exception as exc:
            logger.warning("redis cache write failed, using in-process cache", extra={"error": str(exc)})
            await self._fallback.set(key, payload, ttl_seconds, retain_seconds)


def build_cache(redis_url: str | None) -> BundleCache:
    if not redis_url:
        return InProcessCache()
    try:
        return RedisCache(redis_url)
    except Exception as exc:  # redis package missing or bad URL
        logger.warning(
            "could not initialize redis cache, falling back to in-process",
            extra={"error": str(exc)},
        )
        return InProcessCache()
