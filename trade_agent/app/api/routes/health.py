from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import AppContainer, get_container

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(container: AppContainer = Depends(get_container)) -> dict:
    """Unauthenticated liveness probe. Reports configuration only -- never
    secrets or key material."""
    s = container.settings
    return {
        "status": "ok",
        "trading_mode": s.trading_mode.value,
        "llm_provider": s.llm_provider,
        "market_data_provider": s.market_data_provider,
        "news_provider": s.news_provider,
        "sentiment_provider": s.sentiment_provider,
        "auth_enabled": bool(s.internal_api_key),
        "cache": "redis" if s.redis_url else "in_process",
    }
