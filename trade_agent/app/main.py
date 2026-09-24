from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import AppContainer
from app.api.routes import decisions, health, signals
from app.bootstrap import build_container
from app.config.settings import get_settings
from app.database.session import init_models
from app.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _make_lifespan(container: AppContainer | None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        active = container or build_container(get_settings())
        app.state.container = active
        await init_models(active.engine)
        logger.info(
            "service started",
            extra={
                "trading_mode": active.settings.trading_mode.value,
                "llm_provider": active.settings.llm_provider,
                "market_data_provider": active.settings.market_data_provider,
                "news_provider": active.settings.news_provider,
                "sentiment_provider": active.settings.sentiment_provider,
                "auth_enabled": bool(active.settings.internal_api_key),
            },
        )
        yield

    return lifespan


def create_app(container: AppContainer | None = None) -> FastAPI:
    """Build the FastAPI app. Pass `container` to inject a pre-wired
    AppContainer (e.g. with a MockLLMProvider) for tests instead of
    building one from live environment settings."""
    app = FastAPI(
        title="Multi-Agent Trading Decision System",
        description=(
            "AI decision-support layer that sits between a trading bot and "
            "trade execution. Never executes trades itself -- only "
            "APPROVE/REJECT/WAIT decisions for the bot to act on."
        ),
        version="0.2.0",
        lifespan=_make_lifespan(container),
    )
    app.include_router(health.router)
    app.include_router(signals.router)
    app.include_router(decisions.router)
    return app


app = create_app()
