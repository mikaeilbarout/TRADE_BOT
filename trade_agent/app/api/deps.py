from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.database.repository import DecisionRepository
from app.services.execution_service import ExecutionService
from app.services.pipeline import DecisionPipeline
from app.services.risk_service import RiskService
from app.services.single_agent_pipeline import SingleAgentPipeline


@dataclass
class AppContainer:
    """Everything built once at startup and shared across requests. Kept on
    app.state rather than as globals so tests can construct their own
    container (e.g. with a MockLLMProvider) without touching module state.
    """

    settings: Settings
    pipeline: DecisionPipeline | SingleAgentPipeline
    execution_service: ExecutionService
    risk_service: RiskService
    engine: object  # AsyncEngine
    session_factory: object  # async_sessionmaker[AsyncSession]


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    container: AppContainer = request.app.state.container
    async with container.session_factory() as session:
        yield session


async def get_repository(request: Request) -> AsyncIterator[DecisionRepository]:
    async for session in get_db_session(request):
        yield DecisionRepository(session)


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    """Shared-secret guard between the bot and this service.

    Disabled when INTERNAL_API_KEY is unset (the default for a private
    network / localhost deployment); enforced on every protected route the
    moment a key is configured, so enabling auth is one env var and not a
    code change.
    """
    container: AppContainer = request.app.state.container
    expected = container.settings.internal_api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )
