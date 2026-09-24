from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config.settings import Settings
from app.database.models import Base


def build_engine(settings: Settings):
    is_sqlite_memory = "sqlite" in settings.database_url and ":memory:" in settings.database_url
    if is_sqlite_memory:
        # In-memory SQLite is per-connection; without a shared static pool
        # every checkout would see an empty database (fine for prod
        # Postgres, but required for tests/local dev using :memory:).
        return create_async_engine(
            settings.database_url,
            echo=False,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    return create_async_engine(settings.database_url, echo=False, future=True)


def build_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session
