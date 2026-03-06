"""Async SQLAlchemy engine & session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from underwriting_suite.config import settings

engine = create_async_engine(settings.database_url, echo=(settings.app_env == "development"))
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


async def get_session() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency – yields an async session."""
    async with async_session() as session:
        yield session


async def init_db() -> None:
    """Create all tables (dev convenience; use Alembic in prod)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
