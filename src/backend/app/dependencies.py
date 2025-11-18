"""Shared FastAPI dependencies."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings, get_settings
from .database import get_session
from .services.prediction_service import PredictionService


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


def get_prediction_service(settings: Settings = get_settings()) -> PredictionService:
    return PredictionService(settings=settings)
