"""Liveness and readiness endpoints."""
from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/ping", response_model=HealthResponse)
async def ping() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")
