"""Metadata endpoints for static catalogs."""
from fastapi import APIRouter, Depends

from ..dependencies import get_prediction_service
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.get("/diseases", summary="List supported diseases")
async def list_diseases(
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, list[str]]:
    """Return ocular, neurological, and combined disease catalogs."""
    return service.get_supported_diseases()
