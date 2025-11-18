"""Prediction endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from ..dependencies import get_prediction_service
from ..schemas import PredictionCreate, PredictionResponse
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
async def create_prediction(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        body = await request.json()
    except Exception as exc:  # pragma: no cover - surfaced as HTTP 400
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if isinstance(body, dict) and "payload" in body and isinstance(body["payload"], dict):
        body = body["payload"]

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Prediction payload must be a JSON object")

    try:
        payload = PredictionCreate(**body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    return await service.predict(payload)
