"""Prediction payloads and response models."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Heatmap(BaseModel):
    url: Optional[str] = None
    caption: Optional[str] = None


class PredictionCreate(BaseModel):
    """
    Payload expected from the frontend when running screening.

    This is aligned with `PredictionPayload` built in `UploadForm.tsx`.
    """
    # Forbid unknown keys so we catch typos, but *do* define
    # all the fields the UI sends.
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., description="Patient identifier (MRN / UID)")
    modality: str = Field(..., description="Imaging modality e.g. fundus, oct, slit_lamp")

    # Optional: the set of diseases to evaluate. Frontend currently does not send this,
    # so we default to an empty list and let the service fill it with all supported diseases.
    diseases: List[str] = Field(
        default_factory=list,
        description="Subset of diseases to score; empty = use full catalog",
    )

    # Image info produced by UploadForm
    image_path: Optional[str] = Field(
        default=None,
        description="Logical path or label for the uploaded/captured image",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (data URL or raw base64)",
    )
    image_filename: Optional[str] = Field(
        default=None,
        description="Original filename of the uploaded/captured image",
    )


class PredictionResult(BaseModel):
    disease: str
    probability: float
    triage: str
    heatmap: Heatmap


class PredictionResponse(BaseModel):
    # Allow building from ORM objects
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    created_at: datetime
    results: List[PredictionResult]
