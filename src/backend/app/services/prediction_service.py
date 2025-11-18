"""Business logic bridging API and ML pipeline."""
from __future__ import annotations

from base64 import b64decode
from datetime import datetime
from importlib import import_module
from pathlib import Path
import sys
from typing import List, Optional
from uuid import uuid4

from ..schemas import Heatmap, PredictionCreate, PredictionResponse, PredictionResult


class PredictionService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.pipeline = self._load_pipeline()
        self.uploads_dir = Path(self.settings.uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def get_supported_diseases(self) -> dict[str, List[str]]:
        return self.pipeline.get_supported_diseases()

    def _load_pipeline(self):
        try:
            module = import_module("src.ml.pipeline")
        except ModuleNotFoundError:  # pragma: no cover - fallback path for ad-hoc runs
            project_root = Path(__file__).resolve().parents[4]
            if str(project_root) not in sys.path:
                sys.path.append(str(project_root))
            module = import_module("src.ml.pipeline")
        pipeline_cls = getattr(module, "ScreeningPipeline")
        return pipeline_cls(models_dir=self.settings.models_dir)

    async def predict(self, payload: PredictionCreate) -> PredictionResponse:
        image_path = self._resolve_image_path(payload)

        results = self.pipeline.run_inference(
            patient_id=payload.patient_id,
            modality=payload.modality,
            diseases=payload.diseases,
            image_path=image_path,
        )

        response_results = [
            PredictionResult(
                disease=item.disease,
                probability=item.probability,
                triage=item.triage_label,
                heatmap=Heatmap(url=item.heatmap_path, caption="Grad-CAM"),
            )
            for item in results
        ]

        return PredictionResponse(
            id=payload.patient_id,
            patient_id=payload.patient_id,
            created_at=datetime.utcnow(),
            results=response_results,
        )

    def _resolve_image_path(self, payload: PredictionCreate) -> str:
        if payload.image_base64:
            persisted = self._persist_image_from_base64(
                payload.image_base64,
                payload.image_filename,
                payload.patient_id,
            )
            if persisted:
                return persisted
        return payload.image_path

    def _persist_image_from_base64(
        self,
        data_uri: str,
        provided_filename: Optional[str],
        patient_id: str,
    ) -> Optional[str]:
        if not data_uri:
            return None

        header, _, encoded = data_uri.partition(",")
        # Handle plain base64 without data URI header
        if not encoded:
            encoded = header
            header = ""

        extension = self._infer_extension(header, provided_filename)
        filename = self._sanitize_filename(provided_filename, patient_id, extension)
        destination = self.uploads_dir / filename

        try:
            destination.write_bytes(b64decode(encoded))
        except (ValueError, OSError) as exc:
            raise ValueError("Invalid base64 payload for imaging study") from exc

        return str(destination)

    @staticmethod
    def _infer_extension(header: str, provided_filename: Optional[str]) -> str:
        if provided_filename:
            suffix = Path(provided_filename).suffix
            if suffix:
                return suffix

        if header.startswith("data:image/"):
            try:
                mime = header.split(";", 1)[0].split("/", 1)[1]
                return f".{mime}"
            except (IndexError, ValueError):
                pass

        return ".png"

    @staticmethod
    def _sanitize_filename(provided_filename: Optional[str], patient_id: str, extension: str) -> str:
        if provided_filename:
            candidate = Path(provided_filename).name
            if candidate:
                return candidate
        token = uuid4().hex
        safe_patient_id = "".join(ch for ch in patient_id if ch.isalnum() or ch in ("-", "_"))
        prefix = safe_patient_id or "capture"
        return f"{prefix}-{token}{extension}"
