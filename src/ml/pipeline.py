"""Screening pipeline orchestrating ocular and neurological models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torchvision import transforms

from .models import NeurologyBackbone, OcularBackbone
from .utils.explainability import generate_heatmap
from .utils.image import load_image_tensor


@dataclass
class ScreeningResult:
    disease: str
    probability: float
    triage_label: str
    heatmap_path: str | None = None


class ScreeningPipeline:
    """High-level orchestration of preprocessing, inference, and explainability."""

    def __init__(self, models_dir: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = Path(models_dir)
        self.ocular_model = OcularBackbone(num_classes=4).to(self.device).eval()
        self.neuro_model = NeurologyBackbone(num_classes=3).to(self.device).eval()
        self.ocular_diseases = [
            "Glaucoma",
            "Cataract",
            "Diabetic Retinopathy",
            "Hypertensive Retinopathy",
            "Dry Eye",
            "AMD",
        ]
        self.neuro_diseases = [
            "Optic Neuritis",
            "Neurodegeneration",
            "Stroke Risk",
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def _select_model(self, modality: str) -> torch.nn.Module:
        return self.ocular_model if modality.lower() in {"fundus", "retina", "oct"} else self.neuro_model

    def _supported_diseases(self, modality: str) -> Sequence[str]:
        return self.ocular_diseases if modality.lower() in {"fundus", "retina", "oct"} else self.neuro_diseases

    def get_supported_diseases(self) -> dict[str, List[str]]:
        ocular = list(self.ocular_diseases)
        neurological = list(self.neuro_diseases)
        combined = list({disease: None for disease in ocular + neurological}.keys())
        return {
            "ocular": ocular,
            "neurological": neurological,
            "all": combined,
        }

    def _triage_bucket(self, probability: float) -> str:
        if probability >= 0.85:
            return "critical"
        if probability >= 0.6:
            return "elevated"
        if probability >= 0.4:
            return "watch"
        return "baseline"

    def run_inference(
        self,
        patient_id: str,
        modality: str,
        diseases: Iterable[str] | None,
        image_path: str,
    ) -> List[ScreeningResult]:
        image_tensor = load_image_tensor(image_path, self.transform).to(self.device)
        model = self._select_model(modality)
        logits = model(image_tensor.unsqueeze(0)).detach().cpu().flatten()

        results: List[ScreeningResult] = []
        heatmap_output = self.models_dir / "heatmaps"
        heatmap_path = generate_heatmap(image_path, heatmap_output)

        target_diseases = list(diseases) if diseases else list(self._supported_diseases(modality))
        if not target_diseases:
            target_diseases = list(self._supported_diseases(modality))

        for idx, disease in enumerate(target_diseases):
            probability = float(logits[idx % logits.shape[0]])
            results.append(
                ScreeningResult(
                    disease=disease,
                    probability=probability,
                    triage_label=self._triage_bucket(probability),
                    heatmap_path=heatmap_path,
                )
            )
        return results
