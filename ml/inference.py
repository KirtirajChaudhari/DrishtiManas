"""Load the trained model bundle and classify images. Depends only on NumPy + Pillow."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES
from .features import FeatureExtractor, colourfulness, preprocess_image
from .nn.mlp import MLP, MLPConfig

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.npz"


def save_bundle(path: str | Path, model: MLP, extractor: FeatureExtractor, meta: dict) -> None:
    np.savez_compressed(
        path,
        config=json.dumps(asdict(model.config)),
        feature_kind=extractor.kind,
        feat_mean=extractor.mean.astype(np.float32),
        feat_std=extractor.std.astype(np.float32),
        meta=json.dumps(meta),
        **model.state_dict(),
    )


class OCTClassifier:
    def __init__(self, path: str | Path = MODEL_PATH) -> None:
        data = np.load(path, allow_pickle=False)
        self.model = MLP(MLPConfig(**json.loads(str(data["config"]))))
        self.model.load_state_dict({k: data[k] for k in data.files if k[0] in "Wb" and k[1:].isdigit()})
        self.extractor = FeatureExtractor(
            kind=str(data["feature_kind"]), mean=data["feat_mean"], std=data["feat_std"]
        )
        self.meta = json.loads(str(data["meta"]))
        self.classes = CLASSES

    def predict_image(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        pixels = preprocess_image(image)
        features = self.extractor.transform(pixels[None])
        probs = self.model.predict_proba(features)[0]
        top = int(probs.argmax())

        warnings = []
        if colourfulness(image) > 12:
            warnings.append(
                "This image is in colour. The model was trained on grayscale retinal OCT B-scans, "
                "so predictions for other image types are not meaningful."
            )
        if probs[top] < 0.5:
            warnings.append("Low confidence: the model is unsure between several classes.")

        return {
            "prediction": {**self.classes[top], "probability": float(probs[top])},
            "probabilities": [
                {**c, "probability": float(p)} for c, p in zip(self.classes, probs.tolist())
            ],
            "input_pixels": np.round(pixels * 255).astype(int).tolist(),
            "warnings": warnings,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
